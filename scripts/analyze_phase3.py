"""Derive Phase 3 decision quantities: rungs 4 (H2O-style) and 5 (Quest-style) on the Phase 2 sweep.

Reads   results/phase3/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase3/budget_speed/run_*/metrics.json  (decode latency, fast kernel)
        results/phase2/policy_quality/metrics.json     (cross-phase repeatability check, if present)
Writes  results/phase3/analysis/metrics.json

Aggregation rules are in harness/sweep_analysis.py. Policy differences are paired: every condition
decodes the same prompts from the same prefill, so a difference is bootstrapped over per-prompt
score differences rather than read off two independent intervals.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci  # noqa: E402
from harness.sweep_analysis import across, cond_rows, headline, label, niah_table, per_run, span, speed_table, teacher_forced_table  # noqa: E402

REFERENCE = "window_sink"  # Phase 2's best policy, re-measured in the Phase 3 run


def prompt_key(r: dict[str, Any]) -> tuple[str, float, int]:
    return (r["kind"], r["depth"], r["sample"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="policy_quality")
    p.add_argument("--speed", default="budget_speed")
    args = p.parse_args()
    base = RESULTS_DIR / "phase3"
    q = json.loads((base / args.quality / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / args.speed).glob("run_*/metrics.json"))]
    cfg = q["config"]
    policies = cfg["policies"]
    budgets = sorted(cfg["budgets"])
    depths = cfg["niah"]["depths"]
    bs = cfg["block_size"]

    niah, full_scores = niah_table(q)
    full_acc = statistics.mean(full_scores.values())
    teacher_forced = teacher_forced_table(q)
    speed = speed_table(runs)
    head = headline(niah, speed, policies)

    # ---- paired differences against the reference policy, per budget -------------------------
    scores = {(r["policy"], r["budget"]) + prompt_key(r): r["score"] for r in q["niah"]}
    paired: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        if policy == REFERENCE:
            continue
        rows = []
        for b in budgets:
            diffs = [scores[(policy, b) + k] - scores[(REFERENCE, b) + k] for k in full_scores]
            rows.append({
                "budget": b,
                "accuracy_minus_reference": statistics.mean(diffs),
                "ci95": bootstrap_mean_ci(diffs),
                "prompts_better": sum(1 for d in diffs if d > 0),
                "prompts_worse": sum(1 for d in diffs if d < 0),
            })
        paired[policy] = rows

    # ---- H2O: did accumulated mass keep the sink it is not forced to keep? ---------------------
    num_layers: int | None = None
    h2o_rows = [r for r in q["niah"] if r["policy"] == "h2o"]
    h2o = {}
    if h2o_rows:
        # The 100% pool keeps block 0 in every layer, so its count is the layer count.
        n_layers = max(r["layers_with_block0_resident"] for r in q["niah"] if r["policy"] == "block_full")
        num_layers = n_layers
        h2o = {
            "layers": n_layers,
            "block0_resident_layers_min_by_budget": {f"{b:g}": min(r["layers_with_block0_resident"] for r in h2o_rows if r["budget"] == b) for b in budgets},
            "boundary_evicted_blocks_median_by_budget": {f"{b:g}": statistics.median(r["boundary_evicted_blocks"] for r in h2o_rows if r["budget"] == b) for b in budgets},
            "decode_evictions_median_by_budget": {f"{b:g}": statistics.median(r["evictions"] for r in h2o_rows if r["budget"] == b) for b in budgets},
        }

    # ---- Quest: attended budget vs resident memory ----------------------------------------------
    quest_rows = [r for r in q["niah"] if r["policy"] == "quest"]
    quest = {}
    if quest_rows:
        quest = {
            "dense_layers": quest_rows[0]["dense_layers"],
            "attended_fraction_by_budget": {
                f"{b:g}": statistics.median(
                    r["attended_tokens_selecting_layers"] / (r["prompt_len"] + cfg["niah"]["max_new_tokens"][r["kind"]] - 1) for r in quest_rows if r["budget"] == b
                )
                for b in budgets
            },
            "resident_fraction": span([n["resident_fraction_median"] for n in niah if n["policy"] == "quest"]),
            "selections_zero_prompts": sum(1 for r in quest_rows if r["selections"] == 0),
        }

    # ---- cross-phase repeatability: conditions measured in both phases -----------------------------
    cross: dict[str, Any] | None = None
    p2_path = RESULTS_DIR / "phase2" / "policy_quality" / "metrics.json"
    if p2_path.exists():
        p2 = json.loads(p2_path.read_text(encoding="utf-8"))
        p2_scores = {(r["policy"], r["budget"]) + prompt_key(r): r["score"] for r in p2["niah"]}
        shared = [k for k in scores if k in p2_scores]
        p2_tf = {(r["policy"], r["budget"], r["offset"]): r for r in p2["teacher_forced"]}
        tf_shared = [(r, p2_tf[(r["policy"], r["budget"], r["offset"])]) for r in q["teacher_forced"] if (r["policy"], r["budget"], r["offset"]) in p2_tf]
        cross = {
            "niah_prompt_conditions_compared": len(shared),
            "niah_scored_differently": sum(1 for k in shared if scores[k] != p2_scores[k]),
            "tf_conditions_compared": len(tf_shared),
            "tf_max_abs_kl_diff": max((abs(a["mean_kl"] - b["mean_kl"]) for a, b in tf_shared), default=None),
            "tf_max_abs_top1_diff": max((abs(a["top1_agreement"] - b["top1_agreement"]) for a, b in tf_shared), default=None),
            "phase2_commit": p2["provenance"].get("git_commit"),
        }

    # ---- summary spans the write-up quotes ------------------------------------------------------------
    nb = {pol: cond_rows(niah, pol) for pol in policies}
    tb = {pol: cond_rows(teacher_forced, pol) for pol in policies}
    last_depth = f"{max(depths):g}"
    summary: dict[str, Any] = {
        "accuracy": {pol: {f"{b:g}": nb[pol][b]["accuracy"] for b in budgets} for pol in policies},
        "tf_top1": {pol: span([tb[pol][b]["top1_agreement"] for b in budgets]) for pol in policies},
        "tf_kl": {pol: span([tb[pol][b]["mean_kl"] for b in budgets]) for pol in policies},
        "tf_kl_by_budget": {pol: {f"{b:g}": tb[pol][b]["mean_kl"] for b in budgets} for pol in policies},
        "tf_top1_by_budget": {pol: {f"{b:g}": tb[pol][b]["top1_agreement"] for b in budgets} for pol in policies},
        "at_depth0": {pol: span([nb[pol][b]["by_depth"]["0"] for b in budgets]) for pol in policies},
        "at_last_depth": {pol: span([nb[pol][b]["by_depth"][last_depth] for b in budgets]) for pol in policies},
        "full_by_kind": next(n["by_kind"] for n in niah if n["policy"] == "full"),
        "niah_kinds": len(cfg["niah"]["kinds"]),
        "niah_depths": len(depths),
        "niah_samples": cfg["niah"]["samples"],
        "teacher_forced_documents": len(cfg["teacher_forced"]["offsets"]),
        "teacher_forced_positions": cfg["teacher_forced"]["continuation"],
        "block_size": bs,
    }
    if speed:
        sp = {s["label"]: s for s in speed}
        summary["over_full"] = {pol: span([sp[label(pol, b)]["over_full"] for b in budgets]) for pol in policies}
        summary["over_full_by_budget"] = {pol: {f"{b:g}": sp[label(pol, b)]["over_full"] for b in budgets} for pol in policies}
        summary["manager_ms_per_token"] = {pol: span([sp[label(pol, b)]["manager_host_s_per_token"]["median"] * 1e3 for b in budgets]) for pol in policies}
        summary["tokens_per_s_full"] = sp["full@1"]["tokens_per_s"]["median"]
        summary["conditions_with_spill"] = sum(1 for s in speed if s["any_spill"])
        prefills = [pf for run in runs for pf in run.get("prefills", []) if "scored_wall_s" in pf]
        if prefills:
            summary["prefill_plain_s"] = statistics.median(pf["plain_wall_s"] for pf in prefills)
            summary["prefill_scored_s"] = statistics.median(pf["scored_wall_s"] for pf in prefills)
            summary["prefill_scored_over_plain"] = statistics.median(pf["scored_wall_s"] / pf["plain_wall_s"] for pf in prefills)

    write_metrics(
        base / "analysis",
        {
            "sources": {"policy_quality": q["provenance"], "budget_speed": [r["provenance"] for r in runs]},
            "context": cfg["context"],
            "block_size": bs,
            "kv_bytes_per_token": q["kv_bytes_per_token"],
            "num_layers": num_layers,
            "prefill_scorer": q.get("prefill_scorer"),
            "prompt_len_median": statistics.median(r["prompt_len"] for r in q["niah"]),
            "full_niah_accuracy": full_acc,
            "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
            "niah_prompts_per_condition": len(full_scores),
            "reference_policy": REFERENCE,
            "niah": niah,
            "teacher_forced": teacher_forced,
            "speed": speed,
            "speed_runs": len(runs),
            "full_decode_median_s": across(per_run(runs, "full", 1.0, "decode_wall_s.median")) if runs else None,
            "headline": head,
            "paired_vs_reference": paired,
            "h2o": h2o,
            "quest": quest,
            "cross_phase": cross,
            "summary": summary,
        },
    )
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
