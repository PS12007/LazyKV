"""Derive Phase 2 decision quantities: the budget sweep for policy ladder rungs 1-3 at 32K.

Reads   results/phase2/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase2/budget_speed/run_*/metrics.json  (decode latency, fast kernel)
Writes  results/phase2/analysis/metrics.json

Aggregation rules are in harness/sweep_analysis.py, shared with Phase 3.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci  # noqa: E402
from harness.sweep_analysis import across, cond_rows, headline, label, niah_table, per_run, span, speed_table, teacher_forced_table  # noqa: E402


def main() -> None:
    q = json.loads((RESULTS_DIR / "phase2" / "policy_quality" / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS_DIR / "phase2" / "budget_speed").glob("run_*/metrics.json"))]
    kinds = q["config"]["niah"]["kinds"]
    depths = q["config"]["niah"]["depths"]

    niah, full_scores = niah_table(q)
    full_acc = statistics.mean(full_scores.values())
    prompt_len = statistics.median(r["prompt_len"] for r in q["niah"])
    teacher_forced = teacher_forced_table(q)
    speed = speed_table(runs)
    head = headline(niah, speed, q["config"]["policies"])

    # ---- comparisons the write-up states, computed rather than eyeballed ---------------------
    budgets = sorted(q["config"]["budgets"])
    per_prompt = {(r["policy"], r["budget"], r["kind"], r["depth"], r["sample"]): r["score"] for r in q["niah"]}
    nl, nw = cond_rows(niah, "lru"), cond_rows(niah, "window")
    tl, tw = cond_rows(teacher_forced, "lru"), cond_rows(teacher_forced, "window")
    lru_keys = [key for key in per_prompt if key[0] == "lru"]
    lru_vs_window = {
        "prompts_compared": len(lru_keys),
        "prompts_scored_differently": sum(1 for (_, b, k, d, s) in lru_keys if per_prompt[("lru", b, k, d, s)] != per_prompt[("window", b, k, d, s)]),
        "niah_accuracy_max_abs_diff": max(abs(nl[b]["accuracy"] - nw[b]["accuracy"]) for b in budgets),
        "tf_top1_max_abs_diff": max(abs(tl[b]["top1_agreement"] - tw[b]["top1_agreement"]) for b in budgets),
        "tf_kl_max_rel_diff": max(abs(tl[b]["mean_kl"] - tw[b]["mean_kl"]) / tw[b]["mean_kl"] for b in budgets),
    }
    ws_niah, ws_tf = cond_rows(niah, "window_sink"), cond_rows(teacher_forced, "window_sink")
    last_depth = f"{max(depths):g}"
    summary = {
        "window_sink_tf_top1": span([ws_tf[b]["top1_agreement"] for b in budgets]),
        "window_sink_tf_kl": span([ws_tf[b]["mean_kl"] for b in budgets]),
        "window_tf_top1": span([tw[b]["top1_agreement"] for b in budgets]),
        "window_tf_kl": span([tw[b]["mean_kl"] for b in budgets]),
        "block_full_tf_max_kl": next((r["mean_kl_max_doc"] for r in teacher_forced if r["policy"] == "block_full"), None),
        "block_full_tf_min_top1": next((r["top1_min_doc"] for r in teacher_forced if r["policy"] == "block_full"), None),
        "block_full_niah_prompts_scored_differently": sum(
            1 for (p, b, k, d, s), v in per_prompt.items() if p == "block_full" and v != per_prompt[("full", 1.0, k, d, s)]
        ),
        "teacher_forced_documents": len(q["config"]["teacher_forced"]["offsets"]),
        "teacher_forced_positions": q["config"]["teacher_forced"]["continuation"],
        "niah_kinds": len(kinds),
        "niah_depths": len(depths),
        "niah_samples": q["config"]["niah"]["samples"],
        # Sink effect where the needle is always inside the window: the last depth.
        "last_depth": float(last_depth),
        "window_sink_at_last_depth": span([ws_niah[b]["by_depth"][last_depth] for b in budgets]),
        "window_at_last_depth": span([nw[b]["by_depth"][last_depth] for b in budgets]),
        "full_by_kind": next(n["by_kind"] for n in niah if n["policy"] == "full"),
    }
    if speed:
        sp = {s["label"]: s for s in speed}
        summary["over_full"] = {pol: span([sp[label(pol, b)]["over_full"] for b in budgets]) for pol in q["config"]["policies"]}
        summary["manager_ms_per_token"] = {
            pol: span([sp[label(pol, b)]["manager_host_s_per_token"]["median"] * 1e3 for b in budgets]) for pol in q["config"]["policies"]
        }
        summary["observe_ms_per_token_lru"] = span([sp[label("lru", b)]["manager_observe_s_per_token"]["median"] * 1e3 for b in budgets])
        summary["block_full_over_full"] = sp["block_full@1"]["over_full"]
        summary["conditions_with_spill"] = sum(1 for s in speed if s["any_spill"])

    write_metrics(
        RESULTS_DIR / "phase2" / "analysis",
        {
            "sources": {"policy_quality": q["provenance"], "budget_speed": [r["provenance"] for r in runs]},
            "context": q["config"]["context"],
            "block_size": q["config"]["block_size"],
            "kv_bytes_per_token": q["kv_bytes_per_token"],
            "prompt_len_median": prompt_len,
            "full_niah_accuracy": full_acc,
            "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
            "niah_prompts_per_condition": len(full_scores),
            "niah": niah,
            "teacher_forced": teacher_forced,
            "speed": speed,
            "speed_runs": len(runs),
            "full_decode_median_s": across(per_run(runs, "full", 1.0, "decode_wall_s.median")) if runs else None,
            "headline": head,
            "lru_vs_window": lru_vs_window,
            "summary": summary,
        },
    )
    print("wrote", RESULTS_DIR / "phase2" / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
