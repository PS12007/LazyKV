"""Derive Phase 2 decision quantities: the budget sweep for policy ladder rungs 1-3 at 32K.

Reads   results/phase2/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase2/budget_speed/run_*/metrics.json  (decode latency, fast kernel)
Writes  results/phase2/analysis/metrics.json

Aggregation, speed: within a run each (condition, repeat) gives a median; the run's value is
the median over repeats; the reported value is the median over runs with min/max across runs.
Aggregation, NIAH: mean score over prompts, with a percentile bootstrap CI over prompts.
Retention is a condition's accuracy divided by the full cache's accuracy on the same prompts.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci  # noqa: E402

RETENTION_TARGET = 0.99  # brief §B8 headline: minimum budget retaining >= 99% of baseline NIAH accuracy


def across(values: list[float]) -> dict[str, float | int]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n_runs": 0}
    return {"n_runs": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def label(policy: str, budget: float) -> str:
    return f"{policy}@{budget:g}"


def main() -> None:
    q = json.loads((RESULTS_DIR / "phase2" / "policy_quality" / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS_DIR / "phase2" / "budget_speed").glob("run_*/metrics.json"))]
    kv_per_token = q["kv_bytes_per_token"]

    # ---- NIAH ----------------------------------------------------------------------------
    by_cond: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in q["niah"]:
        by_cond[(r["policy"], r["budget"])].append(r)
    full_scores = {(r["kind"], r["depth"], r["sample"]): r["score"] for r in by_cond[("full", 1.0)]}
    full_acc = statistics.mean(full_scores.values())
    kinds = q["config"]["niah"]["kinds"]
    depths = q["config"]["niah"]["depths"]

    niah = []
    for (policy, budget), rows in sorted(by_cond.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        scores = [r["score"] for r in rows]
        acc = statistics.mean(scores)
        entry = {
            "policy": policy,
            "budget": budget,
            "label": label(policy, budget),
            "prompts": len(rows),
            "accuracy": acc,
            "accuracy_ci95": bootstrap_mean_ci(scores),
            "retention": acc / full_acc if full_acc else None,
            "agrees_with_full": statistics.mean(1.0 if r["score"] == full_scores[(r["kind"], r["depth"], r["sample"])] else 0.0 for r in rows),
            "by_kind": {k: statistics.mean(r["score"] for r in rows if r["kind"] == k) for k in kinds},
            "by_depth": {f"{d:g}": statistics.mean(r["score"] for r in rows if r["depth"] == d) for d in depths},
            "gpu_resident_kv_bytes_median": statistics.median(r["gpu_resident_kv_bytes"] for r in rows),
            # Resident KV at the end of the answer, as a share of the whole sequence's KV.
            "resident_fraction_median": statistics.median(
                r["gpu_resident_kv_bytes"] / (kv_per_token * (r["prompt_len"] + q["config"]["niah"]["max_new_tokens"][r["kind"]] - 1)) for r in rows
            ),
        }
        niah.append(entry)

    # Needle position relative to what each window keeps: explains depth results mechanically.
    prompt_len = statistics.median(r["prompt_len"] for r in q["niah"])

    # ---- teacher-forced -----------------------------------------------------------------
    tf_by: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in q["teacher_forced"]:
        tf_by[(r["policy"], r["budget"])].append(r)
    teacher_forced = [
        {
            "policy": p,
            "budget": b,
            "label": label(p, b),
            "documents": len(rows),
            "top1_agreement": statistics.mean(r["top1_agreement"] for r in rows),
            "top1_min_doc": min(r["top1_agreement"] for r in rows),
            "mean_kl": statistics.mean(r["mean_kl"] for r in rows),
            "mean_kl_max_doc": max(r["mean_kl"] for r in rows),
            "exact_all_docs": all(r["exact_match"] for r in rows),
        }
        for (p, b), rows in sorted(tf_by.items(), key=lambda kv: (kv[0][0], -kv[0][1]))
    ]

    # ---- speed ----------------------------------------------------------------------------
    speed = []
    if runs:
        conds = sorted({(r["policy"], r["budget"]) for r in runs[0]["results"]}, key=lambda c: (c[0], -c[1]))

        def per_run(policy: str, budget: float, key: str) -> list[float]:
            vals = []
            for run in runs:
                rows = [r for r in run["results"] if r["policy"] == policy and r["budget"] == budget and _get(r, key) is not None]
                if rows:
                    vals.append(statistics.median(_get(r, key) for r in rows))
            return vals

        full_med = across(per_run("full", 1.0, "decode_wall_s.median"))["median"]
        for policy, budget in conds:
            med = per_run(policy, budget, "decode_wall_s.median")
            speed.append({
                "policy": policy,
                "budget": budget,
                "label": label(policy, budget),
                "decode_wall_median_s": across(med),
                "decode_wall_p90_s": across(per_run(policy, budget, "decode_wall_p90_s")),
                "tokens_per_s": across([1.0 / m for m in med]),
                "over_full": (statistics.median(med) / full_med) if med and full_med else None,
                "manager_host_s_per_token": across(per_run(policy, budget, "manager_host_s_per_token")),
                "manager_observe_s_per_token": across(per_run(policy, budget, "manager_observe_s_per_token")),
                "boundary_build_s": across(per_run(policy, budget, "boundary_build_s")),
                "gpu_resident_kv_bytes": across(per_run(policy, budget, "gpu_resident_kv_bytes")),
                "max_memory_allocated_bytes": across(per_run(policy, budget, "max_memory_allocated_bytes")),
                "evictions": across(per_run(policy, budget, "evictions")),
                "any_spill": any(r.get("spilled_to_shared") for run in runs for r in run["results"] if r["policy"] == policy and r["budget"] == budget),
            })

    # ---- headline (brief §B8) ----------------------------------------------------------------
    speed_by = {s["label"]: s for s in speed}
    headline = {}
    for policy in q["config"]["policies"]:
        rows = sorted((n for n in niah if n["policy"] == policy), key=lambda n: n["budget"])
        passing = [n for n in rows if n["retention"] is not None and n["retention"] >= RETENTION_TARGET]
        best = passing[0] if passing else None
        headline[policy] = {
            "retention_target": RETENTION_TARGET,
            "min_budget_meeting_target": best["budget"] if best else None,
            "tokens_per_s_at_that_budget": speed_by[best["label"]]["tokens_per_s"]["median"] if best and best["label"] in speed_by else None,
            "best_retention_below_full": max((n["retention"] for n in rows), default=None),
            "best_retention_budget": max(rows, key=lambda n: n["retention"])["budget"] if rows else None,
        }

    # ---- comparisons the write-up states, computed rather than eyeballed ---------------------
    def cond_rows(src: list[dict[str, Any]], policy: str) -> dict[float, dict[str, Any]]:
        return {r["budget"]: r for r in src if r["policy"] == policy}

    def span(vals: list[float]) -> dict[str, float] | None:
        return {"min": min(vals), "max": max(vals)} if vals else None

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
        "full_by_kind": by_cond and next(n["by_kind"] for n in niah if n["policy"] == "full"),
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
            "kv_bytes_per_token": kv_per_token,
            "prompt_len_median": prompt_len,
            "full_niah_accuracy": full_acc,
            "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
            "niah_prompts_per_condition": len(full_scores),
            "niah": niah,
            "teacher_forced": teacher_forced,
            "speed": speed,
            "speed_runs": len(runs),
            "full_decode_median_s": across(per_run("full", 1.0, "decode_wall_s.median")) if runs else None,
            "headline": headline,
            "lru_vs_window": lru_vs_window,
            "summary": summary,
        },
    )
    print("wrote", RESULTS_DIR / "phase2" / "analysis" / "metrics.json")


def _get(row: dict[str, Any], dotted: str) -> Any:
    """Dotted lookup; None when any part is absent (e.g. manager counters on the full cache)."""
    node: Any = row
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


if __name__ == "__main__":
    main()
