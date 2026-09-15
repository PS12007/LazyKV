"""Aggregation shared by the budget-sweep analyses (Phase 2 and Phase 3).

Speed: within a run each (condition, repeat) gives a median; the run's value is the median over
repeats; the reported value is the median over runs with min/max across runs.
NIAH: mean score over prompts, with a percentile bootstrap CI over prompts. Retention is a
condition's accuracy divided by the full cache's accuracy on the same prompts.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from harness.stats import bootstrap_mean_ci

RETENTION_TARGET = 0.99  # brief §B8 headline: minimum budget retaining >= 99% of baseline NIAH accuracy


def across(values: list[float]) -> dict[str, float | int]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n_runs": 0}
    return {"n_runs": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def label(policy: str, budget: float) -> str:
    return f"{policy}@{budget:g}"


def span(vals: list[float]) -> dict[str, float] | None:
    return {"min": min(vals), "max": max(vals)} if vals else None


def get_dotted(row: dict[str, Any], dotted: str) -> Any:
    """Dotted lookup; None when any part is absent (e.g. manager counters on the full cache)."""
    node: Any = row
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _order(kv: tuple[tuple[str, float], Any]) -> tuple[str, float]:
    return (kv[0][0], -kv[0][1])


def niah_table(q: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, float, int], float]]:
    """Per-condition NIAH rows, and the full cache's score per (kind, depth, sample)."""
    by_cond: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in q["niah"]:
        by_cond[(r["policy"], r["budget"])].append(r)
    full_scores = {(r["kind"], r["depth"], r["sample"]): r["score"] for r in by_cond[("full", 1.0)]}
    full_acc = statistics.mean(full_scores.values())
    kinds = q["config"]["niah"]["kinds"]
    depths = q["config"]["niah"]["depths"]
    kv_per_token = q["kv_bytes_per_token"]

    niah = []
    for (policy, budget), rows in sorted(by_cond.items(), key=_order):
        scores = [r["score"] for r in rows]
        acc = statistics.mean(scores)
        niah.append({
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
        })
    return niah, full_scores


def teacher_forced_table(q: dict[str, Any]) -> list[dict[str, Any]]:
    tf_by: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in q["teacher_forced"]:
        tf_by[(r["policy"], r["budget"])].append(r)
    return [
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
        for (p, b), rows in sorted(tf_by.items(), key=_order)
    ]


def per_run(runs: list[dict[str, Any]], policy: str, budget: float, key: str) -> list[float]:
    """One value per run: the median over that run's repeats of `key` for the condition."""
    vals = []
    for run in runs:
        rows = [r for r in run["results"] if r["policy"] == policy and r["budget"] == budget and get_dotted(r, key) is not None]
        if rows:
            vals.append(statistics.median(get_dotted(r, key) for r in rows))
    return vals


def speed_table(runs: list[dict[str, Any]], extra_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if not runs:
        return []
    conds = sorted({(r["policy"], r["budget"]) for r in runs[0]["results"]}, key=lambda c: (c[0], -c[1]))
    full_med = across(per_run(runs, "full", 1.0, "decode_wall_s.median"))["median"]
    speed = []
    for policy, budget in conds:
        med = per_run(runs, policy, budget, "decode_wall_s.median")
        row = {
            "policy": policy,
            "budget": budget,
            "label": label(policy, budget),
            "decode_wall_median_s": across(med),
            "decode_wall_p90_s": across(per_run(runs, policy, budget, "decode_wall_p90_s")),
            "tokens_per_s": across([1.0 / m for m in med]),
            "over_full": (statistics.median(med) / full_med) if med and full_med else None,
            "manager_host_s_per_token": across(per_run(runs, policy, budget, "manager_host_s_per_token")),
            "manager_observe_s_per_token": across(per_run(runs, policy, budget, "manager_observe_s_per_token")),
            "boundary_build_s": across(per_run(runs, policy, budget, "boundary_build_s")),
            "gpu_resident_kv_bytes": across(per_run(runs, policy, budget, "gpu_resident_kv_bytes")),
            "max_memory_allocated_bytes": across(per_run(runs, policy, budget, "max_memory_allocated_bytes")),
            "evictions": across(per_run(runs, policy, budget, "evictions")),
            "any_spill": any(r.get("spilled_to_shared") for run in runs for r in run["results"] if r["policy"] == policy and r["budget"] == budget),
        }
        for key in extra_keys:
            row[key] = across(per_run(runs, policy, budget, key))
        speed.append(row)
    return speed


def headline(niah: list[dict[str, Any]], speed: list[dict[str, Any]], policies: list[str]) -> dict[str, dict[str, Any]]:
    """Brief §B8: the smallest budget retaining >= 99% of full-cache NIAH accuracy, and tokens/s there."""
    speed_by = {s["label"]: s for s in speed}
    out = {}
    for policy in policies:
        rows = sorted((n for n in niah if n["policy"] == policy), key=lambda n: n["budget"])
        passing = [n for n in rows if n["retention"] is not None and n["retention"] >= RETENTION_TARGET]
        best = passing[0] if passing else None
        out[policy] = {
            "retention_target": RETENTION_TARGET,
            "min_budget_meeting_target": best["budget"] if best else None,
            "tokens_per_s_at_that_budget": speed_by[best["label"]]["tokens_per_s"]["median"] if best and best["label"] in speed_by else None,
            "best_retention_below_full": max((n["retention"] for n in rows), default=None),
            "best_retention_budget": max(rows, key=lambda n: n["retention"])["budget"] if rows else None,
        }
    return out


def cond_rows(src: list[dict[str, Any]], policy: str) -> dict[float, dict[str, Any]]:
    return {r["budget"]: r for r in src if r["policy"] == policy}
