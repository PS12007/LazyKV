"""Teacher-forced KL as a supporting quality measure beside the NIAH headline.

Reads   results/phase10/perplexity_ctx*/metrics.json   (KL per condition per window, 4 contexts)
        results/ladder/metrics.json                    (NIAH retention per condition, 32K)
Writes  results/kl/metrics.json

Phase 9 showed the brief's 99% NIAH bar is finer than needle retrieval can resolve near it, and the
owner decided (2026-09-27) to keep that headline and add KL beside it rather than replace it. This
script asks one question of KL, with no threshold chosen: **where retrieval cannot order two
conditions, can KL?**

Phase 10's teacher-forced pass scored the same twelve 256-token windows under every condition, so
every comparison is paired by window. Two orderings are tested with an exact sign test over windows:

- *within a rung*, each budget against the next larger one (does less budget move the
  distribution further, window by window?);
- *across rungs at one budget*, every pair of rungs.

A comparison is "resolved" at p < 0.05. That level is the study's convention, not a tuned threshold.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci, sign_test_p  # noqa: E402

ALPHA = 0.05
POLICIES = ("window_sink", "h2o", "quest", "tiered_int8")


def label(policy: str, budget: float) -> str:
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def kl_by_window(rows: list[dict[str, Any]], policy: str, budget: float) -> dict[tuple[str, int], float]:
    return {(r["book"], r["target"]): r["mean_kl"] for r in rows if r["policy"] == policy and r["budget"] == budget}


def compare(a: dict[tuple[str, int], float], b: dict[tuple[str, int], float]) -> dict[str, Any]:
    """Windows where a's KL exceeds b's, and the reverse, with the exact two-sided sign test."""
    keys = sorted(k for k in a if k in b)
    above = sum(1 for k in keys if a[k] > b[k])
    below = sum(1 for k in keys if a[k] < b[k])
    p = sign_test_p(above, below)
    return {"windows": len(keys), "a_above": above, "a_below": below, "sign_test_p": p, "resolved": p < ALPHA}


def main() -> None:
    runs = {}
    for path in sorted((RESULTS_DIR / "phase10").glob("perplexity_ctx*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        runs[int(m["config"]["context"])] = m["teacher_forced"]
    if not runs:
        raise SystemExit("no Phase 10 runs")
    ladder_path = RESULTS_DIR / "ladder" / "metrics.json"
    ladder = json.loads(ladder_path.read_text(encoding="utf-8")) if ladder_path.exists() else None
    niah = {label(r["policy"], r["budget"]): r["retention"] for r in ladder["rows"]} if ladder else {}
    ladder_ctx = ladder["summary"]["context"] if ladder else None

    per_ctx: dict[str, Any] = {}
    within_all, across_all = [], []
    for ctx, rows in sorted(runs.items()):
        budgets = sorted({r["budget"] for r in rows if r["policy"] in POLICIES}, reverse=True)
        conds = {}
        for pol in POLICIES:
            for b in budgets:
                w = kl_by_window(rows, pol, b)
                if not w:
                    continue
                vals = list(w.values())
                lo, hi = bootstrap_mean_ci(vals, iters=10000, seed=0)
                conds[label(pol, b)] = {"policy": pol, "budget": b, "mean_kl": sum(vals) / len(vals), "kl_ci95": [lo, hi],
                                        "niah_retention": niah.get(label(pol, b)) if ctx == ladder_ctx else None}
        within = []
        for pol in POLICIES:
            for big, small in zip(budgets, budgets[1:]):
                c = compare(kl_by_window(rows, pol, small), kl_by_window(rows, pol, big))
                if c["windows"]:
                    within.append({"context": ctx, "policy": pol, "smaller": small, "larger": big, **c})
        across = []
        for b in budgets:
            for p1, p2 in combinations(POLICIES, 2):
                c = compare(kl_by_window(rows, p1, b), kl_by_window(rows, p2, b))
                if c["windows"]:
                    across.append({"context": ctx, "budget": b, "a": p1, "b": p2, **c})
        per_ctx[str(ctx)] = {"conditions": conds, "within_rung": within, "across_rungs": across}
        within_all += within
        across_all += across

    longest = str(max(runs))
    q = per_ctx[longest]["conditions"]
    q75_vs_50 = next((w for w in per_ctx[longest]["within_rung"] if w["policy"] == "quest" and w["larger"] == 0.75), None)
    summary = {
        "contexts": sorted(runs),
        "longest_context": int(longest),
        "windows": per_ctx[longest]["within_rung"][0]["windows"] if per_ctx[longest]["within_rung"] else None,
        "alpha": ALPHA,
        # Less budget, more KL: how often does that hold window by window, and resolve?
        "within_comparisons": len(within_all),
        "within_resolved": sum(1 for w in within_all if w["resolved"]),
        "within_resolved_wrong_way": sum(1 for w in within_all if w["resolved"] and w["a_below"] > w["a_above"]),
        "across_comparisons": len(across_all),
        "across_resolved": sum(1 for a in across_all if a["resolved"]),
        # Rungs 5 and 8 make the same selection (rung 8 only narrows the cold tier's precision), so
        # their KLs are expected to be indistinguishable; counted apart so the rest can be read alone.
        "across_unresolved": sum(1 for a in across_all if not a["resolved"]),
        "across_unresolved_rung5_vs_rung8": sum(1 for a in across_all if not a["resolved"] and {a["a"], a["b"]} == {"quest", "tiered_int8"}),
        # The case the headline could not settle: rung 5 at 75% against rung 5 at 50%.
        "quest_75_kl": q.get(label("quest", 0.75), {}).get("mean_kl"),
        "quest_75_kl_ci95": q.get(label("quest", 0.75), {}).get("kl_ci95"),
        "quest_50_kl": q.get(label("quest", 0.5), {}).get("mean_kl"),
        "quest_50_kl_ci95": q.get(label("quest", 0.5), {}).get("kl_ci95"),
        "quest_50_over_75_windows_above": None if q75_vs_50 is None else q75_vs_50["a_above"],
        "quest_50_over_75_sign_test_p": None if q75_vs_50 is None else q75_vs_50["sign_test_p"],
    }
    write_metrics(RESULTS_DIR / "kl", {"contexts": per_ctx, "summary": summary})


if __name__ == "__main__":
    main()
