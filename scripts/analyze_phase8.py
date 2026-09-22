"""Derive Phase 8 decision quantities: what the ladder does as context length moves.

Reads   results/phase8/policy_quality_ctx*/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase8/budget_speed_ctx*/run_*/metrics.json (decode latency and tier counters, fast kernel)
Writes  results/phase8/analysis/metrics.json

Every number in Phases 2-7 was measured at 32,768 tokens. That makes two of the study's load-bearing
claims untested in the one direction they could most easily be artefacts of, and this analysis is
built around separating them:

1. **Is retention a function of the budget fraction, or of the blocks actually kept?** The brief
   states its headline in fractions ("minimum GPU KV budget at which policy P retains >= 99%"),
   which quietly assumes the fraction is the thing that governs quality. It need not be: a policy
   might simply need some absolute number of blocks to find a needle, in which case the same
   fraction is generous at 32K and starving at 4K. The sweep discriminates these for free, because
   K = floor(budget x context / block_size) - 2 makes the same K recur along the grid's diagonals:
   K=30 appears at all four contexts, K=14 and K=62 at three. So the same data can be collapsed two
   ways -- by fraction and by K -- and whichever collapse leaves less spread is the one that governs.
   `collapse()` reports both spreads rather than picking a winner in prose.

2. **Does "host-bound" survive?** Phases 4-7 found every attack on bytes moved failing to reduce
   latency, and blamed the host work that decides what to move. That diagnosis has a sharp
   prediction: the manager ranks every block every step, so its cost should grow linearly in the
   block count, i.e. in context. If instead it is largely a fixed per-layer cost, the finding is
   the same but its cause is not, and "host-bound" would mean bound by a constant rather than by
   the ranking. `scaling()` fits both the manager and the full cache against context and reports
   the intercept's share, which is the number that tells those two apart.

Neither question can be answered from a single context, which is why neither was answered before.
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
from harness.sweep_analysis import RETENTION_TARGET, across, headline, label, niah_table, span, speed_table, teacher_forced_table  # noqa: E402
from lazykv.selection import blocks_for_budget  # noqa: E402

FULL = "full"
MIB = 2**20
MS = 1e3


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def _fit(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    """Least-squares y = intercept + slope*x, with r^2. None below three points.

    Deliberately plain: four contexts is not enough data to justify anything cleverer, and the
    quantity actually wanted is the intercept's share of the largest measured y, not the fit.
    """
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {
        "intercept": intercept,
        "slope": slope,
        "r2": (1 - ss_res / ss_tot) if ss_tot else 1.0,
        "n_points": n,
    }


# ---- 1. does the fraction or the block count govern quality? ---------------------------


def collapse(per_ctx: dict[int, dict[str, Any]], policies: list[str], block_size: int) -> dict[str, Any]:
    """The same retentions grouped by budget fraction and by selected blocks K.

    A group is only informative when it spans more than one context, so singleton groups are
    dropped from the spread rather than counted as perfect agreement -- a group of one always has
    zero spread and would flatter whichever collapse happened to have more of them.
    """
    points = []
    for ctx, d in per_ctx.items():
        for row in d["niah"]:
            # The reference retains itself perfectly at every context, and `block_full` is the
            # 100%-budget control rather than a point on any budget curve. Neither carries
            # information about how retention responds to context, so neither is a datum here.
            if row["policy"] not in policies or row["retention"] is None:
                continue
            points.append({
                "context": ctx,
                "policy": row["policy"],
                "budget": row["budget"],
                "k_blocks": blocks_for_budget(row["budget"], ctx, block_size),
                "retention": row["retention"],
                "accuracy": row["accuracy"],
            })

    def groups(key: str) -> list[dict[str, Any]]:
        out = []
        for policy in policies:
            keys = sorted({p[key] for p in points if p["policy"] == policy})
            for k in keys:
                sel = [p for p in points if p["policy"] == policy and p[key] == k]
                if len({p["context"] for p in sel}) < 2:
                    continue
                rets = [p["retention"] for p in sel]
                out.append({
                    "policy": policy,
                    "grouped_by": key,
                    "value": k,
                    "contexts": sorted(p["context"] for p in sel),
                    "retention_by_context": {str(p["context"]): p["retention"] for p in sorted(sel, key=lambda p: p["context"])},
                    "spread_pp": 100 * (max(rets) - min(rets)),
                    "mean_retention": statistics.fmean(rets),
                })
        return out

    by_budget, by_k = groups("budget"), groups("k_blocks")

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        spreads = [r["spread_pp"] for r in rows]
        return {
            "groups": len(rows),
            "comparisons": sum(len(r["contexts"]) for r in rows),
            "mean_spread_pp": statistics.fmean(spreads) if spreads else None,
            "median_spread_pp": statistics.median(spreads) if spreads else None,
            "max_spread_pp": max(spreads) if spreads else None,
        }

    fr, kk = summarize(by_budget), summarize(by_k)
    return {
        "points": points,
        "by_budget": by_budget,
        "by_k_blocks": by_k,
        "summary_by_budget": fr,
        "summary_by_k_blocks": kk,
        # >1 means grouping by K holds retention tighter than grouping by fraction does, i.e. the
        # absolute blocks attended govern quality and the brief's fraction-shaped headline is a
        # statement about 32K rather than about the policy.
        "fraction_over_k_mean_spread": _ratio(fr["mean_spread_pp"], kk["mean_spread_pp"]),
    }


# ---- 2. what scales with context, and what does not? -----------------------------------


def scaling(per_ctx: dict[int, dict[str, Any]], policies: list[str], budgets: list[float], block_size: int) -> dict[str, Any]:
    """Decode latency and manager host time against context, per policy and budget.

    Fitted against the *block count* rather than against the context directly, because the block
    count is what the manager iterates over and so is what the host-bound diagnosis predicts the
    cost to be proportional to. The two differ only by a constant factor, but the slope is then
    readable as microseconds per block per token, which is a quantity that can be sanity-checked.
    """
    contexts = sorted(per_ctx)
    series: list[dict[str, Any]] = []
    for policy in [FULL, *policies]:
        for budget in ([1.0] if policy == FULL else budgets):
            pts = []
            for ctx in contexts:
                row = next((s for s in per_ctx[ctx]["speed"] if s["policy"] == policy and s["budget"] == budget), None)
                if row is None or not row["decode_wall_median_s"].get("n_runs"):
                    continue
                full_row = next((s for s in per_ctx[ctx]["speed"] if s["policy"] == FULL), None)
                pts.append({
                    "context": ctx,
                    "n_blocks": ctx // block_size,
                    "k_blocks": blocks_for_budget(budget, ctx, block_size) if policy != FULL else None,
                    "decode_ms_per_token": MS * row["decode_wall_median_s"]["median"],
                    "manager_ms_per_token": MS * row["manager_host_s_per_token"]["median"] if row["manager_host_s_per_token"].get("n_runs") else None,
                    "tokens_per_s": row["tokens_per_s"]["median"],
                    "resident_mib": row["gpu_resident_kv_bytes"]["median"] / MIB,
                    "over_full": _ratio(row["decode_wall_median_s"]["median"], full_row["decode_wall_median_s"]["median"]) if full_row else None,
                })
            if not pts:
                continue
            blocks = [float(p["n_blocks"]) for p in pts]
            decode_fit = _fit(blocks, [p["decode_ms_per_token"] for p in pts])
            mgr_pts = [p for p in pts if p["manager_ms_per_token"] is not None]
            mgr_fit = _fit([float(p["n_blocks"]) for p in mgr_pts], [p["manager_ms_per_token"] for p in mgr_pts])
            largest = pts[-1]
            series.append({
                "policy": policy,
                "budget": budget,
                "label": label(policy, budget),
                "points": pts,
                "decode_fit": decode_fit,
                "manager_fit": mgr_fit,
                # The number that separates "bound by ranking blocks" from "bound by a per-layer
                # constant": at the largest context measured, how much of the cost the fit puts at
                # zero blocks. Near 1 means context is nearly irrelevant to the manager's cost.
                "decode_intercept_share_at_max_ctx": _ratio(decode_fit["intercept"], largest["decode_ms_per_token"]) if decode_fit else None,
                "manager_intercept_share_at_max_ctx": _ratio(mgr_fit["intercept"], largest["manager_ms_per_token"]) if mgr_fit and largest["manager_ms_per_token"] else None,
                "decode_ms_first_to_last": _ratio(pts[-1]["decode_ms_per_token"], pts[0]["decode_ms_per_token"]),
                "over_full_by_context": {str(p["context"]): p["over_full"] for p in pts},
            })
    return {"contexts": contexts, "series": series}


# ---- 3. the headline, per context -------------------------------------------------------


def _margin_prompts(row: dict[str, Any], full_acc: float, n_prompts: int) -> float | None:
    """How many prompts' worth of score separates a condition from the 99% bar.

    Retention differences of a point or two are the ones this phase turns on -- whether an
    approximating rung clears the bar is decided by about that much -- and a percentage point is
    not a unit anyone can judge. Each prompt contributes 1/n of the accuracy, so expressing the
    margin in prompts says plainly how thin the claim is. Positive means above the bar.
    """
    if row.get("accuracy") is None or not full_acc:
        return None
    return (row["accuracy"] - RETENTION_TARGET * full_acc) * n_prompts


def headline_by_context(per_ctx: dict[int, dict[str, Any]], policies: list[str]) -> dict[str, Any]:
    """Brief §B8's number, recomputed at every context rather than quoted from 32K."""
    out: dict[str, Any] = {"retention_target": RETENTION_TARGET, "by_policy": {}}
    for policy in policies:
        per = {}
        for ctx, d in sorted(per_ctx.items()):
            h = d["headline"].get(policy, {})
            full_acc, n = d["full_niah_accuracy"], d["niah_prompts_per_condition"]
            best = max((r for r in d["niah"] if r["policy"] == policy and r["retention"] is not None),
                       key=lambda r: r["retention"], default=None)
            per[str(ctx)] = {
                "min_budget_meeting_target": h.get("min_budget_meeting_target"),
                "best_retention_below_full": h.get("best_retention_below_full"),
                "best_retention_budget": h.get("best_retention_budget"),
                "tokens_per_s_at_that_budget": h.get("tokens_per_s_at_that_budget"),
                # How thin the verdict is, in prompts, at the budget that scored best.
                "best_margin_prompts": _margin_prompts(best, full_acc, n) if best else None,
                "best_accuracy_ci95": best["accuracy_ci95"] if best else None,
                "prompts": n,
            }
        met = [c for c, v in per.items() if v["min_budget_meeting_target"] is not None]
        out["by_policy"][policy] = {
            "by_context": per,
            "contexts_meeting_target": sorted(int(c) for c in met),
            "meets_target_at_every_context": len(met) == len(per_ctx),
            "meets_target_at_no_context": not met,
            "best_retention_span": span([v["best_retention_below_full"] for v in per.values() if v["best_retention_below_full"] is not None]),
        }
    return out


# ---- 4. does the 32K column reproduce Phase 7? ------------------------------------------


def phase7_agreement(per_ctx: dict[int, dict[str, Any]], context: int = 32768) -> dict[str, Any]:
    """The overlap check the combined frontier needs: this sweep's 32K column against Phase 7's.

    Phase 8 re-measures conditions Phase 7 already published, on the same prompts and the same
    block size. That is a deliberate overlap, not waste: without it the context sweep would be a
    separate study whose 32K point merely resembles the one every other phase is built on.
    """
    src = RESULTS_DIR / "phase7" / "analysis" / "metrics.json"
    if context not in per_ctx or not src.exists():
        return {"compared": False, "reason": "phase7 analysis or 32768 column absent"}
    p7 = json.loads(src.read_text(encoding="utf-8"))
    mine_n = {r["label"]: r for r in per_ctx[context]["niah"]}
    mine_s = {r["label"]: r for r in per_ctx[context]["speed"]}
    p7_n = {r["label"]: r for r in p7["niah"]}
    p7_s = {r["label"]: r for r in p7["speed"]}
    rows = []
    for lab in sorted(set(mine_n) & set(p7_n)):
        a, b = mine_n[lab], p7_n[lab]
        sa, sb = mine_s.get(lab), p7_s.get(lab)
        rows.append({
            "label": lab,
            "prompts": a["prompts"],
            "accuracy_phase8": a["accuracy"],
            "accuracy_phase7": b["accuracy"],
            "accuracy_gap_pp": 100 * abs(a["accuracy"] - b["accuracy"]),
            "tokens_per_s_phase8": sa["tokens_per_s"]["median"] if sa and sa["tokens_per_s"].get("n_runs") else None,
            "tokens_per_s_phase7": sb["tokens_per_s"]["median"] if sb and sb["tokens_per_s"].get("n_runs") else None,
        })
    for r in rows:
        r["tokens_per_s_ratio"] = max(_ratio(r["tokens_per_s_phase8"], r["tokens_per_s_phase7"]) or 1.0,
                                      _ratio(r["tokens_per_s_phase7"], r["tokens_per_s_phase8"]) or 1.0)
    gaps = [r["accuracy_gap_pp"] for r in rows]
    ratios = [r["tokens_per_s_ratio"] for r in rows if r["tokens_per_s_ratio"]]
    return {
        "compared": True,
        "context": context,
        "conditions": len(rows),
        "rows": rows,
        "max_accuracy_gap_pp": max(gaps) if gaps else None,
        "max_tokens_per_s_ratio": max(ratios) if ratios else None,
        "accuracy_reproduces": all(g < 1e-9 for g in gaps) if gaps else None,
    }


# ---- driver -----------------------------------------------------------------------------


def load_context(base: Path, ctx: int, policies: list[str]) -> dict[str, Any] | None:
    """One context's quality run and its speed runs, reduced with the shared sweep helpers."""
    qpath = base / f"policy_quality_ctx{ctx}" / "metrics.json"
    if not qpath.exists():
        return None
    q = json.loads(qpath.read_text(encoding="utf-8"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((base / f"budget_speed_ctx{ctx}").glob("run_*/metrics.json"))]
    niah, full_scores = niah_table(q)
    speed = speed_table(runs)
    return {
        "context": ctx,
        "niah": niah,
        "teacher_forced": teacher_forced_table(q),
        "speed": speed,
        "speed_runs": len(runs),
        "headline": headline(niah, speed, policies),
        "full_niah_accuracy": statistics.fmean(full_scores.values()),
        "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
        "niah_prompts_per_condition": len(full_scores),
        "kv_bytes_per_token": q["kv_bytes_per_token"],
        "quality_provenance": q["provenance"],
        "speed_provenance": [r["provenance"] for r in runs],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(RESULTS_DIR / "phase8"))
    p.add_argument("--config", default=None, help="config the sweep ran (defaults to configs/phase8.yaml)")
    args = p.parse_args()
    base = Path(args.base)

    import yaml

    cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parents[1] / "configs" / "phase8.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    policies, budgets, block_size = cfg["policies"], cfg["budgets"], cfg["block_size"]

    per_ctx: dict[int, dict[str, Any]] = {}
    for ctx in cfg["contexts"]:
        d = load_context(base, ctx, policies)
        if d is not None:
            per_ctx[ctx] = d
    if not per_ctx:
        raise SystemExit(f"no context results under {base}")

    col = collapse(per_ctx, policies, block_size)
    scale = scaling(per_ctx, policies, budgets, block_size)
    head = headline_by_context(per_ctx, policies)
    agree = phase7_agreement(per_ctx)

    contexts = sorted(per_ctx)
    by_label = {s["label"]: s for s in scale["series"]}
    full_series = by_label.get("full@1")
    tps = {
        pol: {str(c): next((s["tokens_per_s"] for s in per_ctx[c]["speed"] if s["policy"] == pol and s["budget"] == b), {}).get("median")
              for c in contexts}
        for pol, b in [(FULL, 1.0)] + [(p_, min(budgets)) for p_ in policies]
    }

    summary = {
        "contexts": contexts,
        "contexts_measured": len(contexts),
        "block_size": block_size,
        "budgets": budgets,
        "policies": policies,
        "niah_prompts_per_condition": per_ctx[contexts[0]]["niah_prompts_per_condition"],
        "speed_runs_per_context": {str(c): per_ctx[c]["speed_runs"] for c in contexts},
        "full_accuracy_by_context": {str(c): per_ctx[c]["full_niah_accuracy"] for c in contexts},
        # 1. Which collapse holds retention tighter.
        "collapse_mean_spread_by_budget_pp": col["summary_by_budget"]["mean_spread_pp"],
        "collapse_mean_spread_by_k_pp": col["summary_by_k_blocks"]["mean_spread_pp"],
        "collapse_max_spread_by_budget_pp": col["summary_by_budget"]["max_spread_pp"],
        "collapse_max_spread_by_k_pp": col["summary_by_k_blocks"]["max_spread_pp"],
        "collapse_fraction_over_k": col["fraction_over_k_mean_spread"],
        "collapse_groups_by_k": col["summary_by_k_blocks"]["groups"],
        # 2. What scales.
        "full_decode_ms_by_context": {str(p["context"]): p["decode_ms_per_token"] for p in full_series["points"]} if full_series else None,
        "full_decode_first_to_last": full_series["decode_ms_first_to_last"] if full_series else None,
        "full_decode_intercept_share": full_series["decode_intercept_share_at_max_ctx"] if full_series else None,
        "manager_intercept_share": span([s["manager_intercept_share_at_max_ctx"] for s in scale["series"] if s["manager_intercept_share_at_max_ctx"] is not None]),
        "over_full_by_context": {s["label"]: s["over_full_by_context"] for s in scale["series"] if s["policy"] != FULL},
        "tokens_per_s_by_context_at_smallest_budget": tps,
        # 3. The headline, per context.
        "headline_meets_target_at_every_context": [p_ for p_ in policies if head["by_policy"][p_]["meets_target_at_every_context"]],
        "headline_meets_target_at_no_context": [p_ for p_ in policies if head["by_policy"][p_]["meets_target_at_no_context"]],
        "headline_min_budget_by_context": {p_: {c: v["min_budget_meeting_target"] for c, v in head["by_policy"][p_]["by_context"].items()} for p_ in policies},
        # The verdict flips between contexts for at least one rung; this says by how little.
        "headline_margin_prompts_by_context": {p_: {c: v["best_margin_prompts"] for c, v in head["by_policy"][p_]["by_context"].items()} for p_ in policies},
        "headline_smallest_abs_margin_prompts": min(
            (abs(v["best_margin_prompts"]) for p_ in policies for v in head["by_policy"][p_]["by_context"].values() if v["best_margin_prompts"] is not None),
            default=None,
        ),
        # 4. The overlap with Phase 7.
        "phase7_max_accuracy_gap_pp": agree.get("max_accuracy_gap_pp"),
        "phase7_max_tokens_per_s_ratio": agree.get("max_tokens_per_s_ratio"),
        "phase7_accuracy_reproduces": agree.get("accuracy_reproduces"),
        "phase7_conditions_compared": agree.get("conditions"),
    }

    write_metrics(
        base / "analysis",
        {
            "sources": {str(c): {"policy_quality": per_ctx[c]["quality_provenance"], "budget_speed": per_ctx[c]["speed_provenance"]} for c in contexts},
            "block_size": block_size,
            "budgets": budgets,
            "policies": policies,
            "per_context": {str(c): {k: v for k, v in per_ctx[c].items() if not k.endswith("provenance")} for c in contexts},
            "collapse": col,
            "scaling": scale,
            "headline_by_context": head,
            "phase7_agreement": agree,
            "summary": summary,
        },
    )
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
