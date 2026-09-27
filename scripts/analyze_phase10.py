"""Derive Phase 10 quantities: long-document perplexity per policy, budget and context (brief §B7.3).

Reads   results/phase10/perplexity_ctx*/metrics.json   (one run per context, quality kernel)
Writes  results/phase10/analysis/metrics.json

Every run scores the same 256-token windows of the same books (`configs/phase10.yaml` fixes the
scored tokens and varies only the context before them), so every comparison here is paired twice
over: a condition against the full cache on the same window, and a context against another on the
same window.

The effect is reported as a paired NLL delta in nats and as the perplexity ratio it exponentiates
to. Its CI is a bootstrap over *windows*, not tokens: the 256 tokens of one window share a context
and are not independent, and resampling them would report an interval several times too narrow.
No pass/fail bar is applied; the config says why.

At the ladder's context the analysis also joins each condition's NIAH retention from
`results/ladder/metrics.json`, measured on the same model, block size and slot setting. That is
the comparison this phase exists for: whether a perplexity budget would have told a reader what the
retrieval budget did.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci, sign_test_p  # noqa: E402
from lazykv.selection import blocks_for_budget  # noqa: E402

FULL = "full"
CONTROL = "block_full"
Window = tuple[str, int]  # (book, target): one scored window


def label(policy: str, budget: float) -> str:
    """A key a dotted template lookup can address: "quest_6_25", not "quest@0.0625"."""
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def load_runs(base: Path) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Teacher-forced rows per context, and each run's provenance.

    Refuses runs from different commits or a dirty tree: the contexts are only one experiment if
    they ran the same code.
    """
    rows: dict[int, list[dict[str, Any]]] = {}
    prov: list[dict[str, Any]] = []
    for path in sorted(base.glob("perplexity_ctx*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        ctx = int(m["config"]["context"])
        rows[ctx] = m["teacher_forced"]
        prov.append({"run": path.parent.name, "context": ctx, "wall_s": m["wall_s"], **m["provenance"]})
    if not prov:
        raise SystemExit(f"no runs under {base}")
    commits = {p["git_commit"] for p in prov}
    if len(commits) != 1 or any(p["git_dirty"] for p in prov):
        raise SystemExit(f"runs disagree on provenance: commits {commits}, dirty {[p['run'] for p in prov if p['git_dirty']]}")
    return rows, prov


def by_window(rows: list[dict[str, Any]], policy: str, budget: float, field: str) -> dict[Window, float]:
    return {(r["book"], r["target"]): r[field] for r in rows if r["policy"] == policy and r["budget"] == budget}


def paired_delta(cond: dict[Window, float], ref: dict[Window, float], iters: int, seed: int) -> dict[str, Any]:
    """Mean paired difference cond - ref over windows, its bootstrap CI, and the windows' signs."""
    keys = sorted(k for k in ref if k in cond)
    d = [cond[k] - ref[k] for k in keys]
    worse, better = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    mean = sum(d) / len(d)
    lo, hi = bootstrap_mean_ci(d, iters=iters, seed=seed)
    return {"windows": len(d), "delta": mean, "delta_ci95": [lo, hi],
            "windows_worse": worse, "windows_better": better, "sign_test_p": sign_test_p(worse, better)}


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, ties given their mean rank. Across conditions, so n is ~20: descriptive."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def ladder_retention(path: Path, context: int) -> dict[str, float]:
    """NIAH retention per condition label from the ladder, if it was measured at `context`."""
    if not path.exists():
        return {}
    m = json.loads(path.read_text(encoding="utf-8"))
    if m["summary"]["context"] != context:
        return {}
    return {label(r["policy"], r["budget"]): r["retention"] for r in m["rows"]}


def condition(rows: list[dict[str, Any]], policy: str, budget: float, iters: int, seed: int) -> dict[str, Any]:
    ref = by_window(rows, FULL, 1.0, "mean_nll")
    nll = by_window(rows, policy, budget, "mean_nll")
    kl = by_window(rows, policy, budget, "mean_kl")
    top1 = by_window(rows, policy, budget, "top1_agreement")
    dn = paired_delta(nll, ref, iters, seed)
    mean_nll = sum(nll.values()) / len(nll)
    return {
        "policy": policy,
        "budget": budget,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "nll_delta": dn["delta"],
        "nll_delta_ci95": dn["delta_ci95"],
        # exp of a mean NLL delta is the ratio of the two perplexities, since every window scores
        # the same number of tokens; its CI is the delta's, exponentiated.
        "ppl_ratio": math.exp(dn["delta"]),
        "ppl_ratio_ci95": [math.exp(x) for x in dn["delta_ci95"]],
        "windows": dn["windows"],
        "windows_worse": dn["windows_worse"],
        "windows_better": dn["windows_better"],
        "sign_test_p": dn["sign_test_p"],
        "mean_kl": sum(kl.values()) / len(kl),
        "top1_agreement": sum(top1.values()) / len(top1),
    }


def collapse(per_ctx: dict[str, Any], policies: list[str], block_size: int) -> dict[str, Any]:
    """Each policy's NLL delta grouped across context by budget fraction, and by blocks kept (K).

    Phase 8 asked the same question of NIAH retention and found the fraction travels: a policy at
    25% retains about the same share of retrieval at 4K as at 32K. Perplexity need not behave the
    same way, since most tokens are predicted from nearby text, and nearby text is an absolute
    amount rather than a share. Whichever grouping leaves less spread is the one that governs.
    K uses Phase 8's definition (`blocks_for_budget`), so the two phases' answers are comparable.
    Groups spanning one context are dropped: they have zero spread by construction.
    """
    points = [
        {"context": int(ctx), "policy": c["policy"], "budget": c["budget"],
         "k_blocks": blocks_for_budget(c["budget"], int(ctx), block_size), "nll_delta": c["nll_delta"]}
        for ctx, d in per_ctx.items() for c in d["conditions"].values() if c["policy"] in policies
    ]

    def spreads(key: str) -> list[dict[str, Any]]:
        out = []
        for pol in policies:
            for k in sorted({q[key] for q in points if q["policy"] == pol}):
                sel = [q for q in points if q["policy"] == pol and q[key] == k]
                if len({q["context"] for q in sel}) < 2:
                    continue
                d = [q["nll_delta"] for q in sel]
                out.append({"policy": pol, "value": k, "contexts": sorted(q["context"] for q in sel), "spread_nats": max(d) - min(d)})
        return out

    by_budget, by_k = spreads("budget"), spreads("k_blocks")
    mean = lambda rows: sum(r["spread_nats"] for r in rows) / len(rows) if rows else None  # noqa: E731
    fr, kk = mean(by_budget), mean(by_k)
    return {
        "by_budget": by_budget,
        "by_k_blocks": by_k,
        "groups_by_budget": len(by_budget),
        "groups_by_k": len(by_k),
        "mean_spread_by_budget_nats": fr,
        "mean_spread_by_k_nats": kk,
        # >1 means grouping by blocks kept holds perplexity tighter than grouping by the fraction:
        # the opposite of Phase 8's answer for retrieval.
        "fraction_over_k_mean_spread": fr / kk if fr is not None and kk else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(RESULTS_DIR / "phase10"))
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase10.yaml"))
    args = p.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iters, seed = cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"]
    runs, prov = load_runs(base)
    contexts = sorted(runs)
    conds = [(CONTROL, 1.0)] + [(pol, b) for pol in cfg["policies"] for b in cfg["budgets"]]

    per_ctx: dict[str, Any] = {}
    for ctx in contexts:
        rows = runs[ctx]
        ref = by_window(rows, FULL, 1.0, "mean_nll")
        full_nll = sum(ref.values()) / len(ref)
        per_ctx[str(ctx)] = {
            "windows": len(ref),
            "tokens": len(ref) * cfg["teacher_forced"]["continuation"],
            "full_nll": full_nll,
            "full_perplexity": math.exp(full_nll),
            "conditions": {label(pol, b): condition(rows, pol, b, iters, seed) for pol, b in conds},
        }

    # The reference against itself across context: the same windows, more text before them. This
    # is the check that the corpus rewards long context at all; if the full cache did not get
    # better with more context, no policy's loss of context could be measured on it.
    longest = str(contexts[-1])
    context_effect = {}
    for ctx in contexts[:-1]:
        d = paired_delta(by_window(runs[ctx], FULL, 1.0, "mean_nll"), by_window(runs[contexts[-1]], FULL, 1.0, "mean_nll"), iters, seed)
        context_effect[str(ctx)] = {"nll_over_longest": d["delta"], "nll_over_longest_ci95": d["delta_ci95"],
                                    "ppl_ratio_over_longest": math.exp(d["delta"]), "windows_worse": d["windows_worse"]}

    niah = ladder_retention(RESULTS_DIR / "ladder" / "metrics.json", contexts[-1])
    for k, c in per_ctx[longest]["conditions"].items():
        c["niah_retention"] = niah.get(k)
    policy_conds = [c for c in per_ctx[longest]["conditions"].values() if c["policy"] != CONTROL]
    col = collapse(per_ctx, cfg["policies"], cfg["block_size"]) if len(contexts) > 1 else None
    joined = [c for c in policy_conds if c["niah_retention"] is not None]
    # "Blind" = NIAH lost at least a tenth of the full cache's retrieval, while the perplexity CI
    # still includes no change: a reader holding only perplexity would call the condition lossless.
    blind = [c for c in joined if c["niah_retention"] < 0.9 and c["ppl_ratio_ci95"][0] <= 1.0 <= c["ppl_ratio_ci95"][1]]
    worst_blind = min(blind, key=lambda c: c["niah_retention"]) if blind else None
    tightest = min(cfg["budgets"])
    summary = {
        "contexts": contexts,
        "contexts_measured": len(contexts),
        "longest_context": contexts[-1],
        "windows": per_ctx[longest]["windows"],
        "tokens_per_condition": per_ctx[longest]["tokens"],
        "full_perplexity_by_context": {str(c): per_ctx[str(c)]["full_perplexity"] for c in contexts},
        "control_max_abs_nll_delta": max(abs(per_ctx[str(c)]["conditions"][label(CONTROL, 1.0)]["nll_delta"]) for c in contexts),
        # At the longest context: each policy's perplexity ratio at every budget, and at the tightest.
        "ppl_ratio_longest": {label(c["policy"], c["budget"]): c["ppl_ratio"] for c in policy_conds},
        "ppl_ratio_tightest_longest": {pol: per_ctx[longest]["conditions"][label(pol, tightest)]["ppl_ratio"] for pol in cfg["policies"]},
        "tightest_budget": tightest,
        # Conditions whose whole CI excludes no change, at the longest context (either direction).
        "resolved_worse_longest": sorted(label(c["policy"], c["budget"]) for c in policy_conds if c["nll_delta_ci95"][0] > 0),
        "resolved_better_longest": sorted(label(c["policy"], c["budget"]) for c in policy_conds if c["nll_delta_ci95"][1] < 0),
        "resolved_worse_count_longest": sum(1 for c in policy_conds if c["nll_delta_ci95"][0] > 0),
        "resolved_better_count_longest": sum(1 for c in policy_conds if c["nll_delta_ci95"][1] < 0),
        "largest_budget_resolved_worse_longest": max((c["budget"] for c in policy_conds if c["nll_delta_ci95"][0] > 0), default=None),
        "niah_joined_conditions": len(joined),
        "ppl_blind_conditions": sorted(label(c["policy"], c["budget"]) for c in blind),
        "ppl_blind_count": len(blind),
        "ppl_blind_worst": None if worst_blind is None else {
            "label": label(worst_blind["policy"], worst_blind["budget"]), "policy": worst_blind["policy"],
            "budget": worst_blind["budget"], "niah_retention": worst_blind["niah_retention"],
            "ppl_ratio": worst_blind["ppl_ratio"], "ppl_ratio_ci95": worst_blind["ppl_ratio_ci95"]},
        # How well each quality signal orders the conditions by their NIAH loss.
        "spearman_nll_vs_niah_loss": spearman([c["nll_delta"] for c in joined], [1 - c["niah_retention"] for c in joined]) if len(joined) > 2 else None,
        "spearman_kl_vs_niah_loss": spearman([c["mean_kl"] for c in joined], [1 - c["niah_retention"] for c in joined]) if len(joined) > 2 else None,
        "resolved_conditions_longest": len(policy_conds),
        "collapse_mean_spread_by_budget_nats": None if col is None else col["mean_spread_by_budget_nats"],
        "collapse_mean_spread_by_k_nats": None if col is None else col["mean_spread_by_k_nats"],
        "collapse_fraction_over_k": None if col is None else col["fraction_over_k_mean_spread"],
        "collapse_groups_by_k": None if col is None else col["groups_by_k"],
        # The noise floor the collapse spreads must be read against: the median half-width of one
        # condition's NLL-delta CI. A spread smaller than this is inside one measurement's noise.
        "median_delta_ci_halfwidth_nats": sorted(
            (c["nll_delta_ci95"][1] - c["nll_delta_ci95"][0]) / 2
            for d in per_ctx.values() for c in d["conditions"].values() if c["policy"] != CONTROL
        )[len(policy_conds) * len(contexts) // 2],
        # Does a fixed fraction cost less perplexity at the longest context than at the shortest?
        # Counted over the conditions whose cost is resolved at the shortest context, since an
        # unresolved one has no cost to shrink.
        "cost_falls_with_context": sum(
            1 for k, c in per_ctx[str(contexts[0])]["conditions"].items()
            if c["policy"] != CONTROL and c["nll_delta_ci95"][0] > 0 and per_ctx[longest]["conditions"][k]["nll_delta"] < c["nll_delta"]
        ),
        "cost_resolved_at_shortest": sum(1 for c in per_ctx[str(contexts[0])]["conditions"].values() if c["policy"] != CONTROL and c["nll_delta_ci95"][0] > 0),
        "shortest_context": contexts[0],
        # Readings better than the full cache with a CI excluding no change, over every condition at
        # every context: the base rate the two at the longest context should be judged against.
        "resolved_better_all_contexts": sum(1 for d in per_ctx.values() for c in d["conditions"].values() if c["policy"] != CONTROL and c["nll_delta_ci95"][1] < 0),
        "policy_conditions_all_contexts": sum(1 for d in per_ctx.values() for c in d["conditions"].values() if c["policy"] != CONTROL),
        "full_ppl_ratio_shortest_over_longest": context_effect[str(contexts[0])]["ppl_ratio_over_longest"] if str(contexts[0]) in context_effect else None,
        "full_ppl_ratio_shortest_over_longest_ci95": [math.exp(x) for x in context_effect[str(contexts[0])]["nll_over_longest_ci95"]] if str(contexts[0]) in context_effect else None,
        "full_windows_worse_shortest": context_effect[str(contexts[0])]["windows_worse"] if str(contexts[0]) in context_effect else None,
        "wall_s": sum(pv["wall_s"] for pv in prov),
    }
    write_metrics(base / "analysis", {"sources": prov, "contexts": per_ctx, "context_effect": context_effect, "collapse": col, "summary": summary})


if __name__ == "__main__":
    main()
