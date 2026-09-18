"""Derive Phase 5 decision quantities: the int8 warm/cold tier (rung 8) against the exact tier (rung 6).

Reads   results/phase5/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase5/budget_speed/run_*/metrics.json (decode latency and tier counters, fast kernel)
Writes  results/phase5/analysis/metrics.json

Rung 8 is the first tier rung that is *not* bit-identical to rung 5, so unlike Phase 4 this
analysis leads with quality rather than checking it away. It computes three things:

1. **Capacity.** What the host pool holds against what the same blocks would cost in bf16, and the
   same ratio in bytes moved: the boundary D2H, the seals and the on-demand fetches.
2. **Quality.** NIAH accuracy against the exact tier and against the full cache, teacher-forced KL
   and top-1 agreement, and how many NIAH answers actually changed. Rung 8 quantizes the block
   metadata as well as the blocks, so it can also *select differently*; the fetch counters show
   that indirectly and are reported next to the quality numbers rather than apart from them.
3. **Latency.** Paired against rung 6 within a repeat, plus where the extra host time goes. Phase 4
   measured this decode as host-bound, so the prediction on the record is that narrowing the bytes
   buys no time and the dequantization costs some.
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
from harness.stats import bootstrap_mean_ci, sign_test_p  # noqa: E402
from harness.sweep_analysis import cond_rows, headline, label, niah_table, span, speed_table, teacher_forced_table  # noqa: E402
from scripts.analyze_phase4 import prompt_key, tier_rates  # noqa: E402

EXACT = "tiered_sync"  # rung 6: bit-identical to rung 5 (Phase 4), so it is this phase's exact reference
QUANT = "tiered_int8"  # rung 8
MIB = 2**20


def divergence(q: dict[str, Any], budgets: list[float]) -> list[dict[str, Any]]:
    """Per budget: how far rung 8's output moved from the exact tier's, prompt by prompt.

    Phase 4 could assert zero here. Rung 8 cannot, so the deltas are the result, reported with the
    direction of the accuracy change rather than only its magnitude.
    """
    answers = {(r["policy"], r["budget"]) + prompt_key(r): r for r in q["niah"]}
    tf = {(r["policy"], r["budget"], r["offset"]): r for r in q["teacher_forced"]}
    out = []
    for b in budgets:
        keys = [k for k in answers if k[0] == QUANT and k[1] == b]
        if not keys or any((EXACT,) + k[1:] not in answers for k in keys):
            continue
        differing = [k for k in keys if answers[k]["answer"] != answers[(EXACT,) + k[1:]]["answer"]]
        # Of the prompts whose answer changed, how many went from right to wrong, and how many back.
        broke = sum(1 for k in differing if answers[k]["score"] < answers[(EXACT,) + k[1:]]["score"])
        fixed = sum(1 for k in differing if answers[k]["score"] > answers[(EXACT,) + k[1:]]["score"])
        offsets = sorted({o for (p, bb, o) in tf if p == QUANT and bb == b})
        rows = [(tf[(QUANT, b, o)], tf[(EXACT, b, o)]) for o in offsets if (EXACT, b, o) in tf]
        out.append({
            "budget": b,
            "prompts": len(keys),
            "answers_differing": len(differing),
            "answers_differing_share": len(differing) / len(keys),
            "answers_broken": broke,
            "answers_fixed": fixed,
            # Paired: the same prompts, so the question is whether the discordant prompts lean one
            # way more than a coin would. Without this a swing of a few prompts in 45 reads as a
            # quality change when it is sampling noise.
            "sign_test_p": sign_test_p(broke, fixed),
            "score_delta_mean": statistics.mean(answers[k]["score"] - answers[(EXACT,) + k[1:]]["score"] for k in keys),
            "tf_mean_kl": statistics.mean(a["mean_kl"] for a, _ in rows) if rows else None,
            "tf_mean_kl_exact": statistics.mean(e["mean_kl"] for _, e in rows) if rows else None,
            "tf_kl_increase": statistics.mean(a["mean_kl"] - e["mean_kl"] for a, e in rows) if rows else None,
            "tf_top1_agreement": statistics.mean(a["top1_agreement"] for a, _ in rows) if rows else None,
            "tf_top1_agreement_exact": statistics.mean(e["top1_agreement"] for _, e in rows) if rows else None,
        })
    return out


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def tier_rates_with_capacity(runs: list[dict[str, Any]], policy: str, budget: float) -> dict[str, Any]:
    """Phase 4's tier rates plus what the host pool holds, which is what this phase is about.

    Kept here rather than added to `tier_rates`: Phase 4's analysis output is rendered into a
    committed document, and widening it would make that document stale for no reason.
    """
    out = tier_rates(runs, policy, budget)
    rows = [r for run in runs for r in run["results"] if r["policy"] == policy and r["budget"] == budget and "tier" in r]
    if not rows:
        return out
    t = [r["tier"] for r in rows]
    out["host_kv_mib"] = statistics.median(x["host_kv_bytes"] / MIB for x in t)
    out["host_kv_logical_mib"] = statistics.median(x.get("host_kv_logical_bytes", x["host_kv_bytes"]) / MIB for x in t)
    out["tier_quant"] = t[0].get("tier_quant", "none")
    return out


def capacity(tiers: dict[str, Any], budgets: list[float]) -> list[dict[str, Any]]:
    """Per budget: what each tier holds off-GPU and what it moves, and rung 8's ratio against rung 6."""
    out = []
    for b in budgets:
        e = tiers.get(EXACT, {}).get(f"{b:g}") or {}
        k = tiers.get(QUANT, {}).get(f"{b:g}") or {}
        if not e or not k:
            continue
        row: dict[str, Any] = {"budget": b}
        for name, t in (("exact", e), ("int8", k)):
            row[f"host_kv_mib_{name}"] = t.get("host_kv_mib")
            row[f"host_pinned_mib_{name}"] = t.get("host_pinned_mib")
            row[f"fetch_mib_per_token_{name}"] = t.get("fetch_mib_per_token")
            row[f"boundary_d2h_mib_{name}"] = t.get("boundary_d2h_mib")
            row[f"pair_kib_{name}"] = t.get("pair_kib")
            row[f"resident_mib_{name}"] = (t.get("gpu_resident_kv_bytes") or 0) / MIB
        # The capacity claim, three ways: what is stored, what is pinned, what crosses the link.
        row["host_kv_ratio"] = _ratio(row["host_kv_mib_int8"], row["host_kv_mib_exact"])
        row["host_pinned_ratio"] = _ratio(row["host_pinned_mib_int8"], row["host_pinned_mib_exact"])
        row["boundary_d2h_ratio"] = _ratio(row["boundary_d2h_mib_int8"], row["boundary_d2h_mib_exact"])
        row["pair_ratio"] = _ratio(row["pair_kib_int8"], row["pair_kib_exact"])
        # Bytes per token is *not* just the record ratio: rung 8 also fetches a different number of
        # pairs, because quantizing the metadata changes what the bound ranks highest.
        row["fetch_mib_ratio"] = _ratio(row["fetch_mib_per_token_int8"], row["fetch_mib_per_token_exact"])
        row["fetched_pairs_ratio"] = _ratio(k.get("fetched_pairs_per_token"), e.get("fetched_pairs_per_token"))
        out.append(row)
    return out


def host_time(tiers: dict[str, Any], budgets: list[float]) -> list[dict[str, Any]]:
    """Where rung 8's extra host time goes: ranking is unchanged, fetching now dequantizes."""
    out = []
    for b in budgets:
        e = tiers.get(EXACT, {}).get(f"{b:g}") or {}
        k = tiers.get(QUANT, {}).get(f"{b:g}") or {}
        if not e or not k:
            continue
        row: dict[str, Any] = {"budget": b}
        for key in ("host_select_ms_per_token", "host_rank_ms_per_token", "host_fetch_ms_per_token", "host_seal_ms_per_token"):
            row[f"{key}_exact"], row[f"{key}_int8"] = e.get(key), k.get(key)
            row[f"{key}_delta"] = None if e.get(key) is None or k.get(key) is None else k[key] - e[key]
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="policy_quality")
    p.add_argument("--speed", default="budget_speed")
    args = p.parse_args()
    base = RESULTS_DIR / "phase5"
    q = json.loads((base / args.quality / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / args.speed).glob("run_*/metrics.json"))]
    cfg = q["config"]
    policies = cfg["policies"]
    budgets = sorted(cfg["budgets"])

    niah, full_scores = niah_table(q)
    full_acc = statistics.mean(full_scores.values())
    teacher_forced = teacher_forced_table(q)
    speed = speed_table(runs, ("tier.fetched_pairs", "tier.host_kv_bytes", "tier.host_kv_logical_bytes", "host_pinned_bytes"))
    head = headline(niah, speed, policies)
    tiers = {pol: {f"{b:g}": tier_rates_with_capacity(runs, pol, b) for b in budgets} for pol in policies}

    # Paired latency: within a repeat both conditions decode from the same prefill, interleaved.
    def repeat_medians(pol: str, b: float) -> dict[tuple[int, int], float]:
        return {(i, r["repeat"]): r["decode_wall_s"]["median"] for i, run in enumerate(runs) for r in run["results"] if r["policy"] == pol and r["budget"] == b}

    latency = []
    for b in budgets:
        ref, got = repeat_medians(EXACT, b), repeat_medians(QUANT, b)
        ratios = [got[k] / ref[k] for k in got if k in ref]
        deltas = [1e3 * (got[k] - ref[k]) for k in got if k in ref]
        latency.append({
            "budget": b,
            "ratio_median": statistics.median(ratios) if ratios else None,
            "ratio_span": span(ratios),
            "delta_ms_median": statistics.median(deltas) if deltas else None,
            "pairs": len(ratios),
        })

    div = divergence(q, budgets)
    cap = capacity(tiers, budgets)
    times = host_time(tiers, budgets)
    # Budgets at which the tier really offloads. Above them the slot count is capped at the candidate
    # block count, every block stays resident, and the tier degenerates to rung 5 plus bookkeeping --
    # so a latency span taken over all budgets would be diluted by conditions that move no bytes.
    # Same definition as Phase 4's `tier_active_budgets`.
    candidates = cfg["context"] // cfg["block_size"] - 1
    active = [b for b in budgets if (tiers[QUANT].get(f"{b:g}") or {}).get("n_slots", 0) < candidates]
    sp = {s["label"]: s for s in speed}
    nb = {pol: cond_rows(niah, pol) for pol in policies}
    dvb = {d["budget"]: d for d in div}
    cpb = {c["budget"]: c for c in cap}
    lat = {r["budget"]: r for r in latency}
    summary: dict[str, Any] = {
        "block_size": cfg["block_size"],
        "context": cfg["context"],
        "niah_prompts": len(full_scores),
        "retention_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["retention"] for b in budgets} for pol in policies},
        "accuracy_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["accuracy"] for b in budgets} for pol in policies},
        "tokens_per_s_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["tokens_per_s"]["median"] for b in budgets} for pol in policies if speed},
        "resident_mib_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["gpu_resident_kv_bytes"]["median"] / MIB for b in budgets} for pol in policies if speed},
        # Capacity: the headline of this phase.
        "host_kv_ratio": span([c["host_kv_ratio"] for c in cap if c["host_kv_ratio"] is not None]),
        "host_pinned_ratio": span([c["host_pinned_ratio"] for c in cap if c["host_pinned_ratio"] is not None]),
        "pair_ratio": span([c["pair_ratio"] for c in cap if c["pair_ratio"] is not None]),
        "boundary_d2h_ratio": span([c["boundary_d2h_ratio"] for c in cap if c["boundary_d2h_ratio"] is not None]),
        "fetch_mib_ratio": span([c["fetch_mib_ratio"] for c in cap if c["fetch_mib_ratio"] is not None]),
        "host_kv_mib_at": {f"b{round(100 * b)}": {"exact": cpb[b]["host_kv_mib_exact"], "int8": cpb[b]["host_kv_mib_int8"]} for b in budgets if b in cpb},
        "host_pinned_mib_at": {f"b{round(100 * b)}": {"exact": cpb[b]["host_pinned_mib_exact"], "int8": cpb[b]["host_pinned_mib_int8"]} for b in budgets if b in cpb},
        "fetch_mib_at": {f"b{round(100 * b)}": {"exact": cpb[b]["fetch_mib_per_token_exact"], "int8": cpb[b]["fetch_mib_per_token_int8"]} for b in budgets if b in cpb},
        # Quality: the axis the tier did not have before rung 8.
        "answers_differing_total": sum(d["answers_differing"] for d in div),
        "answers_differing_share": span([d["answers_differing_share"] for d in div]),
        "answers_broken_total": sum(d["answers_broken"] for d in div),
        "answers_fixed_total": sum(d["answers_fixed"] for d in div),
        "sign_test_p_min": min((d["sign_test_p"] for d in div), default=None),
        "sign_test_p_at": {f"b{round(100 * d['budget'])}": d["sign_test_p"] for d in div},
        "retention_delta_at": {f"b{round(100 * b)}": nb[QUANT][b]["retention"] - nb[EXACT][b]["retention"] for b in budgets},
        "retention_delta": span([nb[QUANT][b]["retention"] - nb[EXACT][b]["retention"] for b in budgets]),
        "tf_kl_increase": span([d["tf_kl_increase"] for d in div if d["tf_kl_increase"] is not None]),
        "tf_top1_agreement": span([d["tf_top1_agreement"] for d in div if d["tf_top1_agreement"] is not None]),
        "tf_kl_at": {f"b{round(100 * b)}": {"exact": dvb[b]["tf_mean_kl_exact"], "int8": dvb[b]["tf_mean_kl"]} for b in budgets if b in dvb},
        # Latency: the prediction Phase 4 put on the record, tested.
        "latency_ratio_vs_exact": span([r["ratio_median"] for r in latency if r["ratio_median"] is not None]),
        "latency_ratio_offloading": span([lat[b]["ratio_median"] for b in active if lat.get(b, {}).get("ratio_median") is not None]),
        "fetch_ms_delta_offloading": span([h["host_fetch_ms_per_token_delta"] for h in times if h["budget"] in active and h["host_fetch_ms_per_token_delta"] is not None]),
        "active_budgets": [f"{b:g}" for b in active],
        "inactive_budgets": [f"{b:g}" for b in budgets if b not in active],
        # The boundary is the one place the narrower record costs time instead of saving it: the
        # prompt KV is packed on the GPU before it crosses the link.
        "boundary_d2h_s": {pol: span([(tiers[pol].get(f"{b:g}") or {}).get("boundary_d2h_s") for b in budgets if (tiers[pol].get(f"{b:g}") or {}).get("boundary_d2h_s") is not None]) for pol in policies},
        "boundary_d2h_mib": {pol: span([(tiers[pol].get(f"{b:g}") or {}).get("boundary_d2h_mib") for b in budgets if (tiers[pol].get(f"{b:g}") or {}).get("boundary_d2h_mib") is not None]) for pol in policies},
        # Pooled over every budget: the overall verdict on whether rung 8 changed accuracy at all.
        "sign_test_p_pooled": sign_test_p(sum(d["answers_broken"] for d in div), sum(d["answers_fixed"] for d in div)),
        "latency_ratio_at": {f"b{round(100 * b)}": lat[b]["ratio_median"] for b in budgets if lat.get(b, {}).get("ratio_median") is not None},
        "latency_delta_ms_at": {f"b{round(100 * b)}": lat[b]["delta_ms_median"] for b in budgets if lat.get(b, {}).get("delta_ms_median") is not None},
        "fetch_ms_delta": span([h["host_fetch_ms_per_token_delta"] for h in times if h["host_fetch_ms_per_token_delta"] is not None]),
        "rank_ms_per_token": {pol: span([(tiers[pol].get(f"{b:g}") or {}).get("host_rank_ms_per_token") for b in budgets if (tiers[pol].get(f"{b:g}") or {}).get("host_rank_ms_per_token") is not None]) for pol in policies},
        "manager_ms_per_token": {pol: span([sp[label(pol, b)]["manager_host_s_per_token"]["median"] * 1e3 for b in budgets]) for pol in policies if speed},
        "fetched_pairs_ratio": span([c["fetched_pairs_ratio"] for c in cap if c["fetched_pairs_ratio"] is not None]),
        "dense_layers": next((r["dense_layers"] for r in q["niah"] if r["policy"] in (EXACT, QUANT)), None),
    }
    if speed:
        summary["full_resident_mib"] = sp["full@1"]["gpu_resident_kv_bytes"]["median"] / MIB
        summary["tokens_per_s_full"] = sp["full@1"]["tokens_per_s"]["median"]
        summary["conditions_with_spill"] = sum(1 for s in speed if s["any_spill"])

    write_metrics(
        base / "analysis",
        {
            "sources": {"policy_quality": q["provenance"], "budget_speed": [r["provenance"] for r in runs]},
            "context": cfg["context"],
            "block_size": cfg["block_size"],
            "kv_bytes_per_token": q["kv_bytes_per_token"],
            "full_niah_accuracy": full_acc,
            "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
            "niah_prompts_per_condition": len(full_scores),
            "niah": niah,
            "teacher_forced": teacher_forced,
            "speed": speed,
            "speed_runs": len(runs),
            "headline": head,
            "tier": tiers,
            "divergence_vs_exact": div,
            "capacity": cap,
            "host_time": times,
            "latency_vs_exact": latency,
            "summary": summary,
        },
    )
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
