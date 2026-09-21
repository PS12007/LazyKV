"""Derive Phase 7 decision quantities: rung 9 (exact partial attention) against rung 6 and the full cache.

Reads   results/phase7/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
        results/phase7/budget_speed/run_*/metrics.json (decode latency and tier counters, fast kernel)
Writes  results/phase7/analysis/metrics.json

Every earlier rung was measured against the rung below it, because each one was a variation on the
same approximation. Rung 9 is not: it is the *full cache*, computed in two places, so the full cache
is what it has to be compared against. Three quantities follow from that:

1. **Exactness, measured rather than asserted.** Rung 9 is exact in exact arithmetic. It is not
   bit-identical to the full cache, because the GPU half runs the bf16 kernel over the gathered set,
   the CPU half runs float32 over the rest, and the two are summed in a different order and rounded
   once more. So the claim has to be stated as a distance, and the number that makes it meaningful
   is rung 6's distance on the same prompts: the question is not "is it zero" but "how many orders
   of magnitude closer than the rung it replaces".
2. **The price, in three currencies.** Decode latency against rung 6 and against the full cache;
   host RAM for the float32 mirror; and CPU time, which is the thing actually being spent.
3. **Whether the CPU pass is flat in the budget.** It should be, by construction: the pass masks
   the resident blocks instead of gathering the cold ones (`lazykv/exact.py`, decision 2). If the
   measurement disagrees with the design, the design note is wrong and this is where it shows.
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
from harness.sweep_analysis import cond_rows, headline, label, niah_table, span, speed_table, teacher_forced_table  # noqa: E402
from scripts.analyze_phase4 import prompt_key, tier_rates  # noqa: E402

APPROX = "tiered_sync"  # rung 6: the approximate tier rung 9 makes exact
EXACT = "tiered_exact"  # rung 9
FULL = "full"
MIB = 2**20


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def distance_from_full(q: dict[str, Any], budgets: list[float], policies: list[str]) -> list[dict[str, Any]]:
    """Per policy and budget: how far the output sits from the full cache's, on the same prompts.

    Reported for rung 9 *and* rung 6, because rung 9's number means nothing on its own. A reader
    who sees "mean KL 1e-3" cannot tell whether that is close; a reader who sees it beside rung 6's
    on the same prompts can.
    """
    answers = {(r["policy"], r["budget"]) + prompt_key(r): r for r in q["niah"]}
    tf = {(r["policy"], r["budget"], r["offset"]): r for r in q["teacher_forced"]}
    out = []
    for policy in policies:
        for b in budgets:
            keys = [k for k in answers if k[0] == policy and k[1] == b]
            ref = [(FULL, 1.0) + k[2:] for k in keys]
            if not keys or any(r not in answers for r in ref):
                continue
            differing = [k for k, r in zip(keys, ref) if answers[k]["answer"] != answers[r]["answer"]]
            broke = sum(1 for k, r in zip(keys, ref) if answers[k]["score"] < answers[r]["score"])
            fixed = sum(1 for k, r in zip(keys, ref) if answers[k]["score"] > answers[r]["score"])
            offsets = sorted({o for (p, bb, o) in tf if p == policy and bb == b})
            rows = [tf[(policy, b, o)] for o in offsets]
            out.append({
                "policy": policy,
                "budget": b,
                "prompts": len(keys),
                "answers_differing": len(differing),
                "answers_differing_share": len(differing) / len(keys),
                "answers_broken": broke,
                "answers_fixed": fixed,
                "score_delta_mean": statistics.mean(answers[k]["score"] - answers[r]["score"] for k, r in zip(keys, ref)),
                "tf_mean_kl": statistics.mean(r["mean_kl"] for r in rows) if rows else None,
                "tf_top1_agreement": statistics.mean(r["top1_agreement"] for r in rows) if rows else None,
                "tf_top1_min_doc": min((r["top1_agreement"] for r in rows), default=None),
            })
    return out


def exact_rates(runs: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    """Rung 9's own counters per decoded token: the CPU pass, the merge around it, the mirror.

    Separate from `tier_rates` because these are additive to it, not a variation on it: rung 9 pays
    everything rung 6 pays and then this.
    """
    rows = [r for run in runs for r in run["results"] if r["policy"] == EXACT and r["budget"] == budget and "tier" in r]
    if not rows:
        return {}
    cfg = runs[0]["config"]["speed"]
    n_decode = cfg["warmup_steps"] + cfg["decode_tokens"]

    def med(key: str, scale: float = 1.0) -> float | None:
        vals = [r["tier"][key] * scale for r in rows if key in r["tier"]]
        return statistics.median(vals) if vals else None

    merges = med("exact_merges")
    layers = merges / n_decode if merges else None
    return {
        "cpu_attention_ms_per_token": med("exact_cpu_attention_s", 1e3 / n_decode),
        "merge_ms_per_token": med("exact_host_merge_s", 1e3 / n_decode),
        # The merge minus the CPU pass: the query pull, the GPU log-sum-exp and the blend.
        "merge_overhead_ms_per_token": None if med("exact_host_merge_s") is None else (med("exact_host_merge_s", 1e3 / n_decode) - med("exact_cpu_attention_s", 1e3 / n_decode)),
        "mirror_ms_per_token": med("exact_mirror_s", 1e3 / n_decode),
        "boundary_mirror_s": med("exact_boundary_mirror_s"),
        "cpu_attention_ms_per_layer": None if not layers else med("exact_cpu_attention_s", 1e3 / n_decode) / layers,
        "selecting_layers": layers,
        "cpu_tokens_per_merge": None if not merges else med("exact_cpu_tokens") / merges,
        "cold_tokens_per_merge": None if not merges else med("exact_cold_tokens") / merges,
        "cold_share_of_cpu_tokens": _ratio(med("exact_cold_tokens"), med("exact_cpu_tokens")),
        # The point of design (c): what comes back is O(head_dim) per query head, not block bytes.
        "returned_kib_per_token": med("exact_returned_bytes", 1.0 / 1024 / n_decode),
        "mirror_mib": med("exact_mirror_bytes", 1.0 / MIB),
        "threads": med("exact_threads"),
    }


def price(tiers: dict[str, Any], exact: dict[str, Any], speed: list[dict[str, Any]], budgets: list[float]) -> list[dict[str, Any]]:
    """Per budget: what rung 9 costs against rung 6 and against the full cache, and in what."""
    sp = {s["label"]: s for s in speed}
    full_ms = 1e3 * sp["full@1"]["decode_wall_median_s"]["median"] if "full@1" in sp else None
    out = []
    for b in budgets:
        a = tiers.get(APPROX, {}).get(f"{b:g}") or {}
        e = tiers.get(EXACT, {}).get(f"{b:g}") or {}
        x = exact.get(f"{b:g}") or {}
        if not a or not e:
            continue
        approx_ms, exact_ms = 1e3 * a["decode_wall_median_s"], 1e3 * e["decode_wall_median_s"]
        out.append({
            "budget": b,
            "approx_ms_per_token": approx_ms,
            "exact_ms_per_token": exact_ms,
            "full_ms_per_token": full_ms,
            "over_approx": _ratio(exact_ms, approx_ms),
            "over_full": _ratio(exact_ms, full_ms),
            "added_ms_per_token": exact_ms - approx_ms,
            "cpu_attention_ms_per_token": x.get("cpu_attention_ms_per_token"),
            "merge_overhead_ms_per_token": x.get("merge_overhead_ms_per_token"),
            # How much of the added time the CPU pass and its merge account for. A large remainder
            # would mean rung 9 is paying somewhere neither counter looks.
            "added_explained_share": _ratio(x.get("merge_ms_per_token"), exact_ms - approx_ms),
            "resident_mib_approx": (a.get("gpu_resident_kv_bytes") or 0) / MIB,
            "resident_mib_exact": (e.get("gpu_resident_kv_bytes") or 0) / MIB,
            "host_pinned_mib": e.get("host_pinned_mib"),
            "mirror_mib": x.get("mirror_mib"),
            # The whole trade, in one row: VRAM saved against host RAM spent.
            "host_mib_per_vram_mib_saved": _ratio((e.get("host_pinned_mib") or 0) + (x.get("mirror_mib") or 0), full_resident_minus(sp, e)),
        })
    return out


def full_resident_minus(sp: dict[str, Any], tier: dict[str, Any]) -> float | None:
    full = sp.get("full@1", {}).get("gpu_resident_kv_bytes", {}).get("median")
    if full is None or tier.get("gpu_resident_kv_bytes") is None:
        return None
    saved = (full - tier["gpu_resident_kv_bytes"]) / MIB
    return saved if saved > 0 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="policy_quality")
    p.add_argument("--speed", default="budget_speed")
    args = p.parse_args()
    base = RESULTS_DIR / "phase7"
    q = json.loads((base / args.quality / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / args.speed).glob("run_*/metrics.json"))]
    cfg = q["config"]
    policies = cfg["policies"]
    budgets = sorted(cfg["budgets"])

    niah, full_scores = niah_table(q)
    full_acc = statistics.mean(full_scores.values())
    teacher_forced = teacher_forced_table(q)
    speed = speed_table(runs, ("tier.host_kv_bytes", "host_pinned_bytes", "tier.exact_mirror_bytes"))
    head = headline(niah, speed, policies)
    tiers = {pol: {f"{b:g}": tier_rates(runs, pol, b) for b in budgets} for pol in policies}
    exact = {f"{b:g}": exact_rates(runs, b) for b in budgets}

    dist = distance_from_full(q, budgets, policies)
    costs = price(tiers, exact, speed, budgets) if speed else []

    sp = {s["label"]: s for s in speed}
    nb = {pol: cond_rows(niah, pol) for pol in policies}
    tfb = {(r["policy"], r["budget"]): r for r in teacher_forced}
    db = {(d["policy"], d["budget"]): d for d in dist}
    cb = {c["budget"]: c for c in costs}
    cpu_ms = [e["cpu_attention_ms_per_token"] for e in exact.values() if e.get("cpu_attention_ms_per_token") is not None]

    summary: dict[str, Any] = {
        "block_size": cfg["block_size"],
        "context": cfg["context"],
        "niah_prompts": len(full_scores),
        "retention_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["retention"] for b in budgets if b in nb[pol]} for pol in policies},
        "accuracy_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["accuracy"] for b in budgets if b in nb[pol]} for pol in policies},
        # 1. Exactness, as a distance and always beside the rung it replaces.
        "retention": {pol: span([nb[pol][b]["retention"] for b in budgets if b in nb[pol]]) for pol in policies},
        "answers_differing_from_full": {pol: sum(d["answers_differing"] for d in dist if d["policy"] == pol) for pol in policies},
        "answers_compared_from_full": {pol: sum(d["prompts"] for d in dist if d["policy"] == pol) for pol in policies},
        "tf_mean_kl": {pol: span([d["tf_mean_kl"] for d in dist if d["policy"] == pol and d["tf_mean_kl"] is not None]) for pol in policies},
        "tf_top1_agreement": {pol: span([d["tf_top1_agreement"] for d in dist if d["policy"] == pol and d["tf_top1_agreement"] is not None]) for pol in policies},
        "kl_ratio_approx_over_exact": span([
            r for b in budgets
            if (r := _ratio((db.get((APPROX, b)) or {}).get("tf_mean_kl"), (db.get((EXACT, b)) or {}).get("tf_mean_kl"))) is not None
        ]),
        "tf_kl_at": {f"b{round(100 * b)}": {pol: (db.get((pol, b)) or {}).get("tf_mean_kl") for pol in policies} for b in budgets},
        # 2. The price.
        "over_approx": span([c["over_approx"] for c in costs if c["over_approx"] is not None]),
        "over_full": span([c["over_full"] for c in costs if c["over_full"] is not None]),
        "added_ms_per_token": span([c["added_ms_per_token"] for c in costs if c["added_ms_per_token"] is not None]),
        "added_explained_share": span([c["added_explained_share"] for c in costs if c["added_explained_share"] is not None]),
        "over_approx_at": {f"b{round(100 * b)}": cb[b]["over_approx"] for b in budgets if b in cb},
        "over_full_at": {f"b{round(100 * b)}": cb[b]["over_full"] for b in budgets if b in cb},
        "tokens_per_s_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["tokens_per_s"]["median"] for b in budgets if label(pol, b) in sp} for pol in policies},
        "resident_mib_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["gpu_resident_kv_bytes"]["median"] / MIB for b in budgets if label(pol, b) in sp} for pol in policies},
        "mirror_mib": span([e["mirror_mib"] for e in exact.values() if e.get("mirror_mib") is not None]),
        "host_pinned_mib": span([c["host_pinned_mib"] for c in costs if c["host_pinned_mib"] is not None]),
        "returned_kib_per_token": span([e["returned_kib_per_token"] for e in exact.values() if e.get("returned_kib_per_token") is not None]),
        # 3. The design note, tested: the CPU pass should not respond to the budget.
        "cpu_attention_ms_per_token": span(cpu_ms),
        "cpu_attention_spread_ratio": _ratio(max(cpu_ms), min(cpu_ms)) if cpu_ms else None,
        "cpu_attention_ms_per_layer": span([e["cpu_attention_ms_per_layer"] for e in exact.values() if e.get("cpu_attention_ms_per_layer") is not None]),
        "cold_share_of_cpu_tokens": span([e["cold_share_of_cpu_tokens"] for e in exact.values() if e.get("cold_share_of_cpu_tokens") is not None]),
        "merge_overhead_ms_per_token": span([e["merge_overhead_ms_per_token"] for e in exact.values() if e.get("merge_overhead_ms_per_token") is not None]),
        "cpu_threads": next((e["threads"] for e in exact.values() if e.get("threads") is not None), None),
        "dense_layers": next((r["dense_layers"] for r in q["niah"] if r["policy"] in (APPROX, EXACT)), None),
        "rung9_meets_target": any(
            nb[EXACT][b]["retention"] is not None and nb[EXACT][b]["retention"] >= 0.99 for b in budgets if b in nb[EXACT]
        ),
        "rung9_min_budget_meeting_target": head.get(EXACT, {}).get("min_budget_meeting_target"),
    }
    if speed:
        summary["full_resident_mib"] = sp["full@1"]["gpu_resident_kv_bytes"]["median"] / MIB
        summary["tokens_per_s_full"] = sp["full@1"]["tokens_per_s"]["median"]
        summary["conditions_with_spill"] = sum(1 for s in speed if s["any_spill"])
    summary["tf_top1_at"] = {f"b{round(100 * b)}": {pol: (tfb.get((pol, b)) or {}).get("top1_agreement") for pol in policies} for b in budgets}

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
            "exact": exact,
            "distance_from_full": dist,
            "price": costs,
            "summary": summary,
        },
    )
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
