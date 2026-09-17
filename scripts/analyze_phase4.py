"""Derive Phase 4 decision quantities: the CPU tier (rungs 6-7) against query-aware selection (rung 5).

Reads   results/phase4/policy_quality/metrics.json          (NIAH + teacher-forced, quality kernel)
        results/phase4/budget_speed/run_*/metrics.json       (decode latency and tier counters, fast kernel)
        results/phase4/blocksize_speed_bs*/run_*/metrics.json (block-size sweep, if present)
        results/phase4/blocksize_quality_bs*/metrics.json     (block-size sweep quality, if present)
Writes  results/phase4/analysis/metrics.json

Rungs 6-7 attend to exactly rung 5's key sequences (tests/test_tiered.py), so the first thing this
analysis computes is whether the real run agrees: identical NIAH answers and identical teacher-forced
divergence. Everything after that is about what the tier costs: bytes moved, host time, latency,
and how much of each prefetch copy finished before the layer that needed it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci  # noqa: E402
from harness.sweep_analysis import across, cond_rows, headline, label, niah_table, per_run, span, speed_table, teacher_forced_table  # noqa: E402

REFERENCE = "quest"  # rung 5; rungs 6-7 must reproduce it
TIERED = ("tiered_sync", "tiered_prefetch")
MIB = 2**20


def prompt_key(r: dict[str, Any]) -> tuple[str, float, int]:
    return (r["kind"], r["depth"], r["sample"])


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def identity(q: dict[str, Any], policies: list[str], budgets: list[float]) -> dict[str, Any]:
    """Per tiered policy: prompts whose answer differs from rung 5's, and teacher-forced deltas."""
    answers = {(r["policy"], r["budget"]) + prompt_key(r): r["answer"] for r in q["niah"]}
    tf = {(r["policy"], r["budget"], r["offset"]): r for r in q["teacher_forced"]}
    out = {}
    for pol in [p for p in policies if p in TIERED]:
        keys = [k for k in answers if k[0] == pol]
        diff = sum(1 for k in keys if answers[k] != answers[(REFERENCE,) + k[1:]])
        tf_rows = [(r, tf[(REFERENCE, r["budget"], r["offset"])]) for (p, _, _), r in tf.items() if p == pol]
        out[pol] = {
            "niah_prompt_conditions": len(keys),
            "niah_answers_differing": diff,
            "tf_conditions": len(tf_rows),
            "tf_max_abs_kl_diff": max((abs(a["mean_kl"] - b["mean_kl"]) for a, b in tf_rows), default=None),
            "tf_max_abs_top1_diff": max((abs(a["top1_agreement"] - b["top1_agreement"]) for a, b in tf_rows), default=None),
        }
    return out


def tier_rates(runs: list[dict[str, Any]], policy: str, budget: float) -> dict[str, Any]:
    """Cache and PCIe behaviour per decoded token, from every repeat of every run (warmup steps excluded where per-step data allows)."""
    rows = [r for run in runs for r in run["results"] if r["policy"] == policy and r["budget"] == budget and "tier" in r]
    if not rows:
        return {}
    warm = runs[0]["config"]["speed"]["warmup_steps"]
    n_decode = warm + runs[0]["config"]["speed"]["decode_tokens"]

    def med(fn: Any) -> float:
        return statistics.median(fn(r) for r in rows)

    steady = [x for r in rows for x in r["tier"]["fetched_pairs_per_step"][warm:]]
    pair_bytes = rows[0]["tier"]["fetch_bytes"] / rows[0]["tier"]["fetched_pairs"] if rows[0]["tier"]["fetched_pairs"] else None
    t = lambda r: r["tier"]  # noqa: E731
    out: dict[str, Any] = {
        "k_blocks": rows[0]["k_blocks"],
        "hit_rate": med(lambda r: t(r)["hit_pairs"] / t(r)["selected_pairs"]),
        "fetched_pairs_per_token": med(lambda r: t(r)["fetched_pairs"] / n_decode),
        "fetched_pairs_per_token_steady_median": statistics.median(steady) if steady else None,
        "fetched_pairs_per_token_steady_p90": pct(steady, 0.9) if steady else None,
        "first_step_fetched_pairs": med(lambda r: t(r)["fetched_pairs_per_step"][0] if t(r)["fetched_pairs_per_step"] else 0),
        "fetch_mib_per_token": med(lambda r: t(r)["fetch_bytes"] / n_decode / MIB),
        "fetch_transfers_per_token": med(lambda r: t(r)["fetch_transfers"] / n_decode),
        "fetch_mean_transfer_kib": med(lambda r: t(r)["fetch_bytes"] / t(r)["fetch_transfers"] / 1024 if t(r)["fetch_transfers"] else 0.0),
        "thrash_share_of_fetches": med(lambda r: t(r)["thrash_pairs"] / t(r)["fetched_pairs"] if t(r)["fetched_pairs"] else 0.0),
        "selected_pairs_per_token": med(lambda r: t(r)["selected_pairs"] / n_decode),
        "host_select_ms_per_token": med(lambda r: 1e3 * t(r)["host_select_s"] / n_decode),
        "host_fetch_ms_per_token": med(lambda r: 1e3 * t(r)["host_fetch_s"] / n_decode),
        "host_prefetch_ms_per_token": med(lambda r: 1e3 * t(r)["host_prefetch_s"] / n_decode),
        "host_seal_ms_per_token": med(lambda r: 1e3 * t(r)["host_seal_s"] / n_decode),
        "boundary_d2h_s": med(lambda r: t(r)["boundary_d2h_s"]),
        "boundary_d2h_mib": med(lambda r: t(r)["boundary_d2h_bytes"] / MIB),
        "host_pinned_mib": med(lambda r: r["host_pinned_bytes"] / MIB),
        "pair_kib": None if pair_bytes is None else pair_bytes / 1024,
    }
    if policy == "tiered_prefetch":
        out.update({
            "prefetched_pairs_per_token": med(lambda r: t(r)["prefetched_pairs"] / n_decode),
            "prefetch_mib_per_token": med(lambda r: t(r)["prefetch_bytes"] / n_decode / MIB),
            "prefetch_transfers_per_token": med(lambda r: t(r)["prefetch_transfers"] / n_decode),
            # Precision: share of prefetched pairs the real selection then chose.
            "prefetch_precision": med(lambda r: t(r)["prefetch_used_pairs"] / t(r)["prefetched_pairs"] if t(r)["prefetched_pairs"] else 0.0),
            # Coverage: share of the pairs that had to come from host that the prefetch brought.
            "prefetch_coverage": med(lambda r: t(r)["prefetch_used_pairs"] / (t(r)["prefetch_used_pairs"] + t(r)["fetched_pairs"]) if t(r)["prefetch_used_pairs"] + t(r)["fetched_pairs"] else 0.0),
        })
    return out


def overlap_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[float, dict[str, list[float]]] = {}
    for run in runs:
        for o in run.get("prefetch_overlap", []):
            if not o.get("prefetches_timed"):
                continue
            acc = by.setdefault(o["budget"], {"overlap_fraction": [], "copy_ms": [], "stall_ms": [], "window_ms": []})
            for k in acc:
                acc[k].extend(o[k])
    out = {}
    for b, acc in sorted(by.items(), reverse=True):
        fr = acc["overlap_fraction"]
        out[f"{b:g}"] = {
            "prefetches": len(fr),
            "overlap_fraction_median": statistics.median(fr),
            "overlap_fraction_p10": pct(fr, 0.1),
            "fully_hidden_share": sum(1 for x in fr if x >= 0.999) / len(fr),
            "copy_ms_median": statistics.median(acc["copy_ms"]),
            "copy_ms_p90": pct(acc["copy_ms"], 0.9),
            "window_ms_median": statistics.median(acc["window_ms"]),
            "stall_ms_median": statistics.median(acc["stall_ms"]),
            "stall_ms_p90": pct(acc["stall_ms"], 0.9),
            "stalled_share": sum(1 for x in acc["stall_ms"] if x > 0) / len(fr),
        }
    return out


def blocksize_sweep(base: Path) -> dict[str, Any] | None:
    speed_dirs = sorted(base.glob("blocksize_speed_bs*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
    if not speed_dirs:
        return None
    rows = []
    for d in speed_dirs:
        bs = int(re.sub(r"\D", "", d.name))
        runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("run_*/metrics.json"))]
        qpath = base / f"blocksize_quality_bs{bs}" / "metrics.json"
        q = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else None
        speed = {s["label"]: s for s in speed_table(runs)}
        niah = {n["label"]: n for n in niah_table(q)[0]} if q else {}
        tf = {t["label"]: t for t in teacher_forced_table(q)} if q else {}
        for policy in runs[0]["config"]["policies"]:
            for b in runs[0]["config"]["budgets"]:
                lab = label(policy, b)
                rows.append({
                    "block_size": bs,
                    "policy": policy,
                    "budget": b,
                    "decode_wall_median_s": speed[lab]["decode_wall_median_s"],
                    "tokens_per_s": speed[lab]["tokens_per_s"],
                    "manager_host_s_per_token": speed[lab]["manager_host_s_per_token"],
                    "niah_accuracy": niah[lab]["accuracy"] if lab in niah else None,
                    "niah_retention": niah[lab]["retention"] if lab in niah else None,
                    "tf_mean_kl": tf[lab]["mean_kl"] if lab in tf else None,
                    "tier": tier_rates(runs, policy, b),
                    "provenance": [r["provenance"] for r in runs] + ([q["provenance"]] if q else []),
                })
    return {"rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="policy_quality")
    p.add_argument("--speed", default="budget_speed")
    args = p.parse_args()
    base = RESULTS_DIR / "phase4"
    q = json.loads((base / args.quality / "metrics.json").read_text(encoding="utf-8"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / args.speed).glob("run_*/metrics.json"))]
    cfg = q["config"]
    policies = cfg["policies"]
    budgets = sorted(cfg["budgets"])

    niah, full_scores = niah_table(q)
    full_acc = statistics.mean(full_scores.values())
    teacher_forced = teacher_forced_table(q)
    extra = ("tier.fetched_pairs", "host_pinned_bytes")
    speed = speed_table(runs, extra)
    head = headline(niah, speed, policies)

    tiers = {pol: {f"{b:g}": tier_rates(runs, pol, b) for b in budgets} for pol in policies if pol in TIERED}

    # Paired latency: within a repeat, both conditions decode from the same prefill, interleaved.
    def repeat_medians(pol: str, b: float) -> dict[tuple[int, int], float]:
        return {(i, r["repeat"]): r["decode_wall_s"]["median"] for i, run in enumerate(runs) for r in run["results"] if r["policy"] == pol and r["budget"] == b}

    latency_vs_quest: dict[str, list[dict[str, Any]]] = {}
    for pol in [p for p in policies if p in TIERED]:
        rows = []
        for b in budgets:
            ref, got = repeat_medians(REFERENCE, b), repeat_medians(pol, b)
            ratios = [got[k] / ref[k] for k in got if k in ref]
            deltas = [1e3 * (got[k] - ref[k]) for k in got if k in ref]
            rows.append({"budget": b, "ratio_median": statistics.median(ratios) if ratios else None, "ratio_span": span(ratios), "delta_ms_median": statistics.median(deltas) if deltas else None, "pairs": len(ratios)})
        latency_vs_quest[pol] = rows

    sp = {s["label"]: s for s in speed}
    nb = {pol: cond_rows(niah, pol) for pol in policies}
    summary: dict[str, Any] = {
        "block_size": cfg["block_size"],
        "niah_prompts": len(full_scores),
        "retention_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["retention"] for b in budgets} for pol in policies},
        "accuracy_at": {pol: {f"b{round(100 * b)}": nb[pol][b]["accuracy"] for b in budgets} for pol in policies},
        "resident_mib_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["gpu_resident_kv_bytes"]["median"] / MIB for b in budgets} for pol in policies if speed},
        "tokens_per_s_at": {pol: {f"b{round(100 * b)}": sp[label(pol, b)]["tokens_per_s"]["median"] for b in budgets} for pol in policies if speed},
        "latency_ratio_vs_quest": {pol: span([r["ratio_median"] for r in rows if r["ratio_median"] is not None]) for pol, rows in latency_vs_quest.items()},
        "manager_ms_per_token": {pol: span([sp[label(pol, b)]["manager_host_s_per_token"]["median"] * 1e3 for b in budgets]) for pol in policies if speed},
        "hit_rate": {pol: span([v["hit_rate"] for v in rows.values() if v]) for pol, rows in tiers.items()},
        "fetch_mib_per_token": {pol: span([v["fetch_mib_per_token"] for v in rows.values() if v]) for pol, rows in tiers.items()},
    }
    if speed:
        summary["full_resident_mib"] = sp["full@1"]["gpu_resident_kv_bytes"]["median"] / MIB
        summary["tokens_per_s_full"] = sp["full@1"]["tokens_per_s"]["median"]
        summary["conditions_with_spill"] = sum(1 for s in speed if s["any_spill"])
        summary["full_repeat_medians"] = span([r["decode_wall_s"]["median"] for run in runs for r in run["results"] if r["policy"] == "full"])
        # Rung 5 is re-measured here: its latency against Phase 3's is the cross-phase repeatability check.
        p3_runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RESULTS_DIR / "phase3" / "budget_speed").glob("run_*/metrics.json"))]
        if p3_runs:
            summary["quest_vs_phase3"] = {
                f"b{round(100 * b)}": across(per_run(runs, REFERENCE, b, "decode_wall_s.median"))["median"] / across(per_run(p3_runs, REFERENCE, b, "decode_wall_s.median"))["median"]
                for b in budgets
            }

    write_metrics(
        base / "analysis",
        {
            "sources": {"policy_quality": q["provenance"], "budget_speed": [r["provenance"] for r in runs]},
            "context": cfg["context"],
            "block_size": cfg["block_size"],
            "kv_bytes_per_token": q["kv_bytes_per_token"],
            "full_niah_accuracy": full_acc,
            "full_niah_accuracy_ci95": bootstrap_mean_ci(list(full_scores.values())),
            "niah": niah,
            "teacher_forced": teacher_forced,
            "speed": speed,
            "speed_runs": len(runs),
            "headline": head,
            "identity_vs_quest": identity(q, policies, budgets),
            "tier": tiers,
            "latency_vs_quest": latency_vs_quest,
            "prefetch_overlap": overlap_summary(runs),
            "blocksize": blocksize_sweep(base),
            "summary": summary,
        },
    )
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
