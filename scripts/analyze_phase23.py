"""Derive Phase 23 quantities: the zero-copy tier against rung 6 and rung 5.

Reads   results/phase23/budget_speed/run_*/metrics.json (decode latency, fast kernel, idle machine)
        results/phase23/policy_quality/metrics.json     (NIAH + teacher-forced, quality kernel)
Writes  results/phase23/analysis/metrics.json

1. Speed: per condition, medians across runs (harness.sweep_analysis.speed_table), and the zero-copy
   tier's latency paired against rung 6 and against rung 5 within each (run, repeat), since within
   a repeat every condition decodes from the same prefill. The crossover: budgets at which the
   zero-copy tier is slower than rung 6 in the median pair.
2. Memory: resident KV bytes per condition, from the same runs.
3. Validity: the zero-copy tier's NIAH answers and teacher-forced KL against rung 5's, prompt by
   prompt and window by window; any difference fails the check.
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
from harness.sweep_analysis import label, niah_table, span, speed_table  # noqa: E402
from scripts.analyze_phase4 import prompt_key  # noqa: E402

ZC, TIER, QUEST = "tiered_zerocopy", "tiered_sync", "quest"
MIB = 2**20


def repeat_medians(runs: list[dict[str, Any]], pol: str, b: float) -> dict[tuple[int, int], float]:
    return {(i, r["repeat"]): r["decode_wall_s"]["median"] for i, run in enumerate(runs) for r in run["results"] if r["policy"] == pol and r["budget"] == b}


def paired(runs: list[dict[str, Any]], ref_pol: str, b: float) -> dict[str, Any]:
    ref, got = repeat_medians(runs, ref_pol, b), repeat_medians(runs, ZC, b)
    keys = [k for k in got if k in ref]
    ratios = [got[k] / ref[k] for k in keys]
    deltas = [1e3 * (got[k] - ref[k]) for k in keys]
    return {
        "ratio_median": statistics.median(ratios) if ratios else None,
        "ratio_span": span(ratios),
        "delta_ms_median": statistics.median(deltas) if deltas else None,
        "zc_faster_pairs": sum(r < 1 for r in ratios),
        "pairs": len(ratios),
    }


def validity(q: dict[str, Any]) -> dict[str, Any]:
    answers = {(r["policy"], r["budget"]) + prompt_key(r): r for r in q["niah"]}
    tf = {(r["policy"], r["budget"], r["offset"]): r for r in q["teacher_forced"]}
    out: dict[str, Any] = {"budgets": {}}
    for b in sorted({r["budget"] for r in q["niah"] if r["policy"] == ZC}, reverse=True):
        keys = [k for k in answers if k[0] == ZC and k[1] == b]
        same = sum(1 for k in keys if (QUEST,) + k[1:] in answers and answers[k]["answer"] == answers[(QUEST,) + k[1:]]["answer"])
        offs = sorted(o for (p, bb, o) in tf if p == ZC and bb == b)
        kl_same = sum(1 for o in offs if (QUEST, b, o) in tf and tf[(ZC, b, o)]["mean_kl"] == tf[(QUEST, b, o)]["mean_kl"])
        out["budgets"][f"{b:g}"] = {"prompts": len(keys), "answers_same": same, "windows": len(offs), "kl_same": kl_same}
    bs = out["budgets"].values()
    out["prompts"] = sum(v["prompts"] for v in bs)
    out["answers_same"] = sum(v["answers_same"] for v in bs)
    out["windows"] = sum(v["windows"] for v in bs)
    out["kl_same"] = sum(v["kl_same"] for v in bs)
    out["all_same"] = out["prompts"] > 0 and out["answers_same"] == out["prompts"] and out["kl_same"] == out["windows"]
    niah, full_scores = niah_table(q)
    out["accuracy"] = {n["label"]: n["accuracy"] for n in niah}
    out["full_accuracy"] = statistics.mean(full_scores.values()) if full_scores else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", default="policy_quality")
    p.add_argument("--speed", default="budget_speed")
    args = p.parse_args()
    base = RESULTS_DIR / "phase23"
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / args.speed).glob("run_*/metrics.json"))]
    qpath = base / args.quality / "metrics.json"
    q = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else None
    cfg = runs[0]["config"]
    budgets = sorted(cfg["budgets"], reverse=True)

    speed = speed_table(runs, ("tier.zerocopy_bytes", "tier.fetched_pairs"))
    sp = {s["label"]: s for s in speed}
    per_budget = []
    for b in budgets:
        row: dict[str, Any] = {"budget": b}
        for pol in (QUEST, TIER, ZC):
            s = sp.get(label(pol, b))
            row[pol] = None if s is None else {
                "decode_ms": 1e3 * s["decode_wall_median_s"]["median"],
                "decode_ms_min": 1e3 * s["decode_wall_median_s"]["min"],
                "decode_ms_max": 1e3 * s["decode_wall_median_s"]["max"],
                "tokens_per_s": s["tokens_per_s"]["median"],
                "manager_host_ms": 1e3 * s["manager_host_s_per_token"]["median"],
                "resident_mib": s["gpu_resident_kv_bytes"]["median"] / MIB,
            }
        row["vs_tier"] = paired(runs, TIER, b)
        row["vs_quest"] = paired(runs, QUEST, b)
        per_budget.append(row)
    full = sp.get(label("full", 1.0))
    summary: dict[str, Any] = {
        "context": cfg["context"],
        "runs": len(runs),
        "repeats": cfg["speed"]["repeats"],
        "full_decode_ms": None if full is None else 1e3 * full["decode_wall_median_s"]["median"],
        "full_resident_mib": None if full is None else full["gpu_resident_kv_bytes"]["median"] / MIB,
        "speedup_vs_tier": {f"{r['budget']:g}": 1 / r["vs_tier"]["ratio_median"] for r in per_budget if r["vs_tier"]["ratio_median"]},
        "ratio_vs_quest": {f"{r['budget']:g}": r["vs_quest"]["ratio_median"] for r in per_budget if r["vs_quest"]["ratio_median"]},
        # Budgets where the median pair has the zero-copy tier slower than rung 6.
        "slower_than_tier_at": [r["budget"] for r in per_budget if r["vs_tier"]["ratio_median"] and r["vs_tier"]["ratio_median"] > 1],
        "any_spill": any(s["any_spill"] for s in speed),
    }
    sv = summary["speedup_vs_tier"].values()
    summary["speedup_vs_tier_min"], summary["speedup_vs_tier_max"] = (min(sv), max(sv)) if sv else (None, None)
    rq = summary["ratio_vs_quest"].values()
    summary["ratio_vs_quest_min"], summary["ratio_vs_quest_max"] = (min(rq), max(rq)) if rq else (None, None)
    out: dict[str, Any] = {"speed_sources": [r["provenance"] for r in runs], "speed": speed, "per_budget": per_budget, "summary": summary}
    if q is not None:
        out["quality_source"] = q["provenance"]
        out["validity"] = validity(q)
        summary["valid"] = out["validity"]["all_same"]
    write_metrics(base / "analysis", out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
