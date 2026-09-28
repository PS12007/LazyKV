"""Derive Phase 15 quantities: the 3B stress model against the 1B, prompt for prompt, at 16K.

Reads   results/phase15/policy_quality/metrics.json          (3B, nf4 weights)
        results/phase8/policy_quality_ctx16384/metrics.json  (1B, same prompts)
Writes  results/phase15/analysis/metrics.json

The two runs answered the same NIAH prompts (seed 0, samples 0-2 per kind and depth) under the same
tokenizer, which the analysis checks rather than assumes: a prompt is paired only if both runs
recorded the same needle values at the same position. Each model's retention is against its own
full cache. The model difference in retention gets a bootstrap CI that resamples prompts jointly
for both models, so a prompt that is hard for both moves both retentions together.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import paired_ratio_ci  # noqa: E402

Key = tuple[str, float, int]


def label(policy: str, budget: float) -> str:
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def scores(rows: list[dict[str, Any]], policy: str, budget: float) -> dict[Key, float]:
    return {(r["kind"], r["depth"], r["sample"]): r["score"] for r in rows if r["policy"] == policy and r["budget"] == budget}


def prompt_identity(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    ident = lambda rows: {(r["kind"], r["depth"], r["sample"]): (tuple(r["values"]), r["target_needle_pos"]) for r in rows if r["policy"] == "full"}  # noqa: E731
    ia, ib = ident(a), ident(b)
    shared = sorted(k for k in ia if k in ib)
    same = [k for k in shared if ia[k] == ib[k]]
    return {"shared": len(shared), "identical": len(same), "keys": same}


def model_gap_ci(num3: list[float], den3: list[float], num1: list[float], den1: list[float], iters: int, seed: int) -> list[float]:
    """CI of retention(3B) - retention(1B), resampling prompts jointly (the same indices for both)."""
    rng = random.Random(seed)
    n = len(num3)
    out = []
    for _ in range(iters):
        ix = [rng.randrange(n) for _ in range(n)]
        d3, d1 = sum(den3[i] for i in ix), sum(den1[i] for i in ix)
        if d3 > 0 and d1 > 0:
            out.append(sum(num3[i] for i in ix) / d3 - sum(num1[i] for i in ix) / d1)
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "phase15.yaml").read_text(encoding="utf-8"))
    iters, seed = cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"]
    m3 = json.loads((RESULTS_DIR / "phase15" / "policy_quality" / "metrics.json").read_text(encoding="utf-8"))
    m1 = json.loads((RESULTS_DIR / "phase8" / "policy_quality_ctx16384" / "metrics.json").read_text(encoding="utf-8"))
    if m3["provenance"]["git_dirty"]:
        raise SystemExit("phase 15 run is from a dirty tree")
    r3, r1 = m3["niah"], m1["niah"]
    ident = prompt_identity(r3, r1)
    keys = ident["keys"]
    full3, full1 = scores(r3, "full", 1.0), scores(r1, "full", 1.0)

    conds3 = sorted({(r["policy"], r["budget"]) for r in r3 if r["policy"] not in ("full",)}, key=lambda c: (c[0], -c[1]))
    per: dict[str, Any] = {}
    for pol, b in conds3:
        s3 = scores(r3, pol, b)
        s1 = scores(r1, pol, b)
        k3 = [k for k in keys if k in s3]
        num3, den3 = [s3[k] for k in k3], [full3[k] for k in k3]
        entry: dict[str, Any] = {"policy": pol, "budget": b, "prompts": len(k3),
                                 "retention_3b": sum(num3) / sum(den3) if sum(den3) else None,
                                 "retention_3b_ci95": list(paired_ratio_ci(num3, den3, iters=iters, seed=seed)) if sum(den3) else None}
        k1 = [k for k in keys if k in s3 and k in s1]
        if k1:
            a3, d3 = [s3[k] for k in k1], [full3[k] for k in k1]
            a1, d1 = [s1[k] for k in k1], [full1[k] for k in k1]
            entry["retention_1b"] = sum(a1) / sum(d1) if sum(d1) else None
            entry["retention_1b_ci95"] = list(paired_ratio_ci(a1, d1, iters=iters, seed=seed)) if sum(d1) else None
            entry["gap_3b_minus_1b"] = entry["retention_3b"] - entry["retention_1b"] if entry["retention_1b"] is not None else None
            entry["gap_ci95"] = model_gap_ci(a3, d3, a1, d1, iters, seed)
            # Post hoc, added after the results were read: each retention is against its own model's
            # full cache, and the 1B's misses some prompts the 3B answers, so the two denominators
            # weight different prompts. Restricting to prompts both full caches answer completely
            # asks whether the gap survives on a common set.
            kb = [k for k in k1 if full3[k] == 1.0 and full1[k] == 1.0]
            if kb:
                b3, b1 = [s3[k] for k in kb], [s1[k] for k in kb]
                entry["both_full_prompts"] = len(kb)
                entry["both_full_gap_ci95"] = model_gap_ci(b3, [1.0] * len(kb), b1, [1.0] * len(kb), iters, seed)
                entry["both_full_gap"] = (sum(b3) - sum(b1)) / len(kb)
        per[label(pol, b)] = entry

    # Question 2: the order at each budget, by point estimate, for each model.
    order = {}
    for b in cfg["budgets"]:
        row = {pol: per.get(label(pol, b)) for pol in cfg["policies"]}
        order[f"{100 * b:g}"] = {
            "3b": sorted((p for p, e in row.items() if e and e["retention_3b"] is not None), key=lambda p: -row[p]["retention_3b"]),
            "1b": sorted((p for p, e in row.items() if e and e.get("retention_1b") is not None), key=lambda p: -row[p]["retention_1b"]),
        }
    # Question 3: rung 6 must answer exactly as rung 5.
    ans = lambda pol, b: {(r["kind"], r["depth"], r["sample"]): r["answer"] for r in r3 if r["policy"] == pol and r["budget"] == b}  # noqa: E731
    identity = {f"{100 * b:g}": sum(1 for k, v in ans("tiered_sync", b).items() if ans("quest", b).get(k) != v) for b in cfg["budgets"]}

    compared = [e for e in per.values() if e.get("gap_ci95")]
    summary = {
        "context": m3["config"]["context"],
        "prompts_shared": ident["shared"],
        "prompts_identical": ident["identical"],
        "full_accuracy_3b": sum(full3[k] for k in keys) / len(keys) if keys else None,
        "full_accuracy_1b": sum(full1[k] for k in keys) / len(keys) if keys else None,
        "kv_bytes_per_token_3b": m3["kv_bytes_per_token"],
        "kv_bytes_per_token_1b": m1["kv_bytes_per_token"],
        "model_gaps_compared": len(compared),
        "model_gaps_resolved": sum(1 for e in compared if e["gap_ci95"][0] > 0 or e["gap_ci95"][1] < 0),
        "model_gaps_3b_better": sorted(k for k, e in per.items() if e.get("gap_ci95") and e["gap_ci95"][0] > 0),
        "model_gaps_3b_worse": sorted(k for k, e in per.items() if e.get("gap_ci95") and e["gap_ci95"][1] < 0),
        "both_full_gaps_3b_better": sorted(k for k, e in per.items() if e.get("both_full_gap_ci95") and e["both_full_gap_ci95"][0] > 0),
        "both_full_gaps_3b_worse": sorted(k for k, e in per.items() if e.get("both_full_gap_ci95") and e["both_full_gap_ci95"][1] < 0),
        "rung6_answers_differing_from_rung5": sum(identity.values()),
        "order_by_budget": order,
        "wall_s": m3["wall_s"],
    }
    write_metrics(RESULTS_DIR / "phase15" / "analysis", {"sources": [m3["provenance"], m1["provenance"]], "conditions": per, "summary": summary})


if __name__ == "__main__":
    main()
