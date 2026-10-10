"""Derive Phase 24 quantities: rung 5 ranking by a float32 Quest bound against bf16 (configs/phase24.yaml).

Reads   results/phase24/policy_quality/metrics.json (question 1: NIAH + teacher-forced, quality kernel)
        results/phase24/multiturn/metrics.json      (question 2: Phase 21's sessions, dense turns)
Writes  results/phase24/analysis/metrics.json

Every comparison is float32 against bf16 on the same prompt (or session and turn) in the same run,
with prompts better / worse and a two-sided sign test. The config's decision rule: float32 is
"better" at a budget only if more improve than get worse and p < 0.05.

Added after the gate (no new runs): a null verdict says nothing about size, so every comparison also
carries the paired accuracy difference (float32 - bf16) with a 95% bootstrap CI, and `pooled` bounds
the effect over all budgets at once. Pooling reuses each prompt (session) at every budget, so the
pooled CI resamples whole prompts (sessions), not rows; otherwise it would be too narrow.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci, sign_test_p  # noqa: E402
from scripts.analyze_phase4 import prompt_key  # noqa: E402
from scripts.analyze_phase21 import KINDS, keyed  # noqa: E402

OLD, NEW = "quest", "quest_f32"
ALPHA = 0.05
BOOT_ITERS = 10000


def cluster_diff_ci(diffs: dict[Any, list[float]], iters: int = BOOT_ITERS, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean per-row difference, resampling whole clusters.

    A cluster is one prompt (or session) with its rows at every budget. Its rows share a haystack and
    a needle, so they are not independent draws; resampling rows would understate the width.
    """
    groups = list(diffs.values())
    if not groups:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(sum(g) for g in pick) / sum(len(g) for g in pick))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def pooled(diffs: dict[Any, list[float]], unit: str) -> dict[str, Any]:
    rows = [d for g in diffs.values() for d in g]
    lo, hi = cluster_diff_ci(diffs)
    return {"unit": unit, "clusters": len(diffs), "rows": len(rows), "diff": statistics.mean(rows),
            "diff_ci95": [lo, hi], "max_abs_effect": max(abs(lo), abs(hi))}


def compare(new: dict[Any, float], old: dict[Any, float], keys: list[Any]) -> dict[str, Any]:
    keys = [k for k in keys if k in new and k in old]
    better = sum(1 for k in keys if new[k] > old[k])
    worse = sum(1 for k in keys if new[k] < old[k])
    p = sign_test_p(worse, better)
    diffs = [new[k] - old[k] for k in keys]
    return {
        "n": len(keys),
        "old_accuracy": statistics.mean(old[k] for k in keys) if keys else None,
        "new_accuracy": statistics.mean(new[k] for k in keys) if keys else None,
        "better": better, "worse": worse, "sign_p": p,
        "diff": statistics.mean(diffs) if diffs else None,
        "diff_ci95": list(bootstrap_mean_ci(diffs, iters=BOOT_ITERS)) if diffs else None,
        # With no changed scores every resample is zero, so the bootstrap CI collapses to [0, 0]. That
        # is not certainty: the rule of three bounds the rate of changed scores at 3/n (95%) instead.
        "rule_of_three": 3 / len(keys) if keys and not any(diffs) else None,
        "verdict": "better" if better > worse and p < ALPHA else ("worse" if worse > better and p < ALPHA else "no measurable effect"),
    }


def niah(q: dict[str, Any]) -> dict[str, Any]:
    score = {(r["policy"], r["budget"]) + prompt_key(r): r["score"] for r in q["niah"]}
    answer = {(r["policy"], r["budget"]) + prompt_key(r): r["answer"] for r in q["niah"]}
    full = {k[2:]: v for k, v in score.items() if k[0] == "full"}
    tf = {(r["policy"], r["budget"], r["offset"]): r for r in q["teacher_forced"]}
    out: dict[str, Any] = {"full_accuracy": statistics.mean(full.values()), "prompts": len(full), "budgets": {}}
    for b in sorted({r["budget"] for r in q["niah"] if r["policy"] == NEW}, reverse=True):
        old = {k[2:]: v for k, v in score.items() if k[:2] == (OLD, b)}
        new = {k[2:]: v for k, v in score.items() if k[:2] == (NEW, b)}
        e = compare(new, old, sorted(full))
        e["answers_changed"] = sum(1 for k in full if answer.get((NEW, b) + k) != answer.get((OLD, b) + k))
        e["old_retention"] = e["old_accuracy"] / out["full_accuracy"]
        e["new_retention"] = e["new_accuracy"] / out["full_accuracy"]
        offs = sorted(o for (p, bb, o) in tf if p == NEW and bb == b)
        e["tf"] = [{"offset": o, "old_kl": tf[(OLD, b, o)]["mean_kl"], "new_kl": tf[(NEW, b, o)]["mean_kl"],
                    "old_top1": tf[(OLD, b, o)]["top1_agreement"], "new_top1": tf[(NEW, b, o)]["top1_agreement"]} for o in offs if (OLD, b, o) in tf]
        out["budgets"][f"{b:g}"] = e
    by_prompt: dict[Any, list[float]] = {}
    for (pol, b, *k), v in score.items():
        if pol == NEW and (OLD, b, *k) in score:
            by_prompt.setdefault(tuple(k), []).append(v - score[(OLD, b, *k)])
    out["pooled"] = pooled(by_prompt, "prompt")
    return out


def multiturn(src: dict[str, Any]) -> dict[str, Any]:
    rows = src["turns"]
    out: dict[str, Any] = {"sessions": len({r["session"] for r in rows}), "budgets": {}}
    for b in sorted({r["budget"] for r in rows if r["policy"] == NEW}, reverse=True):
        old, new = keyed(rows, OLD, b), keyed(rows, NEW, b)
        full = keyed(rows, "full", 1.0)
        e: dict[str, Any] = {}
        for kind, sel in KINDS.items():
            keys = sorted({(r["session"], r["turn"]) for r in rows if sel(r)})
            c = compare(new, old, keys)
            c["full_accuracy"] = statistics.mean(full[k] for k in keys if k in full) if keys else None
            e[kind] = c
        e["answers_changed"] = sum(1 for k, v in keyed(rows, NEW, b, "answer").items() if keyed(rows, OLD, b, "answer").get(k) != v)
        out["budgets"][f"{b:g}"] = e
    by_session: dict[Any, list[float]] = {}
    for b in out["budgets"]:
        old, new = keyed(rows, OLD, float(b)), keyed(rows, NEW, float(b))
        for k in sorted(set(old) & set(new)):
            by_session.setdefault(k[0], []).append(new[k] - old[k])
    out["pooled"] = pooled(by_session, "session")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="policy_quality")
    ap.add_argument("--multiturn", default="multiturn")
    args = ap.parse_args()
    base = RESULTS_DIR / "phase24"
    out: dict[str, Any] = {}
    qp, mp = base / args.quality / "metrics.json", base / args.multiturn / "metrics.json"
    if qp.exists():
        q = json.loads(qp.read_text(encoding="utf-8"))
        out["niah_source"], out["niah"] = q["provenance"], niah(q)
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        out["multiturn_source"], out["multiturn"] = m["provenance"], multiturn(m)
    verdicts = [e["verdict"] for e in out.get("niah", {}).get("budgets", {}).values()]
    verdicts += [e[k]["verdict"] for e in out.get("multiturn", {}).get("budgets", {}).values() for k in ("doc_turn0", "doc_later", "doc_reask", "chat", "after_turn0")]
    out["summary"] = {
        "comparisons": len(verdicts),
        "better": verdicts.count("better"),
        "worse": verdicts.count("worse"),
        "no_effect": verdicts.count("no measurable effect"),
    }
    write_metrics(base / "analysis", out)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
