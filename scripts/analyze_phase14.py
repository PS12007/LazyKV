"""Derive Phase 14 quantities: retention on LongBench QA, multi-hop tasks beside a single-document control.

Reads   results/phase14/longbench/metrics.json
Writes  results/phase14/analysis/metrics.json

Mirrors Phase 12's analysis on real text: per task, each condition's retention of the full cache's
F1 with a paired bootstrap CI (prompts resampled with both conditions together, as in Phase 9), and
per rung and budget the multi-hop tasks' pooled retention against the single-document control's,
under both of Phase 12's tests. A task below the config's viability floor is reported and excluded
from the comparison; the floor was fixed before the run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import paired_ratio_ci, sign_test_p  # noqa: E402

FULL = "full"
CONTROL = "block_full"


def label(policy: str, budget: float) -> str:
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def paired(rows: list[dict[str, Any]], tasks: list[str], policy: str, budget: float) -> tuple[list[float], list[float]]:
    full = {(r["task"], r["index"]): r["score"] for r in rows if r["task"] in tasks and r["policy"] == FULL}
    cond = {(r["task"], r["index"]): r["score"] for r in rows if r["task"] in tasks and r["policy"] == policy and r["budget"] == budget}
    keys = sorted(k for k in full if k in cond)
    return [cond[k] for k in keys], [full[k] for k in keys]


def retention(num: list[float], den: list[float], iters: int, seed: int) -> dict[str, Any]:
    worse = sum(1 for a, b in zip(num, den) if a < b)
    better = sum(1 for a, b in zip(num, den) if a > b)
    ok = sum(den) > 0
    return {"prompts": len(num), "mean_f1": sum(num) / len(num) if num else None,
            "retention": sum(num) / sum(den) if ok else None,
            "retention_ci95": list(paired_ratio_ci(num, den, iters=iters, seed=seed)) if ok else None,
            "worse": worse, "better": better, "sign_test_p": sign_test_p(worse, better)}


def gap_ci(hard: tuple[list[float], list[float]], easy: tuple[list[float], list[float]], iters: int, seed: int) -> list[float]:
    """95% CI of retention(hard) - retention(easy), each group's prompts resampled independently.

    Phase 12 introduced this test post hoc; here it was named in the config before the run.
    """
    rng = random.Random(seed)
    (hn, hd), (en, ed) = hard, easy
    out = []
    for _ in range(iters):
        hi = [rng.randrange(len(hn)) for _ in hn]
        ei = [rng.randrange(len(en)) for _ in en]
        dh, de = sum(hd[i] for i in hi), sum(ed[i] for i in ei)
        if dh > 0 and de > 0:
            out.append(sum(hn[i] for i in hi) / dh - sum(en[i] for i in ei) / de)
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(RESULTS_DIR / "phase14"))
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase14.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iters, seed, floor = cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"], cfg["viability_floor"]
    m = json.loads((Path(args.base) / "longbench" / "metrics.json").read_text(encoding="utf-8"))
    rows = m["rows"]
    tasks = sorted({r["task"] for r in rows})
    conds = sorted({(r["policy"], r["budget"]) for r in rows if r["policy"] != FULL}, key=lambda c: (c[0], -c[1]))

    full_f1 = {t: sum(r["score"] for r in rows if r["task"] == t and r["policy"] == FULL) / sum(1 for r in rows if r["task"] == t and r["policy"] == FULL) for t in tasks}
    viable = sorted(t for t in tasks if full_f1[t] >= floor)
    per_task = {t: {label(pol, b): {"policy": pol, "budget": b, **retention(*paired(rows, [t], pol, b), iters, seed)} for pol, b in conds} for t in tasks}

    hard = [t for t in cfg["hard_tasks"] if t in viable]
    easy = [cfg["control_task"]] if cfg["control_task"] in viable else []
    comparison = {}
    if hard and easy:
        for pol, b in conds:
            if pol == CONTROL:
                continue
            h, e = paired(rows, hard, pol, b), paired(rows, easy, pol, b)
            rh, re_ = retention(*h, iters, seed), retention(*e, iters, seed)
            if rh["retention"] is None or re_["retention"] is None:
                continue
            comparison[label(pol, b)] = {
                "policy": pol, "budget": b, "hard": rh, "easy": re_, "gap": rh["retention"] - re_["retention"],
                "separated": rh["retention_ci95"][1] < re_["retention_ci95"][0] or re_["retention_ci95"][1] < rh["retention_ci95"][0],
                "gap_ci95": gap_ci(h, e, iters, seed),
            }
    tight = min(cfg["budgets"])
    rng = lambda xs: {"min": min(xs), "max": max(xs)} if xs else None  # noqa: E731
    summary = {
        "viability_floor": floor,
        "full_f1": full_f1,
        "viable_tasks": viable,
        "excluded_tasks": sorted(set(tasks) - set(viable)),
        "hard_tasks_used": hard,
        "control_task_used": easy[0] if easy else None,
        "prompts_per_task": {t: sum(1 for r in rows if r["task"] == t and r["policy"] == FULL) for t in tasks},
        "truncated_prompts": len({(r["task"], r["index"]) for r in rows if r["truncated"]}),
        "prompt_len": rng([r["prompt_len"] for r in rows if r["policy"] == FULL]),
        "control_exact": all(v["worse"] == 0 and v["better"] == 0 for t in tasks for k, v in per_task[t].items() if v["policy"] == CONTROL),
        "tightest_budget": tight,
        "comparisons": len(comparison),
        "separated_below": sorted(k for k, c in comparison.items() if c["separated"] and c["gap"] < 0),
        "separated_above": sorted(k for k, c in comparison.items() if c["separated"] and c["gap"] > 0),
        "gap_ci_below": sorted(k for k, c in comparison.items() if c["gap_ci95"][1] < 0),
        "gap_ci_above": sorted(k for k, c in comparison.items() if c["gap_ci95"][0] > 0),
        "wall_s": m["wall_s"],
    }
    write_metrics(Path(args.base) / "analysis", {"source": m["provenance"], "tasks": per_task, "comparison": comparison, "summary": summary})


if __name__ == "__main__":
    main()
