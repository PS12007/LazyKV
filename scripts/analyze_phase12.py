"""Derive Phase 12 quantities: retention on RULER's harder tasks, beside the single needle.

Reads   results/phase12/ruler_ctx*/metrics.json   (one run per context, quality kernel)
        results/phase12_pilot/pilot_ctx*/metrics.json (the full-cache pilot that chose the tasks)
Writes  results/phase12/analysis/metrics.json

Every earlier phase judged residency by needle retrieval, where the answer sits in one or a few
blocks and the question names what to look for. RULER's variable tracking (vt) breaks both
properties: the answer is five blocks' worth of chain, and only the first hop contains the value
the question asks about, so a policy has to keep blocks whose relevance only becomes visible as the
answer is generated. Common words extraction (cwe) spreads the evidence over the whole list.

Each run carries the single-needle kind as an in-run control, so a rung's retention on a hard kind
is compared with its retention on the easy one under the same commit, context and prompts
machinery, rather than against another phase's numbers.

Retention and its paired CI are computed exactly as in Phase 9 (`paired_ratio_ci`, prompts
resampled with both conditions together). A kind is only analyzed at a context where the pilot's
full cache clears `viability_floor` from the config; below it a retention is a ratio of noise.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import paired_ratio_ci, sign_test_p  # noqa: E402

FULL = "full"
CONTROL = "block_full"
Key = tuple[float, int]  # (depth, sample): one prompt of one kind


def label(policy: str, budget: float) -> str:
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def load_runs(base: Path, pattern: str) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """NIAH rows per context and each run's provenance; refuses runs from mixed commits or a dirty tree."""
    rows: dict[int, list[dict[str, Any]]] = {}
    prov: list[dict[str, Any]] = []
    for path in sorted(base.glob(f"{pattern}/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        ctx = int(m["config"]["context"])
        rows[ctx] = m["niah"]
        prov.append({"run": path.parent.name, "context": ctx, "wall_s": m["wall_s"], **m["provenance"]})
    if prov:
        commits = {p["git_commit"] for p in prov}
        if len(commits) != 1 or any(p["git_dirty"] for p in prov):
            raise SystemExit(f"{base}: runs disagree on provenance: commits {commits}, dirty {[p['run'] for p in prov if p['git_dirty']]}")
    return rows, prov


def full_accuracy(rows: list[dict[str, Any]], kind: str) -> float | None:
    s = [r["score"] for r in rows if r["kind"] == kind and r["policy"] == FULL]
    return sum(s) / len(s) if s else None


def paired(rows: list[dict[str, Any]], kind: str, policy: str, budget: float) -> tuple[list[Key], list[float], list[float]]:
    full = {(r["depth"], r["sample"]): r["score"] for r in rows if r["kind"] == kind and r["policy"] == FULL}
    cond = {(r["depth"], r["sample"]): r["score"] for r in rows if r["kind"] == kind and r["policy"] == policy and r["budget"] == budget}
    keys = sorted(k for k in full if k in cond)
    return keys, [cond[k] for k in keys], [full[k] for k in keys]


def retention(rows: list[dict[str, Any]], kind: str, policy: str, budget: float, iters: int, seed: int) -> dict[str, Any]:
    keys, num, den = paired(rows, kind, policy, budget)
    worse = sum(1 for a, b in zip(num, den) if a < b)
    better = sum(1 for a, b in zip(num, den) if a > b)
    by_depth: dict[float, list[int]] = defaultdict(list)
    for i, (depth, _) in enumerate(keys):
        by_depth[depth].append(i)
    return {
        "kind": kind,
        "policy": policy,
        "budget": budget,
        "prompts": len(num),
        "accuracy": sum(num) / len(num) if num else None,
        "retention": sum(num) / sum(den) if sum(den) else None,
        "retention_ci95": list(paired_ratio_ci(num, den, iters=iters, seed=seed)) if sum(den) else None,
        "worse": worse,
        "better": better,
        "sign_test_p": sign_test_p(worse, better),
        # Descriptive: for vt the depth is where the chain starts, for single where the needle sits.
        "retention_by_depth": {str(d): (sum(num[i] for i in ix) / sum(den[i] for i in ix)) if sum(den[i] for i in ix) else None
                               for d, ix in sorted(by_depth.items())},
    }


def gap_ci(hard: tuple[list[float], list[float]], easy: tuple[list[float], list[float]], iters: int, seed: int) -> list[float]:
    """95% CI of retention(hard) - retention(easy), resampling each kind's prompts independently.

    POST HOC. The pre-committed separation test is disjoint CIs (`separated`), which is
    conservative; this interval was added after the 32K run had been read, because the disjoint
    test cannot separate effects of the size seen at 30 prompts per kind. It is reported beside the
    pre-committed test and labelled as post hoc wherever it is quoted, never in place of it.
    Within a kind the pairs (policy score, full score) are resampled together, as in Phase 9.
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


def pilot_reproduction(pilot_rows: dict[int, list[dict[str, Any]]], runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Whether the main run's full cache answered the pilot's prompts identically, text and score.

    Prompts are keyed by (kind, depth, sample) and the quality kernel is deterministic, so a shared
    prompt must reproduce exactly; anything less would mean the pilot and the run measure different
    things, and the viability decision would not carry over.
    """
    key = lambda r: (r["kind"], r["depth"], r["sample"])  # noqa: E731
    compared = identical = 0
    for ctx, rows in runs.items():
        before = {key(r): (r["score"], r["answer"]) for r in pilot_rows.get(ctx, []) if r["policy"] == FULL}
        for r in rows:
            if r["policy"] == FULL and key(r) in before:
                compared += 1
                identical += before[key(r)] == (r["score"], r["answer"])
    return {"compared": compared, "identical": identical, "reproduces": compared > 0 and compared == identical}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(RESULTS_DIR / "phase12"))
    p.add_argument("--pilot", default=str(RESULTS_DIR / "phase12_pilot"))
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase12.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iters, seed, floor = cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"], cfg["viability_floor"]

    pilot_rows, pilot_prov = load_runs(Path(args.pilot), "pilot_ctx*")
    pilot = {str(ctx): {kind: full_accuracy(rows, kind) for kind in sorted({r["kind"] for r in rows})} for ctx, rows in sorted(pilot_rows.items())}
    viable = {ctx: sorted(k for k, a in kinds.items() if a is not None and a >= floor) for ctx, kinds in pilot.items()}

    runs, prov = load_runs(Path(args.base), "ruler_ctx*")
    conds = [(CONTROL, 1.0)] + [(pol, b) for pol in cfg["policies"] for b in cfg["budgets"]]
    per_ctx: dict[str, Any] = {}
    for ctx, rows in sorted(runs.items()):
        kinds = sorted({r["kind"] for r in rows})
        per_ctx[str(ctx)] = {
            "full_accuracy": {k: full_accuracy(rows, k) for k in kinds},
            "kinds": {k: {label(pol, b): retention(rows, k, pol, b, iters, seed) for pol, b in conds} for k in kinds},
        }

    # The comparison the phase exists for: per rung and budget, retention on the hard kind next to
    # the in-run single-needle control, at every context both were run.
    hard, easy = cfg["hard_kind"], cfg["control_kind"]
    gaps = []
    for ctx, d in per_ctx.items():
        if hard not in d["kinds"] or easy not in d["kinds"]:
            continue
        rows = runs[int(ctx)]
        for key, h in d["kinds"][hard].items():
            e = d["kinds"][easy][key]
            if h["policy"] == CONTROL or h["retention"] is None or e["retention"] is None:
                continue
            gaps.append({"context": int(ctx), "label": key, "policy": h["policy"], "budget": h["budget"],
                         "hard": h["retention"], "easy": e["retention"], "gap": h["retention"] - e["retention"],
                         # Disjoint CIs are a conservative separation test for two unpaired retentions.
                         "separated": h["retention_ci95"][1] < e["retention_ci95"][0] or e["retention_ci95"][1] < h["retention_ci95"][0],
                         "gap_ci95_post_hoc": gap_ci(paired(rows, hard, h["policy"], h["budget"])[1:], paired(rows, easy, h["policy"], h["budget"])[1:], iters, seed)})
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gaps:
        by_policy[g["policy"]].append(g)

    repro = pilot_reproduction(pilot_rows, runs)
    summary = {
        "pilot_rows_compared": repro["compared"],
        "pilot_rows_identical": repro["identical"],
        "pilot_reproduces": repro["reproduces"],
        "viability_floor": floor,
        "pilot_full_accuracy": pilot,
        "viable_kinds_by_context": viable,
        "contexts": sorted(int(c) for c in per_ctx),
        "hard_kind": hard,
        "control_kind": easy,
        "prompts_per_kind": {c: {k: next(iter(v.values()))["prompts"] for k, v in d["kinds"].items()} for c, d in per_ctx.items()},
        "control_exact": all(d["kinds"][k][label(CONTROL, 1.0)]["worse"] == 0 and d["kinds"][k][label(CONTROL, 1.0)]["better"] == 0
                             for d in per_ctx.values() for k in d["kinds"]),
        "hard_below_easy": sum(1 for g in gaps if g["gap"] < 0),
        "hard_above_easy": sum(1 for g in gaps if g["gap"] > 0),
        "gaps_compared": len(gaps),
        "gaps_separated": sum(1 for g in gaps if g["separated"]),
        "gaps_separated_post_hoc": sum(1 for g in gaps if g["gap_ci95_post_hoc"][1] < 0 or g["gap_ci95_post_hoc"][0] > 0),
        "gaps_below_post_hoc": sorted(f"{g['label']}@{g['context']}" for g in gaps if g["gap_ci95_post_hoc"][1] < 0),
        "gaps_above_post_hoc": sorted(f"{g['label']}@{g['context']}" for g in gaps if g["gap_ci95_post_hoc"][0] > 0),
        "mean_gap_by_policy": {pol: sum(g["gap"] for g in gs) / len(gs) for pol, gs in by_policy.items()},
        "wall_s": sum(pv["wall_s"] for pv in prov),
    }
    write_metrics(Path(args.base) / "analysis", {"pilot_sources": pilot_prov, "sources": prov, "contexts": per_ctx, "gaps": gaps, "summary": summary})


if __name__ == "__main__":
    main()
