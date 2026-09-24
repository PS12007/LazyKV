"""Derive Phase 9 decision quantities: does the 99% bar resolve with a larger paired prompt set?

Reads   results/phase9/niah_s*/metrics.json          (chunks of one NIAH run, quality kernel)
        results/phase8/policy_quality_ctx32768/metrics.json (the 45 prompts the first chunk repeats)
Writes  results/phase9/analysis/metrics.json

Phase 8 found the query-aware rungs within a fraction of a prompt of the bar at every context. That
is a statement about the sample, not the policy, and the only way to turn it into one about the
policy is more paired prompts. `configs/phase9.yaml` fixes the decision rule before the run:

    meets       paired-bootstrap 95% CI of the retention lies entirely at or above the bar
    fails       ... entirely below it
    unresolved  otherwise

The rule is applied here without adjustment, and the analysis also reports what it would have said
at every smaller prefix of the prompt set (by sample index, so the prefixes are fixed in advance,
not chosen). That series is what shows whether more prompts are converging on a verdict or not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import paired_ratio_ci, prompts_to_resolve, sign_test_p  # noqa: E402
from harness.sweep_analysis import RETENTION_TARGET  # noqa: E402

FULL = "full"
CONTROL = "block_full"
Key = tuple[str, float, int]  # (kind, depth, sample): one prompt
# Prefixes in samples per kind x depth. 3 is Phase 8's set; the rest are the chunk boundaries.
PREFIXES = (3, 5, 10, 15, 20)


def label(policy: str, budget: float) -> str:
    """A key a dotted template lookup can address: "quest_75", not "quest@0.75"."""
    return f"{policy}_{100 * budget:g}".replace(".", "_")


def verdict(ci: tuple[float, float], bar: float) -> str:
    """The pre-registered rule. A CI touching the bar from above counts as meeting it (>=)."""
    lo, hi = ci
    if lo >= bar:
        return "meets"
    if hi < bar:
        return "fails"
    return "unresolved"


def load_chunks(base: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """All chunks' NIAH rows, the teacher-forced rows (first chunk only), and each chunk's provenance.

    Refuses a merge that would silently count a prompt twice or mix commits: the chunks are only one
    run if they ran the same code, and a duplicated prompt would narrow the CI for free.
    """
    niah: list[dict[str, Any]] = []
    tf: list[dict[str, Any]] = []
    prov: list[dict[str, Any]] = []
    for path in sorted(base.glob("niah_s*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        niah += m["niah"]
        tf += m["teacher_forced"]
        prov.append({"chunk": path.parent.name, "sample_start": m["config"]["niah"].get("sample_start", 0),
                     "samples": m["config"]["niah"]["samples"], "wall_s": m["wall_s"], **m["provenance"]})
    if not prov:
        raise SystemExit(f"no chunks under {base}")
    commits = {p["git_commit"] for p in prov}
    if len(commits) != 1 or any(p["git_dirty"] for p in prov):
        raise SystemExit(f"chunks disagree on provenance: commits {commits}, dirty {[p['chunk'] for p in prov if p['git_dirty']]}")
    seen = set()
    for r in niah:
        k = (r["kind"], r["depth"], r["sample"], r["policy"], r["budget"])
        if k in seen:
            raise SystemExit(f"prompt {k} appears in more than one chunk")
        seen.add(k)
    return niah, tf, prov


def paired(rows: list[dict[str, Any]], policy: str, budget: float, max_sample: int | None = None) -> tuple[list[Key], list[float], list[float]]:
    """(prompts, condition scores, full scores), aligned by prompt, optionally a sample-index prefix."""
    full = {(r["kind"], r["depth"], r["sample"]): r["score"] for r in rows if r["policy"] == FULL}
    cond = {(r["kind"], r["depth"], r["sample"]): r["score"] for r in rows if r["policy"] == policy and r["budget"] == budget}
    keys = sorted(k for k in full if k in cond and (max_sample is None or k[2] < max_sample))
    return keys, [cond[k] for k in keys], [full[k] for k in keys]


def resolve(num: list[float], den: list[float], bar: float, iters: int, seed: int) -> dict[str, Any]:
    """Retention, its paired CI, the rule's verdict, and the discordant pairs behind the width."""
    worse = sum(1 for a, b in zip(num, den) if a < b)
    better = sum(1 for a, b in zip(num, den) if a > b)
    ci = paired_ratio_ci(num, den, iters=iters, seed=seed)
    r = sum(num) / sum(den) if sum(den) else None
    return {
        "prompts": len(num),
        "accuracy": sum(num) / len(num) if num else None,
        "full_accuracy": sum(den) / len(den) if den else None,
        "retention": r,
        "retention_ci95": list(ci),
        "ci_width_pp": 100 * (ci[1] - ci[0]),
        "verdict": verdict(ci, bar),
        "worse": worse,
        "better": better,
        "discordant": worse + better,
        "sign_test_p": sign_test_p(worse, better),
        # Positive means above the bar; a prompt is one prompt, whatever N is.
        "margin_prompts": (sum(num) - bar * sum(den)) if den else None,
        "prompts_to_resolve": prompts_to_resolve(num, den, bar) if len(num) >= 2 else None,
    }


def by_kind(rows: list[dict[str, Any]], policy: str, budget: float) -> dict[str, dict[str, Any]]:
    """Retention per needle kind. Descriptive only: a third of the prompts cannot carry a verdict."""
    keys, num, den = paired(rows, policy, budget)
    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k[0]].append(i)
    for kind, idx in groups.items():
        n, d = sum(num[i] for i in idx), sum(den[i] for i in idx)
        out[kind] = {"prompts": len(idx), "retention": n / d if d else None,
                     "worse": sum(1 for i in idx if num[i] < den[i]), "better": sum(1 for i in idx if num[i] > den[i])}
    return out


def phase8_reproduction(rows: list[dict[str, Any]], phase8_path: Path) -> dict[str, Any]:
    """Whether the first chunk's repeat of Phase 8's 45 prompts gave the same answers, token for token.

    The quality kernel is deterministic and the prompts are keyed by sample index, so anything short
    of identical would mean the two phases are not measuring the same thing, and pooling them (which
    the prefix series does) would be wrong.
    """
    if not phase8_path.exists():
        return {"compared": 0}
    old = json.loads(phase8_path.read_text(encoding="utf-8"))["niah"]
    key = lambda r: (r["kind"], r["depth"], r["sample"], r["policy"], r["budget"])  # noqa: E731
    before = {key(r): (r["score"], r["answer"]) for r in old}
    pairs = [(before[key(r)], (r["score"], r["answer"])) for r in rows if key(r) in before]
    same = sum(1 for a, b in pairs if a == b)
    return {"compared": len(pairs), "identical": same, "reproduces": bool(pairs) and same == len(pairs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(RESULTS_DIR / "phase9"))
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase9.yaml"))
    args = p.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bar, iters, seed = cfg["bar"], cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"]
    # The config states the bar so the rule is readable in one place; it must be the study's bar.
    if bar != RETENTION_TARGET:
        raise SystemExit(f"config bar {bar} is not the study's retention target {RETENTION_TARGET}")

    niah, tf, prov = load_chunks(base)
    samples = 1 + max(r["sample"] for r in niah)
    conds = [(pol, b) for pol in cfg["policies"] for b in cfg["budgets"]] + [(CONTROL, 1.0)]

    results: dict[str, Any] = {}
    for pol, b in conds:
        _, num, den = paired(niah, pol, b)
        prefixes = []
        for s in (s for s in PREFIXES if s <= samples):
            _, pn, pd = paired(niah, pol, b, max_sample=s)
            prefixes.append({"samples": s, **resolve(pn, pd, bar, iters, seed)})
        results[label(pol, b)] = {"policy": pol, "budget": b, **resolve(num, den, bar, iters, seed),
                                   "by_kind": by_kind(niah, pol, b), "prefixes": prefixes}

    policy_rows = [v for v in results.values() if v["policy"] != CONTROL]
    meeting = sorted(v["budget"] for v in policy_rows if v["verdict"] == "meets")
    repro = phase8_reproduction(niah, RESULTS_DIR / "phase8" / "policy_quality_ctx32768" / "metrics.json")
    first = {k: v["prefixes"][0] for k, v in results.items()}

    summary = {
        "context": cfg["context"],
        "bar": bar,
        "samples_per_cell": samples,
        "prompts": results[label(cfg["policies"][0], cfg["budgets"][0])]["prompts"],
        "phase8_prompts": first[label(cfg["policies"][0], cfg["budgets"][0])]["prompts"],
        "full_accuracy": policy_rows[0]["full_accuracy"],
        "verdicts": {label(v["policy"], v["budget"]): v["verdict"] for v in policy_rows},
        "verdicts_at_phase8_n": {label(v["policy"], v["budget"]): v["prefixes"][0]["verdict"] for v in policy_rows},
        "min_budget_meeting_bar": meeting[0] if meeting else None,
        "any_unresolved": any(v["verdict"] == "unresolved" for v in policy_rows),
        "control_verdict": results[label(CONTROL, 1.0)]["verdict"],
        "phase8_reproduces": repro.get("reproduces"),
        "phase8_rows_compared": repro.get("compared"),
        "wall_s": sum(pv["wall_s"] for pv in prov),
    }

    write_metrics(base / "analysis", {"sources": prov, "conditions": results, "phase8_reproduction": repro,
                                      "teacher_forced": tf, "summary": summary})
    print("wrote", base / "analysis" / "metrics.json")


if __name__ == "__main__":
    main()
