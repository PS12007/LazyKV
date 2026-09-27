"""Derive Phase 13 quantities: Phase 12's pre-committed vt test, powered with pooled prompts.

Reads   results/phase12/ruler_ctx32768/metrics.json   (samples 0-5, every rung)
        results/phase13/vt_ctx32768/metrics.json      (samples 5-29, rung 5 at 12.5% and 6.25%)
Writes  results/phase13/analysis/metrics.json

The pooled set is Phase 12's samples 0-4 plus this run's 5-29, per kind and depth. Sample 5 ran in
both; pooling is refused unless every condition answered every sample-5 prompt identically (text
and score), since only then are the two runs the same measurement. Retention and both tests are
Phase 12's functions, imported rather than re-implemented, so the rule applied is literally the
one Phase 12 committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_phase12", Path(__file__).with_name("analyze_phase12.py"))
assert _spec is not None and _spec.loader is not None
p12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p12)


def overlap_check(old: list[dict[str, Any]], new: list[dict[str, Any]], policies: set[str]) -> dict[str, Any]:
    key = lambda r: (r["kind"], r["depth"], r["sample"], r["policy"], r["budget"])  # noqa: E731
    before = {key(r): (r["score"], r["answer"]) for r in old if r["policy"] in policies}
    pairs = [(before[key(r)], (r["score"], r["answer"])) for r in new if key(r) in before]
    same = sum(1 for a, b in pairs if a == b)
    return {"compared": len(pairs), "identical": same, "reproduces": bool(pairs) and same == len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase13.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iters, seed = cfg["bootstrap"]["iters"], cfg["bootstrap"]["seed"]
    hard, easy = cfg["hard_kind"], cfg["control_kind"]
    old_m = json.loads((RESULTS_DIR / "phase12" / "ruler_ctx32768" / "metrics.json").read_text(encoding="utf-8"))
    new_m = json.loads((RESULTS_DIR / "phase13" / "vt_ctx32768" / "metrics.json").read_text(encoding="utf-8"))
    if new_m["provenance"]["git_dirty"]:
        raise SystemExit("phase 13 run is from a dirty tree")
    conds = {"full", *cfg["policies"]}
    new_rows = new_m["niah"]
    first_new = min(r["sample"] for r in new_rows)
    repro = overlap_check(old_m["niah"], new_rows, conds)
    if not repro["reproduces"]:
        raise SystemExit(f"overlapping prompts differ between runs: {repro}")
    old_rows = [r for r in old_m["niah"] if r["sample"] < first_new and (r["policy"] == "full" or (r["policy"] in cfg["policies"] and r["budget"] in cfg["budgets"]))]
    rows = old_rows + new_rows

    out: dict[str, Any] = {}
    for pol in cfg["policies"]:
        for b in cfg["budgets"]:
            h = p12.retention(rows, hard, pol, b, iters, seed)
            e = p12.retention(rows, easy, pol, b, iters, seed)
            gap_ci = p12.gap_ci(p12.paired(rows, hard, pol, b)[1:], p12.paired(rows, easy, pol, b)[1:], iters, seed)
            out[p12.label(pol, b)] = {
                "policy": pol, "budget": b, "hard": h, "easy": e, "gap": h["retention"] - e["retention"],
                # Phase 12's pre-committed rule, unchanged.
                "separated": h["retention_ci95"][1] < e["retention_ci95"][0] or e["retention_ci95"][1] < h["retention_ci95"][0],
                "gap_ci95": gap_ci,
                "phase12_separated_at_30": None,
            }
    # What Phase 12's 30 prompts said at 32K, for the before/after.
    p12a = json.loads((RESULTS_DIR / "phase12" / "analysis" / "metrics.json").read_text(encoding="utf-8"))
    for g in p12a["gaps"]:
        if g["context"] == 32768 and g["label"] in out:
            out[g["label"]]["phase12_separated_at_30"] = g["separated"]
            out[g["label"]]["phase12_gap_at_30"] = g["gap"]

    summary = {
        "context": new_m["config"]["context"],
        "prompts_per_kind": {k: sum(1 for r in rows if r["kind"] == k and r["policy"] == "full") for k in (hard, easy)},
        "overlap_compared": repro["compared"],
        "overlap_identical": repro["identical"],
        "full_accuracy": {k: sum(r["score"] for r in rows if r["kind"] == k and r["policy"] == "full") / max(1, sum(1 for r in rows if r["kind"] == k and r["policy"] == "full")) for k in (hard, easy)},
        "established": sorted(k for k, v in out.items() if v["separated"] and v["gap"] < 0),
        "not_established": sorted(k for k, v in out.items() if not (v["separated"] and v["gap"] < 0)),
        "gap_ci_below": sorted(k for k, v in out.items() if v["gap_ci95"][1] < 0),
        "wall_s": new_m["wall_s"],
    }
    write_metrics(RESULTS_DIR / "phase13" / "analysis", {"sources": [old_m["provenance"], new_m["provenance"]], "conditions": out, "summary": summary})


if __name__ == "__main__":
    main()
