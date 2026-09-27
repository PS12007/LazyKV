"""Phase 12 analysis: retention per kind, the in-run control, and the provenance guard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _analysis():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("analyze_phase12", ROOT / "scripts" / "analyze_phase12.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(kind: str, depth: float, sample: int, policy: str, budget: float, score: float) -> dict:
    return {"kind": kind, "depth": depth, "sample": sample, "policy": policy, "budget": budget, "score": score}


def test_retention_pairs_prompts_and_splits_by_depth() -> None:
    a = _analysis()
    rows = []
    for depth in (0.0, 1.0):
        for sample in range(3):
            rows.append(_row("vt", depth, sample, "full", 1.0, 1.0))
            # Loses every prompt whose chain starts at the beginning, keeps the rest.
            rows.append(_row("vt", depth, sample, "quest", 0.5, 0.0 if depth == 0.0 else 1.0))
    r = a.retention(rows, "vt", "quest", 0.5, iters=300, seed=0)
    assert r["prompts"] == 6 and r["retention"] == 0.5 and r["worse"] == 3 and r["better"] == 0
    assert r["retention_by_depth"] == {"0.0": 0.0, "1.0": 1.0}


def test_retention_is_none_when_the_full_cache_scores_nothing() -> None:
    """A floor-level reference makes a retention meaningless; it must not become a number."""
    a = _analysis()
    rows = [_row("cwe", 0.0, 0, "full", 1.0, 0.0), _row("cwe", 0.0, 0, "quest", 0.5, 0.0)]
    r = a.retention(rows, "cwe", "quest", 0.5, iters=100, seed=0)
    assert r["retention"] is None and r["retention_ci95"] is None


def test_mixed_commits_are_refused(tmp_path: Path) -> None:
    for ctx, commit in ((4096, "aaa"), (8192, "bbb")):
        d = tmp_path / f"ruler_ctx{ctx}"
        d.mkdir()
        m = {"config": {"context": ctx}, "niah": [], "wall_s": 1.0, "provenance": {"git_commit": commit, "git_dirty": False}}
        (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree"):
        _analysis().load_runs(tmp_path, "ruler_ctx*")


def test_pilot_reproduction_requires_identical_text_and_score() -> None:
    a = _analysis()
    pilot = {8192: [dict(_row("vt", 0.0, 0, "full", 1.0, 1.0), answer="VAR A, VAR B")]}
    same = {8192: [dict(_row("vt", 0.0, 0, "full", 1.0, 1.0), answer="VAR A, VAR B")]}
    moved = {8192: [dict(_row("vt", 0.0, 0, "full", 1.0, 1.0), answer="VAR A,  VAR B")]}
    assert a.pilot_reproduction(pilot, same)["reproduces"]
    r = a.pilot_reproduction(pilot, moved)
    assert r["compared"] == 1 and r["identical"] == 0 and not r["reproduces"]


def test_gap_ci_excludes_zero_only_for_a_real_difference() -> None:
    a = _analysis()
    full = [1.0] * 20
    same = a.gap_ci(([1.0] * 10 + [0.0] * 10, full), ([1.0] * 10 + [0.0] * 10, full), iters=500, seed=0)
    assert same[0] < 0 < same[1]
    worse = a.gap_ci(([1.0] * 2 + [0.0] * 18, full), ([1.0] * 18 + [0.0] * 2, full), iters=500, seed=0)
    assert worse[1] < 0
