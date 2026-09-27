"""Phase 10: long-document perplexity (brief §B7.3) from the teacher-forced pass."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _driver():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("policy_quality", ROOT / "scripts" / "02_policy_quality.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_targets_score_the_same_tokens_at_every_context() -> None:
    """The pairing the context sweep relies on: a fixed target means fixed scored text."""
    tf = {"targets": {"a": [40000, 90000], "b": [50000]}}
    windows = {ctx: _driver().tf_windows(tf, ctx) for ctx in (4096, 32768)}
    assert [(b, t) for b, _, t in windows[4096]] == [(b, t) for b, _, t in windows[32768]]
    assert all(o + ctx == t for ctx, ws in windows.items() for _, o, t in ws)


def test_offsets_keep_their_old_meaning() -> None:
    assert _driver().tf_windows({"book": "a", "offsets": [0, 60000]}, 32768) == [("a", 0, 32768), ("a", 60000, 92768)]


def test_a_target_without_room_for_its_context_is_refused() -> None:
    with pytest.raises(ValueError, match="need 32768 tokens"):
        _driver().tf_windows({"targets": {"a": [20000]}}, 32768)


def _analysis():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("analyze_phase10", ROOT / "scripts" / "analyze_phase10.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(book: str, target: int, policy: str, budget: float, nll: float) -> dict:
    return {"book": book, "target": target, "policy": policy, "budget": budget, "mean_nll": nll, "mean_kl": 0.0, "top1_agreement": 1.0}


def test_perplexity_ratio_is_the_exponentiated_paired_delta() -> None:
    import math

    a = _analysis()
    rows = []
    for i, ref in enumerate([2.0, 3.0, 2.5, 2.8]):
        rows += [_row("b", i, "full", 1.0, ref), _row("b", i, "block_full", 1.0, ref), _row("b", i, "quest", 0.5, ref + 0.1)]
    c = a.condition(rows, "quest", 0.5, iters=500, seed=0)
    assert math.isclose(c["nll_delta"], 0.1) and math.isclose(c["ppl_ratio"], math.exp(0.1))
    # Every window worse by the same amount: the interval collapses onto the effect.
    assert all(math.isclose(x, 0.1) for x in c["nll_delta_ci95"]) and c["windows_worse"] == 4
    ctrl = a.condition(rows, "block_full", 1.0, iters=500, seed=0)
    assert ctrl["nll_delta"] == 0.0 and ctrl["windows_worse"] == ctrl["windows_better"] == 0


def test_runs_from_different_commits_are_refused(tmp_path) -> None:  # noqa: ANN001
    import json

    for ctx, commit in ((4096, "aaa"), (8192, "bbb")):
        d = tmp_path / f"perplexity_ctx{ctx}"
        d.mkdir()
        m = {"config": {"context": ctx}, "teacher_forced": [], "wall_s": 1.0, "provenance": {"git_commit": commit, "git_dirty": False}}
        (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree"):
        _analysis().load_runs(tmp_path)


def test_spearman_handles_ties_and_order() -> None:
    a = _analysis()
    assert a.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert a.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(a.spearman([1, 1, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-12
