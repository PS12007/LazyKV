"""The two inferences Phase 8 rests on, and the ways each can be made to lie.

`collapse` decides whether retention follows the budget fraction or the blocks actually attended,
and `_fit` supplies the intercept share that separates a manager bound by ranking blocks from one
bound by a per-layer constant. Both turn a handful of points into a single claim, so both are
tested against inputs whose right answer is known by construction rather than only against the run.
"""

from __future__ import annotations

import json

import pytest

from harness.results import RESULTS_DIR
from scripts.analyze_phase8 import _fit, collapse

PHASE8_METRICS = RESULTS_DIR / "phase8" / "analysis" / "metrics.json"
BS = 64


def _ctx(context: int, rows: list[tuple[str, float, float]]) -> dict:
    """One context's reduced quality half: (policy, budget, retention) triples plus the reference."""
    return {
        "niah": [{"policy": "full", "budget": 1.0, "retention": 1.0, "accuracy": 1.0}]
        + [{"policy": p, "budget": b, "retention": r, "accuracy": r} for p, b, r in rows],
    }


# ---- collapse ---------------------------------------------------------------------------


def test_retention_that_follows_the_block_count_collapses_on_k_not_on_the_fraction() -> None:
    """The hypothesis Phase 8 exists to test, injected as ground truth.

    K = floor(budget x context / 64) - 2, so (4096, 0.5) and (8192, 0.25) both select 30 blocks.
    Here retention is a pure function of K, so grouping by K must show no spread and grouping by
    budget must show some.
    """
    per_ctx = {
        4096: _ctx(4096, [("quest", 0.5, 0.90), ("quest", 0.25, 0.70)]),
        8192: _ctx(8192, [("quest", 0.5, 0.98), ("quest", 0.25, 0.90)]),
    }
    out = collapse(per_ctx, ["quest"], BS)
    assert out["summary_by_k_blocks"]["max_spread_pp"] == pytest.approx(0.0)
    assert out["summary_by_budget"]["max_spread_pp"] > 0
    assert out["fraction_over_k_mean_spread"] is None or out["fraction_over_k_mean_spread"] > 1


def test_retention_that_follows_the_fraction_collapses_the_other_way() -> None:
    """The converse, so the metric cannot be one that always favours K."""
    per_ctx = {
        4096: _ctx(4096, [("quest", 0.5, 0.90), ("quest", 0.25, 0.70)]),
        8192: _ctx(8192, [("quest", 0.5, 0.90), ("quest", 0.25, 0.70)]),
    }
    out = collapse(per_ctx, ["quest"], BS)
    assert out["summary_by_budget"]["max_spread_pp"] == pytest.approx(0.0)
    assert out["summary_by_k_blocks"]["max_spread_pp"] > 0


def test_a_group_only_one_context_measured_is_not_evidence_of_agreement() -> None:
    """A singleton group has zero spread by construction. Counting it would let whichever collapse
    happened to be sparser look tighter, which is the exact failure this analysis must not have:
    grouping by K produces many more singletons than grouping by budget does.
    """
    per_ctx = {
        4096: _ctx(4096, [("quest", 0.5, 0.90)]),
        8192: _ctx(8192, [("quest", 0.25, 0.90), ("quest", 0.0625, 0.10)]),
    }
    out = collapse(per_ctx, ["quest"], BS)
    # K=30 spans both contexts; K=6 exists only at 8192 and must not be scored.
    assert [g["value"] for g in out["by_k_blocks"]] == [30]
    assert out["summary_by_k_blocks"]["groups"] == 1
    # No budget appears at more than one context here, so the fraction collapse has nothing to say.
    assert out["by_budget"] == []


def test_the_reference_rung_is_not_a_retention_datum() -> None:
    """The full cache retains itself perfectly at every context, so including it would drag both
    spreads toward zero by exactly the amount that carries no information."""
    per_ctx = {4096: _ctx(4096, [("quest", 0.5, 0.90)]), 8192: _ctx(8192, [("quest", 0.25, 0.90)])}
    out = collapse(per_ctx, ["quest"], BS)
    assert all(p["policy"] != "full" for p in out["points"])


# ---- _fit -------------------------------------------------------------------------------


def test_a_cost_that_does_not_move_with_context_puts_all_of_itself_in_the_intercept() -> None:
    fit = _fit([64.0, 128.0, 256.0, 512.0], [20.0, 20.0, 20.0, 20.0])
    assert fit["slope"] == pytest.approx(0.0)
    assert fit["intercept"] == pytest.approx(20.0)


def test_a_cost_proportional_to_the_block_count_puts_none_of_itself_there() -> None:
    fit = _fit([64.0, 128.0, 256.0], [6.4, 12.8, 25.6])
    assert fit["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert fit["slope"] == pytest.approx(0.1)
    assert fit["r2"] == pytest.approx(1.0)


def test_two_contexts_cannot_separate_a_constant_from_a_slope() -> None:
    """Any two points fit a line exactly, so the intercept would be reported with false confidence."""
    assert _fit([64.0, 128.0], [10.0, 20.0]) is None


# ---- the run ----------------------------------------------------------------------------

needs_phase8 = pytest.mark.skipif(not PHASE8_METRICS.exists(), reason="phase 8 analysis not run")


@needs_phase8
def test_the_sweep_holds_everything_but_context_fixed() -> None:
    """The whole inference is that context is the only axis that moved. One block size, one budget
    set, one prompt count, one model across every context, or the curves are not comparable."""
    m = json.loads(PHASE8_METRICS.read_text(encoding="utf-8"))
    per = m["per_context"]
    assert len({d["niah_prompts_per_condition"] for d in per.values()}) == 1
    assert len({d["kv_bytes_per_token"] for d in per.values()}) == 1
    commits = {p["policy_quality"]["git_commit"] for p in m["sources"].values()}
    assert len(commits) == 1, "contexts measured at different commits are not one sweep"
    assert not any(p["policy_quality"]["git_dirty"] for p in m["sources"].values())


@needs_phase8
def test_the_32k_column_reproduces_the_phase_it_overlaps() -> None:
    """Phase 8 re-measures conditions Phase 7 published. If they disagree, the context sweep is a
    separate study and its 32K point cannot be spliced onto the ladder the other phases built."""
    m = json.loads(PHASE8_METRICS.read_text(encoding="utf-8"))
    agree = m["phase7_agreement"]
    if not agree.get("compared"):
        pytest.skip("phase 7 analysis absent")
    assert agree["conditions"] > 0
    assert agree["max_accuracy_gap_pp"] == pytest.approx(0.0, abs=1e-9)
