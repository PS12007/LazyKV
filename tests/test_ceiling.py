"""The replay ablation: the guards that decide whether its ceiling means anything.

The ablation's whole claim is that pass two changed the *decision* and nothing else. Every test
here is one way that claim can be false while the numbers still look plausible.
"""

from __future__ import annotations

import json

import pytest

from harness.results import RESULTS_DIR

CEILING_METRICS = RESULTS_DIR / "phase5" / "ceiling" / "metrics.json"

needs_ceiling = pytest.mark.skipif(not CEILING_METRICS.exists(), reason="ceiling analysis not run")


@pytest.fixture(scope="module")
def ceiling() -> dict:
    return json.loads(CEILING_METRICS.read_text(encoding="utf-8"))


def _tiered(ceiling: dict) -> list[dict]:
    return [c for c in ceiling["conditions"] if c["ceiling_speedup"] is not None]


@needs_ceiling
def test_replay_moved_exactly_the_same_pairs(ceiling: dict) -> None:
    """If replay fetched a different number of pairs it did different work, and the wall-time
    difference is not the decision cost. This is the one guard that can invalidate the phase."""
    assert ceiling["summary"]["all_fetched_pairs_match"] is True
    for c in _tiered(ceiling):
        assert c["fetched_pairs"] == c["replay_fetched_pairs"], c


@needs_ceiling
def test_the_ceiling_is_a_saving_and_not_a_regression(ceiling: dict) -> None:
    """Removing work cannot make decode slower; a ceiling below 1.0 would mean the replay path
    added cost of its own and the instrument is measuring itself."""
    for c in _tiered(ceiling):
        assert c["ceiling_speedup"] > 1.0, c
        assert c["saved_ms"] == pytest.approx(c["decode_ms"] - c["replay_ms"])


@needs_ceiling
def test_the_measured_ceiling_stays_under_the_arithmetic_one(ceiling: dict) -> None:
    """The replay removes the bound and its sync; the arithmetic bound removes *all* host select
    time. The first is a subset of the second, so a measured ceiling above it would mean the two
    are not measuring the same quantity and neither could be quoted."""
    for c in _tiered(ceiling):
        if c["arithmetic_ceiling_speedup"] is None:
            continue
        assert c["ceiling_speedup"] < c["arithmetic_ceiling_speedup"], c


@needs_ceiling
def test_the_replay_saving_agrees_with_the_independently_counted_rank_time(ceiling: dict) -> None:
    """Two instruments, built for different purposes, on the same quantity: the replay's wall-time
    saving and the tier's own rank+sync counters. They are allowed to disagree by a couple of
    milliseconds on a 50-60 ms token; a larger gap means one of them is wrong."""
    for c in _tiered(ceiling):
        if c.get("saved_vs_counted_rank_ms") is None:
            continue
        assert abs(c["saved_vs_counted_rank_ms"]) < 2.0, c


@needs_ceiling
def test_token_agreement_is_reported_high_but_not_asserted_exact(ceiling: dict) -> None:
    """The fast kernel is not bit-repeatable for single-query decode (Phase 1), so demanding
    equality would fail honestly-identical passes. A collapse is still structural evidence."""
    agreement = ceiling["summary"]["token_agreement"]
    assert 0.9 < agreement["min"] <= agreement["max"] <= 1.0


@needs_ceiling
def test_the_full_cache_is_the_thing_the_ceiling_is_measured_against(ceiling: dict) -> None:
    """The phase's conclusion is that even a perfect decision leaves the tier behind the full
    cache. That only holds if the full cache was measured in the same run set."""
    full = [c for c in ceiling["conditions"] if c["policy"] == "full"]
    assert len(full) == 1
    assert ceiling["summary"]["full_decode_ms"] == pytest.approx(full[0]["decode_ms"])
    for c in _tiered(ceiling):
        assert c["replay_ms"] > full[0]["decode_ms"], c
