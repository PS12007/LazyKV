"""Combining phases into one frontier: the repeatability check, and the two ways it can mislead."""

from __future__ import annotations

import json

import pytest

from harness.results import RESULTS_DIR
from scripts.analyze_ladder import repeatability

LADDER_METRICS = RESULTS_DIR / "ladder" / "metrics.json"


def _phase(policy: str, budget: float, accuracy: float, tokens_per_s: float) -> dict:
    return {
        "niah": [{"policy": policy, "budget": budget, "accuracy": accuracy}],
        "speed": [{"label": f"{policy}@{budget:g}", "tokens_per_s": {"median": tokens_per_s}}],
    }


def test_repeatability_reports_the_gap_between_phases_that_measured_the_same_thing() -> None:
    phases = {"phase3": _phase("quest", 0.25, 0.80, 40.0), "phase4": _phase("quest", 0.25, 0.82, 44.0)}
    (row,) = repeatability(phases)
    assert row["phases"] == ["phase3", "phase4"]
    assert row["accuracy_max_gap_pp"] == pytest.approx(2.0)
    assert row["tokens_per_s_max_ratio"] == pytest.approx(1.1)


def test_a_condition_only_one_phase_measured_is_not_a_repeatability_datum() -> None:
    """Otherwise the check would report a perfect zero gap for conditions nothing corroborates."""
    phases = {"phase3": _phase("quest", 0.25, 0.80, 40.0), "phase4": _phase("h2o", 0.25, 0.60, 44.0)}
    assert repeatability(phases) == []


needs_ladder = pytest.mark.skipif(not LADDER_METRICS.exists(), reason="ladder analysis not run")


@needs_ladder
def test_the_reference_rung_does_not_answer_the_question_about_the_others() -> None:
    """Rung 1 retains itself perfectly and at a 100% budget; letting it into either headline figure
    would overstate the ladder by exactly the amount the study exists to measure."""
    a = json.loads(LADDER_METRICS.read_text(encoding="utf-8"))
    s = a["summary"]
    assert "1" not in s["rungs_meeting_target_below_full"]
    best_others = [v["best_retention"] for k, v in a["best"].items() if k != "1"]
    assert s["best_retention_excluding_full"]["max"] == pytest.approx(max(best_others))
    assert s["best_retention_excluding_full"]["max"] < a["best"]["1"]["best_retention"]


@needs_ladder
def test_the_frontier_quotes_the_slot_variant_that_frees_vram() -> None:
    """Rungs measured at two slot counts must be quoted at the one where the memory axis means
    something, or the frontier's x axis silently mixes two definitions."""
    a = json.loads(LADDER_METRICS.read_text(encoding="utf-8"))
    for rung, v in a["best"].items():
        rows = [r for r in a["rows"] if str(r["rung"]) == rung and r["budget"] == v["best_retention_budget"]]
        if not any(r["slots"] == "= attended" for r in rows):
            continue
        honest = next(r for r in rows if r["slots"] == "= attended")
        assert v["resident_mib_at_best"] == pytest.approx(honest["resident_mib"])


@needs_ladder
def test_the_combined_frontier_is_only_published_because_the_sweeps_match() -> None:
    a = json.loads(LADDER_METRICS.read_text(encoding="utf-8"))
    assert a["summary"]["sweeps_comparable"] is True
    assert a["summary"]["overlapping_conditions"] > 0
    shapes = {(v["context"], v["block_size"], v["prompts"]) for v in a["shape"].values()}
    assert len(shapes) == 1
