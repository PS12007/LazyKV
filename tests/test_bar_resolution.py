"""The statistics Phase 9's verdict rests on, tested against inputs whose answer is known.

Phase 9 exists because a retention within a fraction of a prompt of the bar was being read as a
verdict. Its replacement -- a paired bootstrap CI and a pre-registered three-way rule -- can lie in
the same direction if the pairing is lost, the rule is off by an inequality, or a chunked run counts
a prompt twice, so each of those is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.stats import paired_ratio_ci, prompts_to_resolve
from scripts.analyze_phase9 import label, load_chunks, phase8_reproduction, verdict

# ---- paired_ratio_ci -------------------------------------------------------------------


def test_concordant_pairs_add_no_width() -> None:
    # Half the prompts both fail, half both pass: the retention is exactly 1 on every resample.
    # An unpaired interval on each accuracy would be wide; the paired one must be a point.
    num = den = [0.0, 1.0] * 50
    assert paired_ratio_ci(num, den, iters=500) == (1.0, 1.0)


def test_discordant_pairs_are_what_widen_it() -> None:
    den = [1.0] * 100
    few = paired_ratio_ci([0.0] * 2 + [1.0] * 98, den, iters=2000)
    many = paired_ratio_ci([0.0] * 20 + [1.0] * 80, den, iters=2000)
    assert few[1] - few[0] < many[1] - many[0]
    assert few[0] <= 0.98 <= few[1] and many[0] <= 0.80 <= many[1]


def test_unpaired_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        paired_ratio_ci([1.0], [1.0, 1.0])


# ---- verdict ---------------------------------------------------------------------------


@pytest.mark.parametrize(("ci", "expected"), [
    ((0.99, 1.01), "meets"),       # touching the bar from above meets it (>=, as the headline says)
    ((0.995, 1.0), "meets"),
    ((0.95, 0.989), "fails"),
    ((0.95, 0.99), "unresolved"),  # an upper bound *at* the bar has not excluded it
    ((0.98, 1.00), "unresolved"),
])
def test_the_pre_registered_rule(ci: tuple[float, float], expected: str) -> None:
    assert verdict(ci, 0.99) == expected


# ---- prompts_to_resolve ----------------------------------------------------------------


def test_an_effect_on_the_bar_never_resolves() -> None:
    # 99 of 100: retention is exactly the bar, and no number of prompts separates the two.
    assert prompts_to_resolve([0.0] + [1.0] * 99, [1.0] * 100, 0.99) is None


def test_a_larger_gap_needs_fewer_prompts() -> None:
    den = [1.0] * 100
    near = prompts_to_resolve([0.0] * 3 + [1.0] * 97, den, 0.99)
    far = prompts_to_resolve([0.0] * 10 + [1.0] * 90, den, 0.99)
    assert near is not None and far is not None and far < near


def test_it_agrees_with_the_bootstrap_on_which_side_of_resolved_it_is() -> None:
    # 10% discordant, all against the policy: the CI at n=100 already excludes 0.99,
    # so the estimate must say fewer than 100 prompts were needed.
    num, den = [0.0] * 10 + [1.0] * 90, [1.0] * 100
    assert verdict(paired_ratio_ci(num, den, iters=2000), 0.99) == "fails"
    n = prompts_to_resolve(num, den, 0.99)
    assert n is not None and n < 100


# ---- chunk merging ---------------------------------------------------------------------


def _chunk(base: Path, name: str, samples: list[int], commit: str = "abc", dirty: bool = False) -> None:
    rows = [{"kind": "single", "depth": 0.0, "sample": s, "policy": p, "budget": b, "score": 1.0, "answer": "x"}
            for s in samples for p, b in [("full", 1.0), ("quest", 0.75)]]
    d = base / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "niah": rows, "teacher_forced": [], "wall_s": 1.0,
        "config": {"niah": {"samples": len(samples), "sample_start": samples[0]}},
        "provenance": {"git_commit": commit, "git_dirty": dirty},
    }), encoding="utf-8")


def test_chunks_merge(tmp_path: Path) -> None:
    _chunk(tmp_path, "niah_s00", [0, 1])
    _chunk(tmp_path, "niah_s02", [2, 3])
    niah, _, prov = load_chunks(tmp_path)
    assert len(niah) == 8 and len(prov) == 2


def test_a_prompt_counted_twice_is_refused(tmp_path: Path) -> None:
    # A duplicated prompt would narrow the CI for free.
    _chunk(tmp_path, "niah_s00", [0, 1])
    _chunk(tmp_path, "niah_s01", [1, 2])
    with pytest.raises(SystemExit):
        load_chunks(tmp_path)


@pytest.mark.parametrize(("commit", "dirty"), [("def", False), ("abc", True)])
def test_chunks_from_different_code_are_refused(tmp_path: Path, commit: str, dirty: bool) -> None:
    _chunk(tmp_path, "niah_s00", [0])
    _chunk(tmp_path, "niah_s01", [1], commit=commit, dirty=dirty)
    with pytest.raises(SystemExit):
        load_chunks(tmp_path)


def test_a_changed_answer_breaks_reproduction(tmp_path: Path) -> None:
    row = {"kind": "single", "depth": 0.0, "sample": 0, "policy": "quest", "budget": 0.75, "score": 1.0, "answer": "1234567"}
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"niah": [row]}), encoding="utf-8")
    assert phase8_reproduction([row], old)["reproduces"]
    # Same score, different text: still not the same measurement.
    assert not phase8_reproduction([{**row, "answer": "1234567."}], old)["reproduces"]


def test_labels_survive_a_dotted_lookup() -> None:
    assert "." not in label("quest", 0.75) and "." not in label("quest", 0.0625)
