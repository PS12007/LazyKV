"""configs/suite.yaml must cover what results/ records, or the entry point silently skips work."""

from __future__ import annotations

from harness.suite import analysis_steps, diff_paths, load_suite, normalize_argv, recorded
from harness.results import REPO_ROOT


def test_every_result_directory_has_a_phase() -> None:
    recorded()  # raises KeyError naming the directory if one is unmapped


def test_every_recorded_analysis_is_in_the_order() -> None:
    suite = load_suite()
    ordered = {a for _, a in analysis_steps(suite)}
    missing = sorted({r.argv for r in recorded() if r.is_analysis} - ordered)
    assert not missing, f"add to analysis_order in configs/suite.yaml: {missing}"


def test_every_ordered_step_exists() -> None:
    suite = load_suite()
    for _, a in analysis_steps(suite):
        assert (REPO_ROOT / a[0]).exists(), a
    for a in suite["finish"]:
        assert (REPO_ROOT / a[0]).exists(), a


def test_reported_drivers_are_replayable() -> None:
    # Two old runs used a session-scratchpad config. Both are smoke or probe runs; a new one would be
    # a result nobody can reproduce from a clone, so the list is pinned.
    broken = sorted(r.result for r in recorded() if not r.is_analysis and not r.is_smoke and r.not_reproducible_because())
    assert broken == ["phase4/probe_per_pair_transfers/run_1"]


def test_normalize_argv() -> None:
    assert normalize_argv(["scripts\\02_x.py", "--config", "configs\\p.yaml"]) == ("scripts/02_x.py", "--config", "configs/p.yaml")
    assert normalize_argv(["C:/a/b/analyze_y.py"]) == ("scripts/analyze_y.py",)
    assert normalize_argv([]) == ()


def test_diff_paths() -> None:
    assert diff_paths({"a": [1, 2.0], "b": {"c": "x"}}, {"a": [1, 2.0], "b": {"c": "x"}}) == []
    assert diff_paths({"a": [1, 2]}, {"a": [1, 3]}) == ["a[1]"]
    assert diff_paths({"a": 1}, {"b": 1}) == ["a", "b"]
    assert diff_paths({"a": [1]}, {"a": [1, 2]}) == ["a[len 1 -> 2]"]
    assert diff_paths({"x": float("nan")}, {"x": float("nan")}) == []
