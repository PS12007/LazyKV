"""Phase 15 analysis: prompts are paired across models only when they are provably the same."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("analyze_phase15", ROOT / "scripts" / "analyze_phase15.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _full(kind: str, sample: int, values: list[str], pos: int) -> dict:
    return {"kind": kind, "depth": 0.5, "sample": sample, "policy": "full", "budget": 1.0, "score": 1.0, "values": values, "target_needle_pos": pos}


def test_a_prompt_with_a_moved_needle_is_not_paired() -> None:
    m = _mod()
    a = [_full("single", 0, ["123"], 10), _full("single", 1, ["456"], 20)]
    b = [_full("single", 0, ["123"], 10), _full("single", 1, ["456"], 21)]
    r = m.prompt_identity(a, b)
    assert r["shared"] == 2 and r["identical"] == 1 and r["keys"] == [("single", 0.5, 0)]


def test_joint_resampling_sees_a_real_model_gap() -> None:
    m = _mod()
    den = [1.0] * 30
    better = m.model_gap_ci([1.0] * 27 + [0.0] * 3, den, [1.0] * 12 + [0.0] * 18, den, iters=500, seed=0)
    assert better[0] > 0
    same = m.model_gap_ci([1.0] * 15 + [0.0] * 15, den, [1.0] * 15 + [0.0] * 15, den, iters=500, seed=0)
    assert same[0] <= 0 <= same[1]
