"""The KL ordering test: paired by window, exact sign test, no threshold beyond the study's alpha."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("analyze_kl", ROOT / "scripts" / "analyze_kl.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_consistent_ordering_resolves_and_a_mixed_one_does_not() -> None:
    m = _mod()
    lo = {("b", i): 0.001 for i in range(12)}
    hi = {("b", i): 0.002 for i in range(12)}
    c = m.compare(hi, lo)
    assert c["a_above"] == 12 and c["resolved"]
    mixed = {("b", i): (0.003 if i % 2 else 0.0005) for i in range(12)}
    assert not m.compare(mixed, lo)["resolved"]


def test_ties_count_for_neither_side() -> None:
    m = _mod()
    same = {("b", i): 0.001 for i in range(12)}
    c = m.compare(same, same)
    assert c["a_above"] == c["a_below"] == 0 and not c["resolved"]
