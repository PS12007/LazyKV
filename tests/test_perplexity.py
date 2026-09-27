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
