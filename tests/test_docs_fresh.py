"""Hard rule 1, enforced: committed docs must equal a fresh render from metrics.json."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_renderer():  # scripts/ is not a package; load by path
    spec = importlib.util.spec_from_file_location("render_docs", ROOT / "scripts" / "render_docs.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rendered_docs_match_committed_files() -> None:
    rendered = _load_renderer().render_all()
    if not rendered:
        pytest.skip("no templates yet")
    stale = [p.relative_to(ROOT).as_posix() for p, text in rendered.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    assert not stale, f"docs out of date, run scripts/render_docs.py: {stale}"
