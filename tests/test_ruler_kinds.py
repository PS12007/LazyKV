"""RULER variable tracking and common words extraction, and a guard on the original NIAH prompts.

These need the pinned tokenizer (local Hugging Face cache) and the verified corpus; they skip
cleanly where either is absent, as on a fresh clone without `scripts/fetch_corpus.py`.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


@cache
def _tok_and_book():  # noqa: ANN202
    try:
        from transformers import AutoTokenizer

        from harness.corpus import load_tokens

        tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, local_files_only=True)
        return tok, load_tokens("moby_dick", tok)
    except Exception as e:  # noqa: BLE001 -- any missing piece means "cannot run here"
        pytest.skip(f"tokenizer or corpus unavailable: {e}")


@pytest.mark.parametrize("kind", ["vt", "cwe"])
@pytest.mark.parametrize("ctx", [4096, 8192])
def test_ruler_prompts_have_exact_length_and_are_deterministic(kind: str, ctx: int) -> None:
    from lazykv.niah import build_prompt

    tok, book = _tok_and_book()
    a = build_prompt(tok, book, ctx, kind, 0.5, 0)
    b = build_prompt(tok, book, ctx, kind, 0.5, 0)
    assert a.length == ctx and a.input_ids.equal(b.input_ids) and a.values == b.values
    assert not build_prompt(tok, book, ctx, kind, 0.5, 1).input_ids.equal(a.input_ids)


def test_vt_chain_is_in_order_and_only_its_first_statement_holds_the_value() -> None:
    from lazykv.niah import build_prompt

    tok, book = _tok_and_book()
    for depth in (0.0, 0.5, 1.0):
        p = build_prompt(tok, book, 4096, "vt", depth, 3)
        text = tok.decode(p.input_ids)
        names, value = p.values, p.key
        assert len(names) == 5 and text.count(f"VAR {names[0]} = {value}.") == 1
        for a, b in zip(names, names[1:]):
            assert text.count(f"VAR {b} = VAR {a}.") == 1
        # Statement positions are the chain's order: following it means reading forwards.
        assert list(p.needle_token_positions) == sorted(p.needle_token_positions)
        # Asked by value, which only the first hop contains: the rest must be traced.
        assert text.count(value) == 3  # the statement, the question, the answer prefix


def test_cwe_frequencies_are_rulers() -> None:
    from lazykv.niah import CWE_FREQ_COMMON, CWE_FREQ_UNCOMMON, build_prompt

    tok, book = _tok_and_book()
    p = build_prompt(tok, book, 8192, "cwe", 0.0, 0)
    text = tok.decode(p.input_ids)
    listed = [line.split(". ", 1)[1] for line in text.splitlines() if ". " in line and line.split(". ", 1)[0].isdigit()]
    counts = Counter(listed)
    assert len(p.values) == 10 and all(counts[w] == CWE_FREQ_COMMON for w in p.values)
    assert {c for w, c in counts.items() if w not in p.values} == {CWE_FREQ_UNCOMMON}


def test_scoring_is_case_insensitive_and_partial() -> None:
    from lazykv.niah import NiahPrompt, score

    import torch

    p = NiahPrompt("cwe", 0.0, 0, torch.zeros(1, dtype=torch.long), "", ("harpoon", "whale"))
    assert score(p, "1. Harpoon 2. ship") == 0.5 and score(p, "WHALE, harpoon") == 1.0


def test_the_original_needle_prompts_are_unchanged() -> None:
    """Adding kinds must not move a single needle: Phase 9 pooled prompts across phases by index."""
    from lazykv.niah import build_prompt

    path = ROOT / "results" / "phase9" / "niah_s00" / "metrics.json"
    if not path.exists():
        pytest.skip("Phase 9 results not present")
    tok, book = _tok_and_book()
    m = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in m["niah"] if r["policy"] == "full"][:6]
    for r in rows:
        p = build_prompt(tok, book, m["config"]["context"], r["kind"], r["depth"], r["sample"], seed=m["config"]["niah"]["seed"])
        assert list(p.values) == r["values"] and p.needle_token_positions[0] == r["target_needle_pos"]
