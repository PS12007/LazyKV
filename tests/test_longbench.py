"""LongBench QA prompts and scoring, as the reference implementation defines them."""

from __future__ import annotations

import pytest

from lazykv import longbench as lb


def test_f1_normalizes_like_longbench() -> None:
    assert lb.f1("The Eiffel Tower!", "eiffel tower") == 1.0
    assert lb.f1("Paris", "Paris, France") == pytest.approx(2 / 3)
    assert lb.f1("", "anything") == 0.0


def test_score_takes_the_best_reference_answer() -> None:
    import torch

    p = lb.LongBenchPrompt("hotpotqa", 0, torch.zeros(1, dtype=torch.long), ("Barack Obama", "Obama"), False)
    assert lb.score(p, "obama") == 1.0


def test_long_prompts_are_cut_in_the_middle_before_the_chat_template() -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct", revision="5a8abab4a5d6f164389b1079fb721cfab8d7126c", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")
    row = {"context": " ".join(f"word{i}" for i in range(4000)), "input": "What is the first word?", "answers": ["word0"]}
    short = lb.build_prompt(tok, row, "multifieldqa_en", 0, max_length=10**6)
    cut = lb.build_prompt(tok, row, "multifieldqa_en", 0, max_length=512)
    assert not short.truncated and cut.truncated and cut.length < short.length
    text = tok.decode(cut.input_ids)
    # Both ends survive: the instruction before the context, and the question after it.
    assert "Read the following text" in text and "What is the first word?" in text and "word2000" not in text
