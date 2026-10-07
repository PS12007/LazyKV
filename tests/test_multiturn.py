"""Multi-turn sessions: the builder places what the schedule says, and the runner's KV accounting is exact.

Builder tests need the pinned tokenizer and the corpus and skip where either is absent. Runner tests
use a tiny random model on CUDA.
"""

from __future__ import annotations

import copy
from functools import cache

import pytest
import torch

REPO, REVISION = "unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cache
def _tok_and_books():  # noqa: ANN202
    try:
        from transformers import AutoTokenizer

        from harness.corpus import load_tokens

        tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, local_files_only=True)
        return tok, load_tokens("moby_dick", tok), load_tokens("pride_and_prejudice", tok)
    except Exception as e:  # noqa: BLE001 -- any missing piece means "cannot run here"
        pytest.skip(f"tokenizer or corpus unavailable: {e}")


@pytest.mark.parametrize("ctx", [4096, 8192])
def test_session_document_has_exact_length_every_needle_and_is_deterministic(ctx: int) -> None:
    from lazykv.multiturn import build_session

    tok, book, other = _tok_and_books()
    a = build_session(tok, book, other, ctx, 3)
    b = build_session(tok, book, other, ctx, 3)
    assert a.prefix_ids.numel() == ctx
    assert torch.equal(a.prefix_ids, b.prefix_ids) and [t.feed_ids for t in a.turns] == [t.feed_ids for t in b.turns]
    text = tok.decode(a.prefix_ids)
    for key, value, _, pos in a.doc_needles.values():
        assert f"for {key} is: {value}." in text
        assert f"for {key} is: {value}." in tok.decode(a.prefix_ids[pos : pos + 20])
    assert text.endswith(f"The special magic number for {a.turns[0].key} mentioned in the provided text is")
    assert sorted(d for _, _, d, _ in a.doc_needles.values()) == [0.1, 0.35, 0.6, 0.85]


def test_chat_needles_are_given_before_they_are_asked_and_never_in_the_document() -> None:
    from lazykv.multiturn import build_session

    tok, book, other = _tok_and_books()
    s = build_session(tok, book, other, 4096, 0)
    doc_text = tok.decode(s.prefix_ids)
    keys = [k for k, _, _, _ in s.doc_needles.values()] + [k for k, _, _ in s.chat_needles.values()]
    assert len(set(keys)) == len(keys)
    for cid, (key, value, given) in s.chat_needles.items():
        assert value not in doc_text
        assert f"for {key} is: {value}." in tok.decode(list(s.turns[given].feed_ids))
        asked = [t for t in s.turns if t.asks == cid]
        assert asked and all(t.index > given for t in asked)
        assert all(s.turns_since_given(t) == t.index - given for t in asked)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    for t in s.turns[1:]:
        assert t.feed_ids[0] == eot  # closes the previous answer
        assert t.value in (s.doc_needles.get(t.asks, ("", "", 0, 0))[1], s.chat_needles.get(t.asks, ("", "", 0))[1])


def test_schedule_that_asks_before_giving_is_refused() -> None:
    from lazykv.multiturn import build_session

    tok, book, other = _tok_and_books()
    with pytest.raises(ValueError, match="before it is given"):
        build_session(tok, book, other, 4096, 0, schedule=(("d0", None), ("c0", None), ("d1", "c0")))


@pytest.fixture(scope="module")
def tiny():  # noqa: ANN201
    from transformers import LlamaConfig, LlamaForCausalLM

    from lazykv.attention import IMPLEMENTATION_NAME, install

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=4096)
    install("cudnn_bucketed", cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, bucket=64)
    model = LlamaForCausalLM(copy.deepcopy(cfg))
    model.set_attn_implementation(IMPLEMENTATION_NAME)
    return cfg, model.to(dtype=torch.bfloat16).cuda().eval()


def _synthetic_session(vocab: int, prefix: int, feeds: list[int]):  # noqa: ANN202
    from lazykv.multiturn import Session, Turn

    g = torch.Generator().manual_seed(1)
    turns = [Turn(0, "d0", "k", "v", None, ())]
    turns += [Turn(i + 1, "d0", "k", "v", None, tuple(torch.randint(0, vocab, (n,), generator=g).tolist())) for i, n in enumerate(feeds)]
    return Session(0, prefix, torch.randint(0, vocab, (prefix,), generator=g), tuple(turns), {}, {})


@cuda
@pytest.mark.parametrize("turn_chunk", [None, 16])
@pytest.mark.parametrize("eot", [7, 10_000])  # 10_000 is outside the vocab: every answer runs to max_new
def test_kv_length_is_document_plus_feeds_plus_answers(tiny, eot: int, turn_chunk: int | None) -> None:  # noqa: ANN001
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill
    from lazykv.multiturn import run_session

    cfg, model = tiny
    s = _synthetic_session(cfg.vocab_size, 100, [30, 17, 41])
    full = FullGPUCache(cfg.num_hidden_layers, 512)
    pre = prefill(model, full, s.prefix_ids, chunk_size=32)
    res = list(run_session(model, full, pre.last_logits, s, eot=eot, max_new=6, start_len=100, turn_chunk=turn_chunk))
    assert [r.index for r in res] == [0, 1, 2, 3]
    assert all(len(r.answer_ids) <= 6 and eot not in r.answer_ids for r in res)
    assert full.get_seq_length() == 100 + 30 + 17 + 41 + sum(len(r.answer_ids) for r in res)
    assert res[1].length_before == 100 + len(res[0].answer_ids) + 30
    if eot == 10_000:
        assert all(len(r.answer_ids) == 6 for r in res)


@cuda
def test_tier_answers_every_turn_exactly_as_rung5(tiny) -> None:  # noqa: ANN001
    """Fed turns seal blocks into the tier mid-session; the tier must still match rung 5 token for token."""
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill
    from lazykv.multiturn import run_session
    from lazykv.selection import QuestView
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    s = _synthetic_session(cfg.vocab_size, 100, [30, 17, 41])
    full = FullGPUCache(cfg.num_hidden_layers, 512)
    pre = prefill(model, full, s.prefix_ids, chunk_size=32)
    ref = [r.answer_ids for r in run_session(model, QuestView(full, 3, 8, dense_layers=1), pre.last_logits, s, eot=7, max_new=6, start_len=100)]
    full.truncate(100)
    tier = TieredCache(full, 3, 8, capacity_tokens=512, model=model, dense_layers=1, fetch="gather")
    got = [r.answer_ids for r in run_session(model, tier, pre.last_logits, s, eot=7, max_new=6, start_len=100)]
    tier.close()
    assert got == ref
    assert tier.counters.seals > 0
