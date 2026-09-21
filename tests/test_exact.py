"""Rung 9: the merge algebra, the CPU pass, and the one claim that matters — it is exact.

Rungs 6-8 were tested against rung 5, because they implement rung 5's selection. Rung 9 is not a
selection at all: it is the full cache, computed in two places. So it is tested against the full
cache, and the failure mode the tests exist to catch is a set split that is not a partition —
blocks attended twice, or not at all, either of which produces a plausible-looking wrong answer.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention, pinned memory)")
BS = 8
PROMPT = 100  # 12 full blocks + 4 tokens: 11 candidate blocks
K = 3


def _reference(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor, scaling: float) -> torch.Tensor:
    """Plain softmax attention over the whole key set, in float64: the thing a merge must equal."""
    scores = torch.matmul(query.double(), keys.double().transpose(-1, -2)) * scaling
    return torch.matmul(torch.softmax(scores, dim=-1), values.double())


# -- the algebra -------------------------------------------------------------------------------


def test_merging_two_halves_reconstructs_attention_over_the_whole() -> None:
    from lazykv.exact import cpu_partial_attention, merge_lse

    torch.manual_seed(0)
    h, groups, n_blocks, d = 3, 2, 6, 16
    keys = torch.randn(h, n_blocks * BS, d)
    values = torch.randn(h, n_blocks * BS, d)
    query = torch.randn(h, groups, d)
    scaling = d**-0.5

    first = np.tile(np.arange(0, 2), (h, 1))  # blocks 0,1 to one side
    second = np.tile(np.arange(2, n_blocks), (h, 1))  # the rest to the other
    out_a, lse_a = cpu_partial_attention(keys, values, query, scaling, BS, excluded=second)
    out_b, lse_b = cpu_partial_attention(keys, values, query, scaling, BS, excluded=first)
    merged, _ = merge_lse(out_a, lse_a, out_b, lse_b)

    want = _reference(keys, values, query, scaling)
    assert torch.allclose(merged.double(), want, atol=1e-10)


def test_an_empty_partial_contributes_nothing_instead_of_nan() -> None:
    """At a budget where the selection covers every sealed block, the CPU side has no columns
    left. Softmax over an empty set is undefined, and the naive merge returns nan for the token."""
    from lazykv.exact import cpu_partial_attention, merge_lse

    torch.manual_seed(1)
    h, groups, n_blocks, d = 2, 2, 4, 8
    keys, values = torch.randn(h, n_blocks * BS, d), torch.randn(h, n_blocks * BS, d)
    query = torch.randn(h, groups, d)
    everything = np.tile(np.arange(n_blocks), (h, 1))

    empty_out, empty_lse = cpu_partial_attention(keys, values, query, d**-0.5, BS, excluded=everything)
    assert torch.isneginf(empty_lse).all()
    assert torch.isfinite(empty_out).all()

    full_out, full_lse = cpu_partial_attention(keys, values, query, d**-0.5, BS, excluded=np.empty((h, 0), dtype=np.int64))
    merged, _ = merge_lse(full_out, full_lse, empty_out, empty_lse)
    assert torch.allclose(merged, full_out, atol=1e-6)


def test_the_merge_is_not_a_plain_average() -> None:
    """A weighted mean with the wrong weights passes the disjointness tests on symmetric data and
    fails on real attention. Pin the weighting to the log-sum-exp ratio explicitly."""
    from lazykv.exact import merge_lse

    out_a = torch.tensor([[1.0, 0.0]])
    out_b = torch.tensor([[0.0, 1.0]])
    lse_a, lse_b = torch.tensor([2.0]), torch.tensor([0.0])
    merged, lse = merge_lse(out_a, lse_a, out_b, lse_b)
    wa = float(torch.exp(lse_a - lse))
    assert merged[0, 0] == pytest.approx(wa)
    assert merged[0, 0] > merged[0, 1]  # the heavier partial dominates
    assert float(lse) == pytest.approx(float(torch.logaddexp(lse_a, lse_b)))


def test_excluding_a_block_removes_exactly_that_block() -> None:
    """Per-head exclusion: head 0 drops a block the other heads keep."""
    from lazykv.exact import cpu_partial_attention

    torch.manual_seed(2)
    h, groups, n_blocks, d = 3, 1, 5, 8
    keys, values = torch.randn(h, n_blocks * BS, d), torch.randn(h, n_blocks * BS, d)
    query = torch.randn(h, groups, d)
    excluded = np.array([[2], [4], [4]], dtype=np.int64)
    out, _ = cpu_partial_attention(keys, values, query, d**-0.5, BS, excluded)

    for j, drop in enumerate(excluded[:, 0]):
        keep = [t for t in range(n_blocks * BS) if not (drop * BS <= t < (drop + 1) * BS)]
        want = _reference(keys[j, keep], values[j, keep], query[j], d**-0.5)
        assert torch.allclose(out[j].double(), want, atol=1e-10)


# -- end to end --------------------------------------------------------------------------------


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


def _prefilled(model, cfg, n: int = PROMPT, extra: int = 40):  # noqa: ANN001, ANN202
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    gen = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, n + extra), device="cuda", generator=gen)
    full = FullGPUCache(cfg.num_hidden_layers, 512)
    pre = prefill(model, full, ids[:, :n], chunk_size=32)
    return ids, full, pre


def _forced(model, cache, pre, ids, steps: int):  # noqa: ANN001, ANN202
    """Log-probs at each forced continuation position, so the comparison does not depend on the
    two runs agreeing on what to decode next."""
    from lazykv.generate import teacher_forced_decode

    return torch.stack(list(teacher_forced_decode(model, cache, pre.last_logits, ids[:, PROMPT : PROMPT + steps])))


@cuda
@pytest.mark.parametrize("budget_blocks", [2, K])
def test_rung9_reproduces_the_full_cache(tiny, budget_blocks: int) -> None:  # noqa: ANN001
    """The claim of the rung, tested the only way it can be: against a full cache on the same
    prompt and the same forced tokens. A set split that double-counts or drops a block fails here
    while every residency invariant still holds."""
    from lazykv.exact import ExactTieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    steps = 24
    want = _forced(model, full, pre, ids, steps)
    full.truncate(PROMPT)

    cache = ExactTieredCache(full, budget_blocks, BS, capacity_tokens=512, dense_layers=1, fetch="gather", n_slots=budget_blocks + 1)
    got = _forced(model, cache, pre, ids, steps)
    cache.close()
    full.truncate(PROMPT)

    # Exact in the algebra, not in the bits: the GPU half runs the bf16 kernel, the CPU half runs
    # float32, and they are combined in float32. Demanding bitwise equality would fail a correct
    # merge, so the bar is that the log-probs agree to far inside bf16's resolution, and that any
    # argmax flip is a tie the reference itself could not resolve (Phase 1's near-tie problem).
    delta = (want - got).abs().max().item()
    assert delta < 0.02, delta
    for i in (want.argmax(-1) != got.argmax(-1)).nonzero().flatten().tolist():
        top2 = want[i].topk(2).values
        assert (top2[0] - top2[1]).item() < 2 * delta, (i, top2, delta)
    # Row 0 of a teacher-forced run comes from the prefill logits, so `steps` rows are steps-1 decodes.
    assert cache.exact.merges == (steps - 1) * (cfg.num_hidden_layers - 1)
    assert cache.exact.cold_tokens > 0


@cuda
def test_rung9_is_further_from_the_approximate_tier_than_from_the_full_cache(tiny) -> None:  # noqa: ANN001
    """Guards against the merge silently doing nothing: if `merge` were a no-op, rung 9 would
    reproduce rung 6 exactly and this test would fail while the exactness test also failed."""
    from lazykv.exact import ExactTieredCache
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    steps = 16
    reference = _forced(model, full, pre, ids, steps)
    full.truncate(PROMPT)

    approx = TieredCache(full, 2, BS, capacity_tokens=512, dense_layers=1, fetch="gather", n_slots=3)
    approx_logits = _forced(model, approx, pre, ids, steps)
    approx.close()
    full.truncate(PROMPT)

    exact = ExactTieredCache(full, 2, BS, capacity_tokens=512, dense_layers=1, fetch="gather", n_slots=3)
    exact_logits = _forced(model, exact, pre, ids, steps)
    exact.close()
    full.truncate(PROMPT)

    assert (exact_logits - reference).abs().max() < (approx_logits - reference).abs().max()


@cuda
def test_rung9_refuses_the_configurations_that_would_make_it_inexact(tiny) -> None:  # noqa: ANN001
    """A quantized host pool and a prefetch are both silently wrong for this rung, in opposite
    ways: the first reintroduces an approximation, the second adds a variable without changing
    what is attended."""
    from lazykv.exact import ExactTieredCache

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    with pytest.raises(ValueError, match="exact"):
        ExactTieredCache(full, K, BS, capacity_tokens=512, dense_layers=1, fetch="gather", quant="int8")
    with pytest.raises(ValueError, match="prefetch"):
        ExactTieredCache(full, K, BS, capacity_tokens=512, dense_layers=1, fetch="gather", model=model, prefetch=True)
    full.truncate(PROMPT)


@cuda
def test_the_mirror_holds_what_a_fetch_would_produce(tiny) -> None:  # noqa: ANN001
    """The CPU attends to the mirror and the GPU attends to fetched blocks. If the two ever
    disagree, rung 9 is exact with respect to nothing."""
    from lazykv.exact import ExactTieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    cache = ExactTieredCache(full, K, BS, capacity_tokens=512, dense_layers=1, fetch="gather", n_slots=K + 1)
    _forced(model, cache, pre, ids, 30)  # decodes past several seals
    for i, tier in cache.tiers.items():
        m = cache.mirrors[i]
        for b in range(tier.n_sealed):
            assert torch.equal(m.keys[:, b * BS : (b + 1) * BS], tier.host[b, :, 0].float())
            assert torch.equal(m.values[:, b * BS : (b + 1) * BS], tier.host[b, :, 1].float())
    cache.close()
    full.truncate(PROMPT)
