"""Rungs 6-7 (CPU tier): same attention as rung 5, bit for bit, with a consistent residency table."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention, pinned memory)")
BS = 8
PROMPT = 100  # 12 full blocks + 4 tokens: 11 candidate blocks
K = 3


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


def _check_residency(cache) -> None:  # noqa: ANN001
    for tier in cache.tiers.values():
        for j in range(tier.h):
            for s in range(tier.k):
                b = int(tier.slot_block[j, s])
                assert tier.block_slot[j, b] == s + 1
                assert torch.equal(tier.pool[s + 1, j].cpu(), tier.host[b, j])
        assert (tier.block_slot >= 0).sum() == tier.h * tier.k


@cuda
@pytest.mark.parametrize("strategy", ["cudnn_bucketed", "efficient"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_tier_is_bit_identical_to_rung5(tiny, strategy: str, prefetch: bool) -> None:  # noqa: ANN001
    """Same selection, same gather order, same kernel: the tier may move bytes, never change them."""
    from lazykv.attention import set_strategy_for_test
    from lazykv.generate import teacher_forced_decode
    from lazykv.selection import QuestView
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    restore = set_strategy_for_test(strategy)
    try:
        ids, full, pre = _prefilled(model, cfg)
        cont = ids[:, PROMPT : PROMPT + 40]  # crosses several seals
        quest = QuestView(full, k_blocks=K, block_size=BS, dense_layers=1)
        ref = torch.stack(list(teacher_forced_decode(model, quest, pre.last_logits, cont)))
        assert quest.counters.selections > 0
        full.truncate(PROMPT)
        tier = TieredCache(full, K, BS, capacity_tokens=512, model=model, prefetch=prefetch, dense_layers=1)
        got = torch.stack(list(teacher_forced_decode(model, tier, pre.last_logits, cont)))
        tier.close()
        assert torch.equal(got, ref)
        c = tier.counters
        assert c.selections == quest.counters.selections
        assert c.seals == ((PROMPT + 39) // BS - PROMPT // BS) * 3  # 3 tiered layers
        assert c.hit_pairs + c.fetched_pairs == c.selected_pairs
        assert c.fetched_pairs > 0  # the recency seed cannot be right for a random model
        if prefetch:
            assert c.prefetched_pairs > 0
        _check_residency(tier)
    finally:
        restore()


@cuda
def test_seal_writes_the_exact_block_to_host(tiny) -> None:  # noqa: ANN001
    """Host pool rows equal the KV rung 5 writes for the same decode, for prompt and decode-time blocks."""
    from lazykv.generate import teacher_forced_decode
    from lazykv.selection import QuestView
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    cont = ids[:, PROMPT : PROMPT + 30]
    tier = TieredCache(full, K, BS, capacity_tokens=512, dense_layers=1)
    list(teacher_forced_decode(model, tier, pre.last_logits, cont))
    tier.close()
    # Rung 5 attends to the same sets, so the KV it appends to the full cache is the reference.
    full.truncate(PROMPT)
    list(teacher_forced_decode(model, QuestView(full, K, BS, dense_layers=1), pre.last_logits, cont))
    for i, t in tier.tiers.items():
        layer = full.layers[i]
        assert t.n_sealed == (PROMPT + 29) // BS
        for b in range(t.n_sealed):
            got = t.host[b, :, 0]
            assert torch.equal(got, layer.keys[0, :, b * BS : (b + 1) * BS].cpu())
            assert torch.equal(t.host[b, :, 1], layer.values[0, :, b * BS : (b + 1) * BS].cpu())
            assert torch.equal(t.kmin[0, :, b].cpu(), got.amin(dim=1)) and torch.equal(t.kmax[0, :, b].cpu(), got.amax(dim=1))


@cuda
def test_admit_coalesces_runs_and_evicts_outside_the_selection(tiny) -> None:  # noqa: ANN001
    from lazykv.tiered import TierCounters, TieredCache

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    cache = TieredCache(full, K, BS, capacity_tokens=512, dense_layers=1)
    tier = cache.tiers[1]
    # Seed: blocks 9, 10, 11 resident in slots 1..3 for both heads.
    assert (tier.slot_block == np.array([9, 10, 11])).all()
    chosen = np.array([[2, 3, 11], [2, 3, 11]])  # both heads want blocks 2 and 3, keep 11
    mh, mb, transfers = tier.admit(chosen, step=1, counters=TierCounters(), stream=None)
    torch.cuda.synchronize()
    assert sorted(zip(mh.tolist(), mb.tolist())) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    # Victims are slots 1 and 2 (blocks 9, 10) for both heads, so (2,h0)(2,h1)(3,h0)(3,h1) map to
    # consecutive destinations: a single transfer.
    assert transfers == 1
    assert tier.block_slot[0, 11] == 3 and tier.block_slot[0, 9] == -1 and tier.evicted_at[1, 10] == 1
    _check_residency(cache)
    cache.close()


@cuda
def test_close_removes_prefetch_hooks(tiny) -> None:  # noqa: ANN001
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    before = [len(l._forward_pre_hooks) for l in model.model.layers]
    cache = TieredCache(full, K, BS, capacity_tokens=512, model=model, prefetch=True, dense_layers=1)
    assert sum(len(l._forward_pre_hooks) for l in model.model.layers) == sum(before) + 3
    cache.close()
    assert [len(l._forward_pre_hooks) for l in model.model.layers] == before


@cuda
def test_budget_that_covers_every_block_is_refused(tiny) -> None:  # noqa: ANN001
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    with pytest.raises(ValueError):
        TieredCache(full, 11, BS, capacity_tokens=512, dense_layers=1)
