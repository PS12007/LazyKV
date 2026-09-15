"""Block pool and GPU-only policies, on a tiny random Llama.

Guards what every Phase 2 number depends on: a 100% pool is the full cache, eviction keeps
exactly the blocks the policy names at the right slots, positions stay logical after
eviction, and attention observation only runs for policies that use it.
"""

from __future__ import annotations

import copy

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention)")

BS = 8


@pytest.fixture(scope="module")
def tiny():  # noqa: ANN201
    from transformers import LlamaConfig, LlamaForCausalLM

    from lazykv.attention import IMPLEMENTATION_NAME, install

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=4096)
    install("cudnn_bucketed", cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, bucket=64)
    model = LlamaForCausalLM(copy.deepcopy(cfg))
    model.set_attn_implementation(IMPLEMENTATION_NAME)
    # bf16: the cuDNN kernel under test rejects float32.
    return cfg, model.to(dtype=torch.bfloat16).cuda().eval()


def _prefill(model, cfg, ids, max_len=512):  # noqa: ANN001, ANN202
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    cache = FullGPUCache(cfg.num_hidden_layers, max_len)
    pre = prefill(model, cache, ids, chunk_size=32)
    return cache, pre


def test_slots_for_budget_never_exceeds_budget() -> None:
    from lazykv.blocks import slots_for_budget

    assert slots_for_budget(1.0, 100, 8) == 13  # all 100 tokens fit, rounding up
    for frac in (0.75, 0.5, 0.25, 0.125, 0.0625):
        s = slots_for_budget(frac, 32768, 64)
        assert (s + 1) * 64 <= frac * 32768  # slots plus the tail region stay inside the budget
    with pytest.raises(ValueError):
        slots_for_budget(0.0, 10, 8)


def test_window_sink_boundary_and_victims() -> None:
    from lazykv.policies import WindowSink

    p = WindowSink(sink_blocks=1)
    assert p.initial_blocks(10, 4) == [0, 7, 8, 9]
    assert p.initial_blocks(3, 4) == [0, 1, 2]
    assert p.victim([0, 7, 8, 9]) == 1  # block 7 is the oldest non-sink block


@cuda
def test_lru_victim_is_least_recently_used() -> None:
    from lazykv.policies import LRU

    p = LRU()
    assert p.initial_blocks(10, 4) == [6, 7, 8, 9]
    p.start([6, 7, 8, 9], torch.device("cuda"))
    assert p.victim([6, 7, 8, 9]) == 0  # no use yet: insertion order
    p.used(torch.tensor([True, False, False, False], device="cuda"), step=0)
    assert p.victim([6, 7, 8, 9]) == 1  # block 6 was just used; 7 is now the oldest use
    p.sealed(1, 10, step=1)
    p.used(torch.tensor([False, False, True, True], device="cuda"), step=2)
    assert p.victim([6, 10, 8, 9]) == 0  # 6 used at 0, 10 sealed at 1, 8 and 9 used at 2


@cuda
def test_full_budget_pool_matches_full_cache(tiny) -> None:  # noqa: ANN001
    from lazykv.blocks import BlockPoolCache, slots_for_budget
    from lazykv.generate import teacher_forced_decode
    from lazykv.policies import Policy

    cfg, model = tiny
    ids = torch.randint(0, cfg.vocab_size, (1, 140), device="cuda")
    ctx, cont = ids[:, :101], ids[:, 101:]  # 101 is not a block multiple, so the tail is exercised
    full, pre = _prefill(model, cfg, ctx)
    pool = BlockPoolCache.from_full(full, slots_for_budget(1.0, 140, BS), BS, lambda _: Policy())
    ref = torch.stack(list(teacher_forced_decode(model, full, pre.last_logits, cont)))
    got = torch.stack(list(teacher_forced_decode(model, pool, pre.last_logits, cont)))
    assert (got - ref).abs().max().item() < 5e-2
    assert pool.get_seq_length() == full.get_seq_length() == 139
    assert pool.stats().gpu_resident_kv_bytes == full.stats().gpu_resident_kv_bytes


@cuda
@pytest.mark.parametrize("policy_name", ["window_sink", "lru"])
def test_evicting_pool_keeps_named_blocks_at_logical_positions(tiny, policy_name: str) -> None:  # noqa: ANN001
    """Layer-0 keys depend only on token and position, so every resident slot must hold exactly
    the full cache's layer-0 keys for the logical block the slot table names."""
    from lazykv.blocks import BlockPoolCache
    from lazykv.generate import teacher_forced_decode
    from lazykv.policies import make_policy

    cfg, model = tiny
    ids = torch.randint(0, cfg.vocab_size, (1, 200), device="cuda")
    ctx, cont = ids[:, :100], ids[:, 100:]
    full, pre = _prefill(model, cfg, ids[:, :199])  # reference keys for every position
    ref_keys = full.layers[0].keys[:, :, :199].clone()
    full.truncate(100)
    n_slots = 5
    pool = BlockPoolCache.from_full(full, n_slots, BS, lambda _: make_policy(policy_name))
    rows = list(teacher_forced_decode(model, pool, pre.last_logits, cont))  # 99 decode steps: many evictions
    assert len(rows) == 100
    layer = pool.pool_layers[0]
    assert layer.logical_len == 199 and pool.get_seq_length() == 199
    assert layer.n_used == n_slots and layer.counters.evictions > 0
    assert len(set(layer.slot_blocks)) == n_slots
    for slot, block in enumerate(layer.slot_blocks):
        got = layer.keys[:, :, slot * BS : (slot + 1) * BS]
        want = ref_keys[:, :, block * BS : (block + 1) * BS]
        assert torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2), (policy_name, slot, block)
    tail_start = (layer.next_block_id) * BS
    assert torch.allclose(layer.keys[:, :, n_slots * BS : n_slots * BS + layer.fill].float(), ref_keys[:, :, tail_start : tail_start + layer.fill].float(), rtol=1e-2, atol=1e-2)
    if policy_name == "window_sink":
        assert 0 in layer.slot_blocks  # the sink survived every eviction
    counters = pool.counters()
    assert (counters["host_observe_s"] > 0) == (policy_name == "lru")


@cuda
def test_window_sink_first_decode_step_matches_explicit_reference(tiny) -> None:  # noqa: ANN001
    """One decode step over a window+sink pool equals the full model attending to exactly those
    positions, built independently: a FullGPUCache holding only the kept KV, with explicit
    logical position ids."""
    from lazykv.blocks import BlockPoolCache
    from lazykv.cache import FullGPUCache
    from lazykv.policies import WindowSink

    cfg, model = tiny
    ids = torch.randint(0, cfg.vocab_size, (1, 101), device="cuda")
    full, pre = _prefill(model, cfg, ids[:, :100])
    pool = BlockPoolCache.from_full(full, 4, BS, lambda _: WindowSink())
    keep = pool.pool_layers[0].slot_blocks
    positions = [p for b in keep for p in range(b * BS, (b + 1) * BS)] + list(range(12 * BS, 100))
    idx = torch.tensor(positions, device="cuda")
    manual = FullGPUCache(cfg.num_hidden_layers, 256)
    for src, dst in zip(full.layers, manual.layers):
        k, v = src.keys[:, :, :100].index_select(2, idx), src.values[:, :, :100].index_select(2, idx)
        dst.update(k, v)
    tok = ids[:, 100:101]
    def logp(cache, position: int) -> torch.Tensor:  # noqa: ANN001
        for layer in cache.layers if cache is manual else []:
            layer.truncate(len(positions))
        with torch.inference_mode():
            out = model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1, position_ids=torch.tensor([[position]], device="cuda"))
        return torch.log_softmax(out.logits[0, -1].float(), -1)

    with torch.inference_mode():
        got = torch.log_softmax(model(input_ids=tok, past_key_values=pool, use_cache=True, logits_to_keep=1).logits[0, -1].float(), -1)
    right = (got - logp(manual, 100)).abs().max().item()
    # A random tiny model is not very position-sensitive, so a fixed tolerance alone could
    # pass with wrong positions. Require the match to be far closer than the position bug.
    wrong = (got - logp(manual, len(positions))).abs().max().item()
    assert right < 5e-2 and right < wrong / 5, (right, wrong)


@cuda
def test_pool_storage_is_bucket_aligned(tiny) -> None:  # noqa: ANN001
    """Bucketed decode pads the view inside storage; a pool ending mid-bucket forces a plan per token."""
    from lazykv.attention import current_strategy
    from lazykv.blocks import BlockPoolCache
    from lazykv.policies import WindowSink

    cfg, model = tiny
    full, _ = _prefill(model, cfg, torch.randint(0, cfg.vocab_size, (1, 100), device="cuda"))
    pool = BlockPoolCache.from_full(full, 3, BS, lambda _: WindowSink())
    _, bucket = current_strategy()
    assert pool.pool_layers[0].keys.shape[2] % bucket == 0


@cuda
def test_h2o_boundary_keeps_heavy_and_recent_and_never_evicts_recent() -> None:
    from lazykv.policies import H2O

    bs = 4
    # 10 prompt blocks; blocks 2 and 5 carry the most mass, block 0 the least.
    per_block = torch.tensor([0.0, 1, 9, 2, 3, 8, 1, 1, 1, 1], device="cuda")
    token_mass = per_block.repeat_interleave(bs) / bs
    p = H2O(token_mass=torch.cat([token_mass, torch.full((2,), 0.5, device="cuda")]), block_size=bs)
    keep = p.initial_blocks(10, 4)
    assert keep == [2, 5, 8, 9]  # two recent, two heaviest among the older blocks
    p.start(keep, torch.device("cuda"))
    # Block 10 is being sealed: the window is {9, 10}, so 9 is protected and 8 (mass 1) goes.
    assert keep[p.victim(keep)] == 8
    # Decode attention on block 8 lifts it above block 5, which becomes the lightest outside the window.
    probs = torch.zeros(1, 1, 1, 4 * bs + 2, device="cuda")
    probs[..., 2 * bs] = 20.0  # slot 2 holds block 8
    p.observed(probs, n_used=4, block_size=bs, step=0)
    assert keep[p.victim(keep)] == 5
    p.sealed(2, 10, step=1)  # tail mass (0.5 * 2) moves into slot 2
    assert p._mass is not None and abs(float(p._mass[2]) - 1.0) < 1e-6


@cuda
def test_h2o_pool_keeps_named_blocks_and_its_boundary_matches_exact_mass(tiny) -> None:  # noqa: ANN001
    from lazykv.blocks import BlockPoolCache
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill, teacher_forced_decode
    from lazykv.policies import H2O
    from lazykv.scoring import PrefillScorer

    cfg, model = tiny
    ids = torch.randint(0, cfg.vocab_size, (1, 200), device="cuda")
    ref = FullGPUCache(cfg.num_hidden_layers, 512)
    prefill(model, ref, ids[:, :199], chunk_size=32)
    ref_keys = ref.layers[0].keys[:, :, :199].clone()

    full = FullGPUCache(cfg.num_hidden_layers, 512)
    scorer = PrefillScorer(cfg.num_hidden_layers, 512, torch.device("cuda"), stride=1)
    pre = prefill(model, full, ids[:, :100], chunk_size=32, observer=scorer)
    assert scorer.sampled_queries == [100] * cfg.num_hidden_layers
    n_slots = 5
    pool = BlockPoolCache.from_full(full, n_slots, BS, lambda i: H2O(token_mass=scorer.full_sum_estimate(i, 100), block_size=BS))
    for i, layer in enumerate(pool.pool_layers):
        mass = scorer.mass[i][: 12 * BS].view(12, BS).sum(-1)
        n_recent = round(0.5 * n_slots)
        heavy = mass[: 12 - n_recent].topk(n_slots - n_recent).indices.tolist()
        assert layer.slot_blocks == sorted(heavy + list(range(12 - n_recent, 12)))
    rows = list(teacher_forced_decode(model, pool, pre.last_logits, ids[:, 100:]))
    assert len(rows) == 100
    layer = pool.pool_layers[0]
    assert layer.counters.evictions > 0 and pool.counters()["host_observe_s"] > 0
    for slot, block in enumerate(layer.slot_blocks):
        got = layer.keys[:, :, slot * BS : (slot + 1) * BS]
        want = ref_keys[:, :, block * BS : (block + 1) * BS]
        assert torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2), (slot, block)
    newest = max(layer.slot_blocks)
    assert newest == layer.next_block_id - 1  # the recent window survived every eviction
