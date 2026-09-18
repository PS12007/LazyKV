"""Rung 8 (int8 warm/cold tier): the hot tier stays exact, and a block's contents never depend on residency."""

from __future__ import annotations

import pytest
import torch

from tests.test_tiered import BS, K, PROMPT, _prefilled, cuda, tiny  # noqa: F401


def _dequantized(tier, block: int) -> torch.Tensor:  # noqa: ANN001
    """What a fetch of `block` would put in a slot: [h, 2, bs, d] in the model's dtype."""
    from lazykv.quant import unpack

    return unpack(tier.host[block].cuda(), tier.bs, tier.d, tier.pool.dtype)


def _build(model, full, **kw):  # noqa: ANN001, ANN202
    from lazykv.tiered import TieredCache

    return TieredCache(full, K, BS, capacity_tokens=512, model=model, dense_layers=1, fetch="gather", quant="int8", **kw)


@cuda
def test_the_host_pool_narrows_by_exactly_the_record_width(tiny) -> None:  # noqa: ANN001
    """Rung 8's reason to exist: the capacity claim, end to end through the cache.

    The ratio here is the tiny test geometry's, not the study's: the scale arrays are per channel
    and per token, so their share shrinks as the block grows. test_quant pins the real geometry.
    """
    from lazykv.quant import packed_pair_bytes

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    tier = _build(model, full)
    logical, stored = tier.host_logical_bytes(), tier.stats().host_kv_bytes
    assert stored / logical == packed_pair_bytes(BS, cfg.head_dim) / (2 * BS * cfg.head_dim * 2)
    assert tier.quant == "int8" and tier.policy_name == "tiered_int8"


@cuda
def test_every_resident_slot_holds_exactly_what_a_fetch_would_produce(tiny) -> None:  # noqa: ANN001
    """The invariant the whole rung rests on.

    If a slot seeded at the boundary held exact KV while a re-fetch produced dequantized KV, then
    evicting a block would silently change what the model attends to, and no two runs with
    different eviction histories would be comparable.
    """
    from lazykv.generate import teacher_forced_decode

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    tier = _build(model, full)
    list(teacher_forced_decode(model, tier, pre.last_logits, ids[:, PROMPT : PROMPT + 40]))
    checked = 0
    for layer in tier.tiers.values():
        for j in range(layer.h):
            for s in range(layer.n_slots):
                b = int(layer.slot_block[j, s])
                if b < 0:
                    continue
                assert torch.equal(layer.pool[s + 1, j], _dequantized(layer, b)[j])
                checked += 1
    assert checked > 0


@cuda
def test_refetching_an_evicted_block_returns_the_same_bytes(tiny) -> None:  # noqa: ANN001
    """Quantization happens once, on the way out; a round trip through the tier must be idempotent."""
    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    tier = _build(model, full)
    layer = next(iter(tier.tiers.values()))
    chosen = layer.slot_block[:, :1].copy()  # blocks that are resident right now
    before = {j: layer.pool[1, j].clone() for j in range(layer.h)}  # slot 1 holds chosen[j, 0]
    layer.slot_block[:, :] = -1  # evict everything, then ask for them back
    layer.block_slot[:, :] = -1
    layer.last_used[:, :] = -2
    layer.pool[1:].zero_()
    layer.admit(chosen, step=1, counters=tier.counters, stream=None, stage=tier._stage_compute)
    torch.cuda.synchronize()
    for j in range(layer.h):
        slot = int(layer.block_slot[j, chosen[j, 0]])
        assert torch.equal(layer.pool[slot, j], before[j])


@cuda
def test_the_sink_stays_exact_because_it_never_leaves_vram(tiny) -> None:  # noqa: ANN001
    """"fp16 hot, int8 warm/cold": block 0 is permanently hot, so it is not narrowed."""
    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    exact = {i: full.layers[i].keys[0, :, :BS].clone() for i in range(1, cfg.num_hidden_layers)}
    tier = _build(model, full)
    for i, layer in tier.tiers.items():
        assert torch.equal(layer.pool[0, :, 0], exact[i])


@cuda
def test_the_quest_bound_is_taken_over_the_keys_that_are_actually_attended(tiny) -> None:  # noqa: ANN001
    """The bound must cover the dequantized keys, not the exact ones that stop existing on seal."""
    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    tier = _build(model, full)
    for layer in tier.tiers.values():
        for b in range(1, layer.n_sealed):
            keys = _dequantized(layer, b)[:, 0]  # [h, bs, d]
            assert (keys >= layer.kmin[0, :, b].unsqueeze(1)).all()
            assert (keys <= layer.kmax[0, :, b].unsqueeze(1)).all()


@cuda
def test_decode_differs_from_the_exact_tier_but_not_wildly(tiny) -> None:  # noqa: ANN001
    """Rung 8 is the first tier rung with a quality axis: it must actually change the output, finitely."""
    from lazykv.generate import teacher_forced_decode
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    cont = ids[:, PROMPT : PROMPT + 40]
    exact = TieredCache(full, K, BS, capacity_tokens=512, model=model, dense_layers=1, fetch="gather")
    ref = torch.stack(list(teacher_forced_decode(model, exact, pre.last_logits, cont)))
    full.truncate(PROMPT)
    got = torch.stack(list(teacher_forced_decode(model, _build(model, full), pre.last_logits, cont)))
    full.truncate(PROMPT)
    assert torch.isfinite(got).all()
    assert not torch.equal(got, ref)  # if these matched, the tier would not be quantizing anything
    # Typical logits barely move; the tail is where a *different block* got selected, which is a
    # real consequence of quantizing the metadata and not something to assert away.
    diff, spread = (got.float() - ref.float()).abs(), ref.float().std()
    assert diff.mean() < 0.02 * spread
    assert diff.max() < spread


@cuda
def test_runs_fetch_is_refused(tiny) -> None:  # noqa: ANN001
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    with pytest.raises(ValueError, match="needs fetch='gather'"):
        TieredCache(full, K, BS, capacity_tokens=512, model=model, dense_layers=1, fetch="runs", quant="int8")


@cuda
def test_a_bf16_host_pool_is_refused_for_an_int8_tier(tiny) -> None:  # noqa: ANN001
    """The sweep keeps one pool per precision; handing over the wrong one must fail loudly."""
    from lazykv.tiered import allocate_host_pools

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg)
    wrong = allocate_host_pools(cfg.num_hidden_layers - 1, cfg.num_key_value_heads, cfg.head_dim, BS, 512)
    with pytest.raises(ValueError, match="host pool must be pinned"):
        _build(model, full, host_pools=wrong)


@cuda
def test_the_sweep_keeps_one_host_pool_per_precision(tiny) -> None:  # noqa: ANN001
    """A sweep that measures rung 6 against rung 8 needs both pools alive, and must not swap them."""
    from lazykv.sweep import Condition, build_cache, make_host_memory

    cfg, model = tiny
    conds = [Condition("tiered_sync", 0.25), Condition("tiered_int8", 0.25)]
    host = make_host_memory(conds, model, BS, 512, PROMPT, {"fetch": "gather", "spare": 0.0})
    assert set(host.pools) == {None, "int8"}
    assert host.pools[None][0].dtype == torch.bfloat16
    assert host.pools["int8"][0].dtype == torch.uint8

    _, full, _ = _prefilled(model, cfg)
    for cond, quant in ((conds[0], None), (conds[1], "int8")):
        built = build_cache(full, cond, PROMPT, BS, model=model, host=host, tier={"fetch": "gather", "spare": 0.0})
        assert built.cache.quant == quant
        # Each cache must be using the pool of its own precision, not a copy of it.
        assert next(iter(built.cache.tiers.values())).host.data_ptr() == host.pools[quant][0].data_ptr()
        built.close()
        full.truncate(PROMPT)
