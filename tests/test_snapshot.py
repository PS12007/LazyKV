"""Session snapshots: a restored tier continues exactly as the saved one would have."""

from __future__ import annotations

import copy
import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention, pinned memory)")
BS, K, CAP, PROMPT = 8, 3, 512, 100


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


def _feed(model, cache, toks: list[int]) -> list[torch.Tensor]:  # noqa: ANN001
    from lazykv.generate import cache_model_kwargs

    out = []
    ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for t in toks:
            ids.fill_(t)
            out.append(model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1, **cache_model_kwargs(cache)).logits[0, -1].float())
    return out


@cuda
@pytest.mark.parametrize("unbuffered", [False, pytest.param(True, marks=pytest.mark.skipif(os.name != "nt", reason="Windows-only reader"))])
@pytest.mark.parametrize("mid", [0, 5, 12, 20])  # 20: the tail is exactly full and not yet sealed (lazy seal)
@pytest.mark.parametrize("rebuild", ["store", "boundary"])
def test_restored_session_continues_bit_identically(tiny, tmp_path, mid: int, unbuffered: bool, rebuild: str) -> None:  # noqa: ANN001
    from lazykv import snapshot
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    gen = torch.Generator(device="cuda").manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (PROMPT + 60,), device="cuda", generator=gen)
    full = FullGPUCache(cfg.num_hidden_layers, CAP)
    pre = prefill(model, full, ids[:PROMPT], chunk_size=32)
    tier = TieredCache(full, K, BS, capacity_tokens=CAP, model=model, dense_layers=1, fetch="gather")
    before = _feed(model, tier, ids[PROMPT : PROMPT + mid].tolist())
    logits = before[-1] if before else pre.last_logits
    if mid == 20:
        assert next(iter(tier.tiers.values())).fill == BS  # the case this parameter exists for
    path = tmp_path / "s.lkv"
    st = snapshot.save(tier, logits, path)
    assert path.stat().st_size == st.bytes_written and st.bytes_written % snapshot.ALIGN == 0
    ref = _feed(model, tier, ids[PROMPT + mid :].tolist())
    tier.close()

    got_cache, got_logits, rs = snapshot.restore(path, K, CAP, cfg.num_hidden_layers, model=model, fetch="gather", unbuffered=unbuffered, rebuild=rebuild)
    assert torch.equal(got_logits, logits.float())
    assert got_cache.get_seq_length(0) == PROMPT + mid and got_cache.get_seq_length(2) == PROMPT + mid
    got = _feed(model, got_cache, ids[PROMPT + mid :].tolist())
    got_cache.close()
    assert all(torch.equal(a, b) for a, b in zip(got, ref, strict=True))
    assert rs.bytes_read > 0


@cuda
@pytest.mark.parametrize("n_slots", [None, 7])  # 7: spare slots beyond the budget stay empty
@pytest.mark.parametrize("length", [64, 70, 260])  # 64: no tail; 70: a tail; 260: many chunks
def test_from_store_builds_the_constructors_layer(length: int, n_slots: int | None, monkeypatch) -> None:  # noqa: ANN001
    """A layer rebuilt from its host store equals the one the constructor builds from the same KV,
    in every tensor and table, including across chunk edges."""
    import numpy as np

    from lazykv import tiered
    from lazykv.tiered import TieredLayer

    monkeypatch.setattr(tiered, "STORE_CHUNK", 3)  # several chunks, and seeded slots straddling an edge
    gen = torch.Generator(device="cuda").manual_seed(length)
    h, d = 2, 16
    keys = torch.randn((1, h, length, d), device="cuda", generator=gen).bfloat16()
    values = torch.randn((1, h, length, d), device="cuda", generator=gen).bfloat16()
    ref = TieredLayer(keys, values, K, BS, CAP, n_slots=n_slots)
    n, fill = divmod(length, BS)
    host = tiered.pinned_empty(tuple(ref.host.shape), torch.bfloat16)
    host[:n].copy_(ref.host[:n])
    tail = ref.pool[ref.tail_slot, :, :, :fill].cpu()
    got = TieredLayer.from_store(host, n, tail, K, BS, CAP, torch.device("cuda"), n_slots)
    torch.cuda.synchronize()
    assert (got.n_sealed, got.fill, got.length) == (ref.n_sealed, ref.fill, ref.length)
    assert torch.equal(got.kmin[:, :, :n], ref.kmin[:, :, :n]) and torch.equal(got.kmax[:, :, :n], ref.kmax[:, :, :n])
    seed = min(ref.n_slots, n - 1)
    assert torch.equal(got.pool[: seed + 1], ref.pool[: seed + 1])
    assert torch.equal(got.pool[got.tail_slot], ref.pool[ref.tail_slot])
    for name in ("slot_block", "block_slot", "last_used", "evicted_at"):
        assert np.array_equal(getattr(got, name), getattr(ref, name)), name


@cuda
def test_snapshot_refuses_the_int8_tier_and_foreign_files(tiny, tmp_path) -> None:  # noqa: ANN001
    from lazykv import snapshot
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    full = FullGPUCache(cfg.num_hidden_layers, CAP)
    pre = prefill(model, full, torch.arange(PROMPT, device="cuda") % cfg.vocab_size, chunk_size=32)
    tier = TieredCache(full, K, BS, capacity_tokens=CAP, dense_layers=1, fetch="gather", quant="int8")
    with pytest.raises(ValueError, match="int8"):
        snapshot.save(tier, pre.last_logits, tmp_path / "x.lkv")
    bad = tmp_path / "bad.lkv"
    bad.write_bytes((2).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="not a LazyKV snapshot"):
        snapshot.read_header(bad)
