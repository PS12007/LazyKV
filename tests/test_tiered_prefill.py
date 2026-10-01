"""Layer-major tiered prefill: the same tier, bit for bit, without the full prompt KV in VRAM."""

from __future__ import annotations

import copy

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention, pinned memory)")
BS = 8
PROMPT = 100  # not a multiple of the chunk or the block: a partial last chunk and a partial tail block
K = 3
CAP = 512


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


def _ids(cfg):  # noqa: ANN001, ANN202
    gen = torch.Generator(device="cuda").manual_seed(7)
    return torch.randint(0, cfg.vocab_size, (1, PROMPT + 40), device="cuda", generator=gen)


def _reference(model, cfg, ids, chunk: int, **kw):  # noqa: ANN001, ANN202
    """The existing path: chunk-major prefill into a full GPU cache, then the tier built from it."""
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill
    from lazykv.tiered import TieredCache

    full = FullGPUCache(cfg.num_hidden_layers, CAP)
    pre = prefill(model, full, ids[:, :PROMPT], chunk_size=chunk)
    tier = TieredCache(full, K, BS, capacity_tokens=CAP, model=model, dense_layers=1, **kw)
    return pre.last_logits, tier


@cuda
@pytest.mark.parametrize("chunk", [32, 100, 128])
@pytest.mark.parametrize(("prefetch", "quant"), [(False, None), (True, None), (False, "int8")])
def test_layer_major_prefill_matches_the_full_cache_path(tiny, chunk: int, prefetch: bool, quant: str | None) -> None:  # noqa: ANN001
    from lazykv.generate import teacher_forced_decode
    from lazykv.tiered_prefill import tiered_prefill

    cfg, model = tiny
    ids = _ids(cfg)
    kw = {"prefetch": prefetch, "fetch": "gather", "quant": quant}
    ref_logits, ref = _reference(model, cfg, ids, chunk, **kw)
    got = tiered_prefill(model, ids[:, :PROMPT], chunk, K, BS, CAP, dense_layers=1, **kw)
    assert got.chunks == -(-PROMPT // chunk)
    assert torch.equal(got.last_logits, ref_logits)
    for i, t in got.cache.tiers.items():
        r = ref.tiers[i]
        assert (t.n_sealed, t.fill, t.length) == (r.n_sealed, r.fill, r.length)
        assert torch.equal(t.host[: t.n_sealed], r.host[: r.n_sealed])
        assert torch.equal(t.kmin[:, :, : t.n_sealed], r.kmin[:, :, : r.n_sealed])
        assert torch.equal(t.kmax[:, :, : t.n_sealed], r.kmax[:, :, : r.n_sealed])
        assert torch.equal(t.pool, r.pool)
        assert (t.slot_block == r.slot_block).all()
    assert torch.equal(got.full.layers[0].keys[:, :, :PROMPT], ref.full.layers[0].keys[:, :, :PROMPT])
    # Decode after it: same tokens' log-probs, bit for bit, through seals and fetches.
    cont = ids[:, PROMPT : PROMPT + 40]
    a = torch.stack(list(teacher_forced_decode(model, ref, ref_logits, cont)))
    b = torch.stack(list(teacher_forced_decode(model, got.cache, got.last_logits, cont)))
    ref.close()
    got.cache.close()
    assert torch.equal(a, b)
    assert got.cache.counters.boundary_d2h_bytes == ref.counters.boundary_d2h_bytes


@cuda
def test_selecting_layers_never_allocate_in_the_full_cache(tiny) -> None:  # noqa: ANN001
    """The point of the module: only the dense layers' KV lives in the full cache."""
    from lazykv.tiered_prefill import tiered_prefill

    cfg, model = tiny
    got = tiered_prefill(model, _ids(cfg)[:, :PROMPT], 32, K, BS, CAP, dense_layers=1)
    got.cache.close()
    assert got.full.layers[0].is_initialized and got.full.layers[0].get_seq_length() == PROMPT
    assert not any(layer.is_initialized for layer in got.full.layers[1:])
    assert len(got.layer_s) == cfg.num_hidden_layers


@cuda
def test_prompt_longer_than_capacity_is_refused(tiny) -> None:  # noqa: ANN001
    from lazykv.tiered_prefill import tiered_prefill

    cfg, model = tiny
    with pytest.raises(ValueError, match="exceeds capacity"):
        tiered_prefill(model, _ids(cfg)[:, :PROMPT], 32, K, BS, capacity_tokens=64, dense_layers=1)
