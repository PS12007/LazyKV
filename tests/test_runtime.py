"""Runtime correctness against transformers' own eager attention, on a tiny random Llama.

These guard the three things every later policy builds on: the preallocated cache stores
and returns exactly what was written, chunked prefill with lower-right causality matches a
single full forward, and a decode step matches the full forward at that position.
"""

from __future__ import annotations

import copy

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (cuDNN attention)")


@pytest.fixture(scope="module")
def tiny_models():  # noqa: ANN201
    from transformers import LlamaConfig, LlamaForCausalLM

    from lazykv.attention import IMPLEMENTATION_NAME, install

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        max_position_embeddings=4096,
    )
    ref = LlamaForCausalLM(cfg).to(dtype=torch.float32)
    ref.set_attn_implementation("eager")
    ref = ref.cuda().eval()
    # Small bucket so the padded, masked decode path is exercised inside these short caches.
    install("cudnn_bucketed", cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, bucket=64)
    # Separate config object: set_attn_implementation mutates the config, and a shared one
    # would silently switch the eager reference onto the kernel under test.
    ours = LlamaForCausalLM(copy.deepcopy(cfg))
    ours.load_state_dict(ref.state_dict())
    assert ref.config._attn_implementation == "eager"
    ours.set_attn_implementation(IMPLEMENTATION_NAME)
    ours = ours.to(dtype=torch.bfloat16).cuda().eval()
    assert ref.config._attn_implementation == "eager"
    return cfg, ref, ours


def test_preallocated_layer_roundtrip_and_overflow() -> None:
    from lazykv.cache import PreallocatedLayer

    layer = PreallocatedLayer(max_len=10)
    k = torch.arange(2 * 4 * 3, dtype=torch.float32).view(1, 2, 4, 3)
    v = -k
    keys, values = layer.update(k, v)
    assert layer.get_seq_length() == 4
    assert torch.equal(keys, k) and torch.equal(values, v)
    k2 = torch.ones(1, 2, 6, 3)
    keys, _ = layer.update(k2, k2)
    assert keys.shape[-2] == 10 and torch.equal(keys[:, :, :4], k)
    assert layer.resident_bytes() == 10 * 2 * (2 * 3 * 4)
    with pytest.raises(RuntimeError, match="overflow"):
        layer.update(torch.ones(1, 2, 1, 3), torch.ones(1, 2, 1, 3))
    layer.reset()
    assert layer.get_seq_length() == 0


@cuda
def test_chunked_prefill_matches_full_eager_forward(tiny_models) -> None:  # noqa: ANN001
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    cfg, ref, ours = tiny_models
    ids = torch.randint(0, cfg.vocab_size, (1, 300), device="cuda")
    with torch.inference_mode():
        ref_logp = torch.log_softmax(ref(input_ids=ids).logits[0, -1].float(), -1)
    for chunk in (300, 64, 7):
        cache = FullGPUCache(cfg.num_hidden_layers, max_len=512)
        got = torch.log_softmax(prefill(ours, cache, ids, chunk_size=chunk).last_logits, -1)
        assert (got - ref_logp).abs().max().item() < 5e-2, f"chunk={chunk}"
        assert cache.stats().seq_len == 300


@cuda
def test_decode_step_matches_full_forward(tiny_models) -> None:  # noqa: ANN001
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    cfg, ref, ours = tiny_models
    ids = torch.randint(0, cfg.vocab_size, (1, 200), device="cuda")
    with torch.inference_mode():
        ref_all = torch.log_softmax(ref(input_ids=ids).logits[0].float(), -1)
        cache = FullGPUCache(cfg.num_hidden_layers, max_len=256)
        prefill(ours, cache, ids[:, :150], chunk_size=32)
        for pos in range(150, 200):
            out = ours(input_ids=ids[:, pos : pos + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
            got = torch.log_softmax(out.logits[0, -1].float(), -1)
            assert (got - ref_all[pos]).abs().max().item() < 5e-2, f"pos={pos}"


@cuda
def test_teacher_forced_self_comparison_is_exact(tiny_models) -> None:  # noqa: ANN001
    from lazykv.cache import FullGPUCache
    from lazykv.generate import teacher_forced_logprobs
    from lazykv.quality import compare

    cfg, _, ours = tiny_models
    ids = torch.randint(0, cfg.vocab_size, (1, 120), device="cuda")
    a = teacher_forced_logprobs(ours, FullGPUCache(2, 256), ids[:, :100], ids[:, 100:], chunk_size=32)
    b = teacher_forced_logprobs(ours, FullGPUCache(2, 256), ids[:, :100], ids[:, 100:], chunk_size=32)
    d = compare(a, b)
    assert d.positions == 20 and d.top1_agreement == 1.0 and d.mean_kl == 0.0 and d.exact_match


def test_compare_detects_divergence() -> None:
    from lazykv.quality import compare

    ref = torch.log_softmax(torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), -1)
    pol = torch.log_softmax(torch.tensor([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]), -1)
    d = compare(ref, pol)
    assert d.top1_agreement == 0.5 and d.mean_kl > 0 and not d.exact_match


@cuda
def test_attention_rejects_external_mask(tiny_models) -> None:  # noqa: ANN001
    from lazykv.attention import lazykv_attention_forward

    q = torch.randn(1, 8, 1, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 4, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="owns causality"):
        lazykv_attention_forward(torch.nn.Module(), q, k, k, torch.ones(1, 1, 1, 4, device="cuda", dtype=torch.bool))


@cuda
@pytest.mark.parametrize("strategy", ["cudnn_bucketed", "cudnn", "efficient", "math"])
def test_every_strategy_matches_reference_with_garbage_padding(strategy: str) -> None:
    from lazykv.attention import _reference, attend

    torch.manual_seed(1)
    store_k = torch.randn(1, 2, 256, 16, device="cuda", dtype=torch.bfloat16) * 50  # padding must be masked, not ignored by luck
    store_v = torch.randn_like(store_k) * 50
    live = 100
    k, v = store_k[:, :, :live], store_v[:, :, :live]
    for q_len in (1, 17):
        q = torch.randn(1, 8, q_len, 16, device="cuda", dtype=torch.bfloat16)
        ref = _reference(q, k, v, 16**-0.5)
        got = attend(q, k, v, 16**-0.5, strategy, bucket=64).float()
        assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-2, (strategy, q_len)


def test_extend_view_respects_storage() -> None:
    from lazykv.attention import _extend_view

    base = torch.arange(1 * 2 * 10 * 3, dtype=torch.float32).view(1, 2, 10, 3)
    view = base[:, :, :4]
    ext = _extend_view(view, 8)
    assert ext is not None and torch.equal(ext, base[:, :, :8])
    assert _extend_view(view, 11) is None
    assert _extend_view(torch.zeros(1, 2, 4, 3), 8) is None


@cuda
def test_preallocated_storage_is_finite_even_on_recycled_memory() -> None:
    """Padding rows are read (and masked) by the bucketed path, so they must be finite."""
    from lazykv.cache import PreallocatedLayer

    poison = torch.full((1, 2, 256, 16), float("nan"), device="cuda", dtype=torch.bfloat16)
    del poison  # hand a NaN-filled block back to the caching allocator
    layer = PreallocatedLayer(max_len=256)
    x = torch.randn(1, 2, 3, 16, device="cuda", dtype=torch.bfloat16)
    layer.update(x, x)
    assert torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()


@cuda
def test_streaming_compare_matches_full_compare(tiny_models) -> None:  # noqa: ANN001
    from lazykv.cache import FullGPUCache
    from lazykv.generate import teacher_forced_logprobs, teacher_forced_steps
    from lazykv.quality import compare, compare_stream

    cfg, _, ours = tiny_models
    ids = torch.randint(0, cfg.vocab_size, (1, 90), device="cuda")
    ref = teacher_forced_logprobs(ours, FullGPUCache(2, 256), ids[:, :64], ids[:, 64:], chunk_size=16)
    pol_full = teacher_forced_logprobs(ours, FullGPUCache(2, 256), ids[:, :64], ids[:, 64:], chunk_size=7)
    streamed = compare_stream(ref, teacher_forced_steps(ours, FullGPUCache(2, 256), ids[:, :64], ids[:, 64:], chunk_size=7))
    full = compare(ref, pol_full)
    assert streamed.positions == full.positions == 26
    assert streamed.top1_agreement == full.top1_agreement and streamed.exact_match == full.exact_match
    assert abs(streamed.mean_kl - full.mean_kl) < 1e-9
