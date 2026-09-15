"""Rung 5 (Quest-style selection): the bound, the gather, the mask, and the dense fallback."""

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
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=4096)
    install("cudnn_bucketed", cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, bucket=64)
    model = LlamaForCausalLM(copy.deepcopy(cfg))
    model.set_attn_implementation(IMPLEMENTATION_NAME)
    return cfg, model.to(dtype=torch.bfloat16).cuda().eval()


def _prefilled(model, cfg, n: int):  # noqa: ANN001, ANN202
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    ids = torch.randint(0, cfg.vocab_size, (1, n + 40), device="cuda")
    full = FullGPUCache(cfg.num_hidden_layers, 512)
    pre = prefill(model, full, ids[:, :n], chunk_size=32)
    return ids, full, pre


def test_blocks_for_budget_leaves_room_for_sink_and_current_block() -> None:
    from lazykv.selection import blocks_for_budget

    for frac in (0.75, 0.25, 0.0625):
        k = blocks_for_budget(frac, 32768, 64)
        assert (k + 2) * 64 <= frac * 32768


@cuda
def test_bound_dominates_every_dot_product_in_its_block() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4, 16, device="cuda")
    keys = torch.randn(1, 2, 10, BS, 16, device="cuda")  # [1, kv, blocks, bs, d]
    kmin, kmax = keys.amin(3), keys.amax(3)
    qp = q.clamp_min(0)
    bound = (qp @ kmax.transpose(-1, -2)) + ((q - qp) @ kmin.transpose(-1, -2))  # [1, kv, g, blocks]
    exact = torch.einsum("bhgd,bhntd->bhgnt", q, keys).amax(-1)
    assert (bound >= exact - 1e-5).all()


@cuda
def test_large_budget_falls_back_to_dense_exactly(tiny) -> None:  # noqa: ANN001
    from lazykv.generate import teacher_forced_decode
    from lazykv.selection import QuestView

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg, 100)
    cont = ids[:, 100:130]
    ref = torch.stack(list(teacher_forced_decode(model, full, pre.last_logits, cont)))
    full.truncate(100)
    view = QuestView(full, k_blocks=1000, block_size=BS, dense_layers=0)
    got = torch.stack(list(teacher_forced_decode(model, view, pre.last_logits, cont)))
    assert torch.equal(got, ref)
    assert view.counters.selections == 0


@cuda
def test_gather_takes_sink_selected_and_current_blocks_with_tail_masked(tiny) -> None:  # noqa: ANN001
    from lazykv.selection import QuestView

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg, 101)  # 12 full blocks + 5 tokens
    view = QuestView(full, k_blocks=3, block_size=BS, dense_layers=0)
    layer = full.layers[1]
    key, value = layer.keys[:, :, :101], layer.values[:, :, :101]
    query = torch.randn(1, cfg.num_attention_heads, 1, cfg.head_dim, device="cuda", dtype=torch.bfloat16)
    k_sel, v_sel, mask = view.select(1, query, key, value)
    assert k_sel.shape == (1, cfg.num_key_value_heads, 5 * BS, cfg.head_dim)
    # Sink first, current block (12) last, and the three selected blocks are exactly the top-3 by bound.
    assert torch.equal(k_sel[:, :, :BS], layer.keys[:, :, :BS])
    assert torch.equal(k_sel[:, :, 4 * BS :], layer.keys[:, :, 12 * BS : 13 * BS])
    q = query.view(1, cfg.num_key_value_heads, -1, cfg.head_dim)
    blocks = layer.keys[:, :, : 12 * BS].unflatten(2, (12, BS))
    kmin, kmax = blocks.amin(3)[:, :, 1:], blocks.amax(3)[:, :, 1:]
    qp = q.clamp_min(0)
    want = ((qp @ kmax.transpose(-1, -2)) + ((q - qp) @ kmin.transpose(-1, -2))).amax(2).topk(3, -1).indices + 1
    for h in range(cfg.num_key_value_heads):
        for j, b in enumerate(want[0, h].tolist()):
            assert torch.equal(k_sel[0, h, (1 + j) * BS : (2 + j) * BS], layer.keys[0, h, b * BS : (b + 1) * BS])
            assert torch.equal(v_sel[0, h, (1 + j) * BS : (2 + j) * BS], layer.values[0, h, b * BS : (b + 1) * BS])
    # 5 live tokens in the current block: positions 4*BS+5 onward are masked.
    assert (mask[0, 0, 0, : 4 * BS + 5] == 0).all() and torch.isinf(mask[0, 0, 0, 4 * BS + 5 :]).all()


@cuda
@pytest.mark.parametrize("strategy", ["cudnn_bucketed", "efficient"])  # the fast kernel and the quality kernel
def test_selected_decode_matches_masked_reference_attention(tiny, strategy: str) -> None:  # noqa: ANN001
    """The attention output over the gathered set equals explicit attention restricted to it."""
    from lazykv.attention import attend, _reference
    from lazykv.selection import QuestView

    cfg, model = tiny
    _, full, _ = _prefilled(model, cfg, 101)
    view = QuestView(full, k_blocks=3, block_size=BS, dense_layers=0)
    layer = full.layers[2]
    query = torch.randn(1, cfg.num_attention_heads, 1, cfg.head_dim, device="cuda", dtype=torch.bfloat16)
    k_sel, v_sel, mask = view.select(2, query, layer.keys[:, :, :101], layer.values[:, :, :101])
    got = attend(query, k_sel, v_sel, cfg.head_dim**-0.5, strategy, mask=mask).float()
    live = 4 * BS + 5
    ref = _reference(query, k_sel[:, :, :live], v_sel[:, :, :live], cfg.head_dim**-0.5)
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-2
