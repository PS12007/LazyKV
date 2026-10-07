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
def test_restored_session_continues_bit_identically(tiny, tmp_path, mid: int, unbuffered: bool) -> None:  # noqa: ANN001
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

    got_cache, got_logits, rs = snapshot.restore(path, K, CAP, cfg.num_hidden_layers, model=model, fetch="gather", unbuffered=unbuffered)
    assert torch.equal(got_logits, logits.float())
    assert got_cache.get_seq_length(0) == PROMPT + mid and got_cache.get_seq_length(2) == PROMPT + mid
    got = _feed(model, got_cache, ids[PROMPT + mid :].tolist())
    got_cache.close()
    assert all(torch.equal(a, b) for a, b in zip(got, ref, strict=True))
    assert rs.bytes_read > 0


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
