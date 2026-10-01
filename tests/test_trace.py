"""The trace recorder (lazykv/trace.py) against the runtime it stands in for, on a tiny random Llama."""

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


def _prefilled(model, cfg):  # noqa: ANN001, ANN202
    from lazykv.cache import FullGPUCache
    from lazykv.generate import prefill

    gen = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, PROMPT + 40), device="cuda", generator=gen)
    full = FullGPUCache(cfg.num_hidden_layers, 512)
    pre = prefill(model, full, ids[:, :PROMPT], chunk_size=32)
    return ids, full, pre


@cuda
def test_recording_leaves_the_decode_unchanged_and_mass_is_a_distribution(tiny) -> None:  # noqa: ANN001
    from lazykv.attention import set_strategy_for_test
    from lazykv.generate import teacher_forced_decode
    from lazykv.trace import TraceRecorder

    cfg, model = tiny
    restore = set_strategy_for_test("efficient")
    try:
        ids, full, pre = _prefilled(model, cfg)
        cont = ids[:, PROMPT : PROMPT + 30]  # crosses several seals
        ref = torch.stack(list(teacher_forced_decode(model, full, pre.last_logits, cont)))
        full.truncate(PROMPT)
        rec = TraceRecorder(full, BS, dense_layers=1)
        got = torch.stack(list(teacher_forced_decode(model, rec, pre.last_logits, cont)))
        assert torch.equal(got, ref)
        tr = rec.trace({"note": "tiny"})
        assert tr.steps == 29 and tr.mass.shape[1:3] == (3, 2)
        np.testing.assert_allclose(tr.mass.sum(axis=-1), 1.0, rtol=1e-5)
        # Step t attends over PROMPT + 1 + t tokens; a block that has just filled is still current.
        assert tr.n_full.tolist() == [(PROMPT + t) // BS for t in range(29)]
        assert np.isfinite(tr.bound[:, :, :, 1 : tr.n_full.min()]).all()
        assert np.isneginf(tr.bound[:, :, :, 0]).all()
    finally:
        restore()


@cuda
def test_simulated_quest_selection_is_the_runtime_selection(tiny) -> None:  # noqa: ANN001
    """select_quest over the recorded bound picks what top_blocks picks live, up to ties.

    The runtime ranks a bf16 bound, and bf16 ties are common: two blocks with equal bounds may come
    back in either order, and at the K-th place either may be in. So the check is on the bound
    values chosen, which a tie cannot change, not on the ids.
    """
    from lazykv.generate import teacher_forced_decode
    from lazykv.residency_sim import select_quest
    from lazykv.selection import top_blocks
    from lazykv.trace import TraceRecorder

    class Spy(TraceRecorder):
        def __init__(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
            super().__init__(*a, **kw)
            self.live: list[np.ndarray] = []

        def select(self, layer_idx, query, key, value):  # noqa: ANN001, ANN201
            super().select(layer_idx, query, key, value)
            if query.shape[-2] == 1 and layer_idx >= self.dense_layers:
                n = self._steps[-1][-1][0]
                st = self._state[layer_idx]
                self.live.append((top_blocks(query, st.kmin[:, :, 1:n], st.kmax[:, :, 1:n], K) + 1)[0].cpu().numpy())
            return None

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    spy = Spy(full, BS, dense_layers=1)
    list(teacher_forced_decode(model, spy, pre.last_logits, ids[:, PROMPT : PROMPT + 20]))
    tr = spy.trace()
    sim = np.stack(select_quest(tr, K))  # [steps, layers, kv, K]
    live = np.stack(spy.live).reshape(sim.shape)
    bound = tr.bound[: sim.shape[0]]
    np.testing.assert_array_equal(np.take_along_axis(bound, sim, axis=-1), np.take_along_axis(bound, live, axis=-1))
    assert (sim == live).mean() > 0.8  # and ties are the exception, not the rule


@cuda
def test_simulated_lru_fetches_exactly_what_the_tier_fetched(tiny) -> None:  # noqa: ANN001
    """Replaying the tier's own selection through replay_lru reproduces its per-step fetch count."""
    from lazykv.generate import teacher_forced_decode
    from lazykv.residency_sim import replay_belady, replay_lru
    from lazykv.tiered import TieredCache

    cfg, model = tiny
    ids, full, pre = _prefilled(model, cfg)
    n_slots = K + 3
    tier = TieredCache(full, K, BS, capacity_tokens=512, dense_layers=1, n_slots=n_slots, fetch="gather", record_selection=True)
    list(teacher_forced_decode(model, tier, pre.last_logits, ids[:, PROMPT : PROMPT + 30]))
    tier.close()
    rec = tier.recorded_selection()
    steps = sorted({s for _, s in rec})
    n_blocks = PROMPT // BS
    seed = list(range(n_blocks - min(n_slots, n_blocks - 1), n_blocks))
    lru = np.zeros(len(steps), dtype=np.int64)
    belady = np.zeros(len(steps), dtype=np.int64)
    for layer in tier.tiers:
        for h in range(cfg.num_key_value_heads):
            seq = [rec[(layer, s)][h] for s in steps]
            lru += replay_lru(seq, n_slots, seed)
            belady += replay_belady(seq, n_slots, seed)
    assert lru.tolist() == tier.counters.fetched_pairs_per_step
    assert belady.sum() <= lru.sum()
