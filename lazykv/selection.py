"""Query-aware block selection over a fully resident cache: policy ladder rung 5 (Quest-style).

Quest (arXiv 2406.10774) keeps per-page element-wise key minima m and maxima M and bounds a
page's attention score by U = sum_i max(q_i * m_i, q_i * M_i); the top-K pages are attended and
the rest skipped, with nothing freed. It keeps the first two layers dense because their attention
is not sparse. Those are the paper's; the choices below are LazyKV's and are stated as such:

- Blocks are 64 tokens (Quest's experiments use 16-token pages), the granularity every other
  rung uses, so rungs differ in policy rather than in block size.
- Selection is per KV head, scoring each block by the maximum bound over that head's query group.
  The paper does not state the head granularity of its selection.
- The first block (the attention sink, measured as indispensable in Phase 2) and the block being
  filled are always attended and count against the budget.
- The gathered KV has a constant shape per step: sink + K selected blocks + the whole region of
  the current block, with the unfilled part of that region masked. A constant shape keeps cuDNN
  plans cached; a shape that grew every token would rebuild a plan per token (Phase 1).

Cost model, stated up front: without a sparse attention kernel (out of scope, brief §B9), attending
to K blocks means gathering them, a copy proportional to the attended budget in every layer at
every step. Rung 5 measures what query-aware selection retains; its latency here is an upper bound
on what a sparse kernel would pay.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache

from lazykv.cache import CacheStats, FullGPUCache

QUEST_DENSE_LAYERS = 2  # Quest: "we only apply Quest and all baselines on later layers"


def top_blocks(query: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor, k: int) -> torch.Tensor:
    """Indices [1, kv, k] of the k candidate blocks with the highest Quest bound, per KV head.

    `query` is [1, n_q, 1, d]; `kmin`/`kmax` are [1, kv, n, d] over the candidate blocks. Each
    block is scored by the maximum bound over the head's query group, and `topk` returns the
    blocks in descending bound order: callers gather in that order, so two runtimes that share
    this function attend to identical key sequences.
    """
    n_q, n_kv, d = query.shape[1], kmin.shape[1], query.shape[-1]
    q = query.view(1, n_kv, n_q // n_kv, d)
    qp = q.clamp_min(0)
    # sum_i max(q_i m_i, q_i M_i) = q+ . M + q- . m, per query head and block.
    bound = (qp @ kmax.transpose(-1, -2)) + ((q - qp) @ kmin.transpose(-1, -2))  # [1, kv, g, n]
    return bound.amax(dim=2).topk(k, dim=-1).indices


def current_block_mask(total: int, block_size: int, fill: int, dtype: torch.dtype, device: torch.device, reuse: torch.Tensor | None = None) -> torch.Tensor:
    """Additive mask [1, 1, 1, total] hiding the unfilled part of the current block, which is gathered last."""
    mask = torch.zeros((1, 1, 1, total), dtype=dtype, device=device) if reuse is None else reuse.zero_()
    mask[..., total - block_size + fill :] = float("-inf")
    return mask


def blocks_for_budget(fraction: float, total_tokens: int, block_size: int) -> int:
    """Selected blocks K such that sink + K blocks + the current block stay within the budget."""
    return max(1, math.floor(fraction * total_tokens / block_size) - 2)


@dataclass
class SelectorCounters:
    host_select_s: float = 0.0
    selections: int = 0


@dataclass
class _LayerState:
    kmin: torch.Tensor | None = None  # [1, kv, cap_blocks, d]
    kmax: torch.Tensor | None = None
    n_meta: int = 0  # full blocks with metadata


class QuestView(Cache):
    """A decode-time view over a prefilled FullGPUCache that attends only to selected blocks.

    Shares the full cache's layers: decode appends to them, and the caller truncates the full
    cache afterwards. Nothing is copied at the boundary except block metadata.
    """

    def __init__(self, full: FullGPUCache, k_blocks: int, block_size: int, dense_layers: int = QUEST_DENSE_LAYERS) -> None:
        super().__init__(layers=full.layers)
        self.full, self.k_blocks, self.block_size, self.dense_layers = full, k_blocks, block_size, dense_layers
        self.counters = SelectorCounters()
        self.policy_name = "quest"
        self._state = [_LayerState() for _ in full.layers]
        # Every selecting layer in a step shares one mask (it depends only on the current block's fill).
        self._mask: torch.Tensor | None = None
        self._mask_fill = -1
        self._offsets: torch.Tensor | None = None
        for i, layer in enumerate(full.layers):
            self._refresh_metadata(i, layer.keys, layer.get_seq_length())

    # -- transformers plumbing: delegate storage to the full cache --------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return self.full.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.full.layers[layer_idx].get_seq_length()

    def model_kwargs(self) -> dict[str, object]:
        return {"lazykv_selector": self}

    def stats(self) -> CacheStats:
        # Everything stays resident; the attended budget is what rung 5 varies.
        return self.full.stats()

    def attended_tokens(self) -> int:
        return (self.k_blocks + 2) * self.block_size

    # -- selection ----------------------------------------------------------------------------

    def _refresh_metadata(self, layer_idx: int, keys: torch.Tensor, length: int) -> None:
        st = self._state[layer_idx]
        bs = self.block_size
        n_full = length // bs
        if st.kmin is None:
            cap = -(-keys.shape[2] // bs)
            h, d = keys.shape[1], keys.shape[3]
            st.kmin = torch.empty((1, h, cap, d), dtype=keys.dtype, device=keys.device)
            st.kmax = torch.empty_like(st.kmin)
        if n_full > st.n_meta:
            assert st.kmax is not None
            blocks = keys[:, :, st.n_meta * bs : n_full * bs].unflatten(2, (n_full - st.n_meta, bs))  # [1, h, n, bs, d]
            st.kmin[:, :, st.n_meta : n_full] = blocks.amin(dim=3)
            st.kmax[:, :, st.n_meta : n_full] = blocks.amax(dim=3)
            st.n_meta = n_full

    def select(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Gathered (key, value, additive mask) for this decode step, or None to attend densely."""
        if query.shape[-2] != 1 or layer_idx < self.dense_layers:
            return None
        bs = self.block_size
        length = key.shape[2]
        n_full, fill = divmod(length, bs)
        if fill == 0:  # the newest block just completed; treat it as the current block
            n_full, fill = n_full - 1, bs
        # Sink (block 0) and the current block are always in; K more are chosen among blocks 1..n_full-1.
        if n_full - 1 <= self.k_blocks:
            return None  # the budget covers every block: dense is exact and cheaper
        t0 = time.perf_counter()
        st = self._state[layer_idx]
        self._refresh_metadata(layer_idx, self.full.layers[layer_idx].keys, n_full * bs)
        assert st.kmin is not None and st.kmax is not None
        n_kv, d = key.shape[1], query.shape[-1]
        top = top_blocks(query, st.kmin[:, :, 1:n_full], st.kmax[:, :, 1:n_full], self.k_blocks) + 1  # [1, kv, K], block ids
        if self._offsets is None:
            self._offsets = torch.arange(bs, device=key.device)
        sink = torch.zeros((1, n_kv, 1), dtype=top.dtype, device=top.device)
        current = torch.full((1, n_kv, 1), n_full, dtype=top.dtype, device=top.device)
        blocks = torch.cat([sink, top, current], dim=-1)  # [1, kv, K+2]
        idx = (blocks.unsqueeze(-1) * bs + self._offsets).flatten(2)  # [1, kv, (K+2)*bs]
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        # The current block's region runs past the live length; storage there is zeroed or stale
        # but finite, and masked below.
        storage_k = self.full.layers[layer_idx].keys
        storage_v = self.full.layers[layer_idx].values
        k_sel = torch.gather(storage_k, 2, idx)
        v_sel = torch.gather(storage_v, 2, idx)
        if self._mask_fill != fill:
            self._mask = current_block_mask((self.k_blocks + 2) * bs, bs, fill, query.dtype, query.device, self._mask)
            self._mask_fill = fill
        self.counters.host_select_s += time.perf_counter() - t0
        self.counters.selections += 1
        return k_sel, v_sel, self._mask
