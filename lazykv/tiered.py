"""A pinned CPU tier under query-aware selection: policy ladder rungs 6 and 7.

Rung 5 (lazykv/selection.py) attends to the top-K blocks per KV head but keeps every block in
VRAM, so its budget is attended, not freed. This module keeps only the attended set in VRAM and
the rest in host RAM, and uses the *same* selection (`top_blocks`, same metadata, same gather
order), so rungs 6 and 7 attend to exactly the key sequences rung 5 does. Any quality difference
from rung 5 is therefore a bug, and the tests check for bit-identity. What the tier changes is
where the bytes live and what moving them costs.

- Rung 6 (`prefetch=False`): each selecting layer bounds, ranks, and fetches the blocks its
  selection needs but VRAM lacks, on the compute stream, right before attention.
- Rung 7 (`prefetch=True`): InfiniGen's layer-ahead idea (arXiv 2406.19707). Before layer L runs,
  layer L+1's query is speculated from layer L's input hidden state (the residual stream changes
  little between adjacent layers), layer L+1's selection is estimated with it, and the missing
  blocks are copied on a separate CUDA stream while layer L computes. Layer L+1 then selects with
  its real query; whatever the speculation missed is fetched synchronously.

Layout decisions, each forced by a Phase 0 measurement (docs/IMPLEMENTATION_PLAN.md §3.4):

- Residency is per (layer, KV head, block), not per block. Selection is per KV head, so a
  per-block unit would have to hold the union of eight heads' choices, up to 8x the attended set.
- Host pool: one pinned tensor per layer, [blocks, heads, 2, block_size, head_dim], keys and values
  interleaved so that one (block, head) pair is one contiguous region. GPU slot pool: the same
  per-slot layout. A run of pairs that is consecutive in both pools is one transfer, which matters
  because a single pair (16 KiB at 64 tokens) sits far below the PCIe efficiency knee.
- KV is immutable once sealed, and there is one copy engine: each block is copied D2H exactly once,
  when it seals, and never written back on eviction.
- Slot 0 holds the sink (block 0), the last slot holds the block being filled, and neither is ever
  evicted. Victims are the least recently selected slots outside the current selection.

Not yet here: prefill still runs on a full GPU cache, so peak VRAM during prefill is the full KV.
The tier governs decode-time residency, and experiments report the two separately.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import rotate_half

from lazykv.cache import CacheStats, FullGPUCache
from lazykv.selection import QUEST_DENSE_LAYERS, current_block_mask, top_blocks

NEVER = -(2**40)  # "evicted at" sentinel for pairs that were never evicted


@dataclass
class TierCounters:
    """Host time and transfer activity of the tier (brief §B6: cache behaviour, PCIe, overhead)."""

    host_select_s: float = 0.0  # bound, rank, host sync, residency bookkeeping, gather launch
    host_fetch_s: float = 0.0  # launching on-demand H2D copies (rung 6 path, and rung 7 misses)
    host_prefetch_s: float = 0.0  # speculating a query, ranking, and launching copies on the copy stream
    host_seal_s: float = 0.0
    boundary_d2h_s: float = 0.0
    boundary_d2h_bytes: int = 0
    selections: int = 0
    selected_pairs: int = 0  # (head, block) pairs chosen, summed over selections
    hit_pairs: int = 0  # already resident when chosen (including pairs a prefetch brought in)
    fetched_pairs: int = 0  # missing when chosen, fetched on demand
    fetch_transfers: int = 0
    fetch_bytes: int = 0
    thrash_pairs: int = 0  # fetched on demand within `thrash_window` steps of being evicted
    prefetched_pairs: int = 0
    prefetch_transfers: int = 0
    prefetch_bytes: int = 0
    prefetch_used_pairs: int = 0  # prefetched and then chosen by the real selection
    seals: int = 0
    seal_bytes: int = 0
    # Per decode step, summed over layers: pairs fetched on demand. Separates the cold start after
    # the boundary from steady state.
    fetched_pairs_per_step: list[int] = field(default_factory=list)


@dataclass
class _Timed:
    """CUDA events for one prefetch, resolved after decode (instrumented runs only)."""

    layer: int
    copy_start: torch.cuda.Event
    copy_end: torch.cuda.Event
    window_start: torch.cuda.Event  # compute stream, right after the copies were launched
    window_end: torch.cuda.Event  # compute stream, when the target layer starts selecting


class TieredLayer:
    """One selecting layer: GPU slot pool, pinned host pool, block metadata and the residency table."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, k_blocks: int, block_size: int, capacity_tokens: int, host: torch.Tensor | None = None) -> None:
        _, h, length, d = keys.shape
        bs = block_size
        n_blocks, tail = divmod(length, bs)
        if n_blocks - 1 <= k_blocks:
            # Rung 5 attends densely when the budget covers every block. The tier has no dense copy
            # to fall back to, so it refuses budgets that would need one.
            raise ValueError(f"budget of {k_blocks} blocks covers all {n_blocks - 1} candidate blocks; the tier needs selection")
        self.h, self.d, self.bs, self.k = h, d, bs, k_blocks
        self.cap_blocks = -(-capacity_tokens // bs)
        dev, dt = keys.device, keys.dtype
        self.tail_slot = k_blocks + 1
        # Zeroed: the current block's unfilled region is gathered and masked, and masking cannot
        # neutralize NaN/inf left in recycled allocator memory (Phase 1).
        self.pool = torch.zeros((k_blocks + 2, h, 2, bs, d), dtype=dt, device=dev)
        self.kmin = torch.empty((1, h, self.cap_blocks, d), dtype=dt, device=dev)
        self.kmax = torch.empty_like(self.kmin)
        shape = (self.cap_blocks, h, 2, bs, d)
        if host is None:
            host = torch.empty(shape, dtype=dt, pin_memory=True)
        elif host.shape != shape or not host.is_pinned():
            raise ValueError(f"host pool must be pinned with shape {shape}, got {tuple(host.shape)}")
        self.host = host
        self.pair_bytes = 2 * bs * d * self.pool.element_size()

        # Boundary: every full prompt block goes to host once (seal-time D2H, all at once), and the
        # metadata is built on the GPU where the prompt KV still is.
        blocks = torch.stack([keys[0, :, : n_blocks * bs], values[0, :, : n_blocks * bs]], dim=1)  # [h, 2, B*bs, d]
        blocks = blocks.unflatten(2, (n_blocks, bs)).permute(2, 0, 1, 3, 4)  # [B, h, 2, bs, d]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        self.host[:n_blocks].copy_(blocks)
        self.boundary_d2h_s = time.perf_counter() - t0
        self.boundary_d2h_bytes = n_blocks * h * self.pair_bytes
        kb = keys[:, :, : n_blocks * bs].unflatten(2, (n_blocks, bs))
        self.kmin[:, :, :n_blocks] = kb.amin(dim=3)
        self.kmax[:, :, :n_blocks] = kb.amax(dim=3)
        self.pool[0] = blocks[0]
        # Initial residency: the K most recent candidate blocks, copied device-side. Recency is the
        # cheapest guess available at the boundary; the first decode step's misses are the cost of
        # it, and fetched_pairs_per_step shows that cold start separately.
        recent = np.arange(n_blocks - k_blocks, n_blocks)
        self.pool[1 : k_blocks + 1] = blocks[n_blocks - k_blocks : n_blocks]
        if tail:
            self.pool[self.tail_slot, :, 0, :tail] = keys[0, :, n_blocks * bs : length]
            self.pool[self.tail_slot, :, 1, :tail] = values[0, :, n_blocks * bs : length]
        self.n_sealed, self.fill, self.length = n_blocks, tail, length

        # Residency table on the host: slot numbers are pool indices 1..K.
        self.slot_block = np.tile(recent, (h, 1))  # [h, K]
        self.block_slot = np.full((h, self.cap_blocks), -1, dtype=np.int64)
        self.block_slot[:, recent] = np.arange(1, k_blocks + 1)
        self.last_used = np.full((h, k_blocks), -1, dtype=np.int64)
        self.evicted_at = np.full((h, self.cap_blocks), NEVER, dtype=np.int64)
        self.heads = np.arange(h)
        # Pairs a prefetch brought in and the real selection has not yet been checked against.
        self.pending_prefetch: tuple[np.ndarray, np.ndarray] | None = None
        # Pinned index buffer per layer. A non_blocking H2D from a reused buffer is safe only if
        # the previous copy has run before the buffer is rewritten; every step syncs (the top-K
        # indices must reach the host) long before this layer writes it again.
        self.idx_host = torch.empty((h * (k_blocks + 2),), dtype=torch.int64, pin_memory=True)
        self.idx_dev = torch.empty_like(self.idx_host, device=dev)
        self.flat_head = torch.arange(h, device=dev)

    # -- writes -----------------------------------------------------------------------------

    def append(self, key: torch.Tensor, value: torch.Tensor, counters: TierCounters) -> None:
        """Write one decode token; seal the current block first if it is full (lazily, as the block pool does)."""
        if self.fill == self.bs:
            self._seal(counters)
        t = self.pool[self.tail_slot]
        t[:, 0, self.fill].copy_(key[0, :, 0])
        t[:, 1, self.fill].copy_(value[0, :, 0])
        self.fill += 1
        self.length += 1

    def _seal(self, counters: TierCounters) -> None:
        t0 = time.perf_counter()
        b = self.n_sealed
        if b >= self.cap_blocks:
            raise RuntimeError(f"host pool full: {self.cap_blocks} blocks")
        tail = self.pool[self.tail_slot]
        # Blocking: the host pool must hold the bytes before any later transfer (possibly on the
        # copy stream) reads them. One 2 x block_size x head_dim x heads copy per block_size tokens.
        self.host[b].copy_(tail)
        self.kmin[0, :, b] = tail[:, 0].amin(dim=1)
        self.kmax[0, :, b] = tail[:, 0].amax(dim=1)
        self.n_sealed += 1
        self.fill = 0
        counters.seals += 1
        counters.seal_bytes += self.h * self.pair_bytes
        counters.host_seal_s += time.perf_counter() - t0

    # -- residency ----------------------------------------------------------------------------

    def candidates(self) -> tuple[int, int]:
        """(n_full, fill) in rung 5's convention: blocks 1..n_full-1 are selectable, block n_full is current."""
        return self.n_sealed, self.fill

    def rank(self, query: torch.Tensor) -> np.ndarray:
        """Top-K block ids [h, K] for a query, in descending bound order. Syncs with the GPU."""
        n_full, _ = self.candidates()
        top = top_blocks(query, self.kmin[:, :, 1:n_full], self.kmax[:, :, 1:n_full], self.k) + 1
        return top[0].cpu().numpy()

    def admit(self, chosen: np.ndarray, step: int, counters: TierCounters, stream: torch.cuda.Stream | None) -> tuple[np.ndarray, np.ndarray, int]:
        """Make every (head, block) in `chosen` [h, K] resident, launching copies on `stream`.

        Returns the fetched pairs (heads, blocks) and the number of transfers launched. Victims are
        the least recently used slots outside `chosen`; with K slots and K choices that is exactly
        the set of slots `chosen` does not name.
        """
        slots = np.take_along_axis(self.block_slot, chosen, axis=1)  # [h, K]
        miss = slots < 0
        if not miss.any():
            return np.empty(0, np.int64), np.empty(0, np.int64), 0
        mh, mi = np.nonzero(miss)
        mb = chosen[mh, mi]
        dst = np.empty_like(mb)
        for j in np.unique(mh):
            want = mb[mh == j]
            keep = np.zeros(self.k, dtype=bool)
            keep[slots[j][~miss[j]] - 1] = True
            free = np.flatnonzero(~keep)
            victims = free[np.argsort(self.last_used[j, free], kind="stable")[: want.size]]
            old = self.slot_block[j, victims]
            self.block_slot[j, old] = -1
            self.evicted_at[j, old] = step
            self.slot_block[j, victims] = want
            self.block_slot[j, want] = victims + 1
            self.last_used[j, victims] = step
            dst[mh == j] = victims + 1
        # One transfer per run of pairs consecutive in both pools (flat index block*h+head, slot*h+head).
        src_flat, dst_flat = mb * self.h + mh, dst * self.h + mh
        order = np.argsort(src_flat, kind="stable")
        src_flat, dst_flat = src_flat[order], dst_flat[order]
        breaks = np.flatnonzero((np.diff(src_flat) != 1) | (np.diff(dst_flat) != 1)) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [src_flat.size]))
        host_flat = self.host.view(-1, 2, self.bs, self.d)
        pool_flat = self.pool.view(-1, 2, self.bs, self.d)
        # stream=None is the current (compute) stream: torch.cuda.stream(None) is a no-op.
        with torch.cuda.stream(stream):
            for s, e in zip(starts.tolist(), ends.tolist()):
                a, b = int(src_flat[s]), int(dst_flat[s])
                n = e - s
                pool_flat[b : b + n].copy_(host_flat[a : a + n], non_blocking=True)
        return mh, mb, len(starts)

    def gather(self, chosen: np.ndarray, step: int, query: torch.Tensor, mask_cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention inputs for `chosen`: sink, then the chosen blocks in rank order, then the current block."""
        slots = np.take_along_axis(self.block_slot, chosen, axis=1)
        np.put_along_axis(self.last_used, slots - 1, step, axis=1)
        k2 = self.k + 2
        idx = np.empty((self.h, k2), dtype=np.int64)
        idx[:, 0] = 0
        idx[:, 1:-1] = slots
        idx[:, -1] = self.tail_slot
        flat = idx * self.h + self.heads[:, None]
        self.idx_host.numpy()[:] = flat.reshape(-1)
        self.idx_dev.copy_(self.idx_host, non_blocking=True)
        pool_flat = self.pool.view(-1, 2, self.bs, self.d)
        shape = (1, self.h, k2 * self.bs, self.d)
        k_sel = pool_flat[self.idx_dev, 0].view(shape)
        v_sel = pool_flat[self.idx_dev, 1].view(shape)
        fill = self.fill
        if mask_cache.get("fill") != fill or mask_cache.get("total") != k2 * self.bs:
            mask_cache["mask"] = current_block_mask(k2 * self.bs, self.bs, fill, query.dtype, query.device)
            mask_cache["fill"], mask_cache["total"] = fill, k2 * self.bs
        return k_sel, v_sel, mask_cache["mask"]

    def gpu_bytes(self) -> int:
        return self.pool.numel() * self.pool.element_size()

    def metadata_bytes(self) -> int:
        return 2 * self.kmin.numel() * self.kmin.element_size()

    def host_bytes(self) -> int:
        return self.n_sealed * self.h * self.pair_bytes


class TieredCache(Cache):
    """Decode-time cache: dense layers share the prefilled FullGPUCache, selecting layers are tiered.

    Construct after an exact prefill into `full`. Selecting layers copy their prompt KV to host and
    keep only K blocks per head in VRAM; the caller truncates `full` back afterwards (only its dense
    layers grow). `close()` removes the prefetch hooks and must be called before the model is reused.
    """

    def __init__(
        self,
        full: FullGPUCache,
        k_blocks: int,
        block_size: int,
        capacity_tokens: int,
        model: PreTrainedModel | None = None,
        prefetch: bool = False,
        dense_layers: int = QUEST_DENSE_LAYERS,
        host_pools: list[torch.Tensor] | None = None,
        thrash_window: int = 16,
        instrument: bool = False,
    ) -> None:
        super().__init__(layers=full.layers)
        if prefetch and model is None:
            raise ValueError("prefetch needs the model, to speculate the next layer's query")
        self.full, self.k_blocks, self.block_size, self.dense_layers = full, k_blocks, block_size, dense_layers
        self.prefetch, self.thrash_window, self.instrument = prefetch, thrash_window, instrument
        self.policy_name = "tiered_prefetch" if prefetch else "tiered_sync"
        self.counters = TierCounters()
        self.tiers: dict[int, TieredLayer] = {}
        for i in range(dense_layers, len(full.layers)):
            src = full.layers[i]
            n = src.get_seq_length()
            host = None if host_pools is None else host_pools[i - dense_layers]
            layer = TieredLayer(src.keys[:, :, :n], src.values[:, :, :n], k_blocks, block_size, capacity_tokens, host)
            self.counters.boundary_d2h_s += layer.boundary_d2h_s
            self.counters.boundary_d2h_bytes += layer.boundary_d2h_bytes
            self.tiers[i] = layer
        self._mask_cache: dict[str, Any] = {}
        self._step = 0
        self._step_fetched = 0
        self._hooks: list[Any] = []
        self._timed: list[_Timed] = []
        self._copy_events: dict[int, torch.cuda.Event] = {}
        self._window: dict[int, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = {}
        self.copy_stream: torch.cuda.Stream | None = None
        if prefetch:
            assert model is not None
            self.copy_stream = torch.cuda.Stream()
            decoder = model.model.layers
            # A hook on layer L speculates for L+1, so the copy overlaps layer L's compute.
            for src in range(dense_layers - 1, len(decoder) - 1):
                if src >= 0:
                    self._hooks.append(decoder[src].register_forward_pre_hook(self._make_hook(src + 1, decoder[src + 1]), with_kwargs=True))

    # -- transformers plumbing ---------------------------------------------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            # Layer 0 updates before any prefetch hook fires, so every hook and selection of a
            # decode step sees the same step number.
            self._begin_step()
        tier = self.tiers.get(layer_idx)
        if tier is None:
            return self.full.layers[layer_idx].update(key_states, value_states)
        if key_states.shape[-2] != 1:
            raise RuntimeError("the tier is decode-only; prefill into a FullGPUCache first")
        tier.append(key_states, value_states, self.counters)
        # Attention never reads this: select() replaces the key set for every tiered layer.
        return tier.pool[tier.tail_slot : tier.tail_slot + 1, :, 0], tier.pool[tier.tail_slot : tier.tail_slot + 1, :, 1]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        tier = self.tiers.get(layer_idx)
        return tier.length if tier is not None else self.full.layers[layer_idx].get_seq_length()

    def model_kwargs(self) -> dict[str, object]:
        return {"lazykv_selector": self}

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._step > 0:
            self._flush_step()
            self._step = 0

    # -- decode path ---------------------------------------------------------------------------

    def _begin_step(self) -> None:
        if self._step > 0:
            self._flush_step()
        self._step += 1

    def _flush_step(self) -> None:
        self.counters.fetched_pairs_per_step.append(self._step_fetched)
        self._step_fetched = 0

    def select(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        tier = self.tiers.get(layer_idx)
        if tier is None or query.shape[-2] != 1:
            return None
        c = self.counters
        t0 = time.perf_counter()
        window = self._window.pop(layer_idx, None)
        if window is not None and self.instrument:
            window[1].record()  # compute stream reaches the target layer
        ev = self._copy_events.pop(layer_idx, None)
        if ev is not None:
            # The gather below reads slots the copy stream may still be writing.
            torch.cuda.current_stream().wait_event(ev)
        chosen = tier.rank(query)
        c.selections += 1
        c.selected_pairs += chosen.size
        if tier.pending_prefetch is not None:
            ph, pb = tier.pending_prefetch
            c.prefetch_used_pairs += int((chosen[ph] == pb[:, None]).any(axis=1).sum())
            tier.pending_prefetch = None
        t1 = time.perf_counter()
        mh, mb, transfers = tier.admit(chosen, self._step, c, stream=None)
        c.host_fetch_s += time.perf_counter() - t1
        if mh.size:
            c.fetched_pairs += mh.size
            c.fetch_transfers += transfers
            c.fetch_bytes += mh.size * tier.pair_bytes
            c.thrash_pairs += int((tier.evicted_at[mh, mb] >= self._step - self.thrash_window).sum())
            self._step_fetched += mh.size
        c.hit_pairs += chosen.size - mh.size
        out = tier.gather(chosen, self._step, query, self._mask_cache)
        c.host_select_s += time.perf_counter() - t0  # includes host_fetch_s
        return out

    def _make_hook(self, target: int, target_layer: torch.nn.Module):  # noqa: ANN202
        attn = target_layer.self_attn
        norm = target_layer.input_layernorm

        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.shape[1] != 1 or target not in self.tiers:
                return
            t0 = time.perf_counter()
            tier = self.tiers[target]
            cos, sin = kwargs["position_embeddings"]
            # Layer L+1's query as it would be if layer L changed nothing: the residual stream,
            # through L+1's own norm, projection and rotary embedding. RoPE matters here, because
            # the Quest bound is computed against post-RoPE keys.
            q = attn.q_proj(norm(hidden)).view(1, 1, -1, tier.d).transpose(1, 2)
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            # The tier's current block is sealed lazily inside update(), which runs after this
            # hook; rank against the blocks sealed so far, as the real selection will unless a
            # seal happens this step (then the new block can only be a miss).
            guess = tier.rank(q)
            stream = self.copy_stream
            assert stream is not None
            start = torch.cuda.Event(enable_timing=self.instrument)
            with torch.cuda.stream(stream):
                start.record()
            mh, mb, transfers = tier.admit(guess, self._step, self.counters, stream=stream)
            done = torch.cuda.Event(enable_timing=self.instrument)
            with torch.cuda.stream(stream):
                done.record()
            self._copy_events[target] = done
            c = self.counters
            c.prefetched_pairs += mh.size
            c.prefetch_transfers += transfers
            c.prefetch_bytes += mh.size * tier.pair_bytes
            tier.pending_prefetch = (mh, mb) if mh.size else None
            if self.instrument and mh.size:
                ws, we = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ws.record()
                self._window[target] = (ws, we, done)
                self._timed.append(_Timed(target, start, done, ws, we))
            c.host_prefetch_s += time.perf_counter() - t0

        return hook

    # -- reporting ----------------------------------------------------------------------------

    def overlap(self) -> dict[str, Any]:
        """Measured overlap of each prefetch copy with the compute it was launched to hide behind.

        For every prefetch that moved bytes: the copy's interval on the copy stream, and the compute
        stream's interval from launch until the target layer began selecting. Overlap fraction is
        the share of the copy inside that window; stall is how long the target layer had to wait
        for the copy after it got there. Needs `instrument=True`; syncs the device.
        """
        if not self._timed:
            return {"prefetches_timed": 0}
        torch.cuda.synchronize()
        fractions, copy_ms, stall_ms, window_ms = [], [], [], []
        for t in self._timed:
            ref = t.window_start
            cs, ce = ref.elapsed_time(t.copy_start), ref.elapsed_time(t.copy_end)
            we = ref.elapsed_time(t.window_end)
            dur = ce - cs
            inside = max(0.0, min(ce, we) - max(cs, 0.0))
            fractions.append(1.0 if dur <= 0 else inside / dur)
            copy_ms.append(dur)
            stall_ms.append(max(0.0, ce - we))
            window_ms.append(we)
        return {
            "prefetches_timed": len(self._timed),
            "overlap_fraction": fractions,
            "copy_ms": copy_ms,
            "stall_ms": stall_ms,
            "window_ms": window_ms,
        }

    def counters_dict(self) -> dict[str, Any]:
        return asdict(self.counters)

    def stats(self) -> CacheStats:
        dense = sum(self.full.layers[i].resident_bytes() for i in range(self.dense_layers))
        gpu = dense + sum(t.gpu_bytes() + t.metadata_bytes() for t in self.tiers.values())
        # Host copies of blocks that are also resident are not "offloaded" bytes; count only what
        # VRAM does not hold, so gpu / (gpu + host) is the residency fraction.
        per_pair = next(iter(self.tiers.values())).pair_bytes
        not_resident = sum((t.n_sealed - 1 - t.k) * t.h * per_pair for t in self.tiers.values())
        return CacheStats(seq_len=self.get_seq_length(self.dense_layers), gpu_resident_kv_bytes=gpu, host_kv_bytes=not_resident, gpu_allocated_kv_bytes=gpu)

    def host_pinned_bytes(self) -> int:
        return sum(t.host.numel() * t.host.element_size() for t in self.tiers.values())


def allocate_host_pools(n_layers: int, heads: int, head_dim: int, block_size: int, capacity_tokens: int, dtype: torch.dtype = torch.bfloat16) -> list[torch.Tensor]:
    """Pinned host pools, one per selecting layer, allocated once and reused across conditions.

    Pinning a gigabyte takes on the order of a second and fragments the host; a sweep that built a
    fresh pool per condition would time the pinning, not the policy.
    """
    blocks = -(-capacity_tokens // block_size)
    return [torch.empty((blocks, heads, 2, block_size, head_dim), dtype=dtype, pin_memory=True) for _ in range(n_layers)]
