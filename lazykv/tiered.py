"""A pinned CPU tier under query-aware selection: policy ladder rungs 6, 7 and 8.

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
- Rung 8 (`quant="int8"`): the host pool holds int8 records (lazykv/quant.py) and a fetch widens
  them back into the bf16 slot pool, so the hot tier stays exact and only the warm/cold tier is
  approximate. This is the first rung at which the tier stops being bit-identical to rung 5, so
  it is also the first with a quality axis of its own. The ladder writes it as "rung 7 + mixed
  precision"; Phase 4 measured rung 7 slower than rung 6, so it is built on rung 6 instead.

  Two consequences of that follow through the code and are worth stating once. The selection
  metadata and the slots seeded at the boundary are built from *dequantized* keys, so a block's
  contents never depend on whether it happens to be resident and the Quest bound stays a true
  bound over the keys actually attended. And `fetch="runs"` is not available: it writes host bytes
  straight into their slots, which needs the two pools to share a layout.

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
from lazykv.quant import pack, packed_pair_bytes, unpack
from lazykv.selection import QUEST_DENSE_LAYERS, current_block_mask, top_blocks

NEVER = -(2**40)  # "evicted at" sentinel for pairs that were never evicted
BOUNDARY_CHUNK = 128  # blocks packed at once at the boundary; the float32 intermediate is the reason


@dataclass
class TierCounters:
    """Host time and transfer activity of the tier (brief §B6: cache behaviour, PCIe, overhead)."""

    host_select_s: float = 0.0  # bound, rank, host sync, residency bookkeeping, gather launch
    host_rank_s: float = 0.0  # of which: bound + top-K launch and the host sync that brings the indices back
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
class _Stage:
    """Staging buffers for the "gather" fetch: host-side gather into pinned memory, one H2D, device scatter.

    One set per stream, shared by every layer: the compute stream fetches for one layer at a time, and
    the copy stream is synchronized before its buffers are rewritten.
    """

    buf_host: torch.Tensor  # [cap, 2, bs, d] (bf16 tier) or [cap, packed_pair_bytes] (int8 tier), pinned
    buf_dev: torch.Tensor
    src_host: torch.Tensor  # [cap] int64: flat host-pool indices
    dst_host: torch.Tensor  # [cap] int64, pinned: flat slot-pool indices
    dst_dev: torch.Tensor

    @classmethod
    def allocate(cls, cap: int, bs: int, d: int, dtype: torch.dtype, device: torch.device, quant: str | None = None) -> _Stage:
        shape, dt = ((cap, 2, bs, d), dtype) if quant is None else ((cap, packed_pair_bytes(bs, d)), torch.uint8)
        buf = torch.empty(shape, dtype=dt, pin_memory=True)
        dst = torch.empty((cap,), dtype=torch.int64, pin_memory=True)
        return cls(buf, torch.empty_like(buf, device=device), torch.empty((cap,), dtype=torch.int64), dst, torch.empty_like(dst, device=device))

    def gpu_bytes(self) -> int:
        return self.buf_dev.numel() * self.buf_dev.element_size() + self.dst_dev.numel() * 8


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

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, k_blocks: int, block_size: int, capacity_tokens: int, host: torch.Tensor | None = None, n_slots: int | None = None, quant: str | None = None) -> None:
        _, h, length, d = keys.shape
        bs = block_size
        n_blocks, tail = divmod(length, bs)
        if n_blocks - 1 <= k_blocks:
            # Rung 5 attends densely when the budget covers every block. The tier has no dense copy
            # to fall back to, so it refuses budgets that would need one.
            raise ValueError(f"budget of {k_blocks} blocks covers all {n_blocks - 1} candidate blocks; the tier needs selection")
        n_slots = k_blocks if n_slots is None else n_slots
        if n_slots < k_blocks:
            raise ValueError(f"{n_slots} slots cannot hold a selection of {k_blocks} blocks")
        if quant not in (None, "int8"):
            raise ValueError(f"unknown tier precision {quant!r}")
        self.h, self.d, self.bs, self.k, self.n_slots = h, d, bs, k_blocks, n_slots
        self.quant = quant
        self.cap_blocks = -(-capacity_tokens // bs)
        dev, dt = keys.device, keys.dtype
        self.tail_slot = n_slots + 1
        # Zeroed: the current block's unfilled region is gathered and masked, and masking cannot
        # neutralize NaN/inf left in recycled allocator memory (Phase 1).
        self.pool = torch.zeros((n_slots + 2, h, 2, bs, d), dtype=dt, device=dev)
        self.kmin = torch.empty((1, h, self.cap_blocks, d), dtype=dt, device=dev)
        self.kmax = torch.empty_like(self.kmin)
        self.logical_pair_bytes = 2 * bs * d * self.pool.element_size()  # what one pair would cost in the model's dtype
        shape = (self.cap_blocks, h, 2, bs, d) if quant is None else (self.cap_blocks, h, packed_pair_bytes(bs, d))
        want_dtype = dt if quant is None else torch.uint8
        if host is None:
            host = torch.empty(shape, dtype=want_dtype, pin_memory=True)
        elif host.shape != shape or host.dtype != want_dtype or not host.is_pinned():
            raise ValueError(f"host pool must be pinned {want_dtype} with shape {shape}, got {want_dtype if host.dtype == want_dtype else host.dtype} {tuple(host.shape)}")
        self.host = host
        self.pair_bytes = self.logical_pair_bytes if quant is None else packed_pair_bytes(bs, d)

        # Boundary: every full prompt block goes to host once (seal-time D2H, all at once), and the
        # metadata is built on the GPU where the prompt KV still is.
        blocks = torch.stack([keys[0, :, : n_blocks * bs], values[0, :, : n_blocks * bs]], dim=1)  # [h, 2, B*bs, d]
        blocks = blocks.unflatten(2, (n_blocks, bs)).permute(2, 0, 1, 3, 4)  # [B, h, 2, bs, d]
        # Initial residency: the most recent candidate blocks, copied device-side. Recency is the
        # cheapest guess available at the boundary; the first decode step's misses are the cost of
        # it, and fetched_pairs_per_step shows that cold start separately.
        seed = min(n_slots, n_blocks - 1)
        recent = np.arange(n_blocks - seed, n_blocks)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        if quant is None:
            self.host[:n_blocks].copy_(blocks)
            kb = keys[:, :, : n_blocks * bs].unflatten(2, (n_blocks, bs))
            self.kmin[:, :, :n_blocks] = kb.amin(dim=3)
            self.kmax[:, :, :n_blocks] = kb.amax(dim=3)
            self.pool[1 : seed + 1] = blocks[n_blocks - seed : n_blocks]
        else:
            # Pack on the GPU, where the prompt KV already is, so only the narrowed bytes cross the
            # link. Chunked because pack/unpack work in float32 and a whole 32K prompt's
            # intermediate would be a large fraction of an 8 GB card.
            #
            # Metadata and the seeded slots are built from the *dequantized* keys, not the originals.
            # A block's contents must not depend on whether it happens to be resident: if a seeded
            # slot held exact KV, evicting and re-fetching it would silently change what the model
            # attends to. Deriving the Quest bound from the dequantized keys also keeps it a true
            # bound over the keys that are actually attended.
            for c0 in range(0, n_blocks, BOUNDARY_CHUNK):
                c1 = min(n_blocks, c0 + BOUNDARY_CHUNK)
                chunk = blocks[c0:c1]
                self.host[c0:c1].copy_(pack(chunk))
                deq = unpack(self.host[c0:c1].to(dev).flatten(0, 1), bs, d, dt).unflatten(0, (c1 - c0, h))
                kb = deq[:, :, 0].permute(1, 0, 2, 3)  # [h, n, bs, d]
                self.kmin[0, :, c0:c1] = kb.amin(dim=2)
                self.kmax[0, :, c0:c1] = kb.amax(dim=2)
                lo, hi = max(c0, n_blocks - seed), c1
                if lo < hi:
                    self.pool[1 + lo - (n_blocks - seed) : 1 + hi - (n_blocks - seed)] = deq[lo - c0 : hi - c0]
        self.boundary_d2h_s = time.perf_counter() - t0
        self.boundary_d2h_bytes = n_blocks * h * self.pair_bytes
        # The sink never leaves VRAM (admit writes only slots 1..n_slots), so it is permanently hot
        # and stays exact even under an int8 warm/cold tier.
        self.pool[0] = blocks[0]
        if tail:
            self.pool[self.tail_slot, :, 0, :tail] = keys[0, :, n_blocks * bs : length]
            self.pool[self.tail_slot, :, 1, :tail] = values[0, :, n_blocks * bs : length]
        self.n_sealed, self.fill, self.length = n_blocks, tail, length

        # Residency table on the host: slot numbers are pool indices 1..n_slots; -1 is empty.
        self.slot_block = np.full((h, n_slots), -1, dtype=np.int64)
        self.slot_block[:, :seed] = recent
        self.block_slot = np.full((h, self.cap_blocks), -1, dtype=np.int64)
        self.block_slot[:, recent] = np.arange(1, seed + 1)
        # Empty slots (-2) are taken before any occupied one.
        self.last_used = np.full((h, n_slots), -2, dtype=np.int64)
        self.last_used[:, :seed] = -1
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
        if self.quant is None:
            self.host[b].copy_(tail)
            keys = tail[:, 0]
        else:
            self.host[b].copy_(pack(tail))
            # As at the boundary: the bound must be taken over the keys a fetch will produce, not
            # over the exact keys that are about to stop existing anywhere.
            keys = unpack(self.host[b].to(tail.device), self.bs, self.d, tail.dtype)[:, 0]
        self.kmin[0, :, b] = keys.amin(dim=1)
        self.kmax[0, :, b] = keys.amax(dim=1)
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

    def admit(self, chosen: np.ndarray, step: int, counters: TierCounters, stream: torch.cuda.Stream | None, stage: _Stage | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """Make every (head, block) in `chosen` [h, K] resident, launching copies on `stream`.

        Returns the fetched pairs (heads, blocks) and the number of H2D transfers launched. Victims
        are the least recently used slots outside `chosen`; with K slots and K choices that is
        exactly the set of slots `chosen` does not name.

        Without `stage`, pairs go straight from the host pool to their slots, one transfer per run of
        pairs consecutive in both pools. With `stage`, they are gathered on the host into pinned
        staging memory and sent as one transfer, then scattered on the device: one PCIe copy instead
        of many, at the price of a host memcpy and a staging buffer in VRAM.
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
            keep = np.zeros(self.n_slots, dtype=bool)
            keep[slots[j][~miss[j]] - 1] = True
            free = np.flatnonzero(~keep)
            victims = free[np.argsort(self.last_used[j, free], kind="stable")[: want.size]]
            old = self.slot_block[j, victims]
            old = old[old >= 0]
            self.block_slot[j, old] = -1
            self.evicted_at[j, old] = step
            self.slot_block[j, victims] = want
            self.block_slot[j, want] = victims + 1
            self.last_used[j, victims] = step
            dst[mh == j] = victims + 1
        # Flat indices: block*h+head in the host pool, slot*h+head in the slot pool. Sorted by source,
        # so runs are found and a host-side gather walks the pool forward.
        src_flat, dst_flat = mb * self.h + mh, dst * self.h + mh
        order = np.argsort(src_flat, kind="stable")
        src_flat, dst_flat = src_flat[order], dst_flat[order]
        host_flat = self.host.flatten(0, 1)  # [blocks*h, ...]: one row per pair, whatever its width
        pool_flat = self.pool.view(-1, 2, self.bs, self.d)
        if stage is not None:
            cap = stage.buf_host.shape[0]
            chunks = 0
            with torch.cuda.stream(stream):
                for c0 in range(0, src_flat.size, cap):
                    if chunks:
                        # The previous chunk's non_blocking copy reads the same pinned buffer.
                        torch.cuda.current_stream().synchronize()
                    m = min(cap, src_flat.size - c0)
                    stage.src_host.numpy()[:m] = src_flat[c0 : c0 + m]
                    torch.index_select(host_flat, 0, stage.src_host[:m], out=stage.buf_host[:m])
                    stage.dst_host.numpy()[:m] = dst_flat[c0 : c0 + m]
                    stage.buf_dev[:m].copy_(stage.buf_host[:m], non_blocking=True)
                    stage.dst_dev[:m].copy_(stage.dst_host[:m], non_blocking=True)
                    # The slot pool is always the model's dtype: the hot tier is exact, and an int8
                    # record is widened back on arrival rather than attended in place.
                    arrived = stage.buf_dev[:m] if self.quant is None else unpack(stage.buf_dev[:m], self.bs, self.d, self.pool.dtype)
                    pool_flat.index_copy_(0, stage.dst_dev[:m], arrived)
                    chunks += 1
            return mh, mb, chunks
        if self.quant is not None:
            # "runs" writes host bytes straight into their slots, which only works when the two
            # pools share a layout. An int8 record has to land somewhere it can be widened from.
            raise RuntimeError("the int8 tier needs fetch='gather'; 'runs' has nowhere to dequantize")
        breaks = np.flatnonzero((np.diff(src_flat) != 1) | (np.diff(dst_flat) != 1)) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [src_flat.size]))
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

    def not_resident_pairs(self) -> int:
        """Sealed candidate pairs that VRAM does not hold (the sink is always resident)."""
        return (self.n_sealed - 1) * self.h - int((self.slot_block >= 0).sum())

    def not_resident_bytes(self) -> int:
        return self.not_resident_pairs() * self.pair_bytes

    def not_resident_logical_bytes(self) -> int:
        """What those pairs would occupy in the model's dtype: the denominator of rung 8's saving.

        Reported alongside `not_resident_bytes` because the two are not comparable across rungs --
        a residency fraction computed from narrowed bytes would flatter the int8 tier for free.
        """
        return self.not_resident_pairs() * self.logical_pair_bytes


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
        n_slots: int | None = None,
        fetch: str = "runs",
        stages: tuple[_Stage, _Stage] | None = None,
        quant: str | None = None,
        record_selection: bool = False,
        replay_selection: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> None:
        super().__init__(layers=full.layers)
        if prefetch and model is None:
            raise ValueError("prefetch needs the model, to speculate the next layer's query")
        if fetch not in ("runs", "gather"):
            raise ValueError(f"unknown fetch mode {fetch!r}")
        if quant is not None and fetch != "gather":
            raise ValueError(f"the int8 tier needs fetch='gather', got {fetch!r}")
        self.full, self.k_blocks, self.block_size, self.dense_layers = full, k_blocks, block_size, dense_layers
        self.prefetch, self.thrash_window, self.instrument = prefetch, thrash_window, instrument
        self.quant = quant
        # The ladder writes rung 8 as "rung 7 + mixed precision", but Phase 4 measured rung 7 slower
        # than rung 6, so rung 8 is built on rung 6's synchronous fetch. Stated here and in the
        # phase report rather than silently substituted.
        self.policy_name = "tiered_int8" if quant else ("tiered_prefetch" if prefetch else "tiered_sync")
        self.fetch = fetch
        self.counters = TierCounters()
        self.tiers: dict[int, TieredLayer] = {}
        for i in range(dense_layers, len(full.layers)):
            src = full.layers[i]
            n = src.get_seq_length()
            host = None if host_pools is None else host_pools[i - dense_layers]
            layer = TieredLayer(src.keys[:, :, :n], src.values[:, :, :n], k_blocks, block_size, capacity_tokens, host, n_slots, quant=quant)
            self.counters.boundary_d2h_s += layer.boundary_d2h_s
            self.counters.boundary_d2h_bytes += layer.boundary_d2h_bytes
            self.tiers[i] = layer
        self.n_slots = next(iter(self.tiers.values())).n_slots
        self._stages: list[_Stage] = []
        self._stage_compute: _Stage | None = None
        self._stage_copy: _Stage | None = None
        if fetch == "gather":
            # Sized for the worst case, every slot of every head missing in one layer (the first
            # step); a smaller preallocated stage still works, in chunks.
            t = next(iter(self.tiers.values()))
            if stages is None:
                stages = (_Stage.allocate(t.n_slots * t.h, block_size, t.d, t.pool.dtype, t.pool.device, quant), _Stage.allocate(t.n_slots * t.h, block_size, t.d, t.pool.dtype, t.pool.device, quant))
            self._stage_compute = stages[0]
            self._stages.append(self._stage_compute)
            if prefetch:
                self._stage_copy = stages[1]
                self._stages.append(self._stage_copy)
        self._mask_cache: dict[str, Any] = {}
        # Selection record/replay, used only by the Phase 6 ceiling ablation. Keyed by (layer, step).
        self._record: dict[tuple[int, int], np.ndarray] | None = {} if record_selection else None
        self._replay = replay_selection
        self._step = 0
        self._step_fetched = 0
        self._last_chosen: np.ndarray | None = None
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
        tr = time.perf_counter()
        if self._replay is None:
            chosen = tier.rank(query)
        else:
            # Latency ablation only (scripts/03_host_ceiling.py). Replaying a selection recorded
            # from a real pass reproduces that pass exactly -- same blocks, same fetches, same
            # gathers, same tokens -- while skipping the bound and the host sync that produced it.
            # The difference in decode time is what a device-side residency decision could recover.
            chosen = self._replay[(layer_idx, self._step)]
        if self._record is not None:
            self._record[(layer_idx, self._step)] = chosen
        c.host_rank_s += time.perf_counter() - tr
        # Rung 9 (lazykv/exact.py) needs the same block ids to know which blocks the CPU must
        # *not* attend to; recomputing them there would be a second source of truth.
        self._last_chosen = chosen
        c.selections += 1
        c.selected_pairs += chosen.size
        if tier.pending_prefetch is not None:
            ph, pb = tier.pending_prefetch
            c.prefetch_used_pairs += int((chosen[ph] == pb[:, None]).any(axis=1).sum())
            tier.pending_prefetch = None
        t1 = time.perf_counter()
        mh, mb, transfers = tier.admit(chosen, self._step, c, stream=None, stage=self._stage_compute)
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
            if self._stage_copy is not None:
                # The previous prefetch's non_blocking copies read the same pinned staging buffer.
                stream.synchronize()
            # Window start on the compute stream, before any copy is launched, so the copy interval
            # can be placed inside the window the compute stream spends reaching the target layer.
            ws = torch.cuda.Event(enable_timing=True) if self.instrument else None
            if ws is not None:
                ws.record()
            start = torch.cuda.Event(enable_timing=self.instrument)
            with torch.cuda.stream(stream):
                start.record()
            mh, mb, transfers = tier.admit(guess, self._step, self.counters, stream=stream, stage=self._stage_copy)
            done = torch.cuda.Event(enable_timing=self.instrument)
            with torch.cuda.stream(stream):
                done.record()
            self._copy_events[target] = done
            c = self.counters
            c.prefetched_pairs += mh.size
            c.prefetch_transfers += transfers
            c.prefetch_bytes += mh.size * tier.pair_bytes
            tier.pending_prefetch = (mh, mb) if mh.size else None
            if ws is not None and mh.size:
                we = torch.cuda.Event(enable_timing=True)
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

    def recorded_selection(self) -> dict[tuple[int, int], np.ndarray]:
        """The (layer, step) -> chosen-blocks map a `record_selection=True` run captured."""
        if self._record is None:
            raise RuntimeError("this cache was not built with record_selection=True")
        return self._record

    def counters_dict(self) -> dict[str, Any]:
        return asdict(self.counters)

    def stats(self) -> CacheStats:
        dense = sum(self.full.layers[i].resident_bytes() for i in range(self.dense_layers))
        # Everything the tier holds in VRAM for decode: slot pools, block metadata, staging buffers.
        gpu = dense + sum(t.gpu_bytes() + t.metadata_bytes() for t in self.tiers.values()) + sum(s.gpu_bytes() for s in self._stages)
        # Host copies of blocks that are also resident are not "offloaded" bytes; count only what
        # VRAM does not hold, so gpu / (gpu + host) is the residency fraction.
        not_resident = sum(t.not_resident_bytes() for t in self.tiers.values())
        return CacheStats(seq_len=self.get_seq_length(self.dense_layers), gpu_resident_kv_bytes=gpu, host_kv_bytes=not_resident, gpu_allocated_kv_bytes=gpu)

    def host_logical_bytes(self) -> int:
        """What the off-GPU KV would occupy in the model's dtype, whatever precision the tier stores it in.

        `stats().host_kv_bytes` is what rung 8 actually holds; this is what rungs 6 and 7 would hold
        for the same blocks. The saving is the ratio, and reporting only the first would make the
        residency fraction look better for free.
        """
        return sum(t.not_resident_logical_bytes() for t in self.tiers.values())

    def host_pinned_bytes(self) -> int:
        return sum(t.host.numel() * t.host.element_size() for t in self.tiers.values()) + sum(s.buf_host.numel() * s.buf_host.element_size() for s in self._stages)


def allocate_stages(cap_pairs: int, heads: int, head_dim: int, block_size: int, dtype: torch.dtype = torch.bfloat16, quant: str | None = None) -> tuple[_Stage, _Stage]:
    """Gather-mode staging buffers (compute stream, copy stream), allocated once per process like the host pools.

    Allocating them per condition would put pinning inside the boundary time, and WDDM reports pinned
    host memory as shared GPU memory, which a per-condition spill check would then flag.
    """
    del heads
    dev = torch.device("cuda")
    return _Stage.allocate(cap_pairs, block_size, head_dim, dtype, dev, quant), _Stage.allocate(cap_pairs, block_size, head_dim, dtype, dev, quant)


def allocate_host_pools(n_layers: int, heads: int, head_dim: int, block_size: int, capacity_tokens: int, dtype: torch.dtype = torch.bfloat16, quant: str | None = None) -> list[torch.Tensor]:
    """Pinned host pools, one per selecting layer, allocated once and reused across conditions.

    Pinning a gigabyte takes on the order of a second and fragments the host; a sweep that built a
    fresh pool per condition would time the pinning, not the policy.
    """
    blocks = -(-capacity_tokens // block_size)
    shape, dt = ((blocks, heads, 2, block_size, head_dim), dtype) if quant is None else ((blocks, heads, packed_pair_bytes(block_size, head_dim)), torch.uint8)
    return [torch.empty(shape, dtype=dt, pin_memory=True) for _ in range(n_layers)]
