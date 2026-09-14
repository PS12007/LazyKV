"""Block-granular, fixed-budget GPU KV pool: the substrate for policy ladder rungs 2+.

Layout per layer: one preallocated [1, kv_heads, (n_slots + 1) * block_size, head_dim] tensor
pair. Slots [0, n_used) hold sealed blocks; the region right after them holds the partially
filled tail block that new tokens are written into. Resident KV is therefore always one
contiguous prefix, so attention gets a plain view with no mask and no gather:

- The attention kernel and its startup verification are reused unchanged (lazykv/attention.py).
- At a fixed budget the prefix length cycles within one block, so cuDNN plans stay cached.
- Order inside the prefix does not matter: keys are cached after RoPE, and decode attention
  over a set of positions is permutation-invariant.

When the tail fills: if a slot is free, the tail already sits in it and simply becomes sealed.
Otherwise the policy names a victim slot, and the tail's block_size tokens are copied over it.
That is one small copy per block_size decode steps, instead of moving the resident KV.

Prefill is exact and happens outside this class (FullGPUCache). `from_full` applies the
policy at the prefill→decode boundary. Peak memory during prefill is therefore the full KV,
and experiments report it separately from decode-time residency.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from lazykv.attention import current_strategy
from lazykv.cache import CacheStats, FullGPUCache
from lazykv.policies import Policy


def slots_for_budget(fraction: float, total_tokens: int, block_size: int) -> int:
    """Sealed-block slots such that slots plus the tail region hold `fraction` of the sequence.

    The tail region (one block) is always resident and counts against the budget. At 100%
    the pool must hold every token, so it rounds up; below 100% it rounds down, so the
    budget is never exceeded.
    """
    if not 0 < fraction <= 1:
        raise ValueError("budget fraction must be in (0, 1]")
    if fraction == 1:
        return max(1, math.ceil(total_tokens / block_size))
    return max(1, math.floor(fraction * total_tokens / block_size) - 1)


@dataclass
class ManagerCounters:
    """Host-side cost and activity of the pool manager (brief §B6: manager wall time)."""

    host_update_s: float = 0.0
    host_observe_s: float = 0.0
    seals: int = 0
    evictions: int = 0
    boundary_evicted_blocks: int = 0


class BlockPoolLayer(CacheLayerMixin):
    def __init__(self, n_slots: int, block_size: int, policy: Policy) -> None:
        super().__init__()
        self.n_slots, self.block_size, self.policy = n_slots, block_size, policy
        self.n_used = 0
        self.fill = 0
        self.logical_len = 0
        self.next_block_id = 0
        self.slot_blocks: list[int] = []
        self.step = 0
        self.counters = ManagerCounters()

    # -- construction ------------------------------------------------------------------

    def load(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Fill from exact prefill KV ([1, h, L, d] each), applying the policy's boundary choice."""
        _, h, length, d = keys.shape
        bs = self.block_size
        # Storage rounds up to a whole attention bucket. The bucketed cuDNN decode pads the view
        # to the next bucket boundary inside existing storage; if the pool ended short of it, every
        # step would fall back to unbucketed cuDNN and rebuild a plan per token (measured in the
        # Phase 2 smoke run as roughly 3x decode latency). The padding is never resident KV.
        _, bucket = current_strategy()
        capacity = -(-((self.n_slots + 1) * bs) // bucket) * bucket
        # Zeroed for the same reason as PreallocatedLayer: bucketed decode reads masked padding.
        self.keys = torch.zeros((1, h, capacity, d), dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros((1, values.shape[1], capacity, values.shape[3]), dtype=values.dtype, device=values.device)
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

        n_blocks, tail = divmod(length, bs)
        keep = sorted(self.policy.initial_blocks(n_blocks, self.n_slots))
        if keep:
            # One gather per layer: token indices of every kept block, in slot order.
            idx = (torch.tensor(keep, device=keys.device)[:, None] * bs + torch.arange(bs, device=keys.device)).reshape(-1)
            self.keys[:, :, : len(keep) * bs].copy_(keys.index_select(2, idx))
            self.values[:, :, : len(keep) * bs].copy_(values.index_select(2, idx))
        start = len(keep) * bs
        if tail:
            self.keys[:, :, start : start + tail].copy_(keys[:, :, n_blocks * bs : length])
            self.values[:, :, start : start + tail].copy_(values[:, :, n_blocks * bs : length])
        self.slot_blocks = list(keep)
        self.n_used, self.fill = len(keep), tail
        self.logical_len = length
        self.next_block_id = n_blocks
        self.counters.boundary_evicted_blocks = n_blocks - len(keep)
        self.policy.start(self.slot_blocks, keys.device)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        raise RuntimeError("BlockPoolLayer is built from a prefilled FullGPUCache via BlockPoolCache.from_full")

    # -- transformers Cache API --------------------------------------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter()
        bs = self.block_size
        n = key_states.shape[-2]
        for i in range(n):
            # Seal lazily, when a token needs the space. Sealing right after the tail fills
            # would copy the tail over the victim while the view handed to attention still
            # covers both, so attention would see the tail twice. A full pool plus a full
            # tail is exactly the budget, so waiting one token costs nothing.
            if self.fill == bs:
                self._seal()
            pos = self.n_used * bs + self.fill
            self.keys[:, :, pos : pos + 1].copy_(key_states[:, :, i : i + 1])
            self.values[:, :, pos : pos + 1].copy_(value_states[:, :, i : i + 1])
            self.fill += 1
            self.logical_len += 1
        view_len = self.n_used * bs + self.fill
        self.counters.host_update_s += time.perf_counter() - t0
        return self.keys[:, :, :view_len], self.values[:, :, :view_len]

    def _seal(self) -> None:
        bs = self.block_size
        block_id = self.next_block_id
        self.next_block_id += 1
        self.counters.seals += 1
        if self.n_used < self.n_slots:
            slot = self.n_used  # the tail already occupies this slot
            self.slot_blocks.append(block_id)
            self.n_used += 1
        else:
            slot = self.policy.victim(self.slot_blocks)
            tail = self.n_used * bs
            self.keys[:, :, slot * bs : (slot + 1) * bs].copy_(self.keys[:, :, tail : tail + bs])
            self.values[:, :, slot * bs : (slot + 1) * bs].copy_(self.values[:, :, tail : tail + bs])
            self.slot_blocks[slot] = block_id
            self.counters.evictions += 1
        self.fill = 0
        self.policy.sealed(slot, block_id, self.step)

    def observe(self, query: torch.Tensor, key: torch.Tensor) -> None:
        """Per-slot use for this decode step: some query head gives the block above-uniform attention."""
        if query.shape[-2] != 1 or self.n_used == 0:
            return
        t0 = time.perf_counter()
        bs = self.block_size
        n_q, n_kv, d = query.shape[1], key.shape[1], query.shape[-1]
        groups = n_q // n_kv
        live = key.shape[-2]
        # Scores for the sealed slots only need the full softmax denominator, which spans the tail too.
        q = query.float().view(1, n_kv, groups, 1, d)
        scores = (q @ key.float().unsqueeze(2).transpose(-1, -2)) / math.sqrt(d)  # [1, kv, g, 1, live]
        probs = torch.softmax(scores, dim=-1)[..., : self.n_used * bs]
        mass = probs.reshape(1, n_kv, groups, self.n_used, bs).sum(-1)  # [1, kv, g, n_used]
        blocks_resident = self.n_used + (1 if live > self.n_used * bs else 0)
        used = (mass.amax(dim=(0, 1, 2)) > 1.0 / blocks_resident)
        self.policy.used(used, self.step)
        self.step += 1
        self.counters.host_observe_s += time.perf_counter() - t0

    def get_seq_length(self) -> int:
        # Position ids come from here, so it must be the logical length, not the resident count.
        return self.logical_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.n_used * self.block_size + self.fill + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        raise RuntimeError("build a new BlockPoolCache per condition instead of resetting")

    def resident_tokens(self) -> int:
        return self.n_used * self.block_size + self.fill

    def bytes_per_token(self) -> int:
        return self.keys[0, :, 0].numel() * self.keys.element_size() + self.values[0, :, 0].numel() * self.values.element_size()


class BlockPoolCache(Cache):
    def __init__(self, layers: list[BlockPoolLayer]) -> None:
        super().__init__(layers=layers)
        self.pool_layers = layers

    @classmethod
    def from_full(cls, full: FullGPUCache, n_slots: int, block_size: int, policy_factory: Callable[[], Policy]) -> BlockPoolCache:
        layers = []
        for src in full.layers:
            layer = BlockPoolLayer(n_slots, block_size, policy_factory())
            length = src.get_seq_length()
            layer.load(src.keys[:, :, :length], src.values[:, :, :length])
            layers.append(layer)
        return cls(layers)

    @property
    def policy_name(self) -> str:
        return self.pool_layers[0].policy.name

    def model_kwargs(self) -> dict[str, object]:
        """Extra model() kwargs: hands this cache to the attention function when the policy observes."""
        return {"lazykv_observer": self} if self.pool_layers[0].policy.needs_attention else {}

    def observe(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor) -> None:
        self.pool_layers[layer_idx].observe(query, key)

    def counters(self) -> dict[str, float | int]:
        c = [l.counters for l in self.pool_layers]
        return {
            "host_update_s": sum(x.host_update_s for x in c),
            "host_observe_s": sum(x.host_observe_s for x in c),
            "seals": sum(x.seals for x in c),
            "evictions": sum(x.evictions for x in c),
            "boundary_evicted_blocks": sum(x.boundary_evicted_blocks for x in c),
        }

    def stats(self) -> CacheStats:
        layers = self.pool_layers
        resident = sum(l.resident_tokens() * l.bytes_per_token() for l in layers)
        allocated = sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size() for l in layers)
        return CacheStats(seq_len=layers[0].logical_len, gpu_resident_kv_bytes=resident, host_kv_bytes=0, gpu_allocated_kv_bytes=allocated)
