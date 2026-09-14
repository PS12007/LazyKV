"""GPU-only residency policies (policy ladder rungs 2 and 3) for the block pool.

A policy answers two questions per layer: which prompt blocks survive the prefill→decode
boundary, and which resident block to evict when a newly sealed block needs a slot. Eviction
is permanent in GPU-only rungs: nothing is backed up, so an evicted block never returns.

Rung 3 (LRU) needs a notion of "use", and dense attention touches every resident block every
step. A block counts as used at a step when some query head gives it more than a uniform share
of that head's attention (see BlockPoolLayer.observe). Consequence, stated up front: at the
boundary no decode step has run yet, so LRU has no usage for prompt blocks and falls back to
insertion order. Whatever LRU evicts there is gone before it could ever be used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class Policy:
    """Base: keep everything. Valid only when the pool holds the whole sequence (budget 100%)."""

    name: str = "block_full"
    needs_attention: bool = False

    def initial_blocks(self, n_blocks: int, n_slots: int) -> list[int]:
        if n_blocks > n_slots:
            raise RuntimeError(f"{self.name}: {n_blocks} prompt blocks do not fit in {n_slots} slots")
        return list(range(n_blocks))

    def start(self, slot_blocks: list[int], device: torch.device) -> None:
        """Called once after the boundary copy, with the logical block id in each used slot."""

    def victim(self, slot_blocks: list[int]) -> int:
        raise RuntimeError(f"{self.name}: pool is full and this policy never evicts")

    def sealed(self, slot: int, block_id: int, step: int) -> None:
        """A new block now occupies `slot`."""

    def used(self, used_mask: torch.Tensor, step: int) -> None:
        """used_mask: bool [n_used_slots] on the GPU, for this decode step."""


@dataclass
class WindowSink(Policy):
    """StreamingLLM-style: the first `sink_blocks` blocks plus the most recent blocks.

    Block granularity means the sink is a whole block (it contains the BOS token and the chat
    header), not the paper's four tokens. At the budgets swept here that costs at most one block.
    """

    name: str = "window_sink"
    sink_blocks: int = 1

    def initial_blocks(self, n_blocks: int, n_slots: int) -> list[int]:
        if n_blocks <= n_slots:
            return list(range(n_blocks))
        sinks = list(range(min(self.sink_blocks, n_slots)))
        recent = n_slots - len(sinks)
        return sinks + list(range(n_blocks - recent, n_blocks))

    def victim(self, slot_blocks: list[int]) -> int:
        # Oldest non-sink block. Logical ids grow with position, so the smallest id is oldest.
        candidates = [(b, s) for s, b in enumerate(slot_blocks) if b >= self.sink_blocks]
        if not candidates:
            raise RuntimeError("window_sink: only sink blocks are resident; budget below one window block")
        return min(candidates)[1]


@dataclass
class LRU(Policy):
    """Evict the block whose last observed use is oldest; ties go to the older block."""

    name: str = "lru"
    needs_attention: bool = True
    _last_used: torch.Tensor | None = field(default=None, repr=False)
    _block_ids: torch.Tensor | None = field(default=None, repr=False)

    def initial_blocks(self, n_blocks: int, n_slots: int) -> list[int]:
        # No usage history exists before decode, so recency is insertion order.
        return list(range(max(0, n_blocks - n_slots), n_blocks))

    def start(self, slot_blocks: list[int], device: torch.device) -> None:
        capacity = len(slot_blocks) + 1
        # Kept on the GPU and updated with torch.where, so tracking use costs kernel launches
        # but no host synchronization; only choosing a victim (once per sealed block) syncs.
        self._last_used = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self._block_ids = torch.full((capacity,), -1, dtype=torch.long, device=device)
        if slot_blocks:
            ids = torch.tensor(slot_blocks, dtype=torch.long, device=device)
            self._block_ids[: len(slot_blocks)] = ids
            # Before any decode step, a block's "last use" is when it was written: its index
            # minus the number of blocks, so every prompt block is older than any decode step.
            self._last_used[: len(slot_blocks)] = ids - (int(ids.max().item()) + 1)

    def _grow(self, n: int) -> None:
        assert self._last_used is not None and self._block_ids is not None
        if n > self._last_used.numel():
            pad = n - self._last_used.numel()
            self._last_used = torch.cat([self._last_used, self._last_used.new_full((pad,), -1)])
            self._block_ids = torch.cat([self._block_ids, self._block_ids.new_full((pad,), -1)])

    def victim(self, slot_blocks: list[int]) -> int:
        assert self._last_used is not None and self._block_ids is not None
        n = len(slot_blocks)
        # Lexicographic (last_used, block_id): scale so recency dominates and id breaks ties.
        key = self._last_used[:n] * (1 << 32) + self._block_ids[:n]
        return int(torch.argmin(key).item())

    def sealed(self, slot: int, block_id: int, step: int) -> None:
        self._grow(slot + 1)
        assert self._last_used is not None and self._block_ids is not None
        self._last_used[slot] = step
        self._block_ids[slot] = block_id

    def used(self, used_mask: torch.Tensor, step: int) -> None:
        assert self._last_used is not None
        n = used_mask.numel()
        self._grow(n)
        self._last_used[:n] = torch.where(used_mask, torch.full_like(self._last_used[:n], step), self._last_used[:n])


POLICIES: dict[str, Callable[[], Policy]] = {
    "block_full": Policy,
    "window_sink": WindowSink,
    # Control for rung 3: LRU's boundary choice is "most recent blocks, no sink", so this
    # isolates how much of LRU's result is the lost attention sink rather than LRU itself.
    "window": lambda: WindowSink(name="window", sink_blocks=0),
    "lru": LRU,
}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}") from None
