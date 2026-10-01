"""Offline replay of attention traces: selectors against an oracle, slot replacement against Belady.

Two questions, kept apart because they have different oracles:

1. **Selection.** Which K blocks should a step attend to? Scored by the true attention mass the
   choice captures (sink and current block always included, as in every selecting rung). The oracle
   is the K candidates of highest true mass: no selector of K blocks captures more.
2. **Replacement.** Given the blocks a selector chose at every step, which resident blocks should
   leave the slot pool to make room? Scored by fetches. The oracle is Belady: evict the block whose
   next selection is farthest in the future. With exactly K slots the victims are forced (every slot
   not in the current selection must go), so the rule only matters when the pool is larger than K.

The replacement model mirrors `TieredLayer` (lazykv/tiered.py): per (layer, KV head), slots seeded
at the boundary with the most recent sealed blocks, the sink and the current block outside the pool,
a newly sealed block not resident until something selects it, and LRU meaning least recently
selected.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lazykv.trace import AttentionTrace

# chosen[t] is [layers, kv, K] block ids for step t; steps differ in candidate count, so a list.
Selection = list[np.ndarray]


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    """Ids (offset by 1: candidates start at block 1) of the k highest scores along the last axis, highest first."""
    idx = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1, kind="stable")
    return np.take_along_axis(idx, order, axis=-1) + 1


def _k_for(trace: AttentionTrace, k: int) -> int:
    cands = int(trace.n_full.min()) - 1
    if not 1 <= k <= cands:
        raise ValueError(f"K={k} needs 1..{cands} candidate blocks at every step")
    return k


def select_quest(trace: AttentionTrace, k: int) -> Selection:
    """Rungs 5 to 8: the K candidates with the highest Quest bound."""
    _k_for(trace, k)
    return [_top(trace.bound[t, :, :, 1 : trace.n_full[t]], k) for t in range(trace.steps)]


def select_oracle(trace: AttentionTrace, k: int) -> Selection:
    """The K candidates of highest true mass at this step: the selection ceiling."""
    _k_for(trace, k)
    return [_top(trace.mass[t, :, :, 1 : trace.n_full[t]], k) for t in range(trace.steps)]


def select_stale_oracle(trace: AttentionTrace, k: int) -> Selection:
    """The K candidates of highest true mass at the *previous* step; Quest's choice at step 0.

    What a selector that reads last step's attention weights would attend to, if it could read them
    exactly. Its gap to the oracle is what one step of staleness costs; its gap to Quest says whether
    a bound computed from the current query is worth a per-layer host round trip.
    """
    _k_for(trace, k)
    out = [select_quest(trace, k)[0]] if trace.steps else []
    for t in range(1, trace.steps):
        # Block ids are stable across steps; the block that was current at t-1 has a mass there.
        out.append(_top(trace.mass[t - 1, :, :, 1 : trace.n_full[t]], k))
    return out


def select_window(trace: AttentionTrace, k: int) -> Selection:
    """The K most recent candidates (rung 2's recency rule over the same candidate set)."""
    _k_for(trace, k)
    out = []
    for t in range(trace.steps):
        n = int(trace.n_full[t])
        ids = np.arange(n - 1, n - 1 - k, -1)
        out.append(np.broadcast_to(ids, trace.mass.shape[1:3] + (k,)).copy())
    return out


SELECTORS: dict[str, Callable[[AttentionTrace, int], Selection]] = {
    "oracle": select_oracle,
    "quest": select_quest,
    "stale_oracle": select_stale_oracle,
    "window": select_window,
}


def mass_captured(trace: AttentionTrace, chosen: Selection) -> np.ndarray:
    """[steps, layers, kv] share of true attention mass on the sink, the current block and `chosen`."""
    out = np.empty(trace.mass.shape[:3], dtype=np.float64)
    for t, c in enumerate(chosen):
        m = trace.mass[t]
        out[t] = m[..., 0] + m[..., trace.n_full[t]] + np.take_along_axis(m, c, axis=-1).sum(axis=-1)
    return out


# -- replacement ----------------------------------------------------------------------------------


def _seed(trace: AttentionTrace, n_slots: int) -> list[int]:
    """Blocks resident at the boundary: the most recent sealed candidates, as `TieredLayer` seeds them."""
    n0 = int(trace.n_full[0])
    seed = min(n_slots, n0 - 1)
    return list(range(n0 - seed, n0))


def replay_lru(seq: list[np.ndarray], n_slots: int, seed: list[int]) -> np.ndarray:
    """Fetches per step for one (layer, head) under least-recently-selected eviction.

    Ties (never-selected seeded slots, then empty slots) break by slot order, as in
    `TieredLayer.admit`: seeded slots start at last_used -1 and empty ones at -2.
    """
    slot_block = np.full(n_slots, -1, dtype=np.int64)
    last_used = np.full(n_slots, -2, dtype=np.int64)
    slot_block[: len(seed)] = seed
    last_used[: len(seed)] = -1
    where = {b: i for i, b in enumerate(seed)}
    fetches = np.zeros(len(seq), dtype=np.int64)
    for t, chosen in enumerate(seq):
        want = [int(b) for b in chosen if int(b) not in where]
        if want:
            keep = np.zeros(n_slots, dtype=bool)
            keep[[where[int(b)] for b in chosen if int(b) in where]] = True
            free = np.flatnonzero(~keep)
            victims = free[np.argsort(last_used[free], kind="stable")[: len(want)]]
            for s, b in zip(victims.tolist(), want):
                if slot_block[s] >= 0:
                    del where[int(slot_block[s])]
                slot_block[s] = b
                where[b] = s
        for b in chosen:
            last_used[where[int(b)]] = t
        fetches[t] = len(want)
    return fetches


def replay_belady(seq: list[np.ndarray], n_slots: int, seed: list[int]) -> np.ndarray:
    """Fetches per step for one (layer, head) when every eviction is the block selected farthest ahead.

    The current selection is pinned (it must all be resident at once), so victims come from the
    rest of the pool; empty slots are used first. Blocks never selected again count as infinitely far.
    """
    steps = len(seq)
    # next_use[t][b]: the first step > t at which b is selected.
    upcoming: dict[int, list[int]] = {}
    for t in range(steps - 1, -1, -1):
        for b in seq[t]:
            upcoming.setdefault(int(b), []).append(t)  # descending; the tail is the next use
    resident = set(seed)
    fetches = np.zeros(steps, dtype=np.int64)
    for t, chosen in enumerate(seq):
        current = {int(b) for b in chosen}
        for b in current:  # consume this step's use
            uses = upcoming[b]
            while uses and uses[-1] <= t:
                uses.pop()
        want = [b for b in current if b not in resident]
        room = n_slots - len(resident)
        evict = len(want) - room
        if evict > 0:
            cands = [b for b in resident if b not in current]
            cands.sort(key=lambda b: -(upcoming[b][-1] if upcoming.get(b) else steps + 1))
            for b in cands[:evict]:
                resident.discard(b)
        resident.update(want)
        fetches[t] = len(want)
    return fetches


REPLACEMENTS: dict[str, Callable[[list[np.ndarray], int, list[int]], np.ndarray]] = {
    "lru": replay_lru,
    "belady": replay_belady,
}


def replay(trace: AttentionTrace, chosen: Selection, n_slots: int, rule: str) -> np.ndarray:
    """[steps, layers, kv] fetches under replacement `rule`, for every (layer, head) of the trace."""
    fn = REPLACEMENTS[rule]
    seed = _seed(trace, n_slots)
    _, n_layers, n_kv = trace.mass.shape[:3]
    out = np.zeros((trace.steps, n_layers, n_kv), dtype=np.int64)
    for li in range(n_layers):
        for h in range(n_kv):
            out[:, li, h] = fn([c[li, h] for c in chosen], n_slots, seed)
    return out
