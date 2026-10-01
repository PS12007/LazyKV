"""Offline replay (lazykv/residency_sim.py): selector bookkeeping, and Belady checked against brute force."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from lazykv.residency_sim import SELECTORS, mass_captured, replay, replay_belady, replay_lru
from lazykv.trace import AttentionTrace


def _synthetic(steps: int = 12, layers: int = 2, kv: int = 3, n0: int = 20, seed: int = 0) -> AttentionTrace:
    """A trace with one block sealing every 4 steps and random but normalized mass."""
    rng = np.random.default_rng(seed)
    n_full = n0 + np.arange(steps) // 4
    width = int(n_full.max()) + 1
    mass = np.zeros((steps, layers, kv, width), dtype=np.float32)
    bound = np.full_like(mass, -np.inf)
    for t, n in enumerate(n_full):
        m = rng.gamma(0.3, size=(layers, kv, n + 1)).astype(np.float32)
        mass[t, :, :, : n + 1] = m / m.sum(axis=-1, keepdims=True)
        bound[t, :, :, 1:n] = mass[t, :, :, 1:n] + rng.normal(0, 0.02, size=(layers, kv, n - 1))
    return AttentionTrace(block_size=64, dense_layers=2, n_full=n_full, mass=mass, bound=bound)


def test_every_selector_picks_k_distinct_candidates() -> None:
    tr = _synthetic()
    for name, fn in SELECTORS.items():
        chosen = fn(tr, 5)
        assert len(chosen) == tr.steps, name
        for t, c in enumerate(chosen):
            assert c.shape == (2, 3, 5), name
            assert (c >= 1).all() and (c <= tr.n_full[t] - 1).all(), name
            assert all(len(set(row.tolist())) == 5 for row in c.reshape(-1, 5)), name


def test_the_oracle_captures_the_most_mass_and_all_candidates_capture_everything() -> None:
    tr = _synthetic()
    oracle = mass_captured(tr, SELECTORS["oracle"](tr, 6))
    for name in ("quest", "stale_oracle", "window"):
        assert (oracle >= mass_captured(tr, SELECTORS[name](tr, 6)) - 1e-6).all(), name
    everything = mass_captured(tr, SELECTORS["oracle"](tr, int(tr.n_full.min()) - 1))
    # Only steps whose candidate count equals K select every candidate.
    full_steps = tr.n_full == tr.n_full.min()
    np.testing.assert_allclose(everything[full_steps], 1.0, rtol=1e-5)


def test_window_selects_the_most_recent_candidates() -> None:
    tr = _synthetic()
    c = SELECTORS["window"](tr, 3)
    n = int(tr.n_full[0])
    assert c[0][0, 0].tolist() == [n - 1, n - 2, n - 3]


def test_k_outside_the_candidate_range_is_rejected() -> None:
    tr = _synthetic()
    with pytest.raises(ValueError):
        SELECTORS["quest"](tr, int(tr.n_full.min()))


def test_lru_on_a_hand_worked_sequence() -> None:
    # 3 slots seeded with 7, 8, 9. Step 0 wants {1, 9}: 9 hits, 1 evicts the slot holding 7
    # (seeded slots tie at -1 and break by slot order). Step 1 wants {8, 2}: 8 hits; the victim is the
    # older of the slots outside {8}: slot 0 (1, used at step 0) vs slot 2 (9, used at step 0) -> slot 0.
    seq = [np.array([1, 9]), np.array([8, 2]), np.array([1, 2])]
    assert replay_lru(seq, 3, [7, 8, 9]).tolist() == [1, 1, 1]


def _brute_force_min_fetches(seq: list[np.ndarray], n_slots: int, seed: list[int]) -> int:
    """Exhaustive search over every eviction choice: the true optimum under demand fetching."""
    best = [10**9]

    def go(t: int, resident: frozenset[int], cost: int) -> None:
        if cost >= best[0]:
            return
        if t == len(seq):
            best[0] = cost
            return
        cur = frozenset(int(b) for b in seq[t])
        want = cur - resident
        evict = len(want) - (n_slots - len(resident))
        if evict <= 0:
            go(t + 1, resident | want, cost + len(want))
            return
        for out in itertools.combinations(sorted(resident - cur), evict):
            go(t + 1, (resident - set(out)) | want, cost + len(want))

    go(0, frozenset(seed), 0)
    return best[0]


@pytest.mark.parametrize("trial", range(40))
def test_belady_matches_brute_force_and_never_loses_to_lru(trial: int) -> None:
    rng = np.random.default_rng(trial)
    k, n_slots, universe = 2, int(rng.integers(2, 5)), 7
    seq = [rng.choice(np.arange(1, universe), size=k, replace=False) for _ in range(7)]
    seed = list(range(universe - min(n_slots, universe - 1), universe))
    belady = int(replay_belady(seq, n_slots, seed).sum())
    assert belady == _brute_force_min_fetches(seq, n_slots, seed)
    assert belady <= int(replay_lru(seq, n_slots, seed).sum())


def test_with_k_slots_the_victims_are_forced_and_every_rule_agrees() -> None:
    tr = _synthetic()
    chosen = SELECTORS["quest"](tr, 4)
    np.testing.assert_array_equal(replay(tr, chosen, 4, "lru"), replay(tr, chosen, 4, "belady"))


def test_trace_round_trips_through_disk(tmp_path: Path) -> None:
    tr = _synthetic()
    tr.meta = {"kind": "single", "sample": 3}
    tr.save(tmp_path / "t.npz")
    back = AttentionTrace.load(tmp_path / "t.npz")
    assert back.meta == tr.meta and back.block_size == 64 and back.dense_layers == 2
    np.testing.assert_array_equal(back.mass, tr.mass)
    np.testing.assert_array_equal(back.bound, tr.bound)
    np.testing.assert_array_equal(back.n_full, tr.n_full)
