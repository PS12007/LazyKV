"""Attention traces: what each decode step actually attended to, recorded for offline replay.

A trace is recorded on the full-cache reference decode. At every decode step, in every selecting
layer, and for every KV head, it keeps two things per block:

- the **true attention mass**: the softmax over the whole KV, in float32, averaged over the head's
  query group and summed over the block's tokens. Block 0 is the sink, the last block is the one
  being filled.
- the **Quest bound** (`selection.quest_bound`) on each candidate block, exactly what rungs 5 to 8
  rank by.

With both, any selector that picks K blocks can be scored offline by the mass it would have
captured, against an oracle that picks the K blocks of highest true mass, and any slot-replacement
rule can be replayed against Belady (lazykv/residency_sim.py). Recording on the reference
trajectory is a choice, stated here: a selector that changes the answer would have produced
different later queries. For the first answer token, and for every teacher-forced position, the
two trajectories are the same.

The recorder returns `None` from `select`, so the model attends densely and the decode is the full
cache's own. It costs one extra float32 score pass per selecting layer per step, which is why it is
opt-in and never used in a timed run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lazykv.cache import FullGPUCache
from lazykv.selection import QUEST_DENSE_LAYERS, QuestView, quest_bound


@dataclass
class AttentionTrace:
    """One prompt's decode, as arrays indexed [step, selecting layer, KV head, block].

    `n_full[t]` is the number of full blocks before the current block at step t, so the current
    block's id is `n_full[t]` and the candidates are ids 1 .. n_full[t] - 1. `mass` is zero past the
    current block; `bound` is -inf outside the candidates.
    """

    block_size: int
    dense_layers: int
    n_full: np.ndarray  # [steps] int64
    mass: np.ndarray  # [steps, layers, kv, blocks] float32
    bound: np.ndarray  # [steps, layers, kv, blocks] float32
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.n_full.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, n_full=self.n_full, mass=self.mass, bound=self.bound,
                 header=np.array(json.dumps({"block_size": self.block_size, "dense_layers": self.dense_layers, "meta": self.meta})))

    @classmethod
    def load(cls, path: Path) -> AttentionTrace:
        with np.load(path) as z:
            header = json.loads(str(z["header"]))
            return cls(header["block_size"], header["dense_layers"], z["n_full"], z["mass"], z["bound"], header["meta"])


class TraceRecorder(QuestView):
    """A decode-time view over a prefilled FullGPUCache that records attention and attends densely.

    Built on QuestView for its block metadata (the same min/max the selecting rungs keep), so the
    recorded bound is the one they rank by, not a reimplementation of it.
    """

    def __init__(self, full: FullGPUCache, block_size: int, dense_layers: int = QUEST_DENSE_LAYERS) -> None:
        super().__init__(full, k_blocks=1, block_size=block_size, dense_layers=dense_layers)
        self.policy_name = "trace"
        # Per step, per selecting layer: (n_full, mass [kv, n_full + 1], bound [kv, n_full - 1]),
        # left on the GPU until the decode ends so recording adds no host sync.
        self._steps: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = []

    def select(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        if query.shape[-2] != 1 or layer_idx < self.dense_layers:
            return None
        if layer_idx == self.dense_layers:
            self._steps.append([])
        bs = self.block_size
        length = key.shape[2]
        n_full, fill = divmod(length, bs)
        if fill == 0:  # as in QuestView: the block that just completed is still the current block
            n_full, fill = n_full - 1, bs
        st = self._state[layer_idx]
        self._refresh_metadata(layer_idx, self.full.layers[layer_idx].keys, n_full * bs)
        assert st.kmin is not None and st.kmax is not None
        n_kv, d = key.shape[1], query.shape[-1]
        bound = quest_bound(query, st.kmin[:, :, 1:n_full], st.kmax[:, :, 1:n_full])[0].float()
        q = query[0, :, 0].float().view(n_kv, -1, d)  # [kv, g, d]
        scores = torch.einsum("hgd,hld->hgl", q, key[0].float()) * d**-0.5
        probs = torch.softmax(scores, dim=-1).mean(dim=1)  # [kv, L]
        padded = torch.zeros((n_kv, (n_full + 1) * bs), dtype=torch.float32, device=probs.device)
        padded[:, :length] = probs
        mass = padded.view(n_kv, n_full + 1, bs).sum(dim=-1)
        self._steps[-1].append((n_full, mass, bound))
        return None

    def trace(self, meta: dict[str, Any] | None = None) -> AttentionTrace:
        """Collect the recorded steps into host arrays. Syncs the device."""
        if not self._steps:
            raise RuntimeError("nothing recorded: decode at least one step through this cache")
        n_layers, n_kv = len(self._steps[0]), self._steps[0][0][1].shape[0]
        n_full = np.array([s[0][0] for s in self._steps], dtype=np.int64)
        width = int(n_full.max()) + 1
        mass = np.zeros((len(self._steps), n_layers, n_kv, width), dtype=np.float32)
        bound = np.full_like(mass, -np.inf)
        for t, layers in enumerate(self._steps):
            for li, (n, m, b) in enumerate(layers):
                mass[t, li, :, : n + 1] = m.cpu().numpy()
                bound[t, li, :, 1:n] = b.cpu().numpy()
        return AttentionTrace(self.block_size, self.dense_layers, n_full, mass, bound, dict(meta or {}))
