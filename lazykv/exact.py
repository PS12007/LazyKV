"""Rung 9: exact attention over a tiered cache, by computing the cold part where it lives.

Rungs 6-8 answer "the block is not in VRAM" by not attending to it. That is cheap and it changes
the model's output, which is what the whole accuracy column of this study measures. Rung 9 is the
other answer from the brief's §B1(c): attend to the non-resident blocks **on the CPU**, where they
already are, and merge the two partial outputs with their log-sum-exp normalizers. Softmax is
associative under that rescaling, so the result is the full cache's attention, and only
O(head_dim) bytes come back instead of O(block_bytes).

The set split, per selecting layer and decode step:

- **GPU** attends to what `TieredCache.select` gathered: the sink (block 0), the K selected blocks,
  and the current unsealed block.
- **CPU** attends to every *other* sealed block, 1..n_sealed-1 minus the selected ones.

The two sets are disjoint and their union is the whole sequence, which is the exactness argument
and is what `tests/test_exact.py` checks against a full cache rather than asserting.

Three implementation decisions, each forced by a measurement rather than a preference:

1. **A separate float32 mirror of the sealed KV, contiguous per head.** The tier's pinned host pool
   is bf16 laid out `[blocks, heads, 2, block_size, head_dim]`, so one head's keys are a strided
   batch of small matrices, and bf16 has no CPU GEMM on this machine (Raptor Lake: no AVX512, no
   AMX). Measured on this geometry, one layer's pass costs about 7.6 ms from a contiguous float32
   matrix, about 16 ms from a strided float32 batch, and about 140 ms in bf16. The mirror costs
   host RAM -- about 135 MiB per selecting layer at 32K -- and that cost is the honest price of
   exactness on this hardware, reported rather than hidden.
2. **Masking the resident blocks, not skipping them.** The CPU pass computes scores over every
   sealed block and sets the GPU-held ones to -inf before the softmax. Gathering only the cold
   blocks would make the pass proportional to the budget, but gathering them is the strided case
   above: it costs more than computing the columns and throwing them away. The consequence is that
   rung 9's CPU cost is flat in the budget, which is a finding and not an oversight.
3. **The GPU keeps its own kernel.** The merge needs the GPU side's log-sum-exp, which SDPA does
   not return, so it is recomputed with one extra query-key product over the *selected* set only
   (a few thousand columns). Recomputing that is far cheaper than replacing the pinned attention
   kernel with an explicit one, and it means rung 9's GPU path is bit-identical to rung 6's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lazykv.cache import FullGPUCache
from lazykv.tiered import TieredCache

NEG_INF = float("-inf")


@dataclass
class ExactCounters:
    """What the exact merge costs, kept apart from the tier's own counters so the two add up."""

    merges: int = 0
    cpu_attention_s: float = 0.0  # the CPU partial attention itself
    host_merge_s: float = 0.0  # everything in merge(): the query pull, the GPU lse, the blend
    mirror_s: float = 0.0  # keeping the float32 mirror current at seal time
    boundary_mirror_s: float = 0.0
    cpu_tokens: int = 0  # tokens attended on the CPU, summed over merges (the masked-out ones included)
    cold_tokens: int = 0  # of which actually contributed: the non-resident ones
    returned_bytes: int = 0  # what came back from the CPU: O(head_dim) per query head, the point of (c)


def merge_lse(out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two partial softmax attentions into the attention over their union.

    `out_*` are [..., d] already divided by their own normalizer; `lse_*` are [...] log-sum-exp of
    the scores each one covered. This is the FlashAttention / ring-attention merge, and it is exact
    for disjoint key sets. A partial with an empty key set passes lse = -inf and contributes
    nothing, which is why the weights are computed by subtracting the combined lse rather than by
    exponentiating each one.
    """
    lse = torch.logaddexp(lse_a, lse_b)
    # Both partials empty: logaddexp(-inf, -inf) = -inf, and every weight would be nan.
    finite = torch.isfinite(lse)
    safe = torch.where(finite, lse, torch.zeros_like(lse))
    wa = torch.where(finite, (lse_a - safe).exp(), torch.zeros_like(lse))
    wb = torch.where(finite, (lse_b - safe).exp(), torch.zeros_like(lse))
    return wa.unsqueeze(-1) * out_a + wb.unsqueeze(-1) * out_b, lse


def cpu_partial_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    scaling: float,
    block_size: int,
    excluded: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact attention of `query` over `keys`/`values`, with whole blocks excluded per KV head.

    `keys`/`values` are [kv_heads, n_blocks * block_size, head_dim] float32 and contiguous per head;
    `query` is [kv_heads, groups, head_dim] float32; `excluded` is [kv_heads, e] block indices the
    GPU side already covers. Returns the partial output [kv_heads, groups, head_dim] and its
    log-sum-exp [kv_heads, groups], both float32.

    Excluded columns are set to -inf *before* the row maximum, so a head whose every block is
    excluded returns lse = -inf and an output that `merge_lse` will weight to zero, rather than a
    nan from softmax over an empty set.
    """
    kv_heads, total, head_dim = keys.shape
    n_blocks = total // block_size
    out = torch.empty((kv_heads, query.shape[1], head_dim), dtype=torch.float32)
    lse = torch.empty((kv_heads, query.shape[1]), dtype=torch.float32)
    for j in range(kv_heads):
        scores = torch.matmul(query[j], keys[j].transpose(0, 1))  # [groups, n_blocks * block_size]
        scores *= scaling
        scores.view(-1, n_blocks, block_size)[:, excluded[j]] = NEG_INF
        m = scores.amax(dim=-1, keepdim=True)
        empty = torch.isneginf(m)
        # exp(-inf - -inf) is nan; anchor the empty rows at 0 and zero their weights afterwards.
        probs = (scores - torch.where(empty, torch.zeros_like(m), m)).exp()
        probs = torch.where(empty, torch.zeros_like(probs), probs)
        norm = probs.sum(dim=-1, keepdim=True)
        out[j] = torch.matmul(probs / norm.clamp_min(torch.finfo(torch.float32).tiny), values[j])
        lse[j] = torch.where(empty, torch.full_like(m, NEG_INF), m + norm.log()).squeeze(-1)
    return out, lse


@dataclass
class _Mirror:
    """One selecting layer's float32 copy of its sealed KV, contiguous per KV head."""

    keys: torch.Tensor  # [kv_heads, cap_blocks * block_size, head_dim]
    values: torch.Tensor

    @classmethod
    def allocate(cls, kv_heads: int, cap_blocks: int, block_size: int, head_dim: int) -> _Mirror:
        shape = (kv_heads, cap_blocks * block_size, head_dim)
        return cls(torch.empty(shape, dtype=torch.float32), torch.empty(shape, dtype=torch.float32))

    def host_bytes(self) -> int:
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def write(self, pairs: torch.Tensor, first_block: int, block_size: int) -> None:
        """Write `pairs` [n_blocks, kv_heads, 2, block_size, head_dim] (host, any dtype) at `first_block`."""
        n = pairs.shape[0]
        t0, t1 = first_block * block_size, (first_block + n) * block_size
        # [n, h, 2, bs, d] -> [h, 2, n * bs, d]: one permute, then a single cast per side.
        laid = pairs.permute(1, 2, 0, 3, 4).reshape(pairs.shape[1], 2, n * block_size, pairs.shape[-1])
        self.keys[:, t0:t1] = laid[:, 0].float()
        self.values[:, t0:t1] = laid[:, 1].float()


class ExactTieredCache(TieredCache):
    """Rung 9: rung 6's tier, made exact by attending to the cold blocks on the CPU and merging.

    Everything about residency, selection, fetching and the GPU attention is rung 6's, unchanged --
    which is the point, because it makes the latency difference between the two rungs the price of
    exactness and nothing else. What this class adds is the float32 mirror, the CPU pass over the
    blocks the GPU did not gather, and the log-sum-exp merge.
    """

    def __init__(self, full: FullGPUCache, *args: Any, mirrors: list[_Mirror] | None = None, threads: int | None = 8, **kwargs: Any) -> None:
        if kwargs.get("quant") is not None:
            # Rung 9 exists to remove an approximation; a quantized warm tier would reintroduce one
            # and the exactness test could not be written.
            raise ValueError("rung 9 is exact; it cannot be built on a quantized host pool")
        if kwargs.get("prefetch"):
            # A prefetch changes which blocks are resident when the layer runs, not which are
            # attended: rung 9 attends to all of them either way. It would only add a variable.
            raise ValueError("rung 9 does not prefetch; the cold set is attended, not fetched")
        super().__init__(full, *args, **kwargs)
        self.policy_name = "tiered_exact"
        self.exact = ExactCounters()
        tiers = list(self.tiers.values())
        t = tiers[0]
        if mirrors is None:
            mirrors = [_Mirror.allocate(t.h, t.cap_blocks, t.bs, t.d) for _ in tiers]
        self.mirrors = {i: m for i, m in zip(sorted(self.tiers), mirrors)}
        t0 = time.perf_counter()
        for i, tier in self.tiers.items():
            # The tier has already written every sealed prompt block to its pinned host pool, so the
            # mirror is built from those exact bytes rather than from the GPU a second time. The CPU
            # then attends to what a fetch would have produced, which is what exactness means here.
            self.mirrors[i].write(tier.host[: tier.n_sealed], 0, tier.bs)
        self.exact.boundary_mirror_s = time.perf_counter() - t0
        self._sealed_at_entry = {i: tier.n_sealed for i, tier in self.tiers.items()}
        self._chosen: dict[int, np.ndarray] = {}
        # torch's CPU thread count is process-global, so it is set here and restored by close().
        # 8 was the fastest on this geometry; more threads lose on a single-channel DIMM.
        self._threads_before = torch.get_num_threads()
        self.threads = threads or self._threads_before
        torch.set_num_threads(self.threads)

    # -- keeping the mirror current -------------------------------------------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tier = self.tiers.get(layer_idx)
        before = tier.n_sealed if tier is not None else 0
        out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if tier is not None and tier.n_sealed != before:
            t0 = time.perf_counter()
            self.mirrors[layer_idx].write(tier.host[before : tier.n_sealed], before, tier.bs)
            self.exact.mirror_s += time.perf_counter() - t0
        return out

    # -- the merge ------------------------------------------------------------------------------

    def select(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        out = super().select(layer_idx, query, key, value)
        if out is not None:
            self._chosen[layer_idx] = self._last_chosen
        return out

    def merge(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, out: torch.Tensor, scaling: float | None, mask: torch.Tensor | None) -> torch.Tensor:
        """Fold the cold blocks into `out`, which is attention over the gathered (hot) set alone.

        `key` is the gathered key set the GPU attended to, so the GPU side's log-sum-exp is
        recomputed from it rather than from the whole cache. Returns [1, n_q, 1, d] in `out`'s dtype.
        """
        tier = self.tiers.get(layer_idx)
        chosen = self._chosen.pop(layer_idx, None)
        if tier is None or chosen is None or query.shape[-2] != 1:
            return out
        t0 = time.perf_counter()
        scale = scaling if scaling is not None else tier.d**-0.5
        n_q, h = query.shape[1], tier.h
        groups = n_q // h
        # Pulled before the GPU log-sum-exp is enqueued: this copy blocks on everything already
        # queued, and there is no reason to make it block on work that has not been launched yet.
        q_cpu = query[0, :, 0].float().cpu().view(h, groups, tier.d)

        lse_gpu = self._gpu_lse(query, key, scale, mask)  # [1, n_q], enqueued, not yet read

        n = tier.n_sealed
        m = self.mirrors[layer_idx]
        t1 = time.perf_counter()
        out_cpu, lse_cpu = cpu_partial_attention(m.keys[:, : n * tier.bs], m.values[:, : n * tier.bs], q_cpu, scale, tier.bs, self._excluded(chosen))
        self.exact.cpu_attention_s += time.perf_counter() - t1

        dev = out.device
        merged, _ = merge_lse(
            out[0, :, 0].float(),
            lse_gpu.view(n_q).float(),
            out_cpu.view(n_q, tier.d).to(dev, non_blocking=True),
            lse_cpu.view(n_q).to(dev, non_blocking=True),
        )
        c = self.exact
        c.merges += 1
        c.cpu_tokens += n * tier.bs * h
        c.cold_tokens += (n - 1 - chosen.shape[1]) * tier.bs * h
        c.returned_bytes += out_cpu.numel() * out_cpu.element_size() + lse_cpu.numel() * lse_cpu.element_size()
        c.host_merge_s += time.perf_counter() - t0
        return merged.to(out.dtype).view(1, n_q, 1, tier.d)

    @staticmethod
    def _excluded(chosen: np.ndarray) -> np.ndarray:
        """Blocks per KV head the CPU must not attend to: the sink and the selected set.

        The current (unsealed) block is not in the mirror at all, so it needs no exclusion, and
        blocks past n_sealed are outside the slice handed to the CPU pass.
        """
        return np.concatenate([np.zeros((chosen.shape[0], 1), dtype=chosen.dtype), chosen], axis=1)

    @staticmethod
    def _gpu_lse(query: torch.Tensor, key: torch.Tensor, scale: float, mask: torch.Tensor | None) -> torch.Tensor:
        """Log-sum-exp of the scores the GPU kernel used, which SDPA does not return.

        One extra query-key product over the *gathered* set: at a 25% budget that is a few thousand
        columns per head, against the whole cache the CPU side walks. Computed in float32 because it
        is one half of a merge whose other half is float32.
        """
        h_kv = key.shape[1]
        q = query.view(1, h_kv, query.shape[1] // h_kv, query.shape[-1]).float()
        scores = torch.matmul(q, key.float().transpose(-1, -2)) * scale  # [1, h_kv, groups, kv_len]
        if mask is not None:
            scores = scores + mask.float().view(1, 1, 1, -1)
        return torch.logsumexp(scores, dim=-1).view(1, -1)

    # -- reporting ------------------------------------------------------------------------------

    def close(self) -> None:
        super().close()
        torch.set_num_threads(self._threads_before)

    def mirror_bytes(self) -> int:
        """Pageable host RAM the exactness costs, on top of the tier's pinned pools."""
        return sum(m.host_bytes() for m in self.mirrors.values())

    def exact_counters_dict(self) -> dict[str, Any]:
        d = dict(self.exact.__dict__)
        d["threads"] = self.threads
        d["mirror_bytes"] = self.mirror_bytes()
        return d


def allocate_mirrors(n_layers: int, kv_heads: int, head_dim: int, block_size: int, capacity_tokens: int) -> list[_Mirror]:
    """Float32 mirrors, one per selecting layer, allocated once per process like the pinned pools.

    At 32K tokens and this geometry each one is about 135 MiB, so building them per condition would
    time the allocation rather than the policy, and would fragment a 16 GiB host.
    """
    return [_Mirror.allocate(kv_heads, -(-capacity_tokens // block_size), block_size, head_dim) for _ in range(n_layers)]
