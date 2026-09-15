"""Accumulated attention mass per prompt block, for H2O-style eviction (policy ladder rung 4).

H2O (arXiv 2306.14048) ranks KV entries by attention accumulated over the queries that have
attended to them. Its reference code sums the prompt's full attention matrix over batch and query
positions, per head (`attn_score_cache.sum(0).sum(1)` in h2o_hf/utils_real_drop/modify_llama.py).

LazyKV cannot afford the full matrix at 32K: column sums over every prompt query would cost minutes
per prompt, and the exact prefill uses fused kernels that never materialize attention weights. This
module estimates the same quantity from a stride sample of prompt queries (every `stride`-th
position), with each sampled query causally masked to the keys it could see. The estimate is
unbiased for the sum up to a constant factor, which does not change a ranking. The deviation is
stated wherever results are reported.

Scores are summed over heads because the block pool's slot table is shared by all heads of a layer;
H2O's reference code keeps them per head.
"""

from __future__ import annotations

import math

import torch


class PrefillScorer:
    """Passed as `lazykv_observer` during prefill; afterwards `block_scores(layer)` ranks prompt blocks."""

    def __init__(self, num_layers: int, max_len: int, device: torch.device, stride: int = 32, query_batch: int = 16) -> None:
        self.stride, self.query_batch = stride, query_batch
        self.mass = [torch.zeros(max_len, dtype=torch.float32, device=device) for _ in range(num_layers)]
        self.sampled_queries = [0] * num_layers

    def observe(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor) -> None:
        q_len, kv_len = query.shape[-2], key.shape[-2]
        if q_len == 1:
            return  # decode-time accumulation belongs to the policy, not the prefill scorer
        n_q, n_kv, d = query.shape[1], key.shape[1], query.shape[-1]
        groups = n_q // n_kv
        first_pos = kv_len - q_len  # absolute position of this chunk's first query
        rows = torch.arange((-first_pos) % self.stride, q_len, self.stride, device=query.device)
        if rows.numel() == 0:
            return
        k = key.transpose(-1, -2)  # [1, kv, d, L]
        acc = self.mass[layer_idx]
        for start in range(0, rows.numel(), self.query_batch):
            r = rows[start : start + self.query_batch]
            s = r.numel()
            # Query groups ride in the matrix rows so the keys are never broadcast-expanded.
            q = query[:, :, r].reshape(1, n_kv, groups * s, d)  # [1, kv, g*s, d]
            scores = (q @ k).float() / math.sqrt(d)  # [1, kv, g*s, L]
            # Lower-right causality: query at absolute position p sees keys [0, p].
            positions = (first_pos + r).repeat(groups)  # row order of the reshape: head groups, then samples
            visible = torch.arange(kv_len, device=query.device)[None, :] <= positions[:, None]  # [g*s, L]
            scores = scores.masked_fill(~visible, float("-inf"))
            acc[:kv_len] += torch.softmax(scores, dim=-1).sum(dim=(0, 1, 2))
        self.sampled_queries[layer_idx] += rows.numel()

    def reset(self) -> None:
        for m in self.mass:
            m.zero_()
        self.sampled_queries = [0] * len(self.mass)

    def full_sum_estimate(self, layer_idx: int, length: int) -> torch.Tensor:
        """Mass per prompt position scaled by the stride: an estimate of H2O's sum over every query.

        Scaling matters only because decode adds unscaled per-step mass to the same totals.
        """
        return self.mass[layer_idx][:length] * self.stride

    def block_scores(self, layer_idx: int, n_blocks: int, block_size: int) -> torch.Tensor:
        """Mass per full prompt block, [n_blocks] on the GPU."""
        return self.mass[layer_idx][: n_blocks * block_size].view(n_blocks, block_size).sum(-1)
