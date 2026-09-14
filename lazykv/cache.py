"""Full-GPU, preallocated KV cache: policy ladder rung 1 (the exact reference).

Why not transformers' DynamicCache: it grows by torch.cat on every update, so each decode
step copies the whole layer's KV. At long context that copy is bandwidth-bound work the
reference would pay and a block-managed cache would not, which biases every comparison.

Why not StaticCache: it returns max-length tensors and relies on a mask to hide unused
slots, so attention scans the full preallocated length every step and cannot use the
unmasked cuDNN decode path. This layer returns views of only the filled prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PreallocatedLayer(CacheLayerMixin):
    """One layer's KV in a single preallocated [1, kv_heads, max_len, head_dim] tensor pair."""

    def __init__(self, max_len: int, **kwargs: object) -> None:
        super().__init__()
        self.max_len = max_len
        self.length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        b, h, _, d = key_states.shape
        if b != 1:
            raise ValueError("LazyKV is batch-size-1 only (v1 scope)")
        self.dtype, self.device = key_states.dtype, key_states.device
        # Allocated once, up front: expandable_segments is unsupported on this platform and
        # a growing allocation would fragment the caching allocator mid-run.
        self.keys = torch.empty((1, h, self.max_len, d), dtype=key_states.dtype, device=key_states.device)
        self.values = torch.empty((1, value_states.shape[1], self.max_len, value_states.shape[3]), dtype=value_states.dtype, device=value_states.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        end = self.length + n
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > max_len {self.max_len}")
        self.keys[:, :, self.length : end].copy_(key_states)
        self.values[:, :, self.length : end].copy_(value_states)
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.length + query_length, 0

    def get_seq_length(self) -> int:
        return self.length

    def get_max_length(self) -> int:
        return self.max_len

    def reset(self) -> None:
        # Contents need not be zeroed: only the [:length] prefix is ever read.
        self.length = 0

    def resident_bytes(self) -> int:
        if not self.is_initialized:
            return 0
        per_token = (self.keys[0, :, 0].numel() * self.keys.element_size()) + (self.values[0, :, 0].numel() * self.values.element_size())
        return per_token * self.length


class FullGPUCache(Cache):
    """Every token of every layer stays GPU-resident. Exact by construction."""

    def __init__(self, num_layers: int, max_len: int) -> None:
        super().__init__(layers=[PreallocatedLayer(max_len) for _ in range(num_layers)])
        self.max_len = max_len

    def stats(self) -> CacheStats:
        layers = [l for l in self.layers if isinstance(l, PreallocatedLayer)]
        resident = sum(l.resident_bytes() for l in layers)
        allocated = sum(
            (l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()) for l in layers if l.is_initialized
        )
        return CacheStats(
            seq_len=layers[0].get_seq_length() if layers else 0,
            gpu_resident_kv_bytes=resident,
            host_kv_bytes=0,
            gpu_allocated_kv_bytes=allocated,
        )


@dataclass(frozen=True)
class CacheStats:
    seq_len: int
    gpu_resident_kv_bytes: int
    host_kv_bytes: int
    gpu_allocated_kv_bytes: int

    @property
    def gpu_residency_fraction(self) -> float:
        total = self.gpu_resident_kv_bytes + self.host_kv_bytes
        return 1.0 if total == 0 else self.gpu_resident_kv_bytes / total
