"""Exact-size pinning for the host store (lazykv/pinned.py)."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_pinned_empty_is_pinned_without_the_caching_allocator() -> None:
    from lazykv.pinned import pinned_empty

    before = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
    t = pinned_empty((3, 5 * 2**20 + 7), torch.uint8)  # not a power of two
    assert t.is_pinned()
    # The caching host allocator is not involved, so nothing is rounded up there.
    assert torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) == before
    t.random_(0, 255)
    d = torch.empty_like(t, device="cuda")
    d.copy_(t, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(d.cpu(), t)


@cuda
def test_host_pools_use_exact_size_pinning() -> None:
    from lazykv.tiered import allocate_host_pools

    before = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
    pools = allocate_host_pools(2, heads=2, head_dim=16, block_size=8, capacity_tokens=100)
    assert all(p.is_pinned() for p in pools)
    assert torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) == before
