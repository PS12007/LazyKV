"""Exact-size pinned host memory for the tier's host store.

`torch.empty(..., pin_memory=True)` goes through PyTorch's caching host allocator, which rounds each
request up to the next power of two (measured 2026-10-01: a 300 MiB request reserved 512 MiB). For
small staging buffers that is noise. For the host store it is up to 2x the RAM: a 3B model's 64K
store is ~266 MiB per layer, which pins 512 MiB per layer, 13 GiB in all on a 16 GB machine. The
host store is what bounds the tier's context, so it has to be pinned at its real size.

`cudaHostRegister` page-locks an ordinary allocation in place, at its own size, and the result is
as pinned as the caching allocator's (`is_pinned()` is True, and non_blocking copies are async).
"""

from __future__ import annotations

import weakref

import torch


def _unregister(ptr: int) -> None:
    # Ignoring the result: at interpreter exit the CUDA context may already be gone.
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # noqa: BLE001
        pass


def pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """An uninitialized CPU tensor, page-locked at exactly its size, unpinned when it is collected."""
    t = torch.empty(shape, dtype=dtype)
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        return t
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0)
    if rc != torch.cuda.cudart().cudaError.success:
        raise RuntimeError(f"cudaHostRegister of {nbytes} bytes failed: {rc}")
    # Tied to this tensor object, which the tier keeps for the process's life; views are transient.
    weakref.finalize(t, _unregister, t.data_ptr())
    return t
