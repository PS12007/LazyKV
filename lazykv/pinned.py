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

import ctypes.util
import weakref
from pathlib import Path
from typing import Any

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


_CUDART: Any = None


def _cudart() -> Any:
    """The CUDA runtime DLL that torch ships, for the one call torch.cuda.cudart() does not expose."""
    global _CUDART  # noqa: PLW0603 -- a loaded library handle, not mutable state
    if _CUDART is None:
        import ctypes

        lib_dir = Path(torch.__file__).parent / "lib"
        names = sorted(lib_dir.glob("cudart64_*.dll")) + sorted(lib_dir.glob("libcudart.so*"))
        if names:
            _CUDART = ctypes.CDLL(str(names[-1]))
        else:  # Linux wheels ship the runtime in nvidia-cuda-runtime, found by the loader
            found = ctypes.util.find_library("cudart")
            if found is None:
                raise RuntimeError("CUDA runtime library not found")
            _CUDART = ctypes.CDLL(found)
    return _CUDART


class _DeviceArray:
    """The minimal `__cuda_array_interface__` torch.as_tensor needs to wrap a device pointer."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}


def device_view(t: torch.Tensor) -> torch.Tensor:
    """A CUDA tensor aliasing the pinned CPU tensor `t`: GPU kernels that read it read host RAM over PCIe.

    Page-locked memory is mapped into the device's address space under unified addressing; this only
    asks the runtime for the device-side address (which on this Windows driver is not the host
    address: cudaDevAttrCanUseHostPointerForRegisteredMem is 0). Reads are zero-copy: no transfer is
    launched, the kernel's loads cross the link. The caller keeps `t` alive for as long as the view.
    """
    import ctypes

    if not t.is_pinned() or not t.is_contiguous():
        raise ValueError("device_view needs a contiguous pinned tensor")
    torch.cuda.init()
    ptr = ctypes.c_void_p()
    rc = _cudart().cudaHostGetDevicePointer(ctypes.byref(ptr), ctypes.c_void_p(t.data_ptr()), 0)
    if rc != 0 or ptr.value is None:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {rc}")
    nbytes = t.numel() * t.element_size()
    raw = torch.as_tensor(_DeviceArray(ptr.value, nbytes), device="cuda")
    return raw.view(t.dtype).view(t.shape)
