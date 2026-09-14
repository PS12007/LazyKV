"""Host CPU scheduling facts and controls (Windows, hybrid P/E CPU).

Why this matters here: batch-1 decode of a 1B model is host-bound. Most of a layer's span
on the GPU timeline is Python launching kernels, so decode latency moves with whatever the
OS does to the Python thread: which core class it runs on, and whether Windows applies
power throttling (EcoQoS) to it. A process launched detached with no foreground window is
exactly what Windows' heuristics may classify as background work.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

# PROCESS_INFORMATION_CLASS.ProcessPowerThrottling and PROCESS_POWER_THROTTLING_* from processthreadsapi.h.
_PROCESS_POWER_THROTTLING = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


class _GroupAffinity(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", ctypes.c_ushort), ("Reserved", ctypes.c_ushort * 3)]


class _ProcessorRelationship(ctypes.Structure):
    _fields_ = [
        ("Flags", ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte),
        ("Reserved", ctypes.c_ubyte * 20),
        ("GroupCount", ctypes.c_ushort),
        ("GroupMask", _GroupAffinity * 1),
    ]


def _kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise OSError("Windows only")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    return k32


def core_class_masks() -> dict[int, int]:
    """EfficiencyClass -> logical-processor affinity mask (group 0). Higher class = P-cores."""
    k32 = _kernel32()
    length = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))  # 0 = RelationProcessorCore
    buf = ctypes.create_string_buffer(length.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(length)):
        raise OSError("GetLogicalProcessorInformationEx failed")
    masks: dict[int, int] = {}
    off = 0
    while off < length.value:
        size = ctypes.c_ulong.from_buffer(buf, off + 4).value
        rel = _ProcessorRelationship.from_buffer(buf, off + 8)
        masks[rel.EfficiencyClass] = masks.get(rel.EfficiencyClass, 0) | rel.GroupMask[0].Mask
        off += size
    return masks


def set_power_throttling(opt_out: bool) -> None:
    """Opt this process out of Windows execution-speed throttling, or return control to the OS.

    opt_out=True sets ControlMask=EXECUTION_SPEED, StateMask=0 ("never throttle", i.e. HighQoS).
    opt_out=False clears both masks, which is the system-managed default.
    """
    k32 = _kernel32()
    state = _PowerThrottlingState(_PROCESS_POWER_THROTTLING_CURRENT_VERSION, _PROCESS_POWER_THROTTLING_EXECUTION_SPEED if opt_out else 0, 0)
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    if not k32.SetProcessInformation(k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)):
        raise OSError(f"SetProcessInformation(ProcessPowerThrottling) failed: {ctypes.get_last_error()}")


def current_processor_number() -> int:
    """Logical processor the calling thread is running on right now."""
    return int(_kernel32().GetCurrentProcessorNumber())
