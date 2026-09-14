"""Keep every benchmark inside dedicated VRAM, and prove it per condition.

Phase 1 found that on Windows (WDDM), when this process's CUDA allocations outgrow
dedicated VRAM the driver silently backs them with shared system memory instead of raising
out-of-memory. A 64K run went on for minutes at a fraction of normal speed with 1.2 GB
spilled to host RAM. The allocator's *reserved* pool, not live tensors, drove it: peak
allocated was ~4.7 GiB while reserved grew past what was free.

Two defenses:
- cap PyTorch's caching allocator at the dedicated VRAM actually free at startup, so it
  frees cached blocks and retries, and raises OOM rather than spilling;
- read the per-process "GPU Process Memory" performance counters (dedicated and shared)
  around each condition, so any spill is recorded and the condition can be flagged.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass

import torch


def cap_allocator_to_dedicated(margin_bytes: int = 256 * 2**20) -> dict[str, int | float]:
    """Limit the caching allocator to what is free now plus what it already holds."""
    free, total = torch.cuda.mem_get_info()
    limit = torch.cuda.memory_reserved() + free - margin_bytes
    fraction = max(0.05, min(1.0, limit / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {"free_bytes_at_cap": free, "total_bytes": total, "margin_bytes": margin_bytes, "fraction": fraction, "limit_bytes": int(fraction * total)}


@dataclass(frozen=True)
class GpuProcessMemory:
    dedicated_bytes: int | None
    shared_bytes: int | None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


def process_gpu_memory() -> GpuProcessMemory:
    """Dedicated and shared GPU memory of this process from Windows performance counters.

    Takes about a second (spawns PowerShell); call only outside timed regions.
    """
    if os.name != "nt":
        return GpuProcessMemory(None, None)
    pid = os.getpid()
    script = (
        f"$s=(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples; "
        "'D=' + [int64](($s | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Sum).Sum); "
        "'S=' + [int64](($s | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum)"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return GpuProcessMemory(None, None)
    vals = dict(line.strip().split("=", 1) for line in out.splitlines() if "=" in line)
    try:
        return GpuProcessMemory(int(vals["D"]), int(vals["S"]))
    except (KeyError, ValueError):
        return GpuProcessMemory(None, None)


def allocator_counters() -> dict[str, int]:
    stats = torch.cuda.memory_stats()
    # num_alloc_retries > 0 means the cap forced cache flushes (slow); num_ooms counts failures.
    return {"num_alloc_retries": int(stats.get("num_alloc_retries", 0)), "num_ooms": int(stats.get("num_ooms", 0))}
