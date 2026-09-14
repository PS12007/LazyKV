"""Phase 0 go/no-go microbenchmarks (docs/BRIEF.md section B2).

Measures, on this machine:
  1. PCIe H2D / D2H bandwidth, pageable vs pinned, 64 KiB .. 256 MiB
  2. VRAM bandwidth (device-to-device copy) and decode-attention KV scan rate
  3. (derived in scripts/analyze_phase0.py) the VRAM : PCIe ratio
  4. Async overlap of a compute kernel and a pinned copy on separate streams
  5. CPU-side attention throughput (option (c) in B1), with explicit thread counts
  6. NVMe unbuffered sequential and random read bandwidth

Runtime is several minutes: launch detached (CLAUDE.md rule 4), e.g.
    .venv\\Scripts\\python.exe scripts\\00_feasibility.py > logs\\feasibility.log 2>&1
Use --quick for a <90 s smoke test that writes to results/phase0/feasibility_quick/.

Units: bandwidth in bytes/s (docs render GB/s = 1e9 B/s); sizes in bytes (docs use IEC).
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harness import sysinfo  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from harness.telemetry import TelemetryLogger  # noqa: E402

log = logging.getLogger("feasibility")

KiB = 1024
MiB = 1024**2
GiB = 1024**3


@dataclass(frozen=True)
class Config:
    quick: bool
    seed: int = 0
    warmup: int = 3
    repeats: int = 7

    @property
    def pcie_sizes(self) -> list[int]:
        exps = range(0, 13, 4) if self.quick else range(13)
        return [64 * KiB * 2**i for i in exps]  # 64 KiB .. 256 MiB

    @property
    def reps(self) -> int:
        return 3 if self.quick else self.repeats


def sync() -> None:
    torch.cuda.synchronize()


def timed_wall(fn: Callable[[], None]) -> float:
    """Wall time of fn with deliberate synchronization on both sides.

    Wall time (not CUDA events alone) is the primary metric for transfers because a
    pageable copy spends much of its time in a host-side staging memcpy that a device
    event pair would also span, but that a pure device timer can misattribute. CUDA
    events are recorded alongside as a cross-check.
    """
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def timed_events(fn: Callable[[], None]) -> tuple[float, float]:
    """(wall_s, cuda_event_s) for fn on the current stream."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sync()
    t0 = time.perf_counter()
    start.record()
    fn()
    end.record()
    sync()
    wall = time.perf_counter() - t0
    return wall, start.elapsed_time(end) / 1e3


# --------------------------------------------------------------------------------------
# 1. PCIe
# --------------------------------------------------------------------------------------


def bench_pcie(cfg: Config, tel: TelemetryLogger) -> dict[str, Any]:
    tel.mark("pcie_start")
    sizes = cfg.pcie_sizes
    bufs: dict[int, dict[str, torch.Tensor]] = {}
    for s in sizes:
        pageable = torch.empty(s, dtype=torch.uint8)
        pageable.random_(0, 255)  # touch every page so first-touch faults are not timed
        pinned = torch.empty(s, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(pageable)
        dev = torch.empty(s, dtype=torch.uint8, device="cuda")
        dev.copy_(pinned)
        bufs[s] = {
            "pageable": pageable,
            "pinned": pinned,
            "dev": dev,
            "pageable_dst": torch.empty_like(pageable).fill_(0),
            "pinned_dst": torch.empty(s, dtype=torch.uint8, pin_memory=True).fill_(0),
        }
    sync()

    def make_op(size: int, mode: str, direction: str) -> Callable[[], None]:
        b = bufs[size]
        if direction == "h2d":
            src = b[mode]
            dst = b["dev"]
        else:
            src = b["dev"]
            dst = b[f"{mode}_dst"]
        nb = mode == "pinned"
        return lambda: dst.copy_(src, non_blocking=nb)

    # Each timed sample does k copies so small sizes are not below timer/WDDM jitter.
    # Per-copy overhead is included on purpose: it is exactly what small blocks pay.
    target_bytes = 64 * MiB if cfg.quick else 192 * MiB
    cells = [(s, m, d) for s in sizes for m in ("pageable", "pinned") for d in ("h2d", "d2h")]
    samples: dict[tuple[int, str, str], list[tuple[float, float]]] = {c: [] for c in cells}
    copies_per_sample = {s: max(1, min(4000, math.ceil(target_bytes / s))) for s in sizes}

    rng = random.Random(cfg.seed)
    # Warmup rounds are discarded; also brings the PCIe link out of its idle low-gen state.
    for rep in range(cfg.warmup + cfg.reps):
        order = cells[:]
        rng.shuffle(order)  # interleave + randomize to decorrelate conditions from thermals
        for cell in order:
            size, mode, direction = cell
            op = make_op(size, mode, direction)
            k = copies_per_sample[size]

            def run(op: Callable[[], None] = op, k: int = k) -> None:
                for _ in range(k):
                    op()

            wall, ev = timed_events(run)
            if rep >= cfg.warmup:
                samples[cell].append((wall, ev))
        log.info("pcie rep %d/%d done", rep + 1, cfg.warmup + cfg.reps)

    rows = []
    for (size, mode, direction), xs in samples.items():
        k = copies_per_sample[size]
        wall_bps = [size * k / w for w, _ in xs]
        ev_bps = [size * k / e for _, e in xs if e > 0]
        per_copy_s = [w / k for w, _ in xs]
        rows.append(
            {
                "size_bytes": size,
                "mode": mode,
                "direction": direction,
                "copies_per_sample": k,
                "wall_bps": summarize(wall_bps).to_dict(),
                "event_bps": summarize(ev_bps).to_dict() if ev_bps else None,
                "seconds_per_copy": summarize(per_copy_s).to_dict(),
            }
        )
    rows.sort(key=lambda r: (r["direction"], r["mode"], r["size_bytes"]))
    del bufs
    torch.cuda.empty_cache()
    tel.mark("pcie_end")
    return {"sizes_bytes": sizes, "warmup": cfg.warmup, "repeats": cfg.reps, "rows": rows}


# --------------------------------------------------------------------------------------
# 2. VRAM bandwidth and decode-attention scan rate
# --------------------------------------------------------------------------------------


def bench_vram(cfg: Config, tel: TelemetryLogger) -> dict[str, Any]:
    tel.mark("vram_start")
    gpu_spinup()
    free, _ = torch.cuda.mem_get_info()
    sizes = [64 * MiB, 256 * MiB] if cfg.quick else [64 * MiB, 256 * MiB, 1 * GiB]
    sizes = [s for s in sizes if 2 * s < 0.8 * free]
    copy_rows = []
    for s in sizes:
        src = torch.empty(s, dtype=torch.uint8, device="cuda").random_(0, 255)
        dst = torch.empty_like(src).zero_()
        for _ in range(cfg.warmup):
            timed_events(lambda: dst.copy_(src))
        xs = [timed_events(lambda: dst.copy_(src)) for _ in range(cfg.reps)]
        copy_rows.append(
            {
                "size_bytes": s,
                "wall_bps": summarize([s / w for w, _ in xs]).to_dict(),
                "event_bps": summarize([s / e for _, e in xs]).to_dict(),
            }
        )
        del src, dst
        torch.cuda.empty_cache()

    tel.mark("vram_end")
    return {"d2d_copy": copy_rows, "decode_attention_scan": bench_decode_attention(cfg, tel)}


def bench_decode_attention(cfg: Config, tel: TelemetryLogger) -> dict[str, Any]:
    """GPU decode attention time per layer, per SDPA backend.

    Decode attention is the operation option (a) would have to feed. Backends are timed
    explicitly because on this Windows build FlashAttention is not compiled in and the
    default GQA dispatch lands on the math kernel, which would understate the GPU by an
    order of magnitude and make PCIe look artificially competitive. Every backend's output
    is checked against the math reference so a fast-but-wrong kernel cannot win.
    Geometry is Llama-3.2-1B: 32 query heads, 8 KV heads, head_dim 64, bf16, 1 query token.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    tel.mark("attn_start")
    gpu_spinup()
    F = torch.nn.functional.scaled_dot_product_attention
    n_q, n_kv, hd, group = 32, 8, 64, 4
    ctx_lens = [16_384, 65_536] if cfg.quick else [4_096, 16_384, 65_536, 262_144]

    def variants(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, Callable[[], torch.Tensor]]:
        qg = q.view(1, n_kv, group, 1, hd)
        ku, vu = k.unsqueeze(2), v.unsqueeze(2)

        def manual() -> torch.Tensor:
            s = torch.matmul(qg, ku.transpose(-1, -2)) / math.sqrt(hd)
            return torch.matmul(torch.softmax(s.float(), -1).to(v.dtype), vu).view(1, n_q, 1, hd)

        def with_backend(backend: SDPBackend, expand: bool) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:
                with sdpa_kernel([backend]):
                    if expand:
                        # expand() is a view; no KV bytes are copied before the kernel.
                        ke = ku.expand(1, n_kv, group, -1, hd).reshape(1, n_q, -1, hd)
                        ve = vu.expand(1, n_kv, group, -1, hd).reshape(1, n_q, -1, hd)
                        return F(q, ke, ve)
                    return F(q, k, v, enable_gqa=True)

            return run

        return {
            "default_dispatch_gqa": lambda: F(q, k, v, enable_gqa=True),
            "math_gqa": with_backend(SDPBackend.MATH, False),
            "cudnn_gqa": with_backend(SDPBackend.CUDNN_ATTENTION, False),
            "efficient_expanded": with_backend(SDPBackend.EFFICIENT_ATTENTION, True),
            "manual_grouped_matmul": manual,
        }

    rows = []
    for L in ctx_lens:
        kv_bytes = 2 * L * n_kv * hd * 2
        if 6 * kv_bytes > 0.8 * torch.cuda.mem_get_info()[0]:
            log.info("skip attention L=%d: not enough free VRAM", L)
            continue
        q = torch.randn(1, n_q, 1, hd, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, n_kv, L, hd, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        fns = variants(q, k, v)
        ref = fns["math_gqa"]().float()
        for name, fn in fns.items():
            try:
                err = (fn().float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
            except RuntimeError as exc:
                rows.append({"ctx_len": L, "backend": name, "available": False, "error": str(exc).splitlines()[0]})
                continue
            for _ in range(cfg.warmup):
                timed_events(fn)
            xs = [timed_events(fn)[1] for _ in range(cfg.reps)]
            rows.append(
                {
                    "ctx_len": L,
                    "backend": name,
                    "available": True,
                    "max_rel_err_vs_math": err,
                    "kv_bytes_per_layer": kv_bytes,
                    "seconds_per_call": summarize(xs).to_dict(),
                    "kv_scan_bps": summarize([kv_bytes / x for x in xs]).to_dict(),
                }
            )
        del q, k, v, fns, ref
        torch.cuda.empty_cache()
    tel.mark("attn_end")
    return {
        "geometry": {"n_q_heads": n_q, "n_kv_heads": n_kv, "head_dim": hd, "dtype": "bfloat16"},
        "correctness_tolerance_rel": 1e-2,
        "rows": rows,
    }


def gpu_spinup(seconds: float = 2.0) -> None:
    """Hold the GPU at load before a timed GPU section.

    Laptop GPUs drop to a low clock (observed 180 MHz) within seconds of idle, and a few
    warmup calls do not bring them back. Without this, a GPU section that follows a CPU
    section measures DVFS ramp-up, not the kernel. Observed in the Phase 0 smoke test.
    """
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for _ in range(8):
            a @ a
        sync()


# --------------------------------------------------------------------------------------
# 4. Async overlap
# --------------------------------------------------------------------------------------


def query_driver_attributes() -> dict[str, Any]:
    """Read device attributes straight from the CUDA driver API (nvcuda.dll).

    torch.cuda.get_device_properties does not expose asyncEngineCount, and the brief
    says to verify it rather than assume. Enum values come from the CUdevice_attribute
    header; attributes with independently known values are read as a mapping check.
    """
    attrs = {
        "GPU_OVERLAP": 15,
        "KERNEL_EXEC_TIMEOUT": 17,
        "INTEGRATED": 18,
        "CAN_MAP_HOST_MEMORY": 19,
        "CONCURRENT_KERNELS": 31,
        "PCI_BUS_ID": 33,
        "TCC_DRIVER": 35,
        "ASYNC_ENGINE_COUNT": 40,
        "UNIFIED_ADDRESSING": 41,
    }
    try:
        nv = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        if nv.cuInit(0) != 0:
            return {"ok": False, "error": "cuInit failed"}
        dev = ctypes.c_int()
        if nv.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return {"ok": False, "error": "cuDeviceGet failed"}
        out: dict[str, Any] = {"ok": True, "source": "cuDeviceGetAttribute (CUDA driver API)"}
        for name, code in attrs.items():
            val = ctypes.c_int(-1)
            ret = nv.cuDeviceGetAttribute(ctypes.byref(val), code, dev)
            out[name] = val.value if ret == 0 else None
        return out
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def bench_overlap(cfg: Config, tel: TelemetryLogger, pinned_bps_hint: float) -> dict[str, Any]:
    tel.mark("overlap_start")
    gpu_spinup()
    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    copy_stream_2 = torch.cuda.Stream()

    target_s = 0.15 if cfg.quick else 0.4
    chunk = 128 * MiB
    n_chunks = max(1, round(target_s * pinned_bps_hint / chunk))
    host_src = torch.empty(chunk, dtype=torch.uint8, pin_memory=True).random_(0, 255)
    host_dst = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
    dev_dst = torch.empty(chunk, dtype=torch.uint8, device="cuda")
    dev_src = torch.empty(chunk, dtype=torch.uint8, device="cuda").random_(0, 255)

    a = torch.randn(3072, 3072, device="cuda", dtype=torch.float16)
    # Calibrate compute length to roughly match the copy so neither dominates the ratio.
    one = timed_wall(lambda: a @ a)
    n_mm = max(1, round(target_s / max(one, 1e-4)))
    for _ in range(3):
        one = timed_wall(lambda: [a @ a for _ in range(n_mm)] and None) / n_mm
        n_mm = max(1, round(target_s / max(one, 1e-5)))

    def enqueue_compute() -> torch.cuda.Event:
        with torch.cuda.stream(compute_stream):
            for _ in range(n_mm):
                a @ a
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        return ev

    def enqueue_h2d(stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for _ in range(n_chunks):
                dev_dst.copy_(host_src, non_blocking=True)

    def enqueue_d2h(stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for _ in range(n_chunks):
                host_dst.copy_(dev_src, non_blocking=True)

    scenarios: dict[str, Callable[[], None]] = {
        "compute_only": lambda: enqueue_compute() and None,
        "h2d_only": lambda: enqueue_h2d(copy_stream),
        "d2h_only": lambda: enqueue_d2h(copy_stream_2),
        "compute_plus_h2d": lambda: (enqueue_compute(), enqueue_h2d(copy_stream)) and None,
        "compute_plus_d2h": lambda: (enqueue_compute(), enqueue_d2h(copy_stream_2)) and None,
        "h2d_plus_d2h": lambda: (enqueue_h2d(copy_stream), enqueue_d2h(copy_stream_2)) and None,
    }
    samples: dict[str, list[float]] = {k: [] for k in scenarios}
    rng = random.Random(cfg.seed + 4)
    for rep in range(cfg.warmup + cfg.reps):
        order = list(scenarios)
        rng.shuffle(order)
        for name in order:
            w = timed_wall(scenarios[name])
            if rep >= cfg.warmup:
                samples[name].append(w)
    tel.mark("overlap_end")
    return {
        "driver_attributes": query_driver_attributes(),
        "compute_kernel": {"op": "fp16 matmul 3072x3072", "iterations": n_mm},
        "copy": {"chunk_bytes": chunk, "chunks": n_chunks, "total_bytes": chunk * n_chunks},
        "wall_s": {k: summarize(v).to_dict() for k, v in samples.items()},
        "raw_wall_s": samples,
    }


# --------------------------------------------------------------------------------------
# 5. CPU-side attention throughput
# --------------------------------------------------------------------------------------


def bench_cpu_attention(cfg: Config, tel: TelemetryLogger) -> dict[str, Any]:
    tel.mark("cpu_start")
    n_kv, group, hd = 8, 4, 64  # Llama-3.2-1B: 8 KV heads x 4 query heads each, head_dim 64
    token_counts = [1_024, 16_384] if cfg.quick else [256, 1_024, 4_096, 16_384, 65_536]
    logical = os.cpu_count() or 1
    thread_counts = sorted({1, 8, logical}) if cfg.quick else sorted({1, 4, 8, 16, 20, logical})
    gen = torch.Generator().manual_seed(cfg.seed)

    # Host memory bandwidth context: a large memcpy bounds pageable copies and CPU attention.
    mem_src = torch.empty(512 * MiB, dtype=torch.uint8).random_(0, 255)
    mem_dst = torch.empty_like(mem_src).zero_()
    torch.set_num_threads(1)
    mem_bps = []
    for i in range(cfg.warmup + cfg.reps):
        t0 = time.perf_counter()
        mem_dst.copy_(mem_src)
        dt = time.perf_counter() - t0
        if i >= cfg.warmup:
            mem_bps.append(mem_src.numel() / dt)
    del mem_src, mem_dst

    def partial_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Exact partial attention for merge: returns the output AND its log-sum-exp, which
        # is all that has to cross PCIe back to the GPU (O(head_dim) bytes).
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)  # (kv, group, N)
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)  # (kv, group, hd)
        return out, lse

    rows = []
    cells = [(n, t, kind) for n in token_counts for t in thread_counts for kind in ("gemv", "partial_attn")]
    samples: dict[tuple[int, int, str], list[float]] = {c: [] for c in cells}
    tensors = {}
    for n in token_counts:
        tensors[n] = (
            torch.randn(n_kv, group, hd, generator=gen),
            torch.randn(n_kv, n, hd, generator=gen),
            torch.randn(n_kv, n, hd, generator=gen),
        )
    rng = random.Random(cfg.seed + 5)
    reps = cfg.reps
    inner_for: dict[tuple[int, int, str], int] = {}
    for rep in range(cfg.warmup + reps):
        order = cells[:]
        rng.shuffle(order)
        for cell in order:
            n, t, kind = cell
            torch.set_num_threads(t)
            q, k, v = tensors[n]
            q1 = q[:, :1, :].transpose(-1, -2)  # (kv, hd, 1): literal batched GEMV of B2.5
            if kind == "gemv":
                fn = lambda k=k, q1=q1: torch.matmul(k, q1)  # noqa: E731
            else:
                fn = lambda q=q, k=k, v=v: partial_attention(q, k, v)  # noqa: E731
            if cell not in inner_for:
                # Size each sample to ~40 ms so tiny calls clear timer noise and huge
                # single-thread calls do not blow the run time up.
                t0 = time.perf_counter()
                fn()
                est = time.perf_counter() - t0
                inner_for[cell] = max(1, min(2000, round(0.04 / max(est, 1e-7))))
            inner = inner_for[cell]
            t0 = time.perf_counter()
            for _ in range(inner):
                fn()
            dt = (time.perf_counter() - t0) / inner
            if rep >= cfg.warmup:
                samples[(n, t, kind)].append(dt)
        log.info("cpu rep %d/%d done", rep + 1, cfg.warmup + reps)

    for (n, t, kind), xs in samples.items():
        kv_bytes_fp32 = 2 * n * n_kv * hd * 4
        rows.append(
            {
                "kind": kind,
                "tokens": n,
                "threads": t,
                "seconds_per_call": summarize(xs).to_dict(),
                "tokens_per_s": summarize([n / x for x in xs]).to_dict(),
                "kv_bytes_fp32": kv_bytes_fp32,
                "kv_bps": summarize([kv_bytes_fp32 / x for x in xs]).to_dict(),
            }
        )
    rows.sort(key=lambda r: (r["kind"], r["tokens"], r["threads"]))

    # One numpy data point at default BLAS threading (threadpoolctl is not in the env).
    n = token_counts[-1]
    q, k, v = (x.numpy() for x in tensors[n])
    qn = q[:, :1, :].transpose(0, 2, 1)
    np_x = []
    for i in range(cfg.warmup + reps):
        t0 = time.perf_counter()
        np.matmul(k, qn)
        if i >= cfg.warmup:
            np_x.append(time.perf_counter() - t0)
    torch.set_num_threads(logical)

    # Same partial attention on GPU (fp32) for a like-for-like reference.
    gpu_rows = []
    gpu_spinup()
    for n in token_counts:
        q, k, v = (x.cuda() for x in tensors[n])
        for _ in range(cfg.warmup):
            timed_events(lambda: partial_attention(q, k, v))
        xs = [timed_events(lambda: partial_attention(q, k, v))[1] for _ in range(reps)]
        gpu_rows.append({"tokens": n, "seconds_per_call": summarize(xs).to_dict()})
        del q, k, v
    torch.cuda.empty_cache()
    tel.mark("cpu_end")
    return {
        "geometry": {"n_kv_heads": n_kv, "q_heads_per_kv": group, "head_dim": hd, "dtype": "float32"},
        "logical_cpus": logical,
        "host_memcpy_bps_1thread": summarize(mem_bps).to_dict(),
        "rows": rows,
        "numpy_gemv_default_threads": {"tokens": n, "seconds_per_call": summarize(np_x).to_dict()},
        "gpu_partial_attn_fp32": gpu_rows,
    }


# --------------------------------------------------------------------------------------
# 6. NVMe
# --------------------------------------------------------------------------------------


def bench_nvme(cfg: Config, tel: TelemetryLogger) -> dict[str, Any]:
    """Unbuffered (FILE_FLAG_NO_BUFFERING) reads so the OS page cache cannot inflate them."""
    if os.name != "nt":
        return {"measured": False, "reason": "unbuffered reader implemented for Windows only"}
    tel.mark("nvme_start")
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    GENERIC_READ, OPEN_EXISTING, NO_BUFFERING = 0x80000000, 3, 0x20000000
    INVALID = ctypes.c_void_p(-1).value

    file_bytes = (256 if cfg.quick else 1024) * MiB
    path = Path(tempfile.gettempdir()) / "lazykv_nvme_probe.bin"
    chunk = os.urandom(8 * MiB)
    with path.open("wb") as f:
        for _ in range(file_bytes // len(chunk)):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())

    def open_unbuffered() -> int:
        h = k32.CreateFileW(str(path), GENERIC_READ, 1, None, OPEN_EXISTING, NO_BUFFERING, None)
        if h == INVALID:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        return h

    buf_size = 4 * MiB
    buf = k32.VirtualAlloc(None, buf_size, 0x3000, 0x04)  # page-aligned, as NO_BUFFERING requires
    got = wintypes.DWORD()
    try:
        seq_bps = []
        for i in range(1 + (2 if cfg.quick else 3)):
            h = open_unbuffered()
            total = 0
            t0 = time.perf_counter()
            while total < file_bytes:
                if not k32.ReadFile(h, buf, 1 * MiB, ctypes.byref(got), None) or got.value == 0:
                    break
                total += got.value
            dt = time.perf_counter() - t0
            k32.CloseHandle(h)
            if i >= 1:
                seq_bps.append(total / dt)

        rnd = random.Random(cfg.seed + 6)
        random_rows = []
        # 4 KiB: classic IOPS; 128 KiB: one 64-token Llama-1B block for a single layer.
        for io_size in (4 * KiB, 128 * KiB):
            h = open_unbuffered()
            n_slots = file_bytes // io_size
            per_rep = []
            for _ in range(cfg.reps):
                ops = 0
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < (0.3 if cfg.quick else 1.0):
                    off = rnd.randrange(n_slots) * io_size
                    k32.SetFilePointerEx(h, off, None, 0)
                    k32.ReadFile(h, buf, io_size, ctypes.byref(got), None)
                    ops += 1
                dt = time.perf_counter() - t0
                per_rep.append((ops / dt, ops * io_size / dt))
            k32.CloseHandle(h)
            random_rows.append(
                {
                    "io_bytes": io_size,
                    "iops": summarize([x for x, _ in per_rep]).to_dict(),
                    "bps": summarize([y for _, y in per_rep]).to_dict(),
                    "latency_s": summarize([1 / x for x, _ in per_rep]).to_dict(),
                }
            )
    finally:
        k32.VirtualFree(buf, 0, 0x8000)
        path.unlink(missing_ok=True)
    tel.mark("nvme_end")
    return {
        "measured": True,
        "file_bytes": file_bytes,
        "queue_depth": 1,
        "sequential_1mib_bps": summarize(seq_bps).to_dict(),
        "random": random_rows,
    }


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke test, <90 s")
    parser.add_argument("--sections", default="pcie,vram,overlap,cpu,nvme")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: this benchmark needs the cu130 torch build (see requirements.lock)")
    cap = torch.cuda.get_device_capability()
    log.info("device %s capability %s torch %s", torch.cuda.get_device_name(), cap, torch.__version__)

    cfg = Config(quick=args.quick)
    torch.manual_seed(cfg.seed)
    sections = set(args.sections.split(","))
    out_dir = RESULTS_DIR / "phase0" / ("feasibility_quick" if cfg.quick else "feasibility")

    payload: dict[str, Any] = {"config": {"quick": cfg.quick, "warmup": cfg.warmup, "repeats": cfg.reps, "seed": cfg.seed}}
    t_start = time.perf_counter()
    with TelemetryLogger(out_dir / "telemetry.csv") as tel:
        time.sleep(1.0)  # idle baseline samples before load
        pinned_hint = 10e9
        if "pcie" in sections:
            payload["pcie"] = bench_pcie(cfg, tel)
            pinned = [r for r in payload["pcie"]["rows"] if r["mode"] == "pinned" and r["direction"] == "h2d"]
            pinned_hint = max(r["wall_bps"]["median"] for r in pinned)
        if "vram" in sections:
            payload["vram"] = bench_vram(cfg, tel)
        if "overlap" in sections:
            payload["overlap"] = bench_overlap(cfg, tel, pinned_hint)
        if "cpu" in sections:
            payload["cpu_attention"] = bench_cpu_attention(cfg, tel)
        if "nvme" in sections:
            payload["nvme"] = bench_nvme(cfg, tel)
        time.sleep(1.0)
    payload["telemetry"] = {**tel.summary(), "marks": tel.marks, "csv": "telemetry.csv"}
    payload["system"] = sysinfo.collect()
    payload["wall_seconds_total"] = time.perf_counter() - t_start
    path = write_metrics(out_dir, payload)
    log.info("wrote %s in %.1f s", path, payload["wall_seconds_total"])


if __name__ == "__main__":
    main()
