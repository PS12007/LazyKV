"""Session snapshots: save a tiered session's KV to disk and resume it later without a prefill.

Gap table item C1. A long-document chat pays its prefill once per session; resuming a saved session
should cost a sequential read instead. The tier makes this simple: a sealed block is immutable once
it is on the host, so a snapshot is the host store as it already is, plus the few tokens still on the
GPU (the dense layers' KV and each tiered layer's unsealed tail) and the logits for the next token.

**Restore goes through the same boundary as prefill.** Each tiered layer's KV is rebuilt on the GPU
one layer at a time and handed to `TieredLayer`'s constructor, exactly as `tiered_prefill` hands it
a freshly computed layer: the constructor copies it to the host store, builds the Quest metadata and
seeds the slots. So a restored tier is a tier built by prefill at that length, and the only state not
carried over is slot residency, which decides what is fetched, never what is attended (`rank`
selects from the metadata alone). Rung 8's int8 store is not supported: its host bytes are packed and
its constructor takes exact KV.

**File layout.** A JSON header padded to `ALIGN`, then sections, each starting at a multiple of
`ALIGN`: the next-token logits (float32), then per layer either `[2, h, L, d]` (dense layers: keys,
values) or the host store's own `[B, h, 2, bs, d]` blocks followed by the `[h, 2, fill, d]` tail
(tiered layers). Writing in the store's native layout means the bulk of a snapshot is written from
the pinned pool without a permute; alignment lets a reader bypass the OS page cache.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import torch

from lazykv.cache import FullGPUCache
from lazykv.pinned import pinned_empty
from lazykv.tiered import TieredCache, TieredLayer

MAGIC = "lazykv-snapshot-1"
ALIGN = 4096  # sector and page size: what an unbuffered reader needs
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _pad(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _bytes(t: torch.Tensor) -> memoryview:
    """A CPU tensor's bytes, without a copy for a contiguous one (bfloat16 has no numpy dtype)."""
    t = t.contiguous()
    return memoryview(t.view(torch.uint8).reshape(-1).numpy())


@dataclass
class SaveStats:
    bytes_written: int
    seconds: float  # including fsync: the snapshot is on disk when save returns


@dataclass
class RestoreStats:
    bytes_read: int
    read_s: float
    rebuild_s: float  # everything outside reads: H2D, metadata, seeding, dense layers, the final sync
    sections: dict[str, float] = field(default_factory=dict)


def save(cache: TieredCache, next_logits: torch.Tensor, path: Path) -> SaveStats:
    """Write `cache` (a tiered session at any decode position) and the logits for its next token."""
    if cache.quant is not None:
        raise ValueError("snapshots of the int8 tier are not supported")
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    tiers = cache.tiers
    any_tier = next(iter(tiers.values()))
    length = any_tier.length
    sections: list[tuple[str, Any]] = [("logits", next_logits.float().cpu())]
    layers: list[dict[str, Any]] = []
    for i in range(len(cache.full.layers)):
        if i in tiers:
            t = tiers[i]
            if t.length != length:
                raise RuntimeError("tiered layers disagree on length")
            tail = t.pool[t.tail_slot, :, :, : t.fill].cpu()  # [h, 2, fill, d]
            sections.append((f"L{i}.blocks", t.host[: t.n_sealed]))
            sections.append((f"L{i}.tail", tail))
            layers.append({"kind": "tier", "n_sealed": t.n_sealed, "fill": t.fill})
        else:
            src = cache.full.layers[i]
            if src.get_seq_length() != length:
                raise RuntimeError(f"dense layer {i} has length {src.get_seq_length()}, the tier {length}")
            kv = torch.stack([src.keys[0, :, :length], src.values[0, :, :length]]).cpu()  # [2, h, L, d]
            sections.append((f"L{i}.kv", kv))
            layers.append({"kind": "dense"})
    offsets, cursor = {}, 0
    for name, t in sections:
        offsets[name] = [cursor, t.numel() * t.element_size()]
        cursor += _pad(offsets[name][1])
    header = {
        "magic": MAGIC, "length": length, "block_size": cache.block_size, "dense_layers": cache.dense_layers,
        "heads": any_tier.h, "head_dim": any_tier.d, "dtype": str(any_tier.pool.dtype).removeprefix("torch."),
        "vocab": int(next_logits.numel()), "layers": layers, "sections": offsets,
    }
    head = json.dumps(header).encode()
    base = _pad(len(head) + 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        f.write(b"\0" * (base - len(head) - 8))
        for name, t in sections:
            n = f.write(_bytes(t))
            f.write(b"\0" * (_pad(n) - n))
        f.flush()
        os.fsync(f.fileno())
    return SaveStats(base + cursor, time.perf_counter() - t0)


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    if header.get("magic") != MAGIC:
        raise ValueError(f"{path} is not a LazyKV snapshot")
    return header, _pad(n + 8)


def _read_into(f: BinaryIO, offset: int, out: torch.Tensor) -> None:
    f.seek(offset)
    view = _bytes(out)  # `out` is contiguous, so this aliases its memory
    got = f.readinto(view)
    if got != len(view):
        raise EOFError(f"short read: {got} of {len(view)} bytes at offset {offset}")


class _UnbufferedReader:
    """Windows FILE_FLAG_NO_BUFFERING reads, which bypass the OS page cache even for cached pages.

    Without it, a snapshot read back soon after it was written comes from RAM, and its "disk" time
    is a memcpy. The flag needs sector-aligned offsets, sizes and buffers: sections are aligned and
    padded by `save`, and an aligned bounce buffer takes the reads (torch's CPU allocator aligns to
    64 bytes, not 4096). Same mechanism as Phase 0's NVMe probe.
    """

    CHUNK = 64 * 2**20

    def __init__(self, path: Path) -> None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._ctypes, self._k32, self._dword = ctypes, k32, wintypes.DWORD
        h = k32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x20000000, None)  # GENERIC_READ, OPEN_EXISTING, NO_BUFFERING
        if h == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
        self._h = h
        raw = torch.empty(self.CHUNK + ALIGN, dtype=torch.uint8)
        self._raw = raw  # keeps the bounce buffer alive
        skip = (-raw.data_ptr()) % ALIGN
        self._bounce = raw[skip : skip + self.CHUNK]

    def read_into(self, offset: int, out: torch.Tensor) -> None:
        dst = out.contiguous().view(torch.uint8).reshape(-1)
        n = dst.numel()
        if self._k32.SetFilePointerEx(self._h, offset, None, 0) == 0:
            raise OSError(self._ctypes.get_last_error(), "SetFilePointerEx failed")
        got = self._dword()
        done = 0
        while done < n:
            want = min(self.CHUNK, _pad(n - done))
            if not self._k32.ReadFile(self._h, self._bounce.data_ptr(), want, self._ctypes.byref(got), None) or got.value < min(want, n - done):
                raise OSError(self._ctypes.get_last_error(), f"unbuffered read failed at {offset + done}")
            take = min(got.value, n - done)
            dst[done : done + take].copy_(self._bounce[:take])
            done += take

    def close(self) -> None:
        self._k32.CloseHandle(self._h)


def restore(
    path: Path,
    k_blocks: int,
    capacity_tokens: int,
    num_layers: int,
    model: Any = None,
    host_pools: list[torch.Tensor] | None = None,
    n_slots: int | None = None,
    fetch: str = "gather",
    stages: Any = None,
    unbuffered: bool = False,
    rebuild: str = "store",
) -> tuple[TieredCache, torch.Tensor, RestoreStats]:
    """A TieredCache at the snapshot's position, and the logits for its next token.

    By default reads go through the OS page cache, so a snapshot read soon after it was written is a
    RAM copy. `unbuffered=True` (Windows only) reads from the disk itself, which is what a resumed
    session pays after the cache has moved on.

    `rebuild` picks how each tiered layer is rebuilt. "store" (Phase 22) reads the sealed blocks
    straight into the layer's pinned host store and derives the metadata and seeded slots from it
    on the GPU without a host sync, so that work overlaps the next layer's read
    (`TieredLayer.from_store`). "boundary" (Phase 20) copies each layer to the GPU and hands it to
    the constructor, the same boundary as tiered prefill: the KV crosses the link twice and every
    step runs after the read, one after the other. Both give the same state; `rebuild_s` is the
    restore's time outside reads, which under "store" is only the part the overlap did not hide.
    """
    if rebuild not in ("store", "boundary"):
        raise ValueError(f"unknown rebuild {rebuild!r}")
    if unbuffered and os.name != "nt":
        raise NotImplementedError("the unbuffered reader is implemented for Windows only")
    header, base = read_header(path)
    dt = _DTYPES[header["dtype"]]
    h, d, bs, length = header["heads"], header["head_dim"], header["block_size"], header["length"]
    if len(header["layers"]) != num_layers:
        raise ValueError(f"snapshot has {len(header['layers'])} layers, the model {num_layers}")
    sec = header["sections"]
    stats = RestoreStats(0, 0.0, 0.0)
    full = FullGPUCache(num_layers, capacity_tokens)
    tiers: dict[int, TieredLayer] = {}
    dense = 0
    reader = _UnbufferedReader(path) if unbuffered else None
    with path.open("rb", buffering=0) as f:

        def load(name: str, shape: tuple[int, ...], dtype: torch.dtype, into: torch.Tensor | None = None) -> torch.Tensor:
            out = torch.empty(shape, dtype=dtype) if into is None else into
            t0 = time.perf_counter()
            if reader is not None:
                reader.read_into(base + sec[name][0], out)
            else:
                _read_into(f, base + sec[name][0], out)
            stats.read_s += time.perf_counter() - t0
            stats.bytes_read += sec[name][1]
            return out

        logits = load("logits", (header["vocab"],), torch.float32).cuda()
        for i, spec in enumerate(header["layers"]):
            if spec["kind"] == "dense":
                kv = load(f"L{i}.kv", (2, h, length, d), dt)
                t0 = time.perf_counter()
                g = kv.cuda()
                full.layers[i].update(g[0:1], g[1:2])
                stats.rebuild_s += time.perf_counter() - t0
                dense += 1
                continue
            n, fill = spec["n_sealed"], spec["fill"]
            host = None if host_pools is None else host_pools[i - header["dense_layers"]]
            if rebuild == "store":
                t0 = time.perf_counter()
                if host is None:
                    host = pinned_empty((-(-capacity_tokens // bs), h, 2, bs, d), dt)
                stats.rebuild_s += time.perf_counter() - t0
                load(f"L{i}.blocks", (n, h, 2, bs, d), dt, into=host[:n])
                tail = load(f"L{i}.tail", (h, 2, fill, d), dt)
                t0 = time.perf_counter()
                tiers[i] = TieredLayer.from_store(host, n, tail, k_blocks, bs, capacity_tokens, torch.device("cuda"), n_slots)
                stats.rebuild_s += time.perf_counter() - t0
                continue
            blocks = load(f"L{i}.blocks", (n, h, 2, bs, d), dt)
            tail = load(f"L{i}.tail", (h, 2, fill, d), dt)
            t0 = time.perf_counter()
            g = blocks.cuda().permute(1, 2, 0, 3, 4).reshape(h, 2, n * bs, d)  # [h, 2, B*bs, d]
            kv = torch.cat([g, tail.cuda()], dim=2)
            tiers[i] = TieredLayer(kv[None, :, 0], kv[None, :, 1], k_blocks, bs, capacity_tokens, host, n_slots)
            del g, kv
            stats.rebuild_s += time.perf_counter() - t0
    if reader is not None:
        reader.close()
    if dense != header["dense_layers"]:
        raise ValueError("snapshot's dense layer count disagrees with its header")
    t0 = time.perf_counter()
    cache = TieredCache(full, k_blocks, bs, capacity_tokens, model=model, dense_layers=dense, n_slots=n_slots, fetch=fetch, stages=stages, tiers=tiers)
    torch.cuda.synchronize()
    stats.rebuild_s += time.perf_counter() - t0
    return cache, logits, stats
