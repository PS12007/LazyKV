"""Why rung 9 keeps a float32 mirror: the CPU partial-attention pass, timed on four layouts.

Writes results/phase7/cpu_layout/metrics.json

Phase 0 §5 timed CPU attention on *contiguous float32* with no model loaded. Rung 9 cannot use that
layout for free: the tier's pinned host pool is bf16 laid out `[blocks, heads, 2, block_size,
head_dim]`, so one KV head's keys are a strided batch of small matrices rather than a matrix. The
design decision in `lazykv/exact.py` -- keep a separate float32 mirror, contiguous per head -- costs
real host RAM, so the gap it buys has to be a measurement rather than an assertion.

Four layouts, at the study's geometry, plus one implementation variant:

1. **bf16, in place** -- the tier's pinned pool attended where it is, with no extra memory at all.
   A head's keys cannot be *viewed* as a matrix there (the stride between blocks is the whole pool
   row, the stride inside one is the head dim), so this case uses a per-head batched GEMM over the
   `[blocks, block_size, head_dim]` strided batch. That is the design rung 9 would have used if the
   mirror were not worth its memory.
2. **float32, in place** -- the same batched formulation on a widened copy of the same pool, which
   separates the cost of the dtype from the cost of the layout.
3. **float32, contiguous** -- what the mirror provides, as a fresh allocation.
4. **float32, mirror slice** -- the mirror as rung 9 actually uses it: `mirror[:, :n * block_size]`,
   a prefix of a capacity-sized buffer. This is the number that belongs next to the measured decode,
   and it is not quite (3).

The variant is the per-head Python loop the first implementation used, against the batched GEMM it
was replaced by. That replacement was worth more than any of the layout choices at small contexts,
which is the kind of thing a microbenchmark finds and code review does not.

Cases (1) and (2) assert that their inputs are genuinely views of the pool. A `reshape` of the
permuted pool silently *copies* into a contiguous buffer, which would have measured case (3) twice
and reported the mirror as free.

This is a CPU measurement and it competes for the same cores as anything else running, so run it on
an otherwise idle machine, like every other timing in this project.

    .venv\\Scripts\\python.exe scripts\\04_cpu_layout.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harness.host import set_power_throttling  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.sysinfo import windows_host  # noqa: E402
from lazykv.exact import NEG_INF, cpu_partial_attention  # noqa: E402


def in_place_pool(pool: torch.Tensor, query: torch.Tensor, scaling: float, excluded: np.ndarray) -> None:
    """Attend to the tier's pinned pool where it lies: `[blocks, heads, 2, block_size, head_dim]`.

    Per KV head this is a batched GEMM over a strided batch of `[block_size, head_dim]` matrices,
    because no matrix view of the head's keys exists in that layout. Same arithmetic as
    `cpu_partial_attention`, different memory access pattern, and the difference is the measurement.
    """
    n_blocks, h, _, bs, d = pool.shape
    for j in range(h):
        k = pool[:, j, 0]  # [blocks, bs, d], strided between blocks
        v = pool[:, j, 1]
        assert k.data_ptr() == pool[:, j, 0].data_ptr() and not k.is_contiguous(), "case must attend in place"
        scores = torch.matmul(k, query[j].transpose(0, 1))  # [blocks, bs, groups]
        scores *= scaling
        scores[excluded[j]] = NEG_INF
        flat = scores.permute(2, 0, 1).reshape(query.shape[1], -1)
        m = flat.amax(dim=-1, keepdim=True)
        probs = (flat - m).exp()
        norm = probs.sum(dim=-1, keepdim=True)
        pb = (probs / norm).view(query.shape[1], n_blocks, bs).permute(1, 0, 2)
        torch.matmul(pb, v).sum(0)


def per_head_loop(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor, scaling: float, block_size: int, excluded: np.ndarray) -> None:
    """The first implementation, kept only as the thing the batched one is measured against.

    Identical arithmetic, one KV head at a time. It is not in `lazykv/` because nothing should call
    it; it lives here so the claim that batching mattered is reproducible.
    """
    kv_heads, total, _ = keys.shape
    n_blocks = total // block_size
    for j in range(kv_heads):
        scores = torch.matmul(query[j], keys[j].transpose(0, 1))
        scores *= scaling
        scores.view(-1, n_blocks, block_size)[:, excluded[j]] = NEG_INF
        m = scores.amax(dim=-1, keepdim=True)
        probs = (scores - m).exp()
        norm = probs.sum(dim=-1, keepdim=True)
        torch.matmul(probs / norm, values[j])


def timed(fn: Any, repeats: int) -> dict[str, float]:
    fn()  # warm: first call allocates and faults in the buffers
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts), "repeats": repeats}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--capacity", type=int, default=34816, help="mirror capacity in tokens; the sweep allocates the cache's max_len")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--groups", type=int, default=4, help="query heads per KV head")
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=14, help="selecting layers, for the per-token column")
    p.add_argument("--threads", type=int, nargs="+", default=[4, 8, 14, 20])
    p.add_argument("--k-blocks", type=int, default=128, help="selected blocks per head; only the -inf scatter depends on it")
    p.add_argument("--repeats", type=int, default=8)
    args = p.parse_args()

    set_power_throttling(opt_out=True)
    bs, d, h, g = args.block_size, args.head_dim, args.kv_heads, args.groups
    n = args.context // bs
    scaling = d**-0.5
    excluded = np.tile(np.arange(min(args.k_blocks + 1, n)), (h, 1))
    torch.manual_seed(0)

    # (1) and (2): the tier's own pinned layout, [blocks, heads, 2, bs, d]. A head's keys are
    # pool[:n, j, 0]: contiguous within a block, strided between them.
    pool16 = torch.empty((n, h, 2, bs, d), dtype=torch.bfloat16).normal_()
    pool32 = pool16.float()
    # (3): a fresh contiguous float32 matrix per head.
    keys_c = torch.randn((h, n * bs, d), dtype=torch.float32)
    values_c = torch.randn((h, n * bs, d), dtype=torch.float32)
    # (4): the mirror as rung 9 holds it -- a prefix of a capacity-sized buffer.
    mirror_k = torch.randn((h, args.capacity, d), dtype=torch.float32)
    mirror_v = torch.randn((h, args.capacity, d), dtype=torch.float32)

    q32 = torch.randn((h, g, d), dtype=torch.float32)
    q16 = q32.bfloat16()

    rows = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        cases: list[tuple[str, str, Any]] = [
            ("bf16_in_place", "the tier's pinned pool, attended where it lies", lambda: in_place_pool(pool16, q16, scaling, excluded)),
            ("fp32_in_place", "the same layout widened to float32", lambda: in_place_pool(pool32, q32, scaling, excluded)),
            ("fp32_contiguous", "a fresh contiguous float32 matrix per head", lambda: cpu_partial_attention(keys_c, values_c, q32, scaling, bs, excluded)),
            ("fp32_mirror_slice", "the mirror as rung 9 holds it: a prefix of a capacity-sized buffer", lambda: cpu_partial_attention(mirror_k[:, : n * bs], mirror_v[:, : n * bs], q32, scaling, bs, excluded)),
            ("fp32_contiguous_per_head_loop", "the first implementation: one KV head at a time", lambda: per_head_loop(keys_c, values_c, q32, scaling, bs, excluded)),
        ]
        for name, note, fn in cases:
            t = timed(fn, args.repeats)
            rows.append({
                "layout": name,
                "note": note,
                "threads": threads,
                "ms_per_layer": 1e3 * t["median_s"],
                "ms_per_layer_min": 1e3 * t["min_s"],
                "ms_per_layer_max": 1e3 * t["max_s"],
                "ms_per_token": 1e3 * t["median_s"] * args.layers,
                "repeats": t["repeats"],
            })
            print(f"{threads:>3} thr  {name:<30} {1e3 * t['median_s']:8.2f} ms/layer  {1e3 * t['median_s'] * args.layers:8.1f} ms/token")

    def best(layout: str) -> dict[str, Any] | None:
        got = [r for r in rows if r.get("layout") == layout and "ms_per_layer" in r]
        return min(got, key=lambda r: r["ms_per_layer"]) if got else None

    chosen = best("fp32_mirror_slice")
    summary: dict[str, Any] = {
        "context": args.context,
        "block_size": bs,
        "kv_heads": h,
        "groups": g,
        "head_dim": d,
        "layers": args.layers,
        "k_blocks": args.k_blocks,
        "capacity_tokens": args.capacity,
        "best_threads": chosen["threads"] if chosen else None,
        "best_ms_per_layer": {layout: (best(layout) or {}).get("ms_per_layer") for layout in
                              ("bf16_in_place", "fp32_in_place", "fp32_contiguous", "fp32_mirror_slice", "fp32_contiguous_per_head_loop")},
        "best_ms_per_token": {layout: (best(layout) or {}).get("ms_per_token") for layout in
                              ("bf16_in_place", "fp32_in_place", "fp32_contiguous", "fp32_mirror_slice", "fp32_contiguous_per_head_loop")},
    }
    ref = (best("fp32_mirror_slice") or {}).get("ms_per_layer")
    summary["over_mirror"] = {
        layout: (None if ref is None or (best(layout) or {}).get("ms_per_layer") is None else best(layout)["ms_per_layer"] / ref)
        for layout in ("bf16_in_place", "fp32_in_place", "fp32_contiguous_per_head_loop")
    }
    # Host RAM the mirror costs, for the same blocks, against holding nothing extra at all.
    summary["mirror_bytes_per_layer"] = 2 * h * args.capacity * d * 4
    write_metrics(RESULTS_DIR / "phase7" / "cpu_layout", {"host": windows_host(), "config": vars(args), "rows": rows, "summary": summary})
    print("wrote", RESULTS_DIR / "phase7" / "cpu_layout" / "metrics.json")


if __name__ == "__main__":
    main()
