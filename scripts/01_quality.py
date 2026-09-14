"""Phase 1 quality reference: how much do "exact" paths disagree with each other?

Every later policy's quality delta is measured against the full-cache reference. Before
that means anything, we need the noise floor between paths that are all supposed to be
exact (brief §B6: determinism is not guaranteed across attention paths; report agreement).

The first run of this script found the reference disagreeing with itself. The kernel probe
below isolates why (cuDNN decode is not repeatable for identical inputs), and quality is
now scored on lazykv.quality.QUALITY_STRATEGY, with the fast kernel's floor kept as a
reported comparison rather than hidden.

Comparisons, teacher-forced through the decode path (brief §B7.1). Q = quality strategy.
  self_repeat                 Q vs Q on a fresh cache                  (expected: bit-identical)
  chunk_size_2048_vs_256      Q, prefill chunk 2048 vs 256             (numeric noise only)
  fast_kernel_self_repeat     cudnn_bucketed vs itself                 (floor of the speed path)
  fast_kernel_vs_quality      cudnn_bucketed vs Q, same chunking       (cross-kernel)
  kernel_math_vs_quality      math vs Q, chunk 256                     (cross-kernel)
  hf_stock_sdpa_dynamic_cache transformers' stock sdpa + DynamicCache vs Q

Kernel probe: attention inputs of one decode step are captured from every layer of the real
model, then each strategy is called repeatedly on those exact tensors, interleaved with a
prefill-shaped call. Distinct output bit patterns and each pattern's error against float64
attention are recorded.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\01_quality.py > logs\\quality.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

import lazykv.attention as attention  # noqa: E402
from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import IMPLEMENTATION_NAME, attend, current_strategy, set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import LoadedModel, load, teacher_forced_logprobs, teacher_forced_steps  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY, compare_stream  # noqa: E402

log = logging.getLogger("quality")


def peak_working_set() -> int | None:
    """Peak host RAM of this process (Windows), recorded because host memory is the tight resource here."""
    import ctypes
    import os
    from ctypes import wintypes

    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    return int(c.PeakWorkingSetSize) if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
FAST_STRATEGY = "cudnn_bucketed"


def kernel_probe(lm: LoadedModel, tokens: torch.Tensor, ctx: int, step: int, calls: int, bucket: int) -> dict[str, Any]:
    """Repeatability of each attention strategy on real decode-step inputs from every layer."""
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    decode_calls = 0

    # Capture by wrapping the module-level function the registered forward calls; restored
    # in `finally` so nothing downstream runs through the wrapper.
    def capture(query, key, value, scaling, strategy, bucket=attention.DEFAULT_BUCKET):  # noqa: ANN001, ANN202
        nonlocal decode_calls
        if query.shape[-2] == 1:
            if step * lm.num_layers <= decode_calls < (step + 1) * lm.num_layers:
                captured.append((query.clone(), key.clone(memory_format=torch.contiguous_format), value.clone(memory_format=torch.contiguous_format), scaling))
            decode_calls += 1
        return attend(query, key, value, scaling, strategy, bucket)

    attention.attend = capture
    restore = set_strategy_for_test(FAST_STRATEGY)
    try:
        max_len = -(-(ctx + step + 8) // bucket) * bucket
        # Continuation of step + 2 tokens: row 0 comes from prefill, row i + 1 from decode step i.
        teacher_forced_logprobs(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], tokens[ctx : ctx + step + 2], 2048)
    finally:
        attention.attend = attend
        restore()
    layers = captured
    if len(layers) != lm.num_layers:
        raise RuntimeError(f"captured {len(layers)} decode-step attention calls, expected {lm.num_layers}")

    out: dict[str, Any] = {"ctx_len": ctx, "decode_step": step, "calls_per_layer": calls, "strategies": {}}
    for strategy in (FAST_STRATEGY, "cudnn", "efficient", "math"):
        per_layer = []
        for q, k, v, scaling in layers:
            # Storage with spare rows past the live length, as the preallocated cache has, so
            # the bucketed path takes its padded branch.
            n = k.shape[2]
            spare = -(-n // bucket) * bucket + bucket
            ks = torch.zeros((1, k.shape[1], spare, k.shape[3]), dtype=k.dtype, device=k.device)
            vs = torch.zeros_like(ks)
            ks[:, :, :n], vs[:, :, :n] = k, v
            kv, vv = ks[:, :, :n], vs[:, :, :n]
            groups = q.shape[1] // k.shape[1]
            ref = torch.softmax((q.double() @ k.double().repeat_interleave(groups, 1).transpose(-1, -2)) * scaling, -1) @ v.double().repeat_interleave(groups, 1)
            other_shape = torch.randn((1, q.shape[1], 256, q.shape[3]), dtype=q.dtype, device=q.device)
            patterns: dict[str, float] = {}
            for i in range(calls):
                if i % 3 == 0:
                    # A differently shaped call between repeats, as a real decode step has
                    # (other layers, prefill of the next request).
                    attend(other_shape, kv, vv, scaling, strategy, bucket)
                got = attend(q, kv, vv, scaling, strategy, bucket)
                digest = hashlib.sha256(got.float().cpu().numpy().tobytes()).hexdigest()
                if digest not in patterns:
                    patterns[digest] = ((got.double() - ref).abs().max() / ref.abs().max()).item()
            per_layer.append({"distinct_outputs": len(patterns), "rel_err_vs_float64": sorted(patterns.values())})
            del ks, vs, other_shape
        out["strategies"][strategy] = {
            "layers": per_layer,
            "layers_not_repeatable": sum(1 for p in per_layer if p["distinct_outputs"] > 1),
            "max_distinct_outputs": max(p["distinct_outputs"] for p in per_layer),
            "max_rel_err_vs_float64": max(max(p["rel_err_vs_float64"]) for p in per_layer),
        }
        log.info("kernel probe %s: %d/%d layers not repeatable", strategy, out["strategies"][strategy]["layers_not_repeatable"], len(per_layer))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="4096,16384,32768")
    p.add_argument("--continuation", type=int, default=256)
    p.add_argument("--book", default="pride_and_prejudice")
    p.add_argument("--hf-max-ctx", type=int, default=16384, help="stock sdpa + DynamicCache is only run up to here")
    p.add_argument("--probe-ctx", type=int, default=4096)
    p.add_argument("--probe-step", type=int, default=2)
    p.add_argument("--probe-calls", type=int, default=24)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    lm = load(MODEL_REPO, MODEL_REVISION, strategy=FAST_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    _, bucket = current_strategy()
    tokens = load_tokens(args.book, lm.tokenizer)
    contexts = [int(c) for c in args.contexts.split(",")]
    max_len = -(-(max(contexts) + args.continuation + 8) // bucket) * bucket

    probe = kernel_probe(lm, tokens, args.probe_ctx, args.probe_step, args.probe_calls, bucket)

    def cont(ctx: int) -> torch.Tensor:
        return tokens[ctx : ctx + args.continuation]

    def reference(ctx: int, chunk: int, strategy: str) -> torch.Tensor:
        """Reference rows kept on the host: the only positions x vocab tensors this script holds."""
        torch.cuda.empty_cache()
        restore = set_strategy_for_test(strategy)
        try:
            return teacher_forced_logprobs(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], cont(ctx), chunk)
        finally:
            restore()

    def against(ref: torch.Tensor, ctx: int, chunk: int, strategy: str) -> dict[str, Any]:
        torch.cuda.empty_cache()
        restore = set_strategy_for_test(strategy)
        try:
            rows = teacher_forced_steps(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], cont(ctx), chunk)
            return compare_stream(ref, rows).to_dict()
        finally:
            restore()

    def against_hf_stock(ref: torch.Tensor, ctx: int, chunk: int) -> dict[str, Any]:
        torch.cuda.empty_cache()
        lm.model.set_attn_implementation("sdpa")
        try:
            rows = teacher_forced_steps(lm.model, DynamicCache(config=lm.model.config), tokens[:ctx], cont(ctx), chunk)
            return compare_stream(ref, rows).to_dict()
        finally:
            lm.model.set_attn_implementation(IMPLEMENTATION_NAME)

    Q = QUALITY_STRATEGY
    rows_out: list[dict[str, Any]] = []
    for ctx in contexts:
        t0 = time.perf_counter()
        comparisons: dict[str, Any] = {}
        ref_q = reference(ctx, 2048, Q)
        comparisons["self_repeat"] = against(ref_q, ctx, 2048, Q)
        comparisons["chunk_size_2048_vs_256"] = against(ref_q, ctx, 256, Q)
        comparisons["fast_kernel_vs_quality"] = against(ref_q, ctx, 2048, FAST_STRATEGY)
        if ctx <= args.hf_max_ctx:
            comparisons["hf_stock_sdpa_dynamic_cache"] = against_hf_stock(ref_q, ctx, 2048)
        del ref_q
        ref_fast = reference(ctx, 2048, FAST_STRATEGY)
        comparisons["fast_kernel_self_repeat"] = against(ref_fast, ctx, 2048, FAST_STRATEGY)
        del ref_fast
        ref_q256 = reference(ctx, 256, Q)
        # Math materializes chunk x ctx score matrices; the small chunk keeps it within VRAM.
        comparisons["kernel_math_vs_quality"] = against(ref_q256, ctx, 256, "math")
        del ref_q256
        rows_out.append({"ctx_len": ctx, "continuation": args.continuation, "comparisons": comparisons, "wall_s": time.perf_counter() - t0, "peak_host_working_set_bytes": peak_working_set()})
        log.info("ctx %d done in %.0fs: %s", ctx, rows_out[-1]["wall_s"], {k: (v["top1_agreement"], round(v["mean_kl"], 6), v["exact_match"]) for k, v in comparisons.items()})

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(
        RESULTS_DIR / "phase1" / "quality_reference",
        {
            "config": vars(args),
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "attention": {"quality_strategy": Q, "fast_strategy": FAST_STRATEGY, "bucket": bucket},
            "kl_direction": "KL(first || second), nats, per position; CI is a percentile bootstrap over positions (positions are not independent, so it understates uncertainty)",
            "kernel_probe": probe,
            "rows": rows_out,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": shared_after},
        },
    )


if __name__ == "__main__":
    main()
