"""Phase 1 quality reference: how much do "exact" paths disagree with each other?

Every later policy's quality delta is measured against the full-cache reference. Before
that means anything, we need the noise floor between paths that are all supposed to be
exact (brief §B6: determinism is not guaranteed across attention paths; report agreement).

Comparisons, teacher-forced through the decode path (brief §B7.1):
  self_repeat     reference vs itself on a fresh cache           (expected: bit-identical)
  chunk_size      reference, prefill chunk 2048 vs 256            (numeric noise only)
  kernel_math     cudnn_bucketed vs math kernel, same chunking    (cross-kernel floor)
  hf_stock_sdpa   LazyKV path vs transformers' stock sdpa + DynamicCache

Launch detached:
    .venv\\Scripts\\python.exe scripts\\01_quality.py > logs\\quality.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import IMPLEMENTATION_NAME, current_strategy, set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import load, teacher_forced_logprobs, teacher_forced_steps  # noqa: E402
from lazykv.quality import compare_stream  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="4096,16384,32768")
    p.add_argument("--continuation", type=int, default=256)
    p.add_argument("--book", default="pride_and_prejudice")
    p.add_argument("--hf-max-ctx", type=int, default=16384, help="stock sdpa + DynamicCache is only run up to here")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    lm = load(MODEL_REPO, MODEL_REVISION)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    strategy, bucket = current_strategy()
    tokens = load_tokens(args.book, lm.tokenizer)
    contexts = [int(c) for c in args.contexts.split(",")]
    max_len = -(-(max(contexts) + args.continuation + 8) // bucket) * bucket

    def cont(ctx: int) -> torch.Tensor:
        return tokens[ctx : ctx + args.continuation]

    def reference(ctx: int, chunk: int) -> torch.Tensor:
        """Reference rows kept on the host: the only positions x vocab tensors this script holds."""
        torch.cuda.empty_cache()
        return teacher_forced_logprobs(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], cont(ctx), chunk)

    def against(ref: torch.Tensor, ctx: int, chunk: int, strat: str | None = None) -> dict[str, Any]:
        torch.cuda.empty_cache()
        restore = set_strategy_for_test(strat) if strat else (lambda: None)
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

    rows_out: list[dict[str, Any]] = []
    for ctx in contexts:
        t0 = time.perf_counter()
        ref_2048 = reference(ctx, 2048)
        comparisons: dict[str, Any] = {"self_repeat": against(ref_2048, ctx, 2048)}
        if ctx <= args.hf_max_ctx:
            comparisons["hf_stock_sdpa_dynamic_cache"] = against_hf_stock(ref_2048, ctx, 2048)
        comparisons["chunk_size_2048_vs_256"] = against(ref_2048, ctx, 256)
        del ref_2048
        ref_256 = reference(ctx, 256)
        # Math materializes chunk x ctx score matrices; the small chunk keeps it within VRAM.
        comparisons["kernel_math_vs_" + strategy] = against(ref_256, ctx, 256, "math")
        del ref_256
        rows_out.append({"ctx_len": ctx, "continuation": args.continuation, "comparisons": comparisons, "wall_s": time.perf_counter() - t0, "peak_host_working_set_bytes": peak_working_set()})
        log.info("ctx %d done in %.0fs: %s", ctx, rows_out[-1]["wall_s"], {k: (v["top1_agreement"], round(v["mean_kl"], 6)) for k, v in comparisons.items()})

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(
        RESULTS_DIR / "phase1" / "quality_reference",
        {
            "config": vars(args),
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "attention": {"strategy": strategy, "bucket": bucket},
            "kl_direction": "KL(first || second), nats, per position; CI is a percentile bootstrap over positions (positions are not independent, so it understates uncertainty)",
            "rows": rows_out,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": shared_after},
        },
    )


if __name__ == "__main__":
    main()
