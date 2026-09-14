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
from lazykv.generate import load, teacher_forced_logprobs  # noqa: E402
from lazykv.quality import compare  # noqa: E402

log = logging.getLogger("quality")

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

    def ours(ctx: int, chunk: int, strat: str | None = None) -> torch.Tensor:
        # The previous call's cache is unreferenced by now; return its blocks before allocating anew.
        torch.cuda.empty_cache()
        restore = set_strategy_for_test(strat) if strat else (lambda: None)
        try:
            return teacher_forced_logprobs(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], tokens[ctx : ctx + args.continuation], chunk)
        finally:
            restore()

    def hf_stock(ctx: int, chunk: int) -> torch.Tensor:
        lm.model.set_attn_implementation("sdpa")
        try:
            return teacher_forced_logprobs(lm.model, DynamicCache(config=lm.model.config), tokens[:ctx], tokens[ctx : ctx + args.continuation], chunk)
        finally:
            lm.model.set_attn_implementation(IMPLEMENTATION_NAME)
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for ctx in contexts:
        t0 = time.perf_counter()
        ref_2048 = ours(ctx, 2048)
        comparisons: dict[str, Any] = {
            "self_repeat": compare(ref_2048, ours(ctx, 2048)).to_dict(),
        }
        ref_256 = ours(ctx, 256)
        comparisons["chunk_size_2048_vs_256"] = compare(ref_2048, ref_256).to_dict()
        # Math materializes chunk x ctx score matrices; the small chunk keeps it within VRAM.
        comparisons["kernel_math_vs_" + strategy] = compare(ref_256, ours(ctx, 256, "math")).to_dict()
        if ctx <= args.hf_max_ctx:
            comparisons["hf_stock_sdpa_dynamic_cache"] = compare(ref_2048, hf_stock(ctx, 2048)).to_dict()
        rows.append({"ctx_len": ctx, "continuation": args.continuation, "comparisons": comparisons, "wall_s": time.perf_counter() - t0})
        log.info("ctx %d done in %.0fs: %s", ctx, rows[-1]["wall_s"], {k: (v["top1_agreement"], round(v["mean_kl"], 6)) for k, v in comparisons.items()})

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(
        RESULTS_DIR / "phase1" / "quality_reference",
        {
            "config": vars(args),
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "attention": {"strategy": strategy, "bucket": bucket},
            "kl_direction": "KL(first || second), nats, per position; CI is a percentile bootstrap over positions (positions are not independent, so it understates uncertainty)",
            "rows": rows,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": shared_after},
        },
    )


if __name__ == "__main__":
    main()
