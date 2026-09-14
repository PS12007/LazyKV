"""Is each attention strategy repeatable on the real model's decode inputs?

The first Phase 1 quality run found the full-cache reference disagreeing with itself.
Prefill was bit-identical between runs; decode was not. This isolates the attention kernel
as the cause, in situ: while the real model decodes, every attention call is re-executed
on its live inputs a few more times, with a differently shaped call (a prefill-sized query)
in between, and the outputs are compared bit for bit. Outputs that differ are also compared
to float64 attention on the same inputs, which says whether the variation costs accuracy or
is only rounding.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\01_kernel_repeatability.py > logs\\kernel_repeatability.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import lazykv.attention as attention  # noqa: E402
from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import STRATEGIES, attend, current_strategy, set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import load, teacher_forced_logprobs  # noqa: E402

log = logging.getLogger("kernel_repeatability")

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


def float64_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    kk = k.double().repeat_interleave(groups, 1)
    vv = v.double().repeat_interleave(groups, 1)
    return torch.softmax((q.double() @ kk.transpose(-1, -2)) * scaling, -1) @ vv


def probe(lm, tokens: torch.Tensor, strategy: str, ctx: int, steps: int, repeats: int, bucket: int) -> dict[str, Any]:  # noqa: ANN001
    calls = 0
    unrepeatable: list[dict[str, Any]] = []
    other_q: torch.Tensor | None = None

    # Wraps the module-level function the registered forward calls; restored in `finally`.
    def checked(query, key, value, scaling, strat, bkt=attention.DEFAULT_BUCKET):  # noqa: ANN001, ANN202
        nonlocal calls, other_q
        out = attend(query, key, value, scaling, strat, bkt)
        if query.shape[-2] != 1:
            return out
        layer = calls % lm.num_layers
        calls += 1
        if other_q is None:
            # Math materializes query x KV score matrices; a prefill-sized query would not fit at 16K.
            q_len = 256 if strat == "math" else 2048
            other_q = torch.randn((1, query.shape[1], q_len, query.shape[3]), dtype=query.dtype, device=query.device)
        variants = [out]
        for _ in range(repeats):
            attend(other_q, key, value, scaling, strat, bkt)  # a different shape in between, as in real serving
            again = attend(query, key, value, scaling, strat, bkt)
            if not any(torch.equal(again, v) for v in variants):
                variants.append(again)
        if len(variants) > 1:
            ref = float64_attention(query, key, value, scaling)
            scale = ref.abs().max().item()
            unrepeatable.append({
                "decode_step": (calls - 1) // lm.num_layers,
                "layer": layer,
                "distinct_outputs": len(variants),
                "rel_err_vs_float64": [((v.double() - ref).abs().max().item()) / scale for v in variants],
                "max_abs_between_variants_rel": max((variants[0].double() - v.double()).abs().max().item() for v in variants[1:]) / scale,
            })
        return out

    attention.attend = checked
    restore = set_strategy_for_test(strategy)
    try:
        max_len = -(-(ctx + steps + 8) // bucket) * bucket
        teacher_forced_logprobs(lm.model, FullGPUCache(lm.num_layers, max_len), tokens[:ctx], tokens[ctx : ctx + steps + 1], 2048)
    finally:
        attention.attend = attend
        restore()
    return {
        "strategy": strategy,
        "ctx_len": ctx,
        "decode_calls_checked": calls,
        "repeats_per_call": repeats,
        "calls_not_repeatable": len(unrepeatable),
        "fraction_not_repeatable": len(unrepeatable) / calls if calls else None,
        "layers_affected": sorted({u["layer"] for u in unrepeatable}),
        "max_rel_err_vs_float64_any_variant": max((max(u["rel_err_vs_float64"]) for u in unrepeatable), default=None),
        "min_rel_err_vs_float64_any_variant": min((min(u["rel_err_vs_float64"]) for u in unrepeatable), default=None),
        "max_abs_between_variants_rel": max((u["max_abs_between_variants_rel"] for u in unrepeatable), default=None),
        "unrepeatable_calls": unrepeatable,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--book", default="pride_and_prejudice")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    lm = load(MODEL_REPO, MODEL_REVISION, strategy="cudnn_bucketed")
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    _, bucket = current_strategy()
    tokens = load_tokens(args.book, lm.tokenizer)
    rows = []
    for ctx in (int(c) for c in args.contexts.split(",")):
        for strategy in STRATEGIES:
            torch.cuda.empty_cache()
            r = probe(lm, tokens, strategy, ctx, args.steps, args.repeats, bucket)
            rows.append(r)
            log.info("ctx %d %s: %d/%d decode attention calls not repeatable", ctx, strategy, r["calls_not_repeatable"], r["decode_calls_checked"])
    write_metrics(
        RESULTS_DIR / "phase1" / "kernel_repeatability",
        {
            "config": vars(args),
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "bucket": bucket,
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "rows": rows,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": process_gpu_memory().shared_bytes},
        },
    )


if __name__ == "__main__":
    main()
