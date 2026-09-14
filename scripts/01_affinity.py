"""Decode latency vs CPU core type (Windows, hybrid P/E CPU).

Phase 1 found batch-1 decode is host-bound: most of each layer's GPU-timeline span is
Python/transformers kernel launching, not GPU compute. On a hybrid CPU that makes latency
depend on which core the Python thread lands on. This measures default scheduling vs
P-cores only vs E-cores only, interleaved, at a fixed 4K context.

Core classes come from GetLogicalProcessorInformationEx (EfficiencyClass), not assumed.
"""

from __future__ import annotations

import argparse
import ctypes
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.host import core_class_masks, set_power_throttling  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=5)
    args = p.parse_args()

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    masks = core_class_masks()
    all_mask = 0
    for m in masks.values():
        all_mask |= m
    conditions = {"default": all_mask, "p_cores_only": masks[max(masks)], "e_cores_only": masks[min(masks)]}

    # Without this, Windows power throttling moves the process between core classes on its
    # own schedule, and the "default" and "P-cores only" conditions stop meaning what they say.
    set_power_throttling(opt_out=True)
    lm = load(MODEL_REPO, MODEL_REVISION)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    tokens = load_tokens("moby_dick", lm.tokenizer)
    cache = FullGPUCache(lm.num_layers, 8192)
    samples: dict[str, list[float]] = {k: [] for k in conditions}
    per_repeat_median: dict[str, list[float]] = {k: [] for k in conditions}
    rng = random.Random(0)
    for rep in range(args.repeats + 1):  # repeat 0 is warmup, discarded
        order = list(conditions)
        rng.shuffle(order)
        for name in order:
            k32.SetProcessAffinityMask(k32.GetCurrentProcess(), conditions[name])
            for layer in cache.layers:
                layer.reset()
            pre = prefill(lm.model, cache, tokens[: args.ctx], 2048)
            dec = greedy_decode(lm.model, cache, pre.last_logits, args.warmup_steps + args.decode_tokens)
            if rep > 0:
                wall = dec.wall_s[args.warmup_steps :]
                samples[name] += wall
                per_repeat_median[name].append(statistics.median(wall))
    k32.SetProcessAffinityMask(k32.GetCurrentProcess(), all_mask)

    def pct(xs: list[float], q: float) -> float:
        s = sorted(xs)
        return s[round(q * (len(s) - 1))]

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(
        RESULTS_DIR / "phase1" / "affinity",
        {
            "config": vars(args),
            "core_class_masks": {str(k): hex(v) for k, v in masks.items()},
            "attention_strategy": lm.attention_strategy,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": shared_after},
            "host": {"power_throttling": "opted_out"},
            "conditions": {
                name: {
                    "mask": hex(mask),
                    "decode_wall_s": summarize(samples[name]).to_dict(),
                    "p10_s": pct(samples[name], 0.10),
                    "p90_s": pct(samples[name], 0.90),
                    "per_repeat_median_s": summarize(per_repeat_median[name]).to_dict(),
                }
                for name, mask in conditions.items()
            },
        },
    )


if __name__ == "__main__":
    main()
