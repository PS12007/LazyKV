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
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


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


def core_class_masks() -> dict[int, int]:
    """EfficiencyClass -> logical-processor affinity mask (group 0). Higher class = P-cores."""
    k32 = ctypes.WinDLL("kernel32")
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
