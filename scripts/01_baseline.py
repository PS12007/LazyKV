"""Phase 1 gate experiment: trustworthy full-GPU-KV baseline (policy ladder rung 1).

For each context length (interleaved, randomized order, repeated):
  chunked prefill -> TTFT
  timed greedy decode -> per-token latency distribution (warmup steps discarded)
  profiled decode -> per-layer split: attention kernel / attention rest / MLP / other
  memory: torch peak allocated/reserved, mem_get_info, cache-reported KV bytes

Launch detached (CLAUDE.md rule 4):
    .venv\\Scripts\\python.exe scripts\\01_baseline.py --run-id 1 > logs\\baseline_run_1.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from harness import sysinfo  # noqa: E402
from harness.corpus import load_tokens  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from harness.telemetry import TelemetryLogger  # noqa: E402
from lazykv.attention import current_strategy  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402
from lazykv.profiling import LayerProfiler  # noqa: E402

log = logging.getLogger("baseline")

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


def gpu_spinup(seconds: float = 2.0) -> None:
    """Same rationale as Phase 0: the laptop GPU downclocks within seconds of idle."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for _ in range(8):
            a @ a
        torch.cuda.synchronize()


def nvidia_smi_process_memory() -> str:
    """Per-process GPU memory as nvidia-smi reports it (often [N/A] under WDDM)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    import os

    mine = [line for line in out.splitlines() if line.split(",")[0].strip() == str(os.getpid())]
    return mine[0].split(",")[1].strip() if mine else "process not listed"


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def run_condition(lm, cache: FullGPUCache, tokens: torch.Tensor, ctx: int, args: argparse.Namespace) -> dict[str, Any]:  # noqa: ANN001
    cache_layers_reset(cache)
    torch.cuda.reset_peak_memory_stats()
    gpu_spinup(1.0)
    ids = tokens[:ctx]
    pre = prefill(lm.model, cache, ids, args.chunk_size)
    dec = greedy_decode(lm.model, cache, pre.last_logits, args.warmup_steps + args.decode_tokens)
    wall = dec.wall_s[args.warmup_steps :]
    ev = dec.event_s[args.warmup_steps :]
    stats_after_decode = cache.stats()

    # Profiled steps continue on the same cache; the few extra tokens do not change the
    # context length materially, and a second 64K prefill would double run time.
    prof_wall: list[float] = []
    with LayerProfiler(lm.model) as prof:
        last = torch.zeros(lm.model.config.vocab_size, device="cuda")
        last[dec.tokens[-1]] = 1.0
        for _ in range(args.profile_warmup):
            greedy_decode(lm.model, cache, last, 1)
        prof.reset()
        pdec = greedy_decode(lm.model, cache, last, args.profile_tokens)
        prof_wall = pdec.wall_s
        per_layer = prof.collect()

    layers = sorted(per_layer)
    comp = {k: [per_layer[i][k] for i in layers] for k in ("layer_s", "attention_kernel_s", "attention_other_s", "mlp_s", "other_s")}
    free, total = torch.cuda.mem_get_info()
    return {
        "ctx_len": ctx,
        "ttft_wall_s": pre.wall_s,
        "ttft_event_s": pre.event_s,
        "prefill_chunks": pre.chunks,
        "prefill_tokens_per_s": ctx / pre.wall_s,
        "decode_wall_s": summarize(wall).to_dict(),
        "decode_event_s": summarize(ev).to_dict(),
        "decode_wall_p10_s": pct(wall, 0.10),
        "decode_wall_p90_s": pct(wall, 0.90),
        "decode_wall_p99_s": pct(wall, 0.99),
        "decode_tokens_per_s": 1.0 / statistics.median(wall),
        "decode_wall_raw_s": wall,
        "host_overhead_median_s": statistics.median(w - e for w, e in zip(wall, ev)),
        "profiled_decode_wall_median_s": statistics.median(prof_wall),
        "profiler_overhead_ratio": statistics.median(prof_wall) / statistics.median(wall),
        "per_layer_mean_s": {k: statistics.mean(v) for k, v in comp.items()},
        "per_layer_s": comp,
        "all_layers_sum_s": {k: sum(v) for k, v in comp.items()},
        "memory": {
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "mem_get_info_free_bytes": free,
            "mem_get_info_total_bytes": total,
            "nvidia_smi_process_used_mib": nvidia_smi_process_memory(),
            "gpu_resident_kv_bytes": stats_after_decode.gpu_resident_kv_bytes,
            "host_kv_bytes": stats_after_decode.host_kv_bytes,
            "gpu_residency_fraction": stats_after_decode.gpu_residency_fraction,
            "gpu_allocated_kv_bytes": stats_after_decode.gpu_allocated_kv_bytes,
        },
        "generated_token_ids_head": dec.tokens[:16],
    }


def cache_layers_reset(cache: FullGPUCache) -> None:
    for layer in cache.layers:
        layer.reset()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--contexts", default="4096,8192,16384,32768,65536")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--profile-tokens", type=int, default=24)
    p.add_argument("--profile-warmup", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--book", default="moby_dick")
    p.add_argument("--quick", action="store_true", help="smoke test: small contexts, few tokens")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.quick:
        args.contexts, args.repeats, args.decode_tokens, args.profile_tokens = "1024,4096", 1, 16, 8

    contexts = [int(c) for c in args.contexts.split(",")]
    out_dir = RESULTS_DIR / "phase1" / ("baseline_quick" if args.quick else "baseline") / f"run_{args.run_id}"
    torch.manual_seed(args.seed)

    lm = load(MODEL_REPO, MODEL_REVISION)
    log.info("loaded %s, attention strategy %s", MODEL_REPO, lm.attention_strategy)
    tokens = load_tokens(args.book, lm.tokenizer)
    needed = max(contexts) + args.warmup_steps + args.decode_tokens + args.profile_warmup + args.profile_tokens + 8
    # Whole buckets, so the bucketed decode view always has real storage to pad into.
    _, bucket = current_strategy()
    max_len = -(-needed // bucket) * bucket
    if tokens.numel() < max(contexts):
        raise SystemExit("book too short for requested context")
    cache = FullGPUCache(lm.num_layers, max_len)

    # Global warmup: allocate the full cache, JIT cuDNN plans for both prefill and decode.
    run_condition(lm, cache, tokens, min(contexts), argparse.Namespace(**{**vars(args), "decode_tokens": 8, "profile_tokens": 4}))

    order_rng = random.Random(args.seed + args.run_id)
    results: list[dict[str, Any]] = []
    with TelemetryLogger(out_dir / "telemetry.csv") as tel:
        for rep in range(args.repeats):
            order = contexts[:]
            order_rng.shuffle(order)
            for ctx in order:
                tel.mark(f"ctx{ctx}_rep{rep}_start")
                r = run_condition(lm, cache, tokens, ctx, args)
                r["repeat"] = rep
                results.append(r)
                tel.mark(f"ctx{ctx}_rep{rep}_end")
                log.info(
                    "rep %d ctx %d: TTFT %.2fs, decode median %.2f ms (%.1f tok/s), layer %.3f ms, peak alloc %.2f GiB",
                    rep, ctx, r["ttft_wall_s"], 1e3 * r["decode_wall_s"]["median"], r["decode_tokens_per_s"],
                    1e3 * r["per_layer_mean_s"]["layer_s"], r["memory"]["max_memory_allocated_bytes"] / 2**30,
                )
    write_metrics(
        out_dir,
        {
            "config": {k: v for k, v in vars(args).items()},
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "dtype": str(lm.model.dtype),
                "kv_bytes_per_token": lm.kv_bytes_per_token,
                "num_layers": lm.num_layers,
            },
            "attention": {"strategy": lm.attention_strategy, "bucket": current_strategy()[1], "checks": [c.__dict__ for c in lm.kernel_checks]},
            "cache": {"policy": "full_gpu_preallocated", "max_len": max_len},
            "results": results,
            "telemetry": {**tel.summary(), "marks": tel.marks, "csv": "telemetry.csv"},
            "system": {"software_gpu": sysinfo.gpu_and_software()},
        },
    )
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
