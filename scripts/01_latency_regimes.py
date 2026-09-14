"""Why does batch-1 decode latency switch between two regimes?

The Phase 1 baseline's per-token latencies are not one distribution: stretches of about a
second run near twice the fast latency, at every context length, with the GPU at full
clock but lower utilization. So the GPU is waiting on the host. The affinity experiment
showed P-cores-only still switches, so core class alone is not the explanation.

This tests two host-side hypotheses in a 2x2 design, interleaved and repeated:
  throttling   Windows power throttling (EcoQoS) left to the OS, vs opted out
  nvidia_smi   the telemetry poller running, vs not (a WDDM query may stall the driver)

Per token it records a timestamp, wall latency, and the logical processor the Python thread
was on when the step finished. Throughout, typeperf samples CPU "% Processor Performance"
(frequency relative to nominal) so a slow stretch can be lined up with a frequency drop.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\01_latency_regimes.py > logs\\latency_regimes.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.host import core_class_masks, current_processor_number, set_power_throttling  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from harness.telemetry import TelemetryLogger  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import load, prefill  # noqa: E402

log = logging.getLogger("latency_regimes")

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
COUNTERS = (r"\Processor Information(_Total)\% Processor Performance", r"\Processor Information(_Total)\% Processor Utility")


class CpuCounters:
    """typeperf streaming Windows performance counters; a built-in tool, so no new dependency."""

    def __init__(self, t0: float, interval_s: int = 1) -> None:
        self.t0, self.rows = t0, []
        self._proc = subprocess.Popen(["typeperf", *COUNTERS, "-si", str(interval_s)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = [p.strip('"') for p in line.strip().split(",")]
            if len(parts) != 1 + len(COUNTERS):
                continue
            try:
                perf, util = float(parts[1]), float(parts[2])
            except ValueError:
                continue  # header line
            self.rows.append({"t_s": time.perf_counter() - self.t0, "processor_performance_pct": perf, "processor_utility_pct": util})

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


@torch.inference_mode()
def timed_decode(model, cache: FullGPUCache, first_logits: torch.Tensor, n: int, t0: float) -> dict[str, list[float] | list[int]]:  # noqa: ANN001
    """greedy_decode's loop, plus a timestamp and the current logical processor per token."""
    tok = int(torch.argmax(first_logits).item())
    ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
    wall, stamp, cpu = [], [], []
    for _ in range(n):
        ids.fill_(tok)
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        tok = int(torch.argmax(out.logits[0, -1]).item())
        e = time.perf_counter()
        wall.append(e - s)
        stamp.append(e - t0)
        cpu.append(current_processor_number())
    return {"wall_s": wall, "t_end_s": stamp, "processor": cpu}


def slow_episodes(wall: list[float], threshold: float) -> list[int]:
    """Lengths (tokens) of maximal runs of consecutive tokens above threshold."""
    runs, cur = [], 0
    for w in wall:
        if w > threshold:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--decode-tokens", type=int, default=384)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--book", default="moby_dick")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = RESULTS_DIR / "phase1" / "latency_regimes"

    lm = load(MODEL_REPO, MODEL_REVISION)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    tokens = load_tokens(args.book, lm.tokenizer)
    cache = FullGPUCache(lm.num_layers, -(-(args.ctx + args.warmup_steps + args.decode_tokens + 8) // 1024) * 1024)
    masks = core_class_masks()
    p_mask = masks[max(masks)]

    conditions = [{"throttling": t, "nvidia_smi": s} for t in ("os_default", "opted_out") for s in (True, False)]
    t0 = time.perf_counter()
    counters = CpuCounters(t0)
    blocks: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    try:
        for rep in range(args.repeats):
            order = conditions[:]
            rng.shuffle(order)
            for cond in order:
                set_power_throttling(cond["throttling"] == "opted_out")
                tel_path = out_dir / "telemetry" / f"rep{rep}_{cond['throttling']}.csv"
                tel = TelemetryLogger(tel_path) if cond["nvidia_smi"] else None
                if tel:
                    tel.__enter__()
                try:
                    for layer in cache.layers:
                        layer.reset()
                    pre = prefill(lm.model, cache, tokens[: args.ctx], 2048)
                    dec = timed_decode(lm.model, cache, pre.last_logits, args.warmup_steps + args.decode_tokens, t0)
                finally:
                    if tel:
                        tel.__exit__(None, None, None)
                w = args.warmup_steps
                wall = dec["wall_s"][w:]
                procs = dec["processor"][w:]
                blocks.append({**cond, "repeat": rep, "wall_s": wall, "t_end_s": dec["t_end_s"][w:], "processor": procs})
                log.info("rep %d %s nvidia_smi=%s: median %.2f ms, p90 %.2f ms, on P-cores %.0f%%", rep, cond["throttling"], cond["nvidia_smi"],
                         1e3 * statistics.median(wall), 1e3 * sorted(wall)[int(0.9 * (len(wall) - 1))], 100 * sum(1 for c in procs if p_mask >> c & 1) / len(procs))
    finally:
        set_power_throttling(False)
        counters.close()

    # One fast-regime reference for all conditions: the 10th percentile of every token measured.
    everything = sorted(x for b in blocks for x in b["wall_s"])
    fast = everything[int(0.10 * (len(everything) - 1))]
    threshold = 1.5 * fast

    def cond_summary(sel: list[dict[str, Any]]) -> dict[str, Any]:
        wall = [x for b in sel for x in b["wall_s"]]
        procs = [c for b in sel for c in b["processor"]]
        eps = [e for b in sel for e in slow_episodes(b["wall_s"], threshold)]
        slow_on_p = [p_mask >> c & 1 for b in sel for x, c in zip(b["wall_s"], b["processor"]) if x > threshold]
        fast_on_p = [p_mask >> c & 1 for b in sel for x, c in zip(b["wall_s"], b["processor"]) if x <= threshold]
        return {
            "tokens": len(wall),
            "decode_wall_s": summarize(wall).to_dict(),
            "p90_s": sorted(wall)[int(0.9 * (len(wall) - 1))],
            "slow_fraction": sum(1 for x in wall if x > threshold) / len(wall),
            "per_block_median_s": [statistics.median(b["wall_s"]) for b in sel],
            "per_block_slow_fraction": [sum(1 for x in b["wall_s"] if x > threshold) / len(b["wall_s"]) for b in sel],
            "p_core_fraction": sum(1 for c in procs if p_mask >> c & 1) / len(procs),
            "p_core_fraction_slow_tokens": sum(slow_on_p) / len(slow_on_p) if slow_on_p else None,
            "p_core_fraction_fast_tokens": sum(fast_on_p) / len(fast_on_p) if fast_on_p else None,
            "slow_episode_tokens": summarize(eps).to_dict() if eps else None,
        }

    # CPU frequency during slow vs fast tokens: each token takes the nearest counter sample.
    samples = counters.rows

    def perf_at(t: float) -> float | None:
        if not samples:
            return None
        return min(samples, key=lambda r: abs(r["t_s"] - t))["processor_performance_pct"]

    slow_perf = [perf_at(t) for b in blocks for x, t in zip(b["wall_s"], b["t_end_s"]) if x > threshold]
    fast_perf = [perf_at(t) for b in blocks for x, t in zip(b["wall_s"], b["t_end_s"]) if x <= threshold]
    slow_perf = [v for v in slow_perf if v is not None]
    fast_perf = [v for v in fast_perf if v is not None]

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(
        out_dir,
        {
            "config": vars(args),
            "attention_strategy": lm.attention_strategy,
            "core_class_masks": {str(k): hex(v) for k, v in masks.items()},
            "fast_reference_p10_s": fast,
            "slow_threshold_s": threshold,
            "slow_threshold_rule": "a token is slow if its wall latency exceeds 1.5x the 10th percentile of all tokens measured",
            "conditions": {
                f"{c['throttling']}__nvidia_smi_{'on' if c['nvidia_smi'] else 'off'}": cond_summary([b for b in blocks if b["throttling"] == c["throttling"] and b["nvidia_smi"] == c["nvidia_smi"]])
                for c in conditions
            },
            "factor_throttling": {t: cond_summary([b for b in blocks if b["throttling"] == t]) for t in ("os_default", "opted_out")},
            "factor_nvidia_smi": {s: cond_summary([b for b in blocks if b["nvidia_smi"] == (s == "on")]) for s in ("on", "off")},
            "cpu_performance_pct": {
                "samples": len(samples),
                "during_slow_tokens": summarize(slow_perf).to_dict() if slow_perf else None,
                "during_fast_tokens": summarize(fast_perf).to_dict() if fast_perf else None,
                "trace": samples,
            },
            "blocks": blocks,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": shared_after},
        },
    )
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
