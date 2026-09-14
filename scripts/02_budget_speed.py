"""Phase 2 gate, speed half: decode latency per policy and budget at 32K.

Fast kernel (cudnn_bucketed), Windows power throttling opted out (Phase 1). Each repeat
prefills once, then runs every condition in shuffled order from that prefill. Phase 1
showed decode here is host-bound, so the question is how much latency each policy adds,
and whether its manager's host time explains it.

Recorded per condition: decode wall latency distribution, boundary build time, manager
host time per token (update + observe), evictions, resident KV bytes, peak allocation.

Launch detached, once per run id:
    .venv\\Scripts\\python.exe scripts\\02_budget_speed.py --run-id 1 > logs\\budget_speed_run_1.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import allocator_counters, cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.host import set_power_throttling  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from harness.telemetry import TelemetryLogger  # noqa: E402
from lazykv.blocks import BlockPoolCache  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402
from lazykv.sweep import build_cache, conditions, load_config  # noqa: E402

log = logging.getLogger("budget_speed")
FAST_STRATEGY = "cudnn_bucketed"
SPILL_THRESHOLD_BYTES = 64 * 2**20


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase2.yaml"))
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--out", default="budget_speed")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    sp = cfg["speed"]
    ctx = args.context or cfg["context"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    n_decode = sp["warmup_steps"] + sp["decode_tokens"]
    conds = conditions(cfg)

    set_power_throttling(opt_out=True)
    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=FAST_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    tokens = load_tokens(sp["book"], lm.tokenizer)
    total = ctx + n_decode + 1
    full = FullGPUCache(lm.num_layers, -(-(total + 16) // 1024) * 1024)
    out_dir = RESULTS_DIR / "phase2" / args.out / f"run_{args.run_id}"
    rng = random.Random(1000 + args.run_id)

    # Warmup: plans for prefill and for every condition's decode shape, outside the timed repeats.
    pre = prefill(lm.model, full, tokens[:ctx], chunk)
    for cond in conds:
        built = build_cache(full, cond, total, bs)
        greedy_decode(lm.model, built.cache, pre.last_logits, 4)
        if built.cache is full:
            full.truncate(ctx)
        del built

    rows: list[dict[str, Any]] = []
    with TelemetryLogger(out_dir / "telemetry.csv") as tel:
        for rep in range(sp["repeats"]):
            full.truncate(0)
            tel.mark(f"rep{rep}_prefill")
            pre = prefill(lm.model, full, tokens[:ctx], chunk)
            order = conds[:]
            rng.shuffle(order)
            for cond in order:
                tel.mark(f"rep{rep}_{cond.label}_start")
                torch.cuda.reset_peak_memory_stats()
                retries_before = allocator_counters()
                built = build_cache(full, cond, total, bs)
                before = built.cache.counters() if isinstance(built.cache, BlockPoolCache) else None
                dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_decode)
                wall = dec.wall_s[sp["warmup_steps"] :]
                row: dict[str, Any] = {
                    "repeat": rep,
                    "policy": cond.policy,
                    "budget": cond.budget,
                    "n_slots": built.n_slots,
                    "boundary_build_s": built.build_s,
                    "decode_wall_s": summarize(wall).to_dict(),
                    "decode_wall_p10_s": pct(wall, 0.10),
                    "decode_wall_p90_s": pct(wall, 0.90),
                    "decode_wall_raw_s": wall,
                    "gpu_resident_kv_bytes": built.cache.stats().gpu_resident_kv_bytes,
                    "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                if before is not None:
                    after = built.cache.counters()
                    # Manager host time over every decode step of this condition (warmup included).
                    row["manager_host_s_per_token"] = (after["host_update_s"] + after["host_observe_s"]) / n_decode
                    row["manager_update_s_per_token"] = after["host_update_s"] / n_decode
                    row["manager_observe_s_per_token"] = after["host_observe_s"] / n_decode
                    row["evictions"] = after["evictions"]
                    row["boundary_evicted_blocks"] = after["boundary_evicted_blocks"]
                mem = process_gpu_memory()
                spill = None if mem.shared_bytes is None or shared_baseline is None else mem.shared_bytes - shared_baseline
                row["shared_growth_bytes"] = spill
                row["spilled_to_shared"] = None if spill is None else spill > SPILL_THRESHOLD_BYTES
                row["alloc_retries_during"] = allocator_counters()["num_alloc_retries"] - retries_before["num_alloc_retries"]
                rows.append(row)
                if built.cache is full:
                    full.truncate(ctx)
                del built
                tel.mark(f"rep{rep}_{cond.label}_end")
                log.info("rep %d %s: median %.2f ms, p90 %.2f ms, manager %.3f ms/token, resident %.0f MiB, spill %s",
                         rep, cond.label, 1e3 * row["decode_wall_s"]["median"], 1e3 * row["decode_wall_p90_s"],
                         1e3 * row.get("manager_host_s_per_token", 0.0), row["gpu_resident_kv_bytes"] / 2**20, row["spilled_to_shared"])

    write_metrics(
        out_dir,
        {
            "config": {**cfg, "context": ctx, "run_id": args.run_id},
            "attention_strategy": FAST_STRATEGY,
            "host": {"power_throttling": "opted_out"},
            "kv_bytes_per_token": lm.kv_bytes_per_token,
            "conditions": [c.label for c in conds],
            "prefill_wall_s": pre.wall_s,
            "results": rows,
            "telemetry": {**tel.summary(), "marks": tel.marks, "csv": "telemetry.csv"},
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "spill_threshold_bytes": SPILL_THRESHOLD_BYTES},
        },
    )
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
