"""Phase 17, questions 1 and 2: the longest context each prefill path reaches, and what it costs.

Driver mode (default) runs every (model, method, context) cell of configs/phase17.yaml in its own
process, in ascending context order, and stops a (model, method) ladder at its first failure:
memory use only grows with context, so the cells past a failure would only repeat it. Cell mode
(`--cell`) loads the model, builds one NIAH prompt, prefills it, decodes, and writes one JSON.

  full     chunk-major prefill into a FullGPUCache, full-cache decode (rung 1)
  tiered   layer-major prefill into the CPU tier (lazykv/tiered_prefill.py), rung 6 decode

Why a fresh process per cell: an OOM leaves the caching allocator fragmented, and pinned host pools
cannot be resized, so a cell after a failure, or after a smaller cell, would not measure its own
peak. Model load is the price, about half a minute per cell.

The allocator is capped at the dedicated VRAM free at startup (harness/gpu_memory.py), so a cell
that does not fit raises OOM rather than spilling into shared memory as Phase 15's 32K run did; the
process's shared GPU memory is read after each cell to prove it.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\07_prefill_capacity.py > logs\\phase17_capacity.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from harness.gpu_memory import allocator_counters, cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402

log = logging.getLogger("prefill_capacity")
METHODS = ("full", "tiered")
GIB = 2**30


def capacity_for(ctx: int, decode_tokens: int) -> int:
    """KV capacity in tokens: the prompt plus the decode, rounded up as the other drivers do."""
    return -(-(ctx + decode_tokens + 16) // 1024) * 1024


def run_cell(cfg: dict[str, Any], model_key: str, method: str, ctx: int, strategy: str) -> dict[str, Any]:
    from harness.corpus import load_tokens
    from lazykv.attention import set_strategy_for_test
    from lazykv.cache import FullGPUCache
    from lazykv.generate import greedy_decode, load, prefill
    from lazykv.niah import build_prompt, score
    from lazykv.selection import QUEST_DENSE_LAYERS, blocks_for_budget
    from lazykv.tiered import allocate_host_pools
    from lazykv.tiered_prefill import tiered_prefill

    m = cfg["models"][model_key]
    row: dict[str, Any] = {"model": model_key, "method": method, "context": ctx, "strategy": strategy}
    t_load = time.perf_counter()
    lm = load(m["repo"], m.get("revision"), strategy=strategy, quant=m.get("quant"))
    set_strategy_for_test(strategy)
    row["load_s"] = time.perf_counter() - t_load
    row["cap"] = cap_allocator_to_dedicated()
    row["weights_allocated_bytes"] = torch.cuda.memory_allocated()
    row["kv_bytes_per_token"] = lm.kv_bytes_per_token
    p = cfg["capacity_prompt"]
    book = load_tokens(p["book"], lm.tokenizer)
    prompt = build_prompt(lm.tokenizer, book, ctx, p["kind"], p["depth"], p["sample"], seed=p["seed"])
    n_dec = cfg["decode_tokens"]
    cap = capacity_for(ctx, n_dec)
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    row["capacity_tokens"] = cap
    stage = "setup"
    try:
        if method == "tiered":
            stage = "host_pools"
            c = lm.model.config
            head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
            t0 = time.perf_counter()
            pools = allocate_host_pools(lm.num_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, bs, cap, lm.model.dtype)
            row["host_pin_s"] = time.perf_counter() - t0
            row["host_pinned_bytes"] = sum(t.numel() * t.element_size() for t in pools)
            k = blocks_for_budget(cfg["budget"], ctx + n_dec, bs)
            row["k_blocks"] = k
        torch.cuda.reset_peak_memory_stats()
        stage = "prefill"
        if method == "full":
            cache = FullGPUCache(lm.num_layers, cap)
            pre = prefill(lm.model, cache, prompt.input_ids, chunk)
            first, row["prefill_wall_s"] = pre.last_logits, pre.wall_s
        else:
            res = tiered_prefill(lm.model, prompt.input_ids, chunk, k, bs, cap, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"])
            cache, first, row["prefill_wall_s"] = res.cache, res.last_logits, res.wall_s
            row["boundary_s"] = res.boundary_s
            row["layer_s"] = res.layer_s
        row["prefill_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["prefill_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        stage = "decode"
        dec = greedy_decode(lm.model, cache, first, n_dec - 1)
        row["decode_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        # The first steps include cold misses and plan builds; the median is the steady state.
        row["decode_wall_ms_median"] = 1e3 * statistics.median(dec.wall_s)
        row["decode_wall_ms"] = [1e3 * s for s in dec.wall_s]
        if method == "tiered":
            row["counters"] = {k2: v for k2, v in cache.counters_dict().items() if k2 != "fetched_pairs_per_step"}
            cache.close()
        eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
        text = lm.tokenizer.decode(toks)
        row["answer"], row["score"] = text.strip()[:120], score(prompt, text)
        row["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        row["status"], row["error"] = "gpu_oom", str(exc).splitlines()[0][:300]
    except RuntimeError as exc:
        # Pinning fails with a CUDA host-allocation error, not an OOM type.
        row["status"] = "host_oom" if stage == "host_pools" else "error"
        row["error"] = str(exc).splitlines()[0][:300]
    row["failed_stage"] = None if row["status"] == "ok" else stage
    row["allocator"] = allocator_counters()
    row["process_gpu_memory"] = process_gpu_memory().to_dict()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase17.yaml"))
    ap.add_argument("--cell", nargs=3, metavar=("MODEL", "METHOD", "CONTEXT"))
    ap.add_argument("--cell-out")
    ap.add_argument("--models", nargs="*", help="driver: restrict to these model keys")
    ap.add_argument("--contexts", nargs="*", type=int, help="driver: override every model's context ladder (smoke tests)")
    ap.add_argument("--strategy", default="cudnn_bucketed", help="the speed default; capacity is a deployment question")
    ap.add_argument("--out", default="capacity")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.cell:
        model_key, method, ctx = args.cell
        row = run_cell(cfg, model_key, method, int(ctx), args.strategy)
        Path(args.cell_out).write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        return

    out_dir = RESULTS_DIR / cfg["phase"] / args.out
    cell_dir = out_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for model_key in args.models or list(cfg["models"]):
        ladder = args.contexts or cfg["models"][model_key]["contexts"]
        for method in METHODS:
            failed_at = None
            for ctx in sorted(ladder):
                if failed_at is not None:
                    rows.append({"model": model_key, "method": method, "context": ctx, "status": "skipped", "after_failure_at": failed_at})
                    continue
                path = cell_dir / f"{model_key}_{method}_{ctx}.json"
                cmd = [sys.executable, __file__, "--config", args.config, "--strategy", args.strategy, "--cell", model_key, method, str(ctx), "--cell-out", str(path)]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0 or not path.exists():
                    # A process that dies outright (e.g. the OS refusing pinned memory mid-load) still gets a row.
                    row = {"model": model_key, "method": method, "context": ctx, "status": "crashed", "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:]}
                else:
                    row = json.loads(path.read_text(encoding="utf-8"))
                row["cell_wall_s"] = time.perf_counter() - t0
                rows.append(row)
                log.info("%s %s %d: %s prefill_peak=%.2f GiB prefill=%.1fs decode=%.1f ms/tok score=%s (%.0fs)", model_key, method, ctx, row["status"],
                         row.get("prefill_peak_allocated_bytes", 0) / GIB, row.get("prefill_wall_s", float("nan")), row.get("decode_wall_ms_median", float("nan")),
                         row.get("score"), time.perf_counter() - t_start)
                if row["status"] != "ok":
                    failed_at = ctx
    write_metrics(out_dir, {"config": cfg, "strategy": args.strategy, "cells": rows, "wall_s": time.perf_counter() - t_start})
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
