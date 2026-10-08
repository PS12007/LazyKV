"""Phase 21, question 4: what it costs the tier to read a user's turn densely instead of token by token.

Per context (configs/phase21.yaml, `cost:`), in one process:
  1. One layer-major tiered prefill of a NIAH prompt (Phase 17's path), saved as a snapshot
     (lazykv/snapshot.py, Phase 20). Every measurement below starts from a restore of that snapshot,
     so each one ingests into the same tier state without paying a prefill per repeat.
  2. For each turn length: ingest a passage of that many tokens in one forward (dense) and, for the
     per-token lengths, one token per forward through the decode path. Wall time, bytes moved host to
     device (TierCounters.ingest_h2d_bytes), and the peak VRAM above what the tier held before.
The restore is outside every timed region. The turn tokens are a passage from another book, as in
Phase 19's sessions; their content does not matter for timing.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\10_ingest_cost.py > logs\\phase21_cost.log 2>&1
"""

from __future__ import annotations

import argparse
import gc
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated  # noqa: E402
from harness.host import set_power_throttling  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from lazykv import snapshot  # noqa: E402
from lazykv.attention import set_strategy_for_test  # noqa: E402
from lazykv.generate import cache_model_kwargs, load  # noqa: E402
from lazykv.niah import build_prompt  # noqa: E402
from lazykv.selection import QUEST_DENSE_LAYERS, blocks_for_budget  # noqa: E402
from lazykv.sweep import load_config  # noqa: E402
from lazykv.tiered import allocate_host_pools  # noqa: E402
from lazykv.tiered_prefill import tiered_prefill  # noqa: E402

log = logging.getLogger("ingest_cost")


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def ingest(model: Any, cache: Any, turn: torch.Tensor, dense: bool) -> dict[str, float]:
    """Feed `turn` [1, T] into `cache`; wall time, H2D bytes and peak VRAM above the starting point."""
    extra = cache_model_kwargs(cache)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    c0 = cache.counters.ingest_h2d_bytes
    t0 = time.perf_counter()
    if dense:
        model(input_ids=turn, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
    else:
        for j in range(turn.shape[1]):
            model(input_ids=turn[:, j : j + 1], past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "h2d_bytes": cache.counters.ingest_h2d_bytes - c0, "peak_extra_bytes": torch.cuda.max_memory_allocated() - base}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase21.yaml"))
    ap.add_argument("--contexts", nargs="*", type=int, help="override the context ladder (smoke tests)")
    ap.add_argument("--repeats", type=int, help="override repeats (smoke tests)")
    ap.add_argument("--out", default="ingest_cost")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    cc = cfg["cost"]
    contexts = args.contexts or cc["contexts"]
    repeats = args.repeats or cc["repeats"]
    set_power_throttling(opt_out=True)
    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=cc["strategy"])
    set_strategy_for_test(cc["strategy"])
    cap_info = cap_allocator_to_dedicated()
    c = lm.model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    book = load_tokens(cfg["book"], lm.tokenizer)
    turn_src = load_tokens(cc["turn_book"], lm.tokenizer)
    longest = max(cc["dense_turn_tokens"] + cc["per_token_turn_tokens"])
    snap_dir = Path(tempfile.gettempdir()) / "lazykv_snapshots"
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for ctx in contexts:
        prompt = build_prompt(lm.tokenizer, book, ctx, "single", 0.5, 0, seed=cfg["seed"])
        cap = -(-(ctx + longest + 16) // 1024) * 1024
        # As Phase 19: the budget is a fraction of the session's planned length.
        k = blocks_for_budget(cc["budget"], ctx + longest, bs)
        pools = allocate_host_pools(lm.num_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, bs, cap, lm.model.dtype)
        path = snap_dir / f"ingest_ctx{ctx}.lkv"
        row: dict[str, Any] = {"context": ctx, "k_blocks": k, "capacity_tokens": cap, "kv_bytes": ctx * lm.kv_bytes_per_token, "cells": []}
        try:
            release()
            res = tiered_prefill(lm.model, prompt.input_ids, chunk, k, bs, cap, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"])
            row["prefill_s"] = res.wall_s
            snapshot.save(res.cache, res.last_logits, path)
            res.cache.close()
            del res
            plan = [(n, True) for n in cc["dense_turn_tokens"]] + [(n, False) for n in cc["per_token_turn_tokens"]]
            for n, dense in plan:
                turn = turn_src[1000 : 1000 + n].to("cuda").view(1, -1)  # past the BOS and title page
                cell: dict[str, Any] = {"turn_tokens": n, "mode": "dense" if dense else "per_token", "runs": []}
                for _ in range(repeats):
                    release()
                    cache, _, _ = snapshot.restore(path, k, cap, lm.num_layers, model=lm.model, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"])
                    torch.cuda.synchronize()
                    tier_bytes = torch.cuda.memory_allocated()
                    run = ingest(lm.model, cache, turn, dense)
                    run["allocated_before_bytes"] = tier_bytes
                    run["length_after"] = cache.get_seq_length(QUEST_DENSE_LAYERS)
                    cache.close()
                    del cache
                    cell["runs"].append(run)
                walls = [r["wall_s"] for r in cell["runs"]]
                cell["wall_s_median"], cell["wall_s_min"], cell["wall_s_max"] = statistics.median(walls), min(walls), max(walls)
                cell["ms_per_token_median"] = 1000 * cell["wall_s_median"] / n
                cell["h2d_bytes"] = cell["runs"][0]["h2d_bytes"]
                cell["peak_extra_bytes_max"] = max(r["peak_extra_bytes"] for r in cell["runs"])
                row["cells"].append(cell)
                log.info("ctx %d %s %d tokens: %.3fs median (%.0fs elapsed)", ctx, cell["mode"], n, cell["wall_s_median"], time.perf_counter() - t_start)
            row["status"] = "ok"
        except (torch.OutOfMemoryError, RuntimeError, OSError) as exc:
            row["status"], row["error"] = "error", str(exc).splitlines()[0][:300]
            log.error("ctx %d: %s", ctx, row["error"])
        finally:
            path.unlink(missing_ok=True)
        rows.append(row)
        del pools
        release()

    write_metrics(RESULTS_DIR / cfg["phase"] / args.out, {
        "config": {**cc, "contexts": contexts, "repeats": repeats, "block_size": bs, "prefill_chunk": chunk, "fetch": cfg["tier"]["fetch"]},
        "rows": rows, "allocator_cap": cap_info, "host": {"power_throttling": "opted_out"}, "wall_s": time.perf_counter() - t_start,
    })


if __name__ == "__main__":
    main()
