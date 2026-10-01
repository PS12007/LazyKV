"""Phase 17, questions 3 and 4: NIAH through layer-major tiered prefill, on the 3B model.

For each NIAH prompt of the config and each budget: a layer-major prefill straight into rung 6's tier
(lazykv/tiered_prefill.py), then rung 6's greedy decode. The tier's K depends on the budget, so the
prefill is repeated per budget; it is exact either way, and the logits it produces do not depend on K.

`--methods full` adds the full-cache reference on the same prompts, for the contexts where it fits.

At 16K the rows pair with Phase 15's rung 6 rows prompt by prompt (question 3). Kept identical to
Phase 15 so they can: kernel (lazykv.quality.QUALITY_STRATEGY), prompts, budgets, K (sized from the
prompt plus that kind's decode budget), slots (spare 0), fetch mode and prefill chunk.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\07_tiered_prefill_quality.py --context 16384 > logs\\phase17_q16k.log 2>&1
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

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import allocator_counters, cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402
from lazykv.niah import build_prompt, score  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY  # noqa: E402
from lazykv.selection import QUEST_DENSE_LAYERS, blocks_for_budget  # noqa: E402
from lazykv.sweep import load_config  # noqa: E402
from lazykv.tiered import allocate_host_pools, allocate_stages  # noqa: E402
from lazykv.tiered_prefill import tiered_prefill  # noqa: E402

log = logging.getLogger("tiered_prefill_quality")
# Phase 15 sized its KV capacity for its teacher-forced continuation too; the tier's metadata is
# sized from it, so the same rule keeps every tensor shape Phase 15's.
TF_CONTINUATION = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase17.yaml"))
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=["tiered"], choices=["tiered", "full"])
    ap.add_argument("--samples", type=int, help="override NIAH samples per kind x depth (smoke tests)")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    q = cfg["quality"]
    m = cfg["models"][q["model"]]
    niah = q["niah"]
    samples = args.samples or niah["samples"]
    ctx, bs, chunk = args.context, cfg["block_size"], cfg["prefill_chunk"]

    lm = load(m["repo"], m.get("revision"), strategy=QUALITY_STRATEGY, quant=m.get("quant"))
    set_strategy_for_test(QUALITY_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    max_new = max(niah["max_new_tokens"].values())
    cap = -(-(ctx + max(max_new, TF_CONTINUATION) + 16) // 1024) * 1024
    c = lm.model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    # Allocated once and reused, as every other driver does: pinning per prompt would time the OS.
    pools = allocate_host_pools(lm.num_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, bs, cap, lm.model.dtype) if "tiered" in args.methods else None
    k_max = max(blocks_for_budget(b, cap, bs) for b in q["budgets"])
    stages = allocate_stages(k_max * c.num_key_value_heads, c.num_key_value_heads, head_dim, bs, lm.model.dtype) if pools is not None else None
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    book = load_tokens(niah["book"], lm.tokenizer)
    rows: list[dict[str, Any]] = []
    peaks: list[int] = []
    t_start = time.perf_counter()

    def answer(dec_tokens: list[int]) -> str:
        toks = dec_tokens[: dec_tokens.index(eot)] if eot in dec_tokens else dec_tokens
        return lm.tokenizer.decode(toks)

    for kind in niah["kinds"]:
        n_new = niah["max_new_tokens"][kind]
        for depth in niah["depths"]:
            for sample in range(samples):
                prompt = build_prompt(lm.tokenizer, book, ctx, kind, depth, sample, seed=niah["seed"])
                base = {"kind": kind, "depth": depth, "sample": sample, "prompt_len": prompt.length, "values": list(prompt.values)}
                total = prompt.length + n_new
                if "full" in args.methods:
                    # Allocated per prompt and freed before the tier runs: a full cache kept across
                    # prompts would sit in VRAM beside every tiered decode and set that decode's peak.
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    full = FullGPUCache(lm.num_layers, cap)
                    pre = prefill(lm.model, full, prompt.input_ids, chunk)
                    dec = greedy_decode(lm.model, full, pre.last_logits, n_new - 1)
                    del full, pre
                    text = answer(dec.tokens)
                    peaks.append(torch.cuda.max_memory_allocated())
                    rows.append({**base, "method": "full", "policy": "full", "budget": 1.0, "score": score(prompt, text), "answer": text.strip()[:120],
                                 "prefill_wall_s": pre.wall_s, "peak_allocated_bytes": peaks[-1]})
                for budget in q["budgets"] if pools is not None else []:
                    k = blocks_for_budget(budget, total, bs)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    res = tiered_prefill(lm.model, prompt.input_ids, chunk, k, bs, cap, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"], stages=stages)
                    dec = greedy_decode(lm.model, res.cache, res.last_logits, n_new - 1)
                    res.cache.close()
                    text = answer(dec.tokens)
                    peaks.append(torch.cuda.max_memory_allocated())
                    cnt = res.cache.counters
                    rows.append({**base, "method": "tiered", "policy": "tiered_sync", "budget": budget, "k_blocks": k, "score": score(prompt, text),
                                 "answer": text.strip()[:120], "prefill_wall_s": res.wall_s, "boundary_s": res.boundary_s,
                                 "peak_allocated_bytes": peaks[-1], "fetched_pairs": cnt.fetched_pairs, "selected_pairs": cnt.selected_pairs})
                    del res
                this = {f"{r['policy']}@{r['budget']:g}": r["score"] for r in rows if (r["kind"], r["depth"], r["sample"]) == (kind, depth, sample)}
                log.info("niah %s depth %.2f sample %d (%.0fs): %s", kind, depth, sample, time.perf_counter() - t_start, this)

    shared_after = process_gpu_memory().shared_bytes
    write_metrics(RESULTS_DIR / cfg["phase"] / (args.out or f"quality_ctx{ctx}"), {
        "config": cfg, "context": ctx, "methods": args.methods, "samples": samples, "attention_strategy": QUALITY_STRATEGY,
        "capacity_tokens": cap, "kv_bytes_per_token": lm.kv_bytes_per_token, "niah": rows,
        "peak_allocated_bytes_max": max(peaks) if peaks else None,
        "memory_guard": {"cap": memory_cap, "shared_bytes_baseline": shared_baseline, "shared_bytes_after": shared_after, "allocator": allocator_counters()},
        "wall_s": time.perf_counter() - t_start,
    })
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
