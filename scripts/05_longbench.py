"""LongBench QA under every residency condition: the ladder on real long documents.

For each prompt: one exact prefill into a FullGPUCache, then every condition (shuffled) decodes
greedily from a cache derived from it, exactly as `02_policy_quality.py` does for NIAH. Prompts keep
their natural length (up to `max_length`, cut in the middle beyond it), so a budget is a fraction of
each prompt's own length. Scored on the quality kernel; the score is LongBench's QA F1.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\05_longbench.py --config configs\\phase14.yaml > logs\\p14.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402
from lazykv.longbench import MAX_NEW_TOKENS, build_prompt, load_task, score  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY  # noqa: E402
from lazykv.sweep import build_cache, cache_facts, conditions, load_config, make_host_memory, make_scorer, results_subdir  # noqa: E402

log = logging.getLogger("longbench")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", nargs="+", help="override config longbench.tasks")
    p.add_argument("--samples", type=int, help="override config longbench.samples (first N rows per task)")
    p.add_argument("--skip-block-full", action="store_true", help="drop the 100%% block-pool control")
    p.add_argument("--out", default="longbench")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    lb = cfg["longbench"]
    tasks = args.tasks or lb["tasks"]
    samples = args.samples or lb["samples"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    conds = conditions(cfg, skip_block_full=args.skip_block_full)

    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=QUALITY_STRATEGY,
              cache_dir=cfg["model"].get("cache_dir"), quant=cfg["model"].get("quant"))
    set_strategy_for_test(QUALITY_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # Build every prompt first: the cache and the host pools are sized once, for the longest.
    data_dir = Path(lb["data_dir"])
    # The first `samples` prompts per task, in file order, of at least `min_prompt_tokens`: below
    # that, a 6.25% budget is smaller than a window policy's sink plus one window block, and the
    # condition cannot be built. Length is the only criterion, so the filter is blind to content.
    min_len = lb.get("min_prompt_tokens", 0)
    prompts, skipped_short = [], {}
    for task in tasks:
        kept = 0
        for i, row in enumerate(load_task(data_dir, task)):
            if kept == samples:
                break
            pr = build_prompt(lm.tokenizer, row, task, i, lb["max_length"])
            if pr.length < min_len:
                skipped_short[task] = skipped_short.get(task, 0) + 1
                continue
            prompts.append(pr)
            kept += 1
    longest = max(pr.length for pr in prompts) + max(MAX_NEW_TOKENS[t] for t in tasks) + 16
    full = FullGPUCache(lm.num_layers, -(-longest // 1024) * 1024)
    scorer = make_scorer(cfg, conds, lm.num_layers, full.max_len)
    tier_opts = cfg.get("tier", {})
    host = make_host_memory(conds, lm.model, bs, full.max_len, full.max_len, tier_opts)
    log.info("%d prompts, longest %d tokens, %d truncated", len(prompts), max(pr.length for pr in prompts), sum(pr.truncated for pr in prompts))

    rng = random.Random(0)
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for pr in prompts:
        full.truncate(0)
        if scorer is not None:
            scorer.reset()
        pre = prefill(lm.model, full, pr.input_ids, chunk, observer=scorer)
        n_new = MAX_NEW_TOKENS[pr.task]
        order = conds[:]
        rng.shuffle(order)
        for cond in order:
            built = build_cache(full, cond, pr.length + n_new, bs, scorer, lm.model, host, tier=tier_opts)
            dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_new - 1)
            built.close()
            toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
            text = lm.tokenizer.decode(toks)
            rows.append({"task": pr.task, "index": pr.index, "prompt_len": pr.length, "truncated": pr.truncated,
                         "policy": cond.policy, "budget": cond.budget, "score": score(pr, text),
                         "answer": text.strip()[:200], **cache_facts(built)})
            if built.shares_full:
                full.truncate(pr.length)
            del built
        this = {f"{r['policy']}@{r['budget']:g}": round(r["score"], 2) for r in rows[-len(conds):]}
        log.info("%s #%d (%d tok, %.0fs): %s", pr.task, pr.index, pr.length, time.perf_counter() - t_start, this)

    write_metrics(RESULTS_DIR / results_subdir(cfg) / args.out, {
        "config": {**cfg, "longbench": {**lb, "tasks": tasks, "samples": samples}},
        "attention_strategy": QUALITY_STRATEGY,
        "conditions": [c.label for c in conds],
        "skipped_short": skipped_short,
        "rows": rows,
        "wall_s": time.perf_counter() - t_start,
        "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": process_gpu_memory().shared_bytes},
    })
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
