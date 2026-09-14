"""Phase 2 gate, quality half: NIAH accuracy and teacher-forced divergence per policy and budget.

For each prompt: one exact prefill into a FullGPUCache, then every condition (shuffled) decodes
from a cache derived from that prefill. Scored on lazykv.quality.QUALITY_STRATEGY (Phase 1: the
fast kernel is not bit-repeatable), so a condition's delta against "full" is the policy's alone.

  NIAH             greedy answer after the prompt; score = fraction of needle values present
  teacher-forced   256 continuation positions of a book passage; KL and top-1 vs "full"

Launch detached:
    .venv\\Scripts\\python.exe scripts\\02_policy_quality.py > logs\\policy_quality.log 2>&1
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

import torch  # noqa: E402

from harness.corpus import load_tokens  # noqa: E402
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import set_strategy_for_test  # noqa: E402
from lazykv.blocks import BlockPoolCache  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill, teacher_forced_decode  # noqa: E402
from lazykv.niah import build_prompt, score  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY, compare_stream  # noqa: E402
from lazykv.sweep import build_cache, conditions, load_config  # noqa: E402

log = logging.getLogger("policy_quality")


def cache_facts(cache: Any, total_tokens: int) -> dict[str, Any]:
    st = cache.stats()
    facts: dict[str, Any] = {"gpu_resident_kv_bytes": st.gpu_resident_kv_bytes}
    if isinstance(cache, BlockPoolCache):
        facts.update(cache.counters())
        facts["n_slots"] = cache.pool_layers[0].n_slots
    return facts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase2.yaml"))
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--samples", type=int, help="override NIAH samples per kind x depth")
    p.add_argument("--out", default="policy_quality")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    ctx = args.context or cfg["context"]
    samples = args.samples or cfg["niah"]["samples"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    conds = conditions(cfg)

    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=QUALITY_STRATEGY)
    set_strategy_for_test(QUALITY_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    rng = random.Random(0)
    max_new = max(cfg["niah"]["max_new_tokens"].values())
    full = FullGPUCache(lm.num_layers, -(-(ctx + max(max_new, cfg["teacher_forced"]["continuation"]) + 16) // 1024) * 1024)
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # ---- NIAH ------------------------------------------------------------------------------
    book = load_tokens(cfg["niah"]["book"], lm.tokenizer)
    niah_rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for kind in cfg["niah"]["kinds"]:
        n_new = cfg["niah"]["max_new_tokens"][kind]
        for depth in cfg["niah"]["depths"]:
            for sample in range(samples):
                prompt = build_prompt(lm.tokenizer, book, ctx, kind, depth, sample, seed=cfg["niah"]["seed"])
                full.truncate(0)
                pre = prefill(lm.model, full, prompt.input_ids, chunk)
                total = prompt.length + n_new
                order = conds[:]
                rng.shuffle(order)
                for cond in order:
                    built = build_cache(full, cond, total, bs)
                    dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_new - 1)
                    toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
                    text = lm.tokenizer.decode(toks)
                    niah_rows.append({
                        "kind": kind, "depth": depth, "sample": sample, "prompt_len": prompt.length,
                        "policy": cond.policy, "budget": cond.budget, "score": score(prompt, text),
                        "answer": text.strip()[:120], "values": list(prompt.values),
                        "target_needle_pos": prompt.needle_token_positions[0],
                        **cache_facts(built.cache, total),
                    })
                    if built.cache is full:
                        full.truncate(prompt.length)
                    del built
                this = {f"{r['policy']}@{r['budget']:g}": r["score"] for r in niah_rows[-len(conds):]}
                log.info("niah %s depth %.2f sample %d (%.0fs): %s", kind, depth, sample, time.perf_counter() - t_start, this)

    # ---- teacher-forced divergence --------------------------------------------------------
    tf = cfg["teacher_forced"]
    text_ids = load_tokens(tf["book"], lm.tokenizer)
    tf_rows: list[dict[str, Any]] = []
    for offset in tf["offsets"]:
        # BOS, then a stretch of the book: the same shape as the Phase 1 quality reference.
        ids = torch.cat([text_ids[:1], text_ids[1 + offset : offset + ctx + tf["continuation"]]])
        context, cont = ids[:ctx], ids[ctx : ctx + tf["continuation"]]
        full.truncate(0)
        pre = prefill(lm.model, full, context, chunk)
        total = ctx + tf["continuation"]
        reference = torch.stack([row.cpu() for row in teacher_forced_decode(lm.model, full, pre.last_logits, cont)])
        full.truncate(ctx)
        order = conds[:]
        rng.shuffle(order)
        for cond in order:
            built = build_cache(full, cond, total, bs)
            div = compare_stream(reference, teacher_forced_decode(lm.model, built.cache, pre.last_logits, cont))
            tf_rows.append({"offset": offset, "policy": cond.policy, "budget": cond.budget, **div.to_dict(), **cache_facts(built.cache, total)})
            if built.cache is full:
                full.truncate(ctx)
            del built
            log.info("teacher-forced offset %d %s: top1 %.3f, KL %.2e", offset, cond.label, tf_rows[-1]["top1_agreement"], tf_rows[-1]["mean_kl"])
        del reference

    write_metrics(
        RESULTS_DIR / "phase2" / args.out,
        {
            "config": {**cfg, "context": ctx, "niah": {**cfg["niah"], "samples": samples}},
            "attention_strategy": QUALITY_STRATEGY,
            "conditions": [c.label for c in conds],
            "kv_bytes_per_token": lm.kv_bytes_per_token,
            "niah": niah_rows,
            "teacher_forced": tf_rows,
            "wall_s": time.perf_counter() - t_start,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": process_gpu_memory().shared_bytes},
        },
    )
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
