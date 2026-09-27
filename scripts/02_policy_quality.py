"""Budget sweep, quality half: NIAH accuracy and teacher-forced divergence per policy and budget.

Phase 2 gate by default; `--config configs/phase3.yaml` runs the Phase 3 ladder rungs (h2o, quest).

For each prompt: one exact prefill into a FullGPUCache, then every condition (shuffled) decodes
from a cache derived from that prefill. Scored on lazykv.quality.QUALITY_STRATEGY (Phase 1: the
fast kernel is not bit-repeatable), so a condition's delta against "full" is the policy's alone.

  NIAH             greedy answer after the prompt; score = fraction of needle values present
  teacher-forced   256 continuation positions of a book passage; KL and top-1 vs "full", and the
                   true tokens' NLL (perplexity, brief §B7.3) from the same rows

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
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import PrefillResult, greedy_decode, load, prefill, teacher_forced_decode  # noqa: E402
from lazykv.niah import build_prompt, score  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY, compare_stream  # noqa: E402
from lazykv.sweep import build_cache, cache_facts, conditions, load_config, make_host_memory, make_scorer, results_subdir  # noqa: E402

log = logging.getLogger("policy_quality")


def tf_windows(tf: dict[str, Any], ctx: int) -> list[tuple[str, int, int]]:
    """(book, offset, target) per teacher-forced window; the continuation starts at book token `target`.

    `offsets` fixes where the *context* starts, so a different context length scores different
    tokens. `targets` fixes the *scored tokens* and puts `ctx` tokens of the book before them, so
    runs at different contexts score the same text and their perplexities are paired: any change
    is the context's, not the passage's.
    """
    if "targets" in tf:
        out = [(book, t - ctx, t) for book, ts in tf["targets"].items() for t in ts]
    else:
        out = [(tf["book"], o, o + ctx) for o in tf["offsets"]]
    if bad := [w for w in out if w[1] < 0]:
        raise ValueError(f"teacher-forced targets need {ctx} tokens of context before them: {bad}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase2.yaml"))
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--block-size", type=int, help="override config block_size (the Phase 4 block-size sweep)")
    p.add_argument("--fetch", choices=["runs", "gather"], help="override config tier.fetch")
    p.add_argument("--spare", type=float, help="override config tier.spare")
    p.add_argument("--budgets", type=float, nargs="+", help="override config budgets")
    p.add_argument("--policies", nargs="+", help="override config policies")
    p.add_argument("--samples", type=int, help="override NIAH samples per kind x depth")
    # A kind is only run at a context where the full cache can do it (Phase 12's viability rule),
    # so which kinds run can differ by context within one phase.
    p.add_argument("--kinds", nargs="+", help="override config niah.kinds")
    # A prompt is keyed by its sample index, so a range [start, start + samples) reproduces exactly
    # the prompts a single larger run would have drawn. That lets a many-hour run be split into
    # chunks that each write their own metrics, so a crash costs one chunk rather than the run.
    p.add_argument("--sample-start", type=int, default=0, help="first NIAH sample index (chunked runs)")
    p.add_argument("--skip-teacher-forced", action="store_true", help="NIAH only (every chunk after the first)")
    # The 100% block-pool control is a second full-size copy of the KV. On the 3B model at 32K that
    # copy does not fit beside the reference (as at 64K on the 1B model, Phase 8), so it can be dropped.
    p.add_argument("--skip-block-full", action="store_true", help="drop the 100%% block-pool control")
    p.add_argument("--out", default="policy_quality")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    if args.block_size:
        cfg["block_size"] = args.block_size
    if args.fetch is not None:
        cfg.setdefault("tier", {})["fetch"] = args.fetch
    if args.spare is not None:
        cfg.setdefault("tier", {})["spare"] = args.spare
    if args.budgets:
        cfg["budgets"] = args.budgets
    if args.policies:
        cfg["policies"] = args.policies
    if args.kinds:
        cfg["niah"]["kinds"] = args.kinds
    ctx = args.context or cfg["context"]
    samples = args.samples or cfg["niah"]["samples"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    conds = conditions(cfg, skip_block_full=args.skip_block_full)

    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=QUALITY_STRATEGY,
              cache_dir=cfg["model"].get("cache_dir"), quant=cfg["model"].get("quant"))
    set_strategy_for_test(QUALITY_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    rng = random.Random(0)
    max_new = max(cfg["niah"]["max_new_tokens"].values())
    full = FullGPUCache(lm.num_layers, -(-(ctx + max(max_new, cfg["teacher_forced"]["continuation"]) + 16) // 1024) * 1024)
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    scorer = make_scorer(cfg, conds, lm.num_layers, full.max_len)
    tier_opts = cfg.get("tier", {})
    # Staging sized for the longest sequence any prompt reaches (budgets scale with it).
    host = make_host_memory(conds, lm.model, bs, full.max_len, full.max_len, tier_opts)
    prefill_wall_s: list[float] = []

    def run_prefill(ids: torch.Tensor) -> PrefillResult:
        full.truncate(0)
        if scorer is not None:
            scorer.reset()
        pre = prefill(lm.model, full, ids, chunk, observer=scorer)
        prefill_wall_s.append(pre.wall_s)
        return pre

    # ---- NIAH ------------------------------------------------------------------------------
    book = load_tokens(cfg["niah"]["book"], lm.tokenizer)
    niah_rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for kind in cfg["niah"]["kinds"]:
        n_new = cfg["niah"]["max_new_tokens"][kind]
        for depth in cfg["niah"]["depths"]:
            for sample in range(args.sample_start, args.sample_start + samples):
                prompt = build_prompt(lm.tokenizer, book, ctx, kind, depth, sample, seed=cfg["niah"]["seed"])
                pre = run_prefill(prompt.input_ids)
                total = prompt.length + n_new
                order = conds[:]
                rng.shuffle(order)
                for cond in order:
                    built = build_cache(full, cond, total, bs, scorer, lm.model, host, tier=tier_opts)
                    dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_new - 1)
                    built.close()
                    toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
                    text = lm.tokenizer.decode(toks)
                    niah_rows.append({
                        "kind": kind, "depth": depth, "sample": sample, "prompt_len": prompt.length,
                        "policy": cond.policy, "budget": cond.budget, "score": score(prompt, text),
                        "answer": text.strip()[:120], "values": list(prompt.values),
                        # cwe has no needle: its evidence is the whole list, so there is no position.
                        "target_needle_pos": prompt.needle_token_positions[0] if prompt.needle_token_positions else None,
                        **cache_facts(built),
                    })
                    if built.shares_full:
                        full.truncate(prompt.length)
                    del built
                this = {f"{r['policy']}@{r['budget']:g}": r["score"] for r in niah_rows[-len(conds):]}
                log.info("niah %s depth %.2f sample %d (%.0fs): %s", kind, depth, sample, time.perf_counter() - t_start, this)

    # ---- teacher-forced divergence --------------------------------------------------------
    tf = cfg["teacher_forced"]
    books: dict[str, torch.Tensor] = {}
    tf_rows: list[dict[str, Any]] = []
    for book_name, offset, target in [] if args.skip_teacher_forced else tf_windows(tf, ctx):
        text_ids = books.setdefault(book_name, load_tokens(book_name, lm.tokenizer))
        # BOS, then a stretch of the book: the same shape as the Phase 1 quality reference.
        ids = torch.cat([text_ids[:1], text_ids[1 + offset : offset + ctx + tf["continuation"]]])
        context, cont = ids[:ctx], ids[ctx : ctx + tf["continuation"]]
        if cont.numel() != tf["continuation"]:
            raise ValueError(f"{book_name} target {target}: only {cont.numel()} continuation tokens left")
        pre = run_prefill(context)
        total = ctx + tf["continuation"]
        reference = torch.stack([row.cpu() for row in teacher_forced_decode(lm.model, full, pre.last_logits, cont)])
        full.truncate(ctx)
        order = conds[:]
        rng.shuffle(order)
        for cond in order:
            built = build_cache(full, cond, total, bs, scorer, lm.model, host, tier=tier_opts)
            div = compare_stream(reference, teacher_forced_decode(lm.model, built.cache, pre.last_logits, cont), targets=cont)
            built.close()
            tf_rows.append({"book": book_name, "offset": offset, "target": target, "policy": cond.policy, "budget": cond.budget, **div.to_dict(), **cache_facts(built)})
            if built.shares_full:
                full.truncate(ctx)
            del built
            r = tf_rows[-1]
            log.info("teacher-forced %s @%d %s: top1 %.3f, KL %.2e, NLL %.4f (ref %.4f)", book_name, target, cond.label, r["top1_agreement"], r["mean_kl"], r["mean_nll"], r["ref_mean_nll"])
        del reference

    write_metrics(
        RESULTS_DIR / results_subdir(cfg) / args.out,
        {
            "config": {**cfg, "context": ctx, "niah": {**cfg["niah"], "samples": samples, "sample_start": args.sample_start}},
            "attention_strategy": QUALITY_STRATEGY,
            "conditions": [c.label for c in conds],
            "kv_bytes_per_token": lm.kv_bytes_per_token,
            "niah": niah_rows,
            "teacher_forced": tf_rows,
            "wall_s": time.perf_counter() - t_start,
            # Per prompt, including the h2o scorer when one runs. Quality kernel: not a speed result.
            "prefill_wall_s": prefill_wall_s,
            "prefill_scorer": None if scorer is None else {"stride": scorer.stride, "query_batch": scorer.query_batch},
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": process_gpu_memory().shared_bytes},
        },
    )
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
