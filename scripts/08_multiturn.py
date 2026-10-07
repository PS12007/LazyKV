"""Phase 19: multi-turn retrieval sessions under each residency policy.

For each session (lazykv/multiturn.py), the document is prefilled once into a full GPU cache; every
condition's cache is derived from that prefill, as in the single-turn sweeps, and then driven through
every later turn one token at a time. Conditions run in a shuffled order per session. Scored on the
quality kernel (lazykv.quality.QUALITY_STRATEGY); timings here are not speed results.

Metrics are rewritten after every session, so a crash costs one session, not the run.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\08_multiturn.py > logs\\phase19_multiturn.log 2>&1
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
from lazykv.generate import load, prefill  # noqa: E402
from lazykv.multiturn import build_session, run_session, score_turn  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY  # noqa: E402
from lazykv.sweep import Condition, build_cache, cache_facts, load_config, make_host_memory  # noqa: E402

log = logging.getLogger("multiturn")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase19.yaml"))
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--sessions", type=int, help="override config sessions (smoke tests)")
    p.add_argument("--session-start", type=int, default=0, help="first session index (chunked runs)")
    p.add_argument("--out", default="multiturn")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    ctx = args.context or cfg["context"]
    n_sessions = args.sessions or cfg["sessions"]
    bs, chunk, max_new = cfg["block_size"], cfg["prefill_chunk"], cfg["max_new_tokens"]
    conds = [Condition("full", 1.0)] + [Condition(pol, b) for pol in cfg["policies"] for b in cfg["budgets"]]
    if cfg.get("tier_check"):
        conds.append(Condition(cfg["tier_check"]["policy"], float(cfg["tier_check"]["budget"])))

    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=QUALITY_STRATEGY)
    set_strategy_for_test(QUALITY_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    shared_baseline = process_gpu_memory().shared_bytes
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    book = load_tokens(cfg["book"], lm.tokenizer)
    passages = load_tokens(cfg["passage_book"], lm.tokenizer)
    sessions = [build_session(lm.tokenizer, book, passages, ctx, s, seed=cfg["seed"], passage_tokens=cfg["passage_tokens"])
                for s in range(args.session_start, args.session_start + n_sessions)]
    planned = max(s.planned_tokens(max_new) for s in sessions)
    capacity = -(-(planned + 16) // 1024) * 1024
    full = FullGPUCache(lm.num_layers, capacity)
    tier_opts = cfg.get("tier", {})
    host = make_host_memory(conds, lm.model, bs, capacity, capacity, tier_opts)
    rng = random.Random(cfg["seed"])
    rows: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    prefill_wall_s: list[float] = []
    t_start = time.perf_counter()

    def save() -> None:
        write_metrics(RESULTS_DIR / cfg["phase"] / args.out, {
            "config": {**cfg, "context": ctx, "sessions": n_sessions, "session_start": args.session_start},
            "attention_strategy": QUALITY_STRATEGY,
            "conditions": [c.label for c in conds],
            "schedule": [[t.asks, t.introduces] for t in sessions[0].turns],
            "capacity_tokens": capacity,
            "turns": rows,
            "cache_facts": facts,
            "prefill_wall_s": prefill_wall_s,
            "wall_s": time.perf_counter() - t_start,
            "memory_guard": {"allocator_cap": memory_cap, "shared_baseline_bytes": shared_baseline, "shared_after_bytes": process_gpu_memory().shared_bytes},
        })

    for session in sessions:
        full.truncate(0)
        pre = prefill(lm.model, full, session.prefix_ids, chunk)
        prefill_wall_s.append(pre.wall_s)
        total = session.planned_tokens(max_new)
        order = conds[:]
        rng.shuffle(order)
        summary: dict[str, list[float]] = {}
        for cond in order:
            torch.cuda.empty_cache()
            built = build_cache(full, cond, total, bs, None, lm.model, host, tier=tier_opts)
            t0 = time.perf_counter()
            try:
                for res in run_session(lm.model, built.cache, pre.last_logits, session, eot, max_new, session.ctx):
                    turn = session.turns[res.index]
                    text = lm.tokenizer.decode(res.answer_ids)
                    s = score_turn(turn, text)
                    summary.setdefault(cond.label, []).append(s)
                    depth = session.doc_needles[turn.asks][2] if turn.kind == "doc" else None
                    rows.append({
                        "session": session.sample, "turn": turn.index, "policy": cond.policy, "budget": cond.budget,
                        "kind": turn.kind, "asks": turn.asks, "introduces": turn.introduces, "key": turn.key, "value": turn.value,
                        "depth": depth, "turns_since_given": session.turns_since_given(turn),
                        "score": s, "answer": text.strip()[:120], "length_before": res.length_before,
                    })
                facts.append({"session": session.sample, "policy": cond.policy, "budget": cond.budget,
                              "session_s": time.perf_counter() - t0, "final_length": built.cache.get_seq_length(), **cache_facts(built)})
            finally:
                built.close()
            if built.shares_full:
                full.truncate(session.ctx)
            del built
        log.info("session %d (%.0fs): %s", session.sample, time.perf_counter() - t_start,
                 {k: "".join("1" if x else "0" for x in v) for k, v in summary.items()})
        save()
    save()


if __name__ == "__main__":
    main()
