"""How much could taking the residency decision off the host critical path possibly buy?

Phases 4 and 5 attacked bytes moved four ways -- tenfold fewer fetches, fully hidden prefetch
copies, and an int8 tier that halves the traffic -- and none of them made decode faster. The
measurements point at one remaining lever: every selecting layer brings its top-K block indices to
the CPU before it can decide what to fetch, and the GPU has nothing queued while that happens.

Turning that into a device-side design is a redesign rather than a rung (CLAUDE.md rule 8), so this
script does not build it. It prices it. If the ceiling is small, the redesign is not worth a phase.

**What is measured: a replay ablation.** Pass one decodes normally and records the selection each
layer made at each step. Pass two decodes again from the same prefill, driven by that record, so the
bound and the host sync that produced it never run -- but the blocks chosen, the pairs fetched, the
gathers and the output tokens are all identical (tests/test_tiered_int8.py checks the outputs match
bit for bit and the fetch counts are equal). The difference in decode latency is therefore the cost
of *deciding*, with the cost of *doing* held fixed. That is the ceiling.

**What was tried first, and why it is reported anyway.** The obvious instrument was the GPU-stream
gap between consecutive decoder layers, on the theory that the host round trip leaves the GPU with
nothing queued. It measures near zero, because `select()` runs *inside* the attention module and so
inside the layer's own span: the stall is within a layer, not between layers. The gap is still
recorded here, as the evidence for that, and because a non-zero value would mean something else is
wrong. It is not the ceiling and must not be read as one.

The full cache is profiled alongside as the hook-overhead control: it does no residency work, so its
gap is the floor that the hooks alone produce.

Profiled and unprofiled decode are timed separately and both are reported: the gap comes from the
profiled pass, the honest latency from the unprofiled one.

Launch detached, once per run id:
    .venv\\Scripts\\python.exe scripts\\03_host_ceiling.py --run-id 1 > logs\\host_ceiling_run_1.log 2>&1
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
from harness.gpu_memory import cap_allocator_to_dedicated, process_gpu_memory  # noqa: E402
from harness.host import set_power_throttling  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import summarize  # noqa: E402
from harness.telemetry import TelemetryLogger  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill  # noqa: E402
from lazykv.profiling import LayerProfiler  # noqa: E402
from lazykv.sweep import Condition, build_cache, cache_facts, load_config, make_host_memory, results_subdir  # noqa: E402
from lazykv.tiered import TieredCache  # noqa: E402

log = logging.getLogger("host_ceiling")
FAST_STRATEGY = "cudnn_bucketed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase5.yaml"))
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--budgets", type=float, nargs="+", help="override config budgets")
    p.add_argument("--policies", nargs="+", help="override config policies")
    p.add_argument("--profile-tokens", type=int, default=32)
    p.add_argument("--profile-warmup", type=int, default=3)
    p.add_argument("--out", default="host_ceiling")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    if args.budgets:
        cfg["budgets"] = args.budgets
    if args.policies:
        cfg["policies"] = args.policies
    sp = cfg["speed"]
    ctx = args.context or cfg["context"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    n_decode = sp["warmup_steps"] + sp["decode_tokens"]
    # The full cache is the hook-overhead control, not a condition of interest in its own right.
    conds = [Condition("full", 1.0)] + [Condition(pol, b) for pol in cfg["policies"] for b in cfg["budgets"]]

    set_power_throttling(opt_out=True)
    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=FAST_STRATEGY)
    memory_cap = cap_allocator_to_dedicated()
    tokens = load_tokens(sp["book"], lm.tokenizer)
    total = ctx + n_decode + args.profile_tokens + args.profile_warmup + 8
    full = FullGPUCache(lm.num_layers, -(-(total + 16) // 1024) * 1024)
    out_dir = RESULTS_DIR / results_subdir(cfg) / args.out / f"run_{args.run_id}"
    tier_opts = cfg.get("tier", {})
    host = make_host_memory(conds, lm.model, bs, full.max_len, total, tier_opts)
    shared_baseline = process_gpu_memory().shared_bytes
    rng = random.Random(2000 + args.run_id)

    pre = prefill(lm.model, full, tokens[:ctx], chunk)
    for cond in conds:  # plan warmup, outside the timed repeats
        built = build_cache(full, cond, total, bs, None, lm.model, host, tier=tier_opts)
        greedy_decode(lm.model, built.cache, pre.last_logits, 4)
        built.close()
        if built.shares_full:
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
                built = build_cache(full, cond, total, bs, None, lm.model, host, tier=tier_opts)
                if isinstance(built.cache, TieredCache):
                    built.cache._record = {}  # noqa: SLF001  -- record this pass to replay it below
                # Unprofiled first: the honest latency, with no hooks on the model.
                dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_decode)
                wall = dec.wall_s[sp["warmup_steps"] :]
                selection = built.cache.recorded_selection() if isinstance(built.cache, TieredCache) else None
                last = torch.zeros(lm.model.config.vocab_size, device="cuda")
                last[dec.tokens[-1]] = 1.0
                with LayerProfiler(lm.model) as prof:
                    for _ in range(args.profile_warmup):
                        greedy_decode(lm.model, built.cache, last, 1)
                    prof.reset()
                    pdec = greedy_decode(lm.model, built.cache, last, args.profile_tokens)
                    gaps = prof.step_gaps()
                    per_layer = prof.collect()
                facts = cache_facts(built)
                built.close()
                layers = sorted(per_layer)

                # The ceiling: the same run with the per-layer bound and host sync removed, driven
                # by the selection the pass above recorded. Identical fetches, identical outputs.
                replay_wall = None
                if selection is not None:
                    if built.shares_full:
                        full.truncate(ctx)
                    rbuilt = build_cache(full, cond, total, bs, None, lm.model, host, tier=tier_opts)
                    rbuilt.cache._replay = selection  # noqa: SLF001
                    rdec = greedy_decode(lm.model, rbuilt.cache, pre.last_logits, n_decode)
                    replay_wall = summarize(rdec.wall_s[sp["warmup_steps"] :]).to_dict()
                    assert rdec.tokens == dec.tokens, "replay diverged; the ablation would not be comparable"
                    rbuilt.close()
                    del rbuilt
                rows.append({
                    "repeat": rep,
                    "policy": cond.policy,
                    "budget": cond.budget,
                    "k_blocks": built.k_blocks,
                    "n_slots": built.n_slots,
                    "decode_wall_s": summarize(wall).to_dict(),
                    "profiled_wall_s": summarize(pdec.wall_s).to_dict(),
                    # The headline: GPU-stream time per token with nothing queued, between layers.
                    "inter_layer_gap_s": summarize(gaps).to_dict() if gaps else None,
                    "layer_sum_s": sum(per_layer[i]["layer_s"] for i in layers),
                    "profiled_steps": len(gaps),
                    "replay_wall_s": replay_wall,
                    **facts,
                })
                log.info(
                    "rep %d %s: decode %.2f ms, replay %s, inter-layer gap %.3f ms/token",
                    rep, cond.label, 1e3 * statistics.median(wall),
                    f"{1e3 * replay_wall['median']:.2f} ms" if replay_wall else "-",
                    1e3 * statistics.median(gaps) if gaps else float("nan"),
                )
                if built.shares_full:
                    full.truncate(ctx)
                del built

    write_metrics(out_dir, {
        "config": {**cfg, "context": ctx, "run_id": args.run_id, "profile_tokens": args.profile_tokens},
        "attention_strategy": FAST_STRATEGY,
        "n_layers": lm.num_layers,
        "dense_layers": next((r.get("dense_layers") for r in rows if r.get("dense_layers") is not None), None),
        "memory_cap_bytes": memory_cap,
        "shared_baseline_bytes": shared_baseline,
        "results": rows,
    })
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
