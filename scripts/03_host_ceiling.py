"""How much could taking the residency decision off the host critical path possibly buy?

Phases 4 and 5 attacked bytes moved four ways -- tenfold fewer fetches, fully hidden prefetch
copies, and an int8 tier that halves the traffic -- and none of them made decode faster. The
measurements point at one remaining lever: every selecting layer brings its top-K block indices to
the CPU before it can decide what to fetch, and the GPU has nothing queued while that happens.

Turning that into a device-side design is a redesign rather than a rung (CLAUDE.md rule 8), so this
script does not build it. It prices it. If the ceiling is small, the redesign is not worth a phase.

**What is measured.** `LayerProfiler` puts CUDA events around every decoder layer, so the time
between the end of one layer's span and the start of the next is GPU-stream time with nothing
running. That inter-layer gap is where a per-layer host round trip lands, and its total per token is
an upper bound on what any device-side residency decision could recover.

**Why the full cache is profiled too.** The hooks themselves run host code between layers, which
inflates every gap. The full cache does no residency work at all, so its gap under the same hooks is
the hook-overhead control, and the difference is the part the tier actually owns. Without that
control this measurement would overstate the ceiling and there would be no way to tell by how much.

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
                # Unprofiled first: the honest latency, with no hooks on the model.
                dec = greedy_decode(lm.model, built.cache, pre.last_logits, n_decode)
                wall = dec.wall_s[sp["warmup_steps"] :]
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
                    **facts,
                })
                log.info(
                    "rep %d %s: decode %.2f ms, profiled %.2f ms, inter-layer gap %.2f ms/token",
                    rep, cond.label, 1e3 * statistics.median(wall), 1e3 * statistics.median(pdec.wall_s),
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
