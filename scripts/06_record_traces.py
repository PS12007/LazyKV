"""Record attention traces on the full-cache reference decode (Phase 16; upgrade plan A2).

For each NIAH prompt and each teacher-forced window of the config: one exact prefill, then a decode
through `lazykv.trace.TraceRecorder`, which attends densely and logs the true attention mass and the
Quest bound per (step, selecting layer, KV head, block). Each trace is saved as one .npz under the
config's `trace_dir`; metrics.json lists them with the full cache's own NIAH score, so the offline
analysis (scripts/analyze_phase16.py) never needs the GPU.

Scored on lazykv.quality.QUALITY_STRATEGY, the kernel every quality phase used, so the full cache's
answers here can be checked against Phase 8's on the same prompts.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\06_record_traces.py > logs\\phase16_traces.log 2>&1
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
from harness.gpu_memory import cap_allocator_to_dedicated  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.attention import set_strategy_for_test  # noqa: E402
from lazykv.cache import FullGPUCache  # noqa: E402
from lazykv.generate import greedy_decode, load, prefill, teacher_forced_decode  # noqa: E402
from lazykv.niah import build_prompt, score  # noqa: E402
from lazykv.quality import QUALITY_STRATEGY  # noqa: E402
from lazykv.sweep import load_config, results_subdir  # noqa: E402
from lazykv.trace import TraceRecorder  # noqa: E402

log = logging.getLogger("record_traces")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase16.yaml"))
    p.add_argument("--context", type=int, help="override config context (smoke tests)")
    p.add_argument("--samples", type=int, help="override NIAH samples per kind x depth")
    p.add_argument("--skip-teacher-forced", action="store_true")
    p.add_argument("--out", default="traces")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    ctx = args.context or cfg["context"]
    samples = args.samples or cfg["niah"]["samples"]
    bs, chunk = cfg["block_size"], cfg["prefill_chunk"]
    trace_dir = Path(cfg["trace_dir"]) / f"ctx{ctx}"

    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=QUALITY_STRATEGY,
              cache_dir=cfg["model"].get("cache_dir"), quant=cfg["model"].get("quant"))
    set_strategy_for_test(QUALITY_STRATEGY)
    cap_allocator_to_dedicated()
    tf = cfg["teacher_forced"]
    max_new = max(cfg["niah"]["max_new_tokens"].values())
    full = FullGPUCache(lm.num_layers, -(-(ctx + max(max_new, tf["continuation"]) + 16) // 1024) * 1024)
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    def keep(rec: TraceRecorder, name: str, meta: dict[str, Any]) -> None:
        tr = rec.trace(meta)
        path = trace_dir / f"{name}.npz"
        tr.save(path)
        rows.append({**meta, "file": path.name, "bytes": path.stat().st_size, "steps": tr.steps,
                     "selecting_layers": int(tr.mass.shape[1]), "kv_heads": int(tr.mass.shape[2]), "n_full_first": int(tr.n_full[0])})
        log.info("%s: %d steps, %.1f MiB (%.0fs)", name, tr.steps, path.stat().st_size / 2**20, time.perf_counter() - t_start)

    book = load_tokens(cfg["niah"]["book"], lm.tokenizer)
    for kind in cfg["niah"]["kinds"]:
        n_new = cfg["niah"]["max_new_tokens"][kind]
        for depth in cfg["niah"]["depths"]:
            for sample in range(samples):
                prompt = build_prompt(lm.tokenizer, book, ctx, kind, depth, sample, seed=cfg["niah"]["seed"])
                full.truncate(0)
                pre = prefill(lm.model, full, prompt.input_ids, chunk)
                rec = TraceRecorder(full, bs)
                dec = greedy_decode(lm.model, rec, pre.last_logits, n_new - 1)
                toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
                text = lm.tokenizer.decode(toks)
                keep(rec, f"niah_{kind}_d{depth:g}_s{sample}", {
                    "source": "niah", "kind": kind, "depth": depth, "sample": sample, "prompt_len": prompt.length,
                    # The selecting rungs size K from prompt + generated tokens (lazykv.sweep.build_cache).
                    "total_tokens": prompt.length + n_new, "full_score": score(prompt, text), "answer": text.strip()[:120],
                    "answer_tokens": len(toks),
                })
                del rec

    books: dict[str, torch.Tensor] = {}
    for offset in [] if args.skip_teacher_forced else tf["offsets"]:
        text_ids = books.setdefault(tf["book"], load_tokens(tf["book"], lm.tokenizer))
        ids = torch.cat([text_ids[:1], text_ids[1 + offset : offset + ctx + tf["continuation"]]])
        context, cont = ids[:ctx], ids[ctx : ctx + tf["continuation"]]
        full.truncate(0)
        pre = prefill(lm.model, full, context, chunk)
        rec = TraceRecorder(full, bs)
        for _ in teacher_forced_decode(lm.model, rec, pre.last_logits, cont):
            pass
        keep(rec, f"tf_{tf['book']}_o{offset}", {"source": "teacher_forced", "book": tf["book"], "offset": offset,
                                                 "prompt_len": ctx, "total_tokens": ctx + tf["continuation"]})
        del rec

    write_metrics(RESULTS_DIR / results_subdir(cfg) / args.out, {
        "config": {**cfg, "context": ctx, "niah": {**cfg["niah"], "samples": samples}},
        "attention_strategy": QUALITY_STRATEGY,
        "trace_dir": trace_dir.as_posix(),
        "traces": rows,
        "total_bytes": sum(r["bytes"] for r in rows),
        "wall_s": time.perf_counter() - t_start,
    })
    log.info("done in %.0fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
