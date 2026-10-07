"""Phase 20: resuming a long session from a snapshot instead of prefilling it again (gap table C1).

Per context, in one process (configs/phase20.yaml):
  1. Layer-major tiered prefill of one NIAH prompt (Phase 17's path), `repeats` times. The first
     prefill's tier is saved (lazykv/snapshot.py), then decoded: its tokens are the reference.
  2. `repeats` restores from the snapshot with the unbuffered reader, so every read comes from the
     disk and not from the page cache the save just filled. Each restored tier is decoded and must
     produce the reference tokens exactly.
Two checks. The exact one: a hash of the whole tier state (host blocks, Quest metadata, sink and
tail slots, dense layers' KV) after the prefill and after every restore must be equal. The decoded
tokens are compared too, but cuDNN's single-query decode is not bit-repeatable (Phase 1), so a token
mismatch alone would not implicate the snapshot.

Prefill and restore write into the same pinned host pools, allocated once per context, so neither
pays the (slow, one-off) pinning. The snapshot file is deleted afterwards.

Launch detached:
    .venv\\Scripts\\python.exe scripts\\09_snapshot.py > logs\\phase20_snapshot.log 2>&1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
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
from lazykv.generate import greedy_decode, load  # noqa: E402
from lazykv.niah import build_prompt, score  # noqa: E402
from lazykv.selection import QUEST_DENSE_LAYERS, blocks_for_budget  # noqa: E402
from lazykv.sweep import load_config  # noqa: E402
from lazykv.tiered import allocate_host_pools  # noqa: E402
from lazykv.tiered_prefill import tiered_prefill  # noqa: E402

log = logging.getLogger("snapshot")


def state_digest(cache: Any) -> str:
    """Exact hash of everything a tier attends from or ranks by. Residency (which slots hold which
    blocks) is left out: it decides what is fetched, never what is attended."""
    h = hashlib.blake2b(digest_size=16)

    def add(x: torch.Tensor) -> None:
        h.update(memoryview(x.detach().contiguous().cpu().view(torch.uint8).reshape(-1).numpy()))

    for i in sorted(cache.tiers):
        t = cache.tiers[i]
        h.update(f"{i}:{t.n_sealed}:{t.fill}:{t.length}".encode())
        add(t.host[: t.n_sealed])
        add(t.kmin[:, :, : t.n_sealed])
        add(t.kmax[:, :, : t.n_sealed])
        add(t.pool[0])
        add(t.pool[t.tail_slot, :, :, : t.fill])
    for i in range(cache.dense_layers):
        layer = cache.full.layers[i]
        add(layer.keys[:, :, : layer.length])
        add(layer.values[:, :, : layer.length])
    return h.hexdigest()


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase20.yaml"))
    ap.add_argument("--contexts", nargs="*", type=int, help="override the context ladder (smoke tests)")
    ap.add_argument("--repeats", type=int, help="override repeats (smoke tests)")
    ap.add_argument("--out", default="snapshot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path(args.config))
    contexts = args.contexts or cfg["contexts"]
    repeats = args.repeats or cfg["repeats"]
    set_power_throttling(opt_out=True)
    lm = load(cfg["model"]["repo"], cfg["model"]["revision"], strategy=cfg["strategy"])
    set_strategy_for_test(cfg["strategy"])
    cap_info = cap_allocator_to_dedicated()
    c = lm.model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    bs, chunk, n_dec = cfg["block_size"], cfg["prefill_chunk"], cfg["decode_tokens"]
    book = load_tokens(cfg["prompt"]["book"], lm.tokenizer)
    snap_dir = Path(cfg.get("snapshot_dir") or tempfile.gettempdir()) / "lazykv_snapshots"
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for ctx in contexts:
        p = cfg["prompt"]
        prompt = build_prompt(lm.tokenizer, book, ctx, p["kind"], p["depth"], p["sample"], seed=p["seed"])
        cap = -(-(ctx + n_dec + 16) // 1024) * 1024
        k = blocks_for_budget(cfg["budget"], ctx + n_dec, bs)
        pools = allocate_host_pools(lm.num_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, bs, cap, lm.model.dtype)
        path = snap_dir / f"ctx{ctx}.lkv"
        row: dict[str, Any] = {"context": ctx, "capacity_tokens": cap, "k_blocks": k,
                               "host_pinned_bytes": sum(t.numel() * t.element_size() for t in pools), "kv_bytes": ctx * lm.kv_bytes_per_token,
                               "prefill_s": [], "restore": [], "restore_tokens_match": [], "restore_state_match": []}
        ref: list[int] | None = None
        try:
            for r in range(repeats):
                release()
                res = tiered_prefill(lm.model, prompt.input_ids, chunk, k, bs, cap, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"])
                row["prefill_s"].append(res.wall_s)
                if r == 0:
                    row["state_digest"] = state_digest(res.cache)
                    st = snapshot.save(res.cache, res.last_logits, path)
                    row["save_s"], row["snapshot_bytes"] = st.seconds, st.bytes_written
                    dec = greedy_decode(lm.model, res.cache, res.last_logits, n_dec - 1)
                    ref = dec.tokens
                    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    text = lm.tokenizer.decode(ref[: ref.index(eot)] if eot in ref else ref)
                    row["answer"], row["score"] = text.strip()[:120], score(prompt, text)
                res.cache.close()
                del res
            for r in range(repeats):
                release()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cache, logits, rs = snapshot.restore(path, k, cap, lm.num_layers, model=lm.model, host_pools=pools, n_slots=k, fetch=cfg["tier"]["fetch"], unbuffered=True)
                wall = time.perf_counter() - t0
                row["restore_state_match"].append(state_digest(cache) == row["state_digest"])
                dec = greedy_decode(lm.model, cache, logits, n_dec - 1)
                cache.close()
                row["restore"].append({"wall_s": wall, "read_s": rs.read_s, "rebuild_s": rs.rebuild_s, "bytes_read": rs.bytes_read})
                row["restore_tokens_match"].append(dec.tokens == ref)
                del cache, logits
            row["status"] = "ok"
        except (torch.OutOfMemoryError, RuntimeError, OSError) as exc:
            row["status"], row["error"] = "error", str(exc).splitlines()[0][:300]
        finally:
            path.unlink(missing_ok=True)
        if row["prefill_s"] and row["restore"]:
            pre = statistics.median(row["prefill_s"])
            rst = statistics.median(x["wall_s"] for x in row["restore"])
            row["prefill_s_median"], row["restore_s_median"] = pre, rst
            row["read_gbps_median"] = statistics.median(x["bytes_read"] / x["read_s"] / 1e9 for x in row["restore"])
            row["speedup"] = pre / rst
        log.info("ctx %d (%.0fs): %s", ctx, time.perf_counter() - t_start, {k2: row.get(k2) for k2 in ("status", "prefill_s_median", "restore_s_median", "speedup", "restore_state_match", "restore_tokens_match")})
        rows.append(row)
        del pools
        release()

    write_metrics(RESULTS_DIR / cfg["phase"] / args.out, {
        "config": {**cfg, "contexts": contexts, "repeats": repeats}, "rows": rows, "allocator_cap": cap_info,
        "snapshot_dir": str(snap_dir), "host": {"power_throttling": "opted_out"}, "wall_s": time.perf_counter() - t_start,
    })


if __name__ == "__main__":
    main()
