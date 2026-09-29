"""Decode from a long prompt twice: once from the full GPU KV cache, once from LazyKV's CPU tier.

What this shows, and what it does not:
- After prefill, the tiered cache keeps only a budget fraction of the KV in VRAM (plus the dense
  layers and the newest block) and fetches the blocks each step selects from pinned host RAM.
  The printout compares the VRAM each cache holds and the tokens each one produces.
- Prefill is not tiered: the full-size KV exists on the GPU while the prompt is read, so this
  does not let a prompt longer than the GPU can prefill fit. And the tier decodes slower than
  the full cache on the hardware this was measured on (docs/FINDINGS.md). It is the study's
  runtime, not a speedup.

Needs a CUDA GPU and Llama-3.2-1B-Instruct (downloaded from the Hugging Face Hub on first run):

    python examples/tiered_decode.py --context 16384 --budget 0.25
"""

from __future__ import annotations

import argparse

import torch

from lazykv.cache import FullGPUCache
from lazykv.generate import greedy_decode, load, prefill
from lazykv.sweep import Condition, build_cache, make_host_memory

REPO = "unsloth/Llama-3.2-1B-Instruct"
REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"  # the revision every result in this repo used
BLOCK_SIZE = 64
# Phase 4's chosen tier: gather fetched pairs into one staging copy, and give VRAM one spare slot
# per selected block so a block selected on consecutive steps is not fetched twice.
TIER = {"fetch": "gather", "spare": 1.0}


def needle_prompt(tokenizer, context: int, secret: str) -> torch.Tensor:  # noqa: ANN001
    """A synthetic haystack with one fact halfway in, cut to about `context` tokens."""
    filler = " ".join(f"Entry {i}: the warehouse log recorded shipment {i * 7919 % 10007} as delivered." for i in range(context // 12))
    ids = tokenizer(filler, add_special_tokens=False).input_ids[: context - 200]
    half = len(ids) // 2
    body = tokenizer.decode(ids[:half]) + f" The secret code is {secret}. " + tokenizer.decode(ids[half:])
    messages = [{"role": "user", "content": body + "\n\nWhat is the secret code? Answer with the code only."}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, default=16384, help="prompt length in tokens")
    ap.add_argument("--budget", type=float, default=0.25, help="fraction of the KV the tier keeps selected in VRAM")
    ap.add_argument("--new-tokens", type=int, default=12)
    args = ap.parse_args()

    lm = load(REPO, REVISION)
    ids = needle_prompt(lm.tokenizer, args.context, "4417-XQ")
    total = len(ids) + args.new_tokens
    # Preallocated once, rounded up to a 1024-token multiple; every condition decodes on top of it.
    full = FullGPUCache(lm.num_layers, -(-(total + 16) // 1024) * 1024)
    conds = [Condition("full", 1.0), Condition("tiered_sync", args.budget)]
    host = make_host_memory(conds, lm.model, BLOCK_SIZE, full.max_len, full.max_len, TIER)
    eot = lm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    pre = prefill(lm.model, full, ids, chunk_size=2048)
    print(f"prompt: {len(ids):,} tokens, prefilled in {pre.wall_s:.1f} s")
    for cond in conds:
        built = build_cache(full, cond, total, BLOCK_SIZE, model=lm.model, host=host, tier=TIER)
        dec = greedy_decode(lm.model, built.cache, pre.last_logits, args.new_tokens - 1)
        vram = built.cache.stats().gpu_resident_kv_bytes
        built.close()
        full.truncate(len(ids))  # both caches decode on top of the shared prefill; drop what they appended
        ms = 1e3 * sorted(dec.wall_s)[len(dec.wall_s) // 2]
        toks = dec.tokens[: dec.tokens.index(eot)] if eot in dec.tokens else dec.tokens
        answer = lm.tokenizer.decode(toks).strip()
        print(f"{cond.label:>18}: {vram / 2**20:7.1f} MiB KV in VRAM, {ms:5.1f} ms/token (median), answer: {answer!r}")


if __name__ == "__main__":
    main()
