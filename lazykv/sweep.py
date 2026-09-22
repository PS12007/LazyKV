"""Budget-sweep conditions and per-condition cache construction, shared by Phase 2 experiments.

Every condition starts from the same exact prefill: the quality and speed scripts prefill
once into a FullGPUCache and derive each condition's cache from it. The prefill is identical
by construction, so differences between conditions come from decode-time residency alone,
and the sweep avoids repeating a multi-second 32K prefill for every condition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers.cache_utils import Cache

from lazykv.blocks import BlockPoolCache, slots_for_budget
from lazykv.cache import FullGPUCache
from lazykv.policies import H2O, make_policy
from lazykv.scoring import PrefillScorer
from lazykv.selection import QUEST_DENSE_LAYERS, QuestView, blocks_for_budget
from lazykv.exact import ExactTieredCache, allocate_mirrors
from lazykv.tiered import TieredCache, allocate_host_pools, allocate_stages

# Policies whose decode cache is a view over the full cache rather than a block pool.
SELECTORS = ("quest",)
# Rungs 6-9: rung 5's selection over a CPU tier. Same K per budget as quest, so attended = resident.
TIERED = ("tiered_sync", "tiered_prefetch", "tiered_int8", "tiered_exact")
# Rung 9 keeps rung 6's tier and adds an exact CPU pass over the blocks VRAM does not hold, so it
# needs a float32 mirror of the sealed KV in pageable host RAM on top of the pinned pools.
EXACT = "tiered_exact"
# Host-pool precision per policy. Rung 8 narrows the warm/cold tier to int8; the VRAM slots stay bf16.
TIER_QUANT: dict[str, str | None] = {"tiered_int8": "int8"}


@dataclass(frozen=True)
class Condition:
    policy: str  # "full" means the FullGPUCache itself
    budget: float

    @property
    def label(self) -> str:
        return f"{self.policy}@{self.budget:g}"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def conditions(cfg: dict[str, Any], skip_block_full: bool = False) -> list[Condition]:
    """The sweep's conditions: the reference, the 100%-budget block-pool control, then the policies.

    `skip_block_full` drops the control only. It exists for one measured reason: the control is a
    second full-size copy of the KV, so at 64K it and the reference together are ~4 GiB on top of
    the weights and the run dies in `BlockPoolCache.from_full` before any policy is reached. The
    reference is not optional -- every retention and latency figure in this study is relative to it
    -- but the control is, and dropping it is what lets a 64K point exist at all. A run that uses
    this flag is measuring policies against the full cache without the block-pool overhead control.
    """
    out = [Condition("full", 1.0)]
    if not skip_block_full:
        out.append(Condition("block_full", 1.0))
    out += [Condition(p, b) for p in cfg["policies"] for b in cfg["budgets"]]
    return out


def needs_scorer(conds: list[Condition]) -> bool:
    return any(c.policy == "h2o" for c in conds)


def make_scorer(cfg: dict[str, Any], conds: list[Condition], num_layers: int, max_len: int) -> PrefillScorer | None:
    """The prefill scorer h2o conditions need, or None. Config `h2o: {stride, query_batch}`."""
    if not needs_scorer(conds):
        return None
    h = cfg.get("h2o", {})
    return PrefillScorer(num_layers, max_len, torch.device("cuda"), stride=h.get("stride", 32), query_batch=h.get("query_batch", 4))


@dataclass
class HostMemory:
    """Pinned memory the tiered conditions share: per-layer host pools and gather staging buffers.

    Keyed by tier precision, because a bf16 host pool and an int8 one are different buffers and a
    sweep that measures rung 6 against rung 8 needs both alive at once.
    """

    pools: dict[str | None, list[torch.Tensor]]
    stages: dict[str | None, Any]  # (compute, copy) staging, when tier.fetch is "gather"
    mirrors: list[Any] | None = None  # rung 9's float32 mirrors, one per selecting layer


def make_host_memory(conds: list[Condition], model: Any, block_size: int, max_len: int, total_tokens: int, tier: dict[str, Any] | None = None) -> HostMemory | None:
    """Pinned memory for the tiered conditions, allocated once per process (pinning is slow), or None."""
    tiered = [c for c in conds if c.policy in TIERED]
    if not tiered:
        return None
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    gather = (tier or {}).get("fetch", "runs") == "gather"
    pools: dict[str | None, list[torch.Tensor]] = {}
    stages: dict[str | None, Any] = {}
    for quant in dict.fromkeys(TIER_QUANT.get(cond.policy) for cond in tiered):
        pools[quant] = allocate_host_pools(c.num_hidden_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, block_size, max_len, model.dtype, quant)
        if gather or quant is not None:
            slots = max(n_slots_for(cond, total_tokens, block_size, tier) for cond in tiered if TIER_QUANT.get(cond.policy) == quant)
            stages[quant] = allocate_stages(slots * c.num_key_value_heads, c.num_key_value_heads, head_dim, block_size, model.dtype, quant)
    # Allocated once per process for the same reason the pinned pools are: at 32K each mirror is
    # about 135 MiB, and building them per condition would time the allocation, not the policy.
    mirrors = allocate_mirrors(c.num_hidden_layers - QUEST_DENSE_LAYERS, c.num_key_value_heads, head_dim, block_size, max_len) if any(cond.policy == EXACT for cond in tiered) else None
    return HostMemory(pools, stages, mirrors)


def n_slots_for(cond: Condition, total_tokens: int, block_size: int, tier: dict[str, Any] | None) -> int:
    """Slots per head: K selected blocks plus `spare` x K, capped at the candidate blocks the sequence can ever have
    (at a 75% budget, 1x spare would otherwise allocate slots for more blocks than exist)."""
    k = blocks_for_budget(cond.budget, total_tokens, block_size)
    candidates = total_tokens // block_size - 1
    return max(k, min(k + round(float((tier or {}).get("spare", 0.0)) * k), candidates))


def results_subdir(cfg: dict[str, Any]) -> str:
    """Which results/<phase>/ a sweep config writes to (configs predating the key are Phase 2)."""
    return str(cfg.get("phase", "phase2"))


@dataclass
class Built:
    cache: Cache
    build_s: float
    n_slots: int | None  # block pool slots (evicting policies)
    k_blocks: int | None = None  # selected blocks per step (query-aware selection)

    @property
    def shares_full(self) -> bool:
        """Decoding appends to the full cache itself, so the caller must truncate it back afterwards."""
        return isinstance(self.cache, (FullGPUCache, QuestView, TieredCache))

    def close(self) -> None:
        """Release model hooks a cache installed (rung 7). Call before the model decodes anything else."""
        if isinstance(self.cache, TieredCache):
            self.cache.close()


def build_cache(
    full: FullGPUCache,
    cond: Condition,
    total_tokens: int,
    block_size: int,
    scorer: PrefillScorer | None = None,
    model: Any = None,
    host: HostMemory | None = None,
    instrument: bool = False,
    tier: dict[str, Any] | None = None,
) -> Built:
    """The cache a condition decodes from. The full cache is returned as is; callers truncate it back afterwards.

    `scorer` must have observed this prompt's prefill when the condition is h2o. Tiered conditions
    need `host` (see make_host_memory) and, for prefetch, the `model`; callers must `close()` them.
    `tier` is the config's `tier:` section: `fetch` ("runs" or "gather") and `spare`, extra slots as a
    multiple of the selected blocks K (0 means VRAM holds exactly the attended set).
    """
    if cond.policy == "full":
        return Built(full, 0.0, None)
    # Synchronized: the boundary is mostly GPU gathers, and an unsynchronized clock would only
    # time their launch.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if cond.policy in TIERED:
        k = blocks_for_budget(cond.budget, total_tokens, block_size)
        n_slots = n_slots_for(cond, total_tokens, block_size, tier)
        quant = TIER_QUANT.get(cond.policy)
        kind = ExactTieredCache if cond.policy == EXACT else TieredCache
        extra = {"mirrors": None if host is None else host.mirrors, "threads": (tier or {}).get("cpu_threads", 8)} if cond.policy == EXACT else {}
        cache = kind(
            full, k, block_size, full.max_len, model=model, prefetch=cond.policy == "tiered_prefetch",
            host_pools=None if host is None else host.pools[quant], instrument=instrument, n_slots=n_slots,
            # Rung 8 has nowhere to dequantize in "runs" mode, so it always gathers.
            fetch="gather" if quant else (tier or {}).get("fetch", "runs"),
            stages=None if host is None else host.stages.get(quant), quant=quant, **extra,
        )
        torch.cuda.synchronize()
        return Built(cache, time.perf_counter() - t0, n_slots, k)
    if cond.policy in SELECTORS:
        k = blocks_for_budget(cond.budget, total_tokens, block_size)
        view = QuestView(full, k, block_size)
        torch.cuda.synchronize()
        return Built(view, time.perf_counter() - t0, None, k)
    n_slots = slots_for_budget(cond.budget, total_tokens, block_size)
    if cond.policy == "h2o":
        if scorer is None:
            raise ValueError("h2o needs a PrefillScorer that observed the prefill")
        length = full.get_seq_length()
        factory = lambda i: H2O(token_mass=scorer.full_sum_estimate(i, length), block_size=block_size)  # noqa: E731
    else:
        factory = lambda i: make_policy(cond.policy)  # noqa: E731
    cache = BlockPoolCache.from_full(full, n_slots, block_size, factory)
    torch.cuda.synchronize()
    return Built(cache, time.perf_counter() - t0, n_slots)


def cache_facts(built: Built) -> dict[str, Any]:
    """Residency and manager counters a sweep row records for any condition's cache."""
    cache = built.cache
    facts: dict[str, Any] = {"gpu_resident_kv_bytes": cache.stats().gpu_resident_kv_bytes}
    if isinstance(cache, BlockPoolCache):
        facts.update(cache.counters())
        facts["n_slots"] = cache.pool_layers[0].n_slots
        # Whether a policy that does not force the sink kept it anyway (h2o), per layer.
        facts["layers_with_block0_resident"] = sum(1 for l in cache.pool_layers if 0 in l.slot_blocks)
    elif isinstance(cache, QuestView):
        facts["host_select_s"] = cache.counters.host_select_s
        facts["selections"] = cache.counters.selections
        facts["k_blocks"] = cache.k_blocks
        facts["attended_tokens_selecting_layers"] = cache.attended_tokens()
        facts["dense_layers"] = cache.dense_layers
    elif isinstance(cache, TieredCache):
        c = cache.counters_dict()
        facts.update({k: v for k, v in c.items() if k != "fetched_pairs_per_step"})
        facts["fetched_pairs_per_step"] = c["fetched_pairs_per_step"]
        facts["k_blocks"] = cache.k_blocks
        facts["n_slots"] = cache.n_slots
        facts["fetch"] = cache.fetch
        facts["attended_tokens_selecting_layers"] = (cache.k_blocks + 2) * cache.block_size
        facts["dense_layers"] = cache.dense_layers
        facts["host_kv_bytes"] = cache.stats().host_kv_bytes
        # What those same blocks would cost in the model's dtype: the denominator of rung 8's saving.
        facts["host_kv_logical_bytes"] = cache.host_logical_bytes()
        facts["tier_quant"] = cache.quant or "none"
        if isinstance(cache, ExactTieredCache):
            # Rung 9's own cost, kept separate from the tier's so the two are additive.
            facts.update({f"exact_{k}": v for k, v in cache.exact_counters_dict().items()})
    return facts
