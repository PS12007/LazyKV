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
from lazykv.policies import make_policy


@dataclass(frozen=True)
class Condition:
    policy: str  # "full" means the FullGPUCache itself
    budget: float

    @property
    def label(self) -> str:
        return f"{self.policy}@{self.budget:g}"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def conditions(cfg: dict[str, Any]) -> list[Condition]:
    out = [Condition("full", 1.0), Condition("block_full", 1.0)]
    out += [Condition(p, b) for p in cfg["policies"] for b in cfg["budgets"]]
    return out


@dataclass
class Built:
    cache: Cache
    build_s: float
    n_slots: int | None


def build_cache(full: FullGPUCache, cond: Condition, total_tokens: int, block_size: int) -> Built:
    """The cache a condition decodes from. The full cache is returned as is; callers truncate it back afterwards."""
    if cond.policy == "full":
        return Built(full, 0.0, None)
    n_slots = slots_for_budget(cond.budget, total_tokens, block_size)
    # Synchronized: the boundary is mostly GPU gathers, and an unsynchronized clock would only
    # time their launch.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = BlockPoolCache.from_full(full, n_slots, block_size, lambda: make_policy(cond.policy))
    torch.cuda.synchronize()
    return Built(cache, time.perf_counter() - t0, n_slots)
