"""Layer-major prefill straight into the CPU tier, so the prompt's full KV never sits in VRAM.

The tier (lazykv/tiered.py) was built from an exact prefill into a FullGPUCache: every layer's
prompt KV on the GPU at once, then copied to host. So the longest context the tier could serve was
the longest one the full cache could prefill, and the tier saved decode-time VRAM only. This
module removes that limit.

Why layer-major, not chunk-major with the old KV streamed back. A chunk-major prefill runs every
layer on chunk c before chunk c+1, so chunk c's attention at layer L needs layer L's KV for every
earlier chunk. Keeping that KV on the host means fetching all of it back for every chunk: traffic
grows with the square of the context, on the same per-layer host round trip Phase 11 found to be
the decode bottleneck. Layer-major turns the loops round (FlexGen's schedule, arXiv 2303.06865):
run *all* chunks through layer L, holding the hidden states of every prompt token, then move on to
L+1. Only layer L's KV has to be on the GPU while layer L runs. When it finishes, its KV is handed
to `TieredLayer`, which copies it to host, builds the Quest metadata and seeds the slots, exactly
as at the old boundary, and the buffer is reused for layer L+1. Every KV byte crosses PCIe once,
device to host, and never comes back during prefill.

What it costs instead is the hidden state for every prompt token, [n, hidden] in the model's dtype:
for a 3B model at 64K that is 0.4 GiB, against 0.11 MiB per token of KV. And the dense layers
(Quest's first two) keep their full KV on the GPU, because decode attends to all of it.

Bit-identity. Each layer sees the same chunk boundaries, the same input chunks and the same KV
prefix it would see in the chunk-major prefill, so every kernel runs on the same shapes and the
same values. The KV, the prompt's last logits and therefore every decoded token match the
FullGPUCache path bit for bit; tests/test_tiered_prefill.py checks that, including decode after it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from lazykv.cache import FullGPUCache, PreallocatedLayer
from lazykv.selection import QUEST_DENSE_LAYERS
from lazykv.tiered import TieredCache, TieredLayer


class _OneLayerCache(Cache):
    """Routes each layer's update to its own store: dense layers to `full`, the layer being run to `scratch`.

    Every selecting layer shares the one scratch buffer, which is reset before each of them runs.
    """

    def __init__(self, full: FullGPUCache, scratch: PreallocatedLayer, dense_layers: int) -> None:
        super().__init__(layers=full.layers)
        self.full, self.scratch, self.dense_layers = full, scratch, dense_layers

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        store = self.full.layers[layer_idx] if layer_idx < self.dense_layers else self.scratch
        return store.update(key_states, value_states)


@dataclass
class TieredPrefillResult:
    last_logits: torch.Tensor  # [vocab], float32, on GPU
    cache: TieredCache
    full: FullGPUCache  # the dense layers' store; `cache` owns it, it is returned for truncate()
    wall_s: float
    chunks: int
    boundary_s: float  # inside wall_s: building the tiers (D2H, metadata, seeding), summed over layers
    layer_s: list[float] = field(default_factory=list)  # wall time per decoder layer, boundary included


@torch.inference_mode()
def tiered_prefill(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    chunk_size: int,
    k_blocks: int,
    block_size: int,
    capacity_tokens: int,
    dense_layers: int = QUEST_DENSE_LAYERS,
    host_pools: list[torch.Tensor] | None = None,
    n_slots: int | None = None,
    quant: str | None = None,
    **cache_kwargs: object,
) -> TieredPrefillResult:
    """Prefill `input_ids` layer by layer and return a decode-ready TieredCache.

    `cache_kwargs` go to TieredCache (prefetch, fetch, stages, instrument, ...); the arguments named
    here are the ones that also shape the tiers, so they are not passed twice.
    """
    inner = model.model
    layers = inner.layers
    ids = input_ids.to("cuda").view(1, -1)
    n = ids.shape[1]
    if n > capacity_tokens:
        raise ValueError(f"prompt of {n} tokens exceeds capacity {capacity_tokens}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Dense layers grow during decode too, so their stores are sized for the full capacity. The full
    # cache allocates a layer only on its first update, so the selecting layers' entries stay empty.
    full = FullGPUCache(len(layers), capacity_tokens)
    scratch = PreallocatedLayer(n)
    router = _OneLayerCache(full, scratch, dense_layers)
    bounds = [(s, min(n, s + chunk_size)) for s in range(0, n, chunk_size)]
    hidden = inner.embed_tokens(ids)
    rope = []
    for s, e in bounds:
        pos = torch.arange(s, e, device=ids.device).unsqueeze(0)
        rope.append((pos, inner.rotary_emb(hidden[:, s:e], position_ids=pos)))
    tiers: dict[int, TieredLayer] = {}
    layer_s: list[float] = []
    boundary_s = 0.0
    for i, layer in enumerate(layers):
        tl = time.perf_counter()
        scratch.reset()
        for (s, e), (pos, pe) in zip(bounds, rope):
            # Written back in place: chunk c's output is never an input to another chunk at this layer.
            hidden[:, s:e] = layer(hidden[:, s:e], attention_mask=None, position_ids=pos, past_key_values=router, use_cache=True, position_embeddings=pe)
        if i >= dense_layers:
            host = None if host_pools is None else host_pools[i - dense_layers]
            tiers[i] = TieredLayer(scratch.keys[:, :, :n], scratch.values[:, :, :n], k_blocks, block_size, capacity_tokens, host, n_slots, quant=quant)
            boundary_s += tiers[i].boundary_d2h_s
        torch.cuda.synchronize()
        layer_s.append(time.perf_counter() - tl)
    del scratch, router
    # As LlamaForCausalLM does with logits_to_keep=1: norm over the last chunk, project its last row.
    s, e = bounds[-1]
    logits = model.lm_head(inner.norm(hidden[:, s:e])[:, -1:])
    last = logits[0, -1].float()
    del hidden
    cache = TieredCache(full, k_blocks, block_size, capacity_tokens, model=model, dense_layers=dense_layers, n_slots=n_slots, quant=quant, tiers=tiers, **cache_kwargs)  # type: ignore[arg-type]
    torch.cuda.synchronize()
    return TieredPrefillResult(last, cache, full, time.perf_counter() - t0, len(bounds), boundary_s, layer_s)
