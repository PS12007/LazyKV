"""Attention function registered with transformers' AttentionInterface.

Why a registered function instead of attn_implementation="sdpa":
- Phase 0 measured default SDPA dispatch with GQA landing on the math kernel on this
  Windows build, 33x slower than cuDNN at 65K tokens. A baseline left on it would inflate
  every offloading policy's apparent benefit. Here the kernel is chosen explicitly and
  verified against a float32 reference at startup.
- A custom implementation name is not in transformers' mask registry, so the model passes
  attention_mask=None and this function owns causality. That is exactly what block-managed
  policies need later: the mask is a function of which blocks are selected.

Causality: decode (q_len == 1) needs no causal mask at batch size 1. Chunked prefill
(q_len > 1, kv_len >= q_len) needs *lower-right* alignment, because the chunk's queries sit
at the end of the KV. SDPA's is_causal=True is upper-left and would be wrong.

cuDNN plan caching (measured in Phase 1): cuDNN SDPA builds an execution plan per distinct
input shape. A plain growing decode presents a new KV length every token, and each plan
build cost tens of milliseconds, while a cached plan runs in about a tenth of a millisecond.
A fixed-budget block policy would get constant shapes, and cached plans, for free, so a
baseline without the same trick would make offloading look better than it is. The
"cudnn_bucketed" strategy pads the decode KV view to a bucket boundary inside the
preallocated storage and masks the padding, so plans are rebuilt once per bucket.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers import AttentionInterface

log = logging.getLogger(__name__)

IMPLEMENTATION_NAME = "lazykv_sdpa"
DEFAULT_BUCKET = 1024

STRATEGIES = ("cudnn_bucketed", "cudnn", "efficient", "math")


@dataclass
class KernelTimer:
    """Optional CUDA-event timing of the attention kernel call alone (profiling mode only)."""

    enabled: bool = False
    events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)

    def reset(self) -> None:
        self.events.clear()


@dataclass
class AttentionConfig:
    strategy: str = "cudnn_bucketed"
    bucket: int = DEFAULT_BUCKET
    timer: KernelTimer = field(default_factory=KernelTimer)
    # One-entry memo: all layers of a decode step share (length, bucket), so the padding
    # mask is built once per step instead of once per layer.
    _mask_key: tuple[int, int, torch.dtype, torch.device] | None = None
    _mask: torch.Tensor | None = None


# Module-level holder, written only by install()/set_strategy_for_test(). Transformers
# calls a plain function, so its configuration has to be reachable from module scope.
_CONFIG = AttentionConfig()


def _extend_view(t: torch.Tensor, new_len: int) -> torch.Tensor | None:
    """View `t` ([1, h, L, d]) as [1, h, new_len, d] over the same storage, if it fits.

    The preallocated cache returns prefix views of larger tensors, so the rows past L
    already exist. Tensors without spare storage (e.g. from DynamicCache) return None.
    """
    size = (t.shape[0], t.shape[1], new_len, t.shape[3])
    needed = t.storage_offset() + sum((s - 1) * st for s, st in zip(size, t.stride())) + 1
    if needed > t.untyped_storage().nbytes() // t.element_size():
        return None
    return torch.as_strided(t, size, t.stride(), t.storage_offset())


def _padding_mask(length: int, bucket_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (length, bucket_len, dtype, device)
    if _CONFIG._mask_key != key:
        mask = torch.zeros((1, 1, 1, bucket_len), dtype=dtype, device=device)
        mask[..., length:] = float("-inf")
        _CONFIG._mask, _CONFIG._mask_key = mask, key
    assert _CONFIG._mask is not None
    return _CONFIG._mask


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float | None, strategy: str, bucket: int = DEFAULT_BUCKET) -> torch.Tensor:
    """Exact attention of `query` over all of `key`/`value` with lower-right causality."""
    q_len, kv_len = query.shape[-2], key.shape[-2]
    F = torch.nn.functional.scaled_dot_product_attention
    if strategy in ("cudnn", "cudnn_bucketed"):
        backend = SDPBackend.CUDNN_ATTENTION
    elif strategy == "efficient":
        backend = SDPBackend.EFFICIENT_ATTENTION
    elif strategy == "math":
        backend = SDPBackend.MATH
    else:
        raise ValueError(f"unknown attention strategy {strategy!r}")

    if strategy == "efficient":
        # The memory-efficient kernel rejects mismatched head counts; expand() is a view.
        groups = query.shape[1] // key.shape[1]
        key = key[:, :, None].expand(-1, -1, groups, -1, -1).reshape(key.shape[0], -1, kv_len, key.shape[-1])
        value = value[:, :, None].expand(-1, -1, groups, -1, -1).reshape(value.shape[0], -1, kv_len, value.shape[-1])

    if q_len > 1:
        with sdpa_kernel([backend]):
            return F(query, key, value, attn_mask=causal_lower_right(q_len, kv_len), scale=scaling, enable_gqa=strategy != "efficient")

    mask = None
    if strategy == "cudnn_bucketed":
        bucket_len = -(-kv_len // bucket) * bucket
        if bucket_len != kv_len:
            k_ext, v_ext = _extend_view(key, bucket_len), _extend_view(value, bucket_len)
            if k_ext is not None and v_ext is not None:
                key, value = k_ext, v_ext
                mask = _padding_mask(kv_len, bucket_len, query.dtype, query.device)
    with sdpa_kernel([backend]):
        return F(query, key, value, attn_mask=mask, scale=scaling, enable_gqa=strategy != "efficient")


def lazykv_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None:
        raise RuntimeError("lazykv attention owns causality; a mask should never be passed")
    if query.shape[0] != 1:
        raise RuntimeError("batch size 1 only")
    timer = _CONFIG.timer
    if timer.enabled:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    out = attend(query, key, value, scaling, _CONFIG.strategy, _CONFIG.bucket)
    if timer.enabled:
        end.record()
        timer.events.append((getattr(module, "layer_idx", -1), start, end))
    # Policies that learn from attention (LRU usage, later H2O scores) need the post-RoPE
    # query, which only exists here. Transformers forwards extra model() kwargs down to this
    # function, so the cache is handed in per call rather than registered globally.
    observer = kwargs.get("lazykv_observer")
    if observer is not None:
        observer.observe(int(getattr(module, "layer_idx")), query, key)
    return out.transpose(1, 2).contiguous(), None


@dataclass(frozen=True)
class KernelCheck:
    strategy: str
    ok: bool
    max_rel_err_decode: float | None
    max_rel_err_prefill: float | None
    seconds_decode_growing_median: float | None
    seconds_decode_growing_max: float | None
    error: str | None = None


def _reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float) -> torch.Tensor:
    """Float32 explicit attention with lower-right causal masking: the ground truth."""
    q, k, v = query.float(), key.float(), value.float()
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = (q @ k.transpose(-1, -2)) * scaling
    q_len, kv_len = q.shape[-2], k.shape[-2]
    q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
    k_pos = torch.arange(kv_len, device=q.device)[None, :]
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref).abs().max() / ref.abs().max()).item()


def _check_one(strategy: str, n_q: int, n_kv: int, head_dim: int, dtype: torch.dtype, probe_ctx: int, steps: int, tolerance: float) -> KernelCheck:
    gen = torch.Generator(device="cuda").manual_seed(0)
    scaling = head_dim**-0.5

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    try:
        # Correctness with spare storage past the live length, as the preallocated cache has,
        # so the bucketed path is exercised with garbage padding that must be masked out.
        store_k, store_v = rand(1, n_kv, 1024, head_dim), rand(1, n_kv, 1024, head_dim)
        live = 509
        k, v = store_k[:, :, :live], store_v[:, :, :live]
        pq, dq = rand(1, n_q, 64, head_dim), rand(1, n_q, 1, head_dim)
        err_p = _rel_err(attend(pq, k, v, scaling, strategy, bucket=256), _reference(pq, k, v, scaling))
        err_d = _rel_err(attend(dq, k, v, scaling, strategy, bucket=256), _reference(dq, k, v, scaling))

        # Speed on the realistic pattern: KV length grows by one every step.
        big_k, big_v = rand(1, n_kv, probe_ctx + steps + DEFAULT_BUCKET, head_dim), rand(1, n_kv, probe_ctx + steps + DEFAULT_BUCKET, head_dim)
        times = []
        for i in range(steps):
            L = probe_ctx + i
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            attend(dq, big_k[:, :, :L], big_v[:, :, :L], scaling, strategy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    except RuntimeError as exc:
        return KernelCheck(strategy, False, None, None, None, None, str(exc).splitlines()[0])
    ok = math.isfinite(err_p) and math.isfinite(err_d) and max(err_p, err_d) <= tolerance
    steady = sorted(times[1:])  # first call may build a plan for every strategy
    return KernelCheck(strategy, ok, err_d, err_p, steady[len(steady) // 2], max(times))


def verify_strategies(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    tolerance: float = 1e-2,
    probe_ctx: int = 16_000,
    steps: int = 24,
    candidates: tuple[str, ...] = STRATEGIES,
) -> list[KernelCheck]:
    """Correctness (decode + chunked prefill vs float32 reference) and growing-length decode speed."""
    results = []
    for name in candidates:
        # Unavailable kernels emit bursts of "kernel not used because" warnings; the outcome
        # is captured in KernelCheck.error instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results.append(_check_one(name, n_q_heads, n_kv_heads, head_dim, dtype, probe_ctx, steps, tolerance))
    return results


def install(strategy: str | None, n_q_heads: int, n_kv_heads: int, head_dim: int, bucket: int = DEFAULT_BUCKET) -> tuple[str, list[KernelCheck]]:
    """Register the attention function and pin the strategy.

    strategy=None picks the fastest strategy (median growing-length decode) that passes
    verification. An explicit strategy that fails verification is a hard error: silent
    fallback is how the Phase 0 kernel trap happens.
    """
    checks = verify_strategies(n_q_heads, n_kv_heads, head_dim)
    passing = [c for c in checks if c.ok]
    if strategy is None:
        if not passing:
            raise RuntimeError(f"no attention strategy passed verification: {checks}")
        chosen = min(passing, key=lambda c: c.seconds_decode_growing_median or float("inf")).strategy
    else:
        match = next((c for c in checks if c.strategy == strategy), None)
        if match is None or not match.ok:
            raise RuntimeError(f"requested attention strategy {strategy!r} failed verification: {match}")
        chosen = strategy
    _CONFIG.strategy, _CONFIG.bucket = chosen, bucket
    AttentionInterface.register(IMPLEMENTATION_NAME, lazykv_attention_forward)
    log.info("attention strategy pinned to %s (bucket %d); checks=%s", chosen, bucket, checks)
    return chosen, checks


def kernel_timer() -> KernelTimer:
    return _CONFIG.timer


def current_strategy() -> tuple[str, int]:
    return _CONFIG.strategy, _CONFIG.bucket


def set_strategy_for_test(strategy: str) -> Callable[[], None]:
    """Temporarily override the strategy (tests and cross-kernel checks); returns a restore function."""
    previous = _CONFIG.strategy
    _CONFIG.strategy = strategy
    AttentionInterface.register(IMPLEMENTATION_NAME, lazykv_attention_forward)

    def restore() -> None:
        _CONFIG.strategy = previous

    return restore
