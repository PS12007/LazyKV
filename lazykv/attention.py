"""Attention function registered with transformers' AttentionInterface.

Why a registered function instead of attn_implementation="sdpa":
- Phase 0 measured default SDPA dispatch with GQA landing on the math kernel on this
  Windows build, 33x slower than cuDNN at 65K tokens. A baseline left on it would inflate
  every offloading policy's apparent benefit. Here the kernel is chosen explicitly and
  verified against math at startup.
- A custom implementation name is not in transformers' mask registry, so the model passes
  attention_mask=None and this function owns causality. That is exactly what block-managed
  policies need later: the mask is a function of which blocks are selected, not of the
  full sequence.

Causality: decode (q_len == 1) needs no mask at batch size 1. Chunked prefill (q_len > 1,
kv_len >= q_len) needs *lower-right* causal alignment, because the chunk's queries sit at
the end of the KV. SDPA's is_causal=True is upper-left and would be wrong whenever
kv_len > q_len.
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

_BACKENDS: dict[str, SDPBackend] = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}


@dataclass
class KernelTimer:
    """Optional CUDA-event timing of the attention kernel call alone (profiling mode only)."""

    enabled: bool = False
    events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)

    def reset(self) -> None:
        self.events.clear()


@dataclass
class AttentionConfig:
    backend: str = "cudnn"
    timer: KernelTimer = field(default_factory=KernelTimer)


# Module-level holder, set once by `install()`. Transformers' interface calls a plain
# function, so configuration has to be reachable from it; it is written only at install.
_CONFIG = AttentionConfig()


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float | None, backend: SDPBackend) -> torch.Tensor:
    q_len, kv_len = query.shape[-2], key.shape[-2]
    mask = None if q_len == 1 else causal_lower_right(q_len, kv_len)
    with sdpa_kernel([backend]):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling, enable_gqa=True)


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
    out = _sdpa(query, key, value, scaling, _BACKENDS[_CONFIG.backend])
    if timer.enabled:
        end.record()
        timer.events.append((getattr(module, "layer_idx", -1), start, end))
    return out.transpose(1, 2).contiguous(), None


@dataclass(frozen=True)
class KernelCheck:
    backend: str
    ok: bool
    max_rel_err_decode: float | None
    max_rel_err_prefill: float | None
    seconds_decode: float | None
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


def verify_backends(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    tolerance: float = 1e-2,
    probe_ctx: int = 16_384,
    candidates: tuple[str, ...] = ("cudnn", "efficient", "flash", "math"),
) -> list[KernelCheck]:
    """Check every candidate kernel for correctness (decode + chunked prefill) and decode speed."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    scaling = head_dim**-0.5

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    small_kv = 512
    kq, kk, kv_ = rand(1, n_q_heads, 64, head_dim), rand(1, n_kv_heads, small_kv, head_dim), rand(1, n_kv_heads, small_kv, head_dim)
    ref_prefill = _reference(kq, kk, kv_, scaling)
    dq = kq[:, :, -1:]
    ref_decode = _reference(dq, kk, kv_, scaling)
    big_q, big_k, big_v = rand(1, n_q_heads, 1, head_dim), rand(1, n_kv_heads, probe_ctx, head_dim), rand(1, n_kv_heads, probe_ctx, head_dim)

    probes = _Probes(kq, kk, kv_, dq, ref_prefill, ref_decode, big_q, big_k, big_v, scaling)
    results = []
    for name in candidates:
        # Unavailable kernels emit a burst of "kernel not used because" warnings; the
        # outcome is captured in KernelCheck.error instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results.append(_check_one(name, probes, tolerance))
    return results


@dataclass(frozen=True)
class _Probes:
    prefill_q: torch.Tensor
    small_k: torch.Tensor
    small_v: torch.Tensor
    decode_q: torch.Tensor
    ref_prefill: torch.Tensor
    ref_decode: torch.Tensor
    big_q: torch.Tensor
    big_k: torch.Tensor
    big_v: torch.Tensor
    scaling: float


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref).abs().max() / ref.abs().max()).item()


def _check_one(name: str, p: _Probes, tolerance: float) -> KernelCheck:
    backend = _BACKENDS[name]
    try:
        err_p = _rel_err(_sdpa(p.prefill_q, p.small_k, p.small_v, p.scaling, backend), p.ref_prefill)
        err_d = _rel_err(_sdpa(p.decode_q, p.small_k, p.small_v, p.scaling, backend), p.ref_decode)
        for _ in range(3):
            _sdpa(p.big_q, p.big_k, p.big_v, p.scaling, backend)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _sdpa(p.big_q, p.big_k, p.big_v, p.scaling, backend)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
    except RuntimeError as exc:
        return KernelCheck(name, False, None, None, None, str(exc).splitlines()[0])
    ok = math.isfinite(err_p) and math.isfinite(err_d) and max(err_p, err_d) <= tolerance
    return KernelCheck(name, ok, err_d, err_p, dt)


def install(backend: str | None, n_q_heads: int, n_kv_heads: int, head_dim: int) -> tuple[str, list[KernelCheck]]:
    """Register the attention function and pin the kernel.

    backend=None picks the fastest kernel that passes verification. An explicit backend
    that fails verification is a hard error: silently falling back is how the Phase 0
    kernel trap happens.
    """
    checks = verify_backends(n_q_heads, n_kv_heads, head_dim)
    passing = [c for c in checks if c.ok]
    if backend is None:
        if not passing:
            raise RuntimeError(f"no attention kernel passed verification: {checks}")
        chosen = min(passing, key=lambda c: c.seconds_decode or float("inf")).backend
    else:
        match = next((c for c in checks if c.backend == backend), None)
        if match is None or not match.ok:
            raise RuntimeError(f"requested attention backend {backend!r} failed verification: {match}")
        chosen = backend
    _CONFIG.backend = chosen
    AttentionInterface.register(IMPLEMENTATION_NAME, lazykv_attention_forward)
    log.info("attention kernel pinned to %s; checks=%s", chosen, checks)
    return chosen, checks


def kernel_timer() -> KernelTimer:
    return _CONFIG.timer


def set_backend_for_test(backend: str) -> Callable[[], None]:
    """Temporarily override the backend (tests only); returns a restore function."""
    previous = _CONFIG.backend
    _CONFIG.backend = backend
    AttentionInterface.register(IMPLEMENTATION_NAME, lazykv_attention_forward)

    def restore() -> None:
        _CONFIG.backend = previous

    return restore
