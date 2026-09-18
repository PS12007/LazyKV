"""int8 packing for the warm/cold tier: policy ladder rung 8 (mixed precision by tier).

Rungs 6 and 7 keep host copies in the model's own dtype, so the host pool is exactly as large as
the KV it mirrors and every fetch moves full-width bytes. Rung 8 stores the host copy as int8 and
dequantizes on arrival, which halves the host pool and the PCIe traffic. VRAM is untouched: the
slot pool stays bf16, so the "hot" tier is exact and only the warm/cold tier is approximate.

Phase 4 measured that this decode is host-bound, not link-bound (4.6 MiB/token is under a
millisecond of link time against 5.0-6.8 ms of per-layer host round trip), so this is expected to
be a *capacity* result and not a latency one. It is run anyway because it is the ladder's last
memory rung and because it is the first rung at which the tier stops being exact, which puts a
quality axis back on the tier that rungs 6 and 7 did not have.

Scheme, following KIVI (arXiv 2402.02750), which measured that key and value tensors have
differently-shaped outliers:

- **Keys: per channel.** Key outliers are concentrated in a few channels that are consistently
  large across tokens, so a scale shared along the token axis wastes almost the whole range on
  those channels. Reduce over the block's tokens: one (scale, zero) per head_dim channel.
- **Values: per token.** Value outliers are per token rather than per channel. Reduce over
  head_dim: one (scale, zero) per token in the block.

Asymmetric (min/max) rather than symmetric: KV distributions are not centred on zero, and an
asymmetric affine map costs one extra stored number per group for a real accuracy gain.

Scales and zeros are stored as float16, not bf16. bf16 has 8 mantissa bits, the same as the int8
payload it is scaling, so a bf16 scale would contribute about as much error as the quantization
itself; float16's 11 bits make the scale's own error negligible, and KV magnitudes here are far
inside float16's range.

**One pair, one contiguous record.** Everything for one (block, head) pair -- both payloads and all
four scale arrays -- is packed into a single uint8 record. The tier's transfer machinery indexes
the host pool by pair and copies runs of pairs; keeping the format opaque to it means rung 8
changes the *width* of a transfer and nothing else, so the per-transfer count and the gather path
measured in Phase 4 carry over unchanged.
"""

from __future__ import annotations

import torch

SCALE_DTYPE = torch.float16
_LEVELS = 255  # int8 range [-128, 127] used as 256 levels; 255 intervals between min and max


def packed_pair_bytes(block_size: int, head_dim: int) -> int:
    """Bytes one packed (block, head) pair occupies: both int8 payloads plus the four scale arrays."""
    scale = SCALE_DTYPE.itemsize
    return 2 * block_size * head_dim + 2 * scale * head_dim + 2 * scale * block_size


def _offsets(block_size: int, head_dim: int) -> tuple[int, ...]:
    """Cut points of the packed record: k payload, v payload, k scale, k zero, v scale, v zero."""
    s = SCALE_DTYPE.itemsize
    n = block_size * head_dim
    return (n, 2 * n, 2 * n + s * head_dim, 2 * n + 2 * s * head_dim, 2 * n + 2 * s * head_dim + s * block_size)


def _affine(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric int8 quantization of `x` [n, bs, d] with the range taken over `dim`.

    Returns (q int8, scale, zero), scales shaped [n, g] over the surviving axis. The arithmetic is
    float32: in bf16 the division by the scale would itself cost about one int8 level.
    """
    mn = x.amin(dim=dim, keepdim=True).float()
    mx = x.amax(dim=dim, keepdim=True).float()
    # A constant group has no range; any positive scale maps it back to its own value via `zero`.
    scale = ((mx - mn) / _LEVELS).clamp_min(torch.finfo(torch.float32).tiny)
    q = ((x.float() - mn) / scale).round_().clamp_(0, _LEVELS).sub_(128).to(torch.int8)
    return q, scale.squeeze(dim).to(SCALE_DTYPE), mn.squeeze(dim).to(SCALE_DTYPE)


def pack(blocks: torch.Tensor) -> torch.Tensor:
    """Pack KV pairs `[..., 2, block_size, head_dim]` into uint8 records `[..., packed_pair_bytes]`.

    Runs on whatever device `blocks` is on; the tier packs on the GPU, where the KV already is, so
    only the narrowed bytes cross the link.
    """
    *lead, two, bs, d = blocks.shape
    if two != 2:
        raise ValueError(f"expected a [..., 2, block_size, head_dim] KV pair, got {tuple(blocks.shape)}")
    x = blocks.reshape(-1, 2, bs, d)
    kq, ks, kz = _affine(x[:, 0], dim=1)  # per channel: the range is taken over the block's tokens
    vq, vs, vz = _affine(x[:, 1], dim=2)  # per token: the range is taken over head_dim
    parts = [kq.flatten(1), vq.flatten(1), ks, kz, vs, vz]
    packed = torch.cat([p.contiguous().view(torch.uint8) for p in parts], dim=1)
    return packed.view(*lead, packed.shape[-1])


def unpack(packed: torch.Tensor, block_size: int, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize uint8 records `[m, packed_pair_bytes]` back to `[m, 2, block_size, head_dim]`.

    The inverse of `pack`. Reads the payloads and scales as views into the record rather than
    copies, so the only allocation is the output and the float32 arithmetic behind it.
    """
    if packed.ndim != 2 or packed.shape[1] != packed_pair_bytes(block_size, head_dim):
        raise ValueError(f"expected [m, {packed_pair_bytes(block_size, head_dim)}] uint8 records, got {tuple(packed.shape)}")
    o1, o2, o3, o4, o5 = _offsets(block_size, head_dim)
    kq = packed[:, :o1].view(torch.int8).unflatten(1, (block_size, head_dim))
    vq = packed[:, o1:o2].view(torch.int8).unflatten(1, (block_size, head_dim))
    ks, kz = packed[:, o2:o3].view(SCALE_DTYPE), packed[:, o3:o4].view(SCALE_DTYPE)
    vs, vz = packed[:, o4:o5].view(SCALE_DTYPE), packed[:, o5:].view(SCALE_DTYPE)
    k = (kq.float() + 128) * ks.float().unsqueeze(1) + kz.float().unsqueeze(1)  # scales are per channel
    v = (vq.float() + 128) * vs.float().unsqueeze(2) + vz.float().unsqueeze(2)  # per token
    return torch.stack([k, v], dim=1).to(dtype)
