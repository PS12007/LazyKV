"""Teacher-forced divergence between a policy and the full-cache reference (brief §B7.1)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch

# Quality is scored on the memory-efficient SDPA kernel, not the fastest one.
# Phase 1 measured cuDNN's single-query decode returning one of two bit patterns for
# bit-identical inputs (both equally close to a float64 reference, and unaffected by
# PyTorch's determinism flags). Through 16 layers that compounds into several percent
# top-1 flips between two runs of the *same* configuration, a noise floor that would hide
# the small effects later policies must be judged on. The memory-efficient kernel repeats
# bit-exactly, so a policy's divergence from the reference is the policy's alone. Residency
# policies select which KV attention sees; they do not depend on which exact kernel computes it.
QUALITY_STRATEGY = "efficient"


@dataclass(frozen=True)
class Divergence:
    positions: int
    top1_agreement: float
    mean_kl: float
    max_kl: float
    kl_ci95: tuple[float, float]
    top1_ci95: tuple[float, float]
    exact_match: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def per_position_kl(ref_logp: torch.Tensor, pol_logp: torch.Tensor) -> torch.Tensor:
    """KL(ref || policy) per position, in nats. Inputs are log-probs [positions, vocab]."""
    ref = ref_logp.double()
    pol = pol_logp.double()
    return (ref.exp() * (ref - pol)).sum(dim=-1).clamp_min(0.0)


def _bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Positions are not independent (same document),
    so this understates uncertainty; it is reported as a lower bound on spread."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def compare_stream(ref_logp: torch.Tensor, policy_rows: Iterable[torch.Tensor]) -> Divergence:
    """Same result as `compare`, but consumes policy rows one at a time on their device.

    Each reference row is moved next to the policy row and compared in float64 there, so
    host memory holds only the reference tensor. The full-tensor version needed several
    positions x vocab float64 temporaries in RAM, which on a 16 GB laptop was enough to get
    the Phase 1 quality run killed for low memory.
    """
    kl: list[float] = []
    agree: list[float] = []
    exact = True
    n = 0
    for i, row in enumerate(policy_rows):
        ref = ref_logp[i].to(row.device, non_blocking=True)
        exact = exact and bool(torch.equal(ref, row))
        r, p = ref.double(), row.double()
        kl.append(float((r.exp() * (r - p)).sum().clamp_min(0.0)))
        agree.append(float(ref.argmax() == row.argmax()))
        n += 1
    if n != ref_logp.shape[0]:
        raise ValueError(f"policy produced {n} rows, reference has {ref_logp.shape[0]}")
    return Divergence(
        positions=n,
        top1_agreement=sum(agree) / n,
        mean_kl=sum(kl) / n,
        max_kl=max(kl),
        kl_ci95=_bootstrap_ci(kl),
        top1_ci95=_bootstrap_ci(agree),
        exact_match=exact,
    )


def compare(ref_logp: torch.Tensor, pol_logp: torch.Tensor) -> Divergence:
    if ref_logp.shape != pol_logp.shape:
        raise ValueError(f"shape mismatch {tuple(ref_logp.shape)} vs {tuple(pol_logp.shape)}")
    kl = per_position_kl(ref_logp, pol_logp).tolist()
    agree = (ref_logp.argmax(-1) == pol_logp.argmax(-1)).double().tolist()
    return Divergence(
        positions=len(kl),
        top1_agreement=sum(agree) / len(agree),
        mean_kl=sum(kl) / len(kl),
        max_kl=max(kl),
        kl_ci95=_bootstrap_ci(kl),
        top1_ci95=_bootstrap_ci(agree),
        exact_match=bool(torch.equal(ref_logp, pol_logp)),
    )
