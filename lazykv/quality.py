"""Teacher-forced divergence between a policy and the full-cache reference (brief §B7.1)."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch


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
