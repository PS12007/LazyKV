"""Robust summary statistics and curve analysis used by every benchmark.

We report median and IQR rather than mean and stddev because laptop GPU timings are
heavy-tailed (WDDM scheduling, thermal clock changes): a single stalled sample should
not move the headline number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Summary:
    n: int
    median: float
    q25: float
    q75: float
    iqr: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (same definition as numpy's default)."""
    if not sorted_xs:
        raise ValueError("quantile of empty sequence")
    pos = (len(sorted_xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def summarize(xs: Sequence[float]) -> Summary:
    if not xs:
        raise ValueError("cannot summarize an empty sample")
    s = sorted(xs)
    q25, med, q75 = quantile(s, 0.25), quantile(s, 0.5), quantile(s, 0.75)
    return Summary(n=len(s), median=med, q25=q25, q75=q75, iqr=q75 - q25, min=s[0], max=s[-1])


def knee_size(
    sizes: Sequence[float], values: Sequence[float], fraction: float, asymptote: float
) -> float | None:
    """Smallest size at which ``values`` first reaches ``fraction * asymptote``.

    Interpolates linearly in log2(size) between the bracketing points, because transfer
    sizes are swept geometrically. Returns None if the curve never reaches the target.
    """
    if len(sizes) != len(values):
        raise ValueError("sizes and values must have equal length")
    target = fraction * asymptote
    for i, (s, v) in enumerate(zip(sizes, values)):
        if v >= target:
            if i == 0:
                return float(s)
            s0, v0 = sizes[i - 1], values[i - 1]
            t = (target - v0) / (v - v0)
            return 2.0 ** (math.log2(s0) + t * (math.log2(s) - math.log2(s0)))
    return None


@dataclass(frozen=True)
class OverheadFit:
    """Per-transfer cost model: t(s) = t0 + s / bandwidth."""

    t0_s: float
    bandwidth_bps: float

    def knee_bytes(self, fraction: float) -> float:
        # Effective bandwidth s / t(s) equals fraction * B when s = f/(1-f) * t0 * B.
        return fraction / (1.0 - fraction) * self.t0_s * self.bandwidth_bps


def fit_overhead(sizes: Sequence[float], seconds_per_transfer: Sequence[float]) -> OverheadFit:
    """Fit t = t0 + s/B with relative-error weighting.

    Unweighted least squares would let the 256 MiB points (hundreds of ms) swamp the
    64 KiB points (tens of microseconds), leaving t0 essentially unconstrained. Dividing
    each residual by its own t makes every size count equally.
    """
    if len(sizes) < 2:
        raise ValueError("need at least two points to fit")
    # Solve min sum(((t0 + s*c) - t) / t)^2 for (t0, c) with c = 1/B: 2x2 normal equations.
    a11 = a12 = a22 = b1 = b2 = 0.0
    for s, t in zip(sizes, seconds_per_transfer):
        w = 1.0 / (t * t)
        a11 += w
        a12 += w * s
        a22 += w * s * s
        b1 += w * t
        b2 += w * s * t
    det = a11 * a22 - a12 * a12
    if det == 0:
        raise ValueError("degenerate fit")
    t0 = (b1 * a22 - b2 * a12) / det
    c = (a11 * b2 - a12 * b1) / det
    if c <= 0:
        raise ValueError("fit produced non-positive bandwidth")
    return OverheadFit(t0_s=t0, bandwidth_bps=1.0 / c)


def overlap_fraction(t_compute: float, t_copy: float, t_both: float) -> float:
    """How much of the shorter operation ran concurrently with the longer one.

    1.0 means wall time was max(compute, copy) (perfect overlap); 0.0 means it was
    compute + copy (fully serialized). Values slightly outside [0, 1] are noise and are
    reported unclipped so the noise stays visible.
    """
    shorter = min(t_compute, t_copy)
    if shorter <= 0:
        raise ValueError("durations must be positive")
    return (t_compute + t_copy - t_both) / shorter
