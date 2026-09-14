from __future__ import annotations

import math

import pytest

from harness.render import NOT_MEASURED, iec, render
from harness.stats import fit_overhead, knee_size, overlap_fraction, summarize


def test_summarize_odd_sample() -> None:
    s = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s.median == 3.0
    assert s.q25 == 2.0 and s.q75 == 4.0 and s.iqr == 2.0
    assert s.n == 5 and s.min == 1.0 and s.max == 5.0


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_knee_interpolates_in_log_size() -> None:
    sizes = [1024, 4096]
    values = [0.0, 1.0]
    # Target 0.5 sits halfway in log2 space: 2^11.
    assert knee_size(sizes, values, 0.5, 1.0) == pytest.approx(2048)


def test_knee_none_when_never_reached() -> None:
    assert knee_size([1, 2, 4], [0.1, 0.2, 0.3], 0.8, 1.0) is None


def test_fit_overhead_recovers_synthetic_model() -> None:
    t0, bw = 50e-6, 12e9
    sizes = [2**k for k in range(16, 29)]
    times = [t0 + s / bw for s in sizes]
    fit = fit_overhead(sizes, times)
    assert fit.t0_s == pytest.approx(t0, rel=1e-6)
    assert fit.bandwidth_bps == pytest.approx(bw, rel=1e-6)
    # At the 80% knee the effective bandwidth s / t(s) must equal 0.8 * B.
    s80 = fit.knee_bytes(0.8)
    assert s80 / (t0 + s80 / bw) == pytest.approx(0.8 * bw, rel=1e-9)


def test_overlap_fraction_extremes() -> None:
    assert overlap_fraction(1.0, 0.5, 1.0) == pytest.approx(1.0)  # wall = max
    assert overlap_fraction(1.0, 0.5, 1.5) == pytest.approx(0.0)  # wall = sum


def test_render_values_blocks_and_missing() -> None:
    ctx = {"a": {"b": 1.23456, "lst": [10, 20]}, "none": None}
    out = render(
        "v={{ a.b | .2f }} i={{a.lst.1}} m={{ a.nope }} n={{none}} {{> tbl }} {{> absent }}",
        ctx,
        {"tbl": "|x|"},
    )
    assert out == f"v=1.23 i=20 m={NOT_MEASURED} n={NOT_MEASURED} |x| _{NOT_MEASURED}_"


def test_block_content_is_not_reparsed() -> None:
    assert render("{{> t }}", {}, {"t": "{{ a }}"}) == "{{ a }}"


@pytest.mark.parametrize(
    ("n", "expected"),
    [(512, "512 B"), (64 * 1024, "64 KiB"), (int(1.5 * 2**30), "1.5 GiB"), (256 * 2**20, "256 MiB")],
)
def test_iec(n: int, expected: str) -> None:
    assert iec(n) == expected


def test_named_formatters() -> None:
    assert render("{{ v | pct }}", {"v": 0.1234}) == "12.3%"
    assert render("{{ v | gbps }}", {"v": 12.5e9}) == "12.50 GB/s"
    assert math.isfinite(float(render("{{ v | .3f }}", {"v": 2.0})))
