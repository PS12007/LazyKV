"""Phase 0-4 figures, rendered in light and dark variants for the README's <picture> tags.

Reads results/phase0/analysis/metrics.json, run_1 telemetry, results/phase1/analysis/metrics.json,
results/phase1/latency_regimes/metrics.json, and results/phase{2,3,4}/analysis/metrics.json.
Writes docs/figures/*.png.
Palette: validated categorical slots (fixed order), recessive hairline grid, 2px lines.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from harness.render import iec  # noqa: E402
from harness.results import REPO_ROOT, RESULTS_DIR  # noqa: E402

FIG_DIR = REPO_ROOT / "docs" / "figures"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str, str]
    band: str


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), "#f0efec")
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835", ("#3987e5", "#d95926", "#199e70", "#c98500"), "#262624")


def style_axes(ax: plt.Axes, t: Theme) -> None:
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t.muted, labelcolor=t.ink2, labelsize=9, length=3)
    ax.grid(True, color=t.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t.ink2)
    ax.yaxis.label.set_color(t.ink2)


def new_fig(t: Theme, w: float = 9.0, h: float = 4.8) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(t.surface)
    style_axes(ax, t)
    return fig, ax


def title(fig: plt.Figure, t: Theme, main: str, sub: str) -> None:
    fig.text(0.012, 0.975, main, color=t.ink, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.925, sub, color=t.ink2, fontsize=9.5, ha="left", va="top")


def save(fig: plt.Figure, name: str, t: Theme) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}-{t.name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_pcie(a: dict[str, Any], t: Theme) -> None:
    curve = a["pcie"]["curve"]
    sizes = [r["size_bytes"] for r in curve]
    fig, ax = new_fig(t)
    series = [
        ("pinned_h2d", "Pinned H2D"),
        ("pinned_d2h", "Pinned D2H"),
        ("pageable_h2d", "Pageable H2D"),
        ("pageable_d2h", "Pageable D2H"),
    ]
    ends: list[tuple[float, str]] = []
    for (key, label), color in zip(series, t.series):
        med = [r[key]["median"] / 1e9 for r in curve]
        lo = [r[key]["min"] / 1e9 for r in curve]
        hi = [r[key]["max"] / 1e9 for r in curve]
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(sizes, med, color=color, linewidth=2, marker="o", markersize=4.5, markeredgecolor=t.surface, markeredgewidth=1.2, label=label)
        ends.append((med[-1], label))
    # De-collide end labels: pinned H2D/D2H (and pageable H2D/D2H) finish within a label height.
    min_gap = 1.6
    placed: list[float] = []
    for y, label in sorted(ends):
        y_lab = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_lab)
        ax.annotate(label, (sizes[-1], y), xytext=(sizes[-1] * 1.25, y_lab), textcoords="data", color=t.ink2, fontsize=9, va="center")
    knee = a["pcie"]["pinned_h2d"]["knee80_bytes"]["median"]
    if knee:
        ax.axvline(knee, color=t.muted, linewidth=1)
        ax.annotate(f"pinned H2D reaches\n80% of asymptote\nat {iec(knee)}", (knee, 0.42), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", color=t.ink2, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: iec(v)))
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Transfer size")
    ax.set_ylabel("GB/s (1e9 B/s)")
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.84, 0.3), frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    n = a["pcie"]["pinned_h2d"]["asymptote_bps"]["n_runs"]
    title(fig, t, "PCIe bandwidth vs transfer size", f"Median across {n} independent runs (band = min–max); each run: 3 warmup + 7 interleaved, randomized repeats")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.12)
    save(fig, "pcie_bandwidth", t)


def fig_ladder(a: dict[str, Any], t: Theme) -> None:
    big = max(a["vram"]["decode_attention"], key=lambda r: r["ctx_len"])
    cpu64 = max(a["cpu"]["partial_attn_best"], key=lambda r: r["tokens"])
    kv_fp32 = 2 * cpu64["tokens"] * 8 * 64 * 4
    items = [
        ("VRAM device-to-device copy", a["vram"]["d2d_copy_bps"]["median"], False),
        (f"GPU decode-attention KV scan ({big['ctx_len'] // 1024}K tok)", big["backends"][big["best_backend"]]["kv_scan_bps"]["median"], False),
        ("PCIe pinned H2D", a["pcie"]["pinned_h2d"]["asymptote_bps"]["median"], True),
        ("PCIe pinned D2H", a["pcie"]["pinned_d2h"]["asymptote_bps"]["median"], True),
        ("Host memcpy (1 thread)", a["cpu"]["host_memcpy_bps_1thread"]["median"], False),
        ("PCIe pageable H2D", a["pcie"]["pageable_h2d"]["asymptote_bps"]["median"], False),
        (f"CPU partial attention ({cpu64['tokens'] // 1024}K tok, fp32)", kv_fp32 / cpu64["best_seconds"]["median"], False),
    ]
    if a["nvme"].get("measured"):
        items += [
            ("NVMe sequential read", a["nvme"]["sequential_bps"]["median"], False),
            ("NVMe random 128 KiB read", a["nvme"]["random"]["131072"]["bps"]["median"], False),
        ]
    items.sort(key=lambda x: x[1])
    fig, ax = new_fig(t, 9.0, 4.9)
    ys = range(len(items))
    for y, (label, v, hl) in zip(ys, items):
        ax.barh(y, v / 1e9, height=0.62, color=t.series[0] if hl else t.axis, left=0.01)
        ax.annotate(f"{v / 1e9:.3g}", (v / 1e9, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=t.ink2)
    ax.set_yticks(list(ys), [x[0] for x in items])
    ax.set_xscale("log")
    ax.set_xlim(0.1, max(x[1] for x in items) / 1e9 * 3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("GB/s, log scale (median across runs)")
    ax.grid(axis="y", visible=False)
    title(fig, t, "Where the bytes can move", "Every path LazyKV could use, on this laptop. Highlighted: the PCIe link a CPU tier must cross.")
    fig.subplots_adjust(left=0.33, right=0.97, top=0.85, bottom=0.12)
    save(fig, "bandwidth_ladder", t)


def fig_design(a: dict[str, Any], t: Theme) -> None:
    d = a["design"]
    if not d.get("available"):
        return
    rows = d["rows"]
    fig, ax = new_fig(t)
    series = [
        ("fetch_one_transfer_s", "(a) fetch non-resident KV, 1 batched transfer"),
        ("fetch_64tok_blocks_s", "(a) fetch as 64-token blocks"),
        ("gpu_attention_s", "GPU attention over the same tokens"),
        ("cpu_partial_attn_s", "(c) exact partial attention on CPU"),
    ]
    for (key, label), color in zip(series, t.series):
        pts = [(r["tokens_per_layer"], r[key] * 1e3) for r in rows if key in r]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5, markeredgecolor=t.surface, markeredgewidth=1.2, label=label)
        # Direct end labels: light-mode aqua/yellow sit below 3:1 contrast (validator WARN),
        # so identity must not rest on line color + legend alone.
        short = label.split(",")[0].replace("over the same tokens", "").strip()
        ax.annotate(short, (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(right=max(r["tokens_per_layer"] for r in rows) * 5.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:g}K" if v >= 1024 else f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"Non-resident tokens in one layer ({d['model']} geometry)")
    ax.set_ylabel("Milliseconds per layer, per decode step")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, "Cost per layer of each way to handle non-resident KV", "Composed from measured PCIe bandwidth + per-transfer overhead, best correct GPU kernel, best CPU thread count")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12)
    save(fig, "design_costs", t)


def fig_telemetry(run_dir: Path, t: Theme) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "telemetry.csv").open(encoding="utf-8")))
    marks = metrics["telemetry"]["marks"]

    def col(k: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(float(r["t_s"]))
                ys.append(float(r[k]))
            except (ValueError, KeyError):
                pass
        return xs, ys

    panels = [("temperature.gpu", "GPU temp (°C)"), ("power.draw", "GPU power (W)"), ("clocks.current.graphics", "Graphics clock (MHz)"), ("pcie.link.gen.current", "PCIe link gen")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 6.4), dpi=160, sharex=True)
    fig.patch.set_facecolor(t.surface)
    spans = [(m["t_s"], m["label"].removesuffix("_start")) for m in marks if m["label"].endswith("_start")]
    ends = {m["label"].removesuffix("_end"): m["t_s"] for m in marks if m["label"].endswith("_end")}
    for i, (ax, (key, label)) in enumerate(zip(axes, panels)):
        style_axes(ax, t)
        for j, (start, name) in enumerate(spans):
            if j % 2 == 0:
                ax.axvspan(start, ends.get(name, start), color=t.band, linewidth=0)
            if i == 0:
                ax.annotate(name, ((start + ends.get(name, start)) / 2, 1.02), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=8, color=t.ink2)
        xs, ys = col(key)
        ax.plot(xs, ys, color=t.series[0], linewidth=1.6, drawstyle="steps-post" if "gen" in key else "default")
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Seconds since benchmark start")
    title(fig, t, "Thermal and link-state trace, feasibility run 1", "Sampled every 500 ms by nvidia-smi; shaded bands mark benchmark sections")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.08, hspace=0.18)
    save(fig, "telemetry_run1", t)


def _k(v: float) -> str:
    return f"{v / 1024:g}K"


def fig_p1_latency(a: dict[str, Any], t: Theme) -> None:
    rows = a["contexts"]
    xs = [r["ctx_len"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=160)
    fig.patch.set_facecolor(t.surface)
    ax = axes[0]
    style_axes(ax, t)
    med = [r["decode_wall_median_s"]["median"] * 1e3 for r in rows]
    p10 = [r["decode_wall_p10_s"]["median"] * 1e3 for r in rows]
    p90 = [r["decode_wall_p90_s"]["median"] * 1e3 for r in rows]
    ax.fill_between(xs, p10, p90, color=t.series[0], alpha=0.16, linewidth=0)
    ax.plot(xs, med, color=t.series[0], linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
    ax.annotate("median", (xs[-1], med[-1]), xytext=(6, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    ax.annotate("p10–p90", (xs[-1], p90[-1]), xytext=(6, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylim(bottom=0)
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 2.2)
    ax.set_title("Decode, ms per token", color=t.ink, fontsize=10, loc="left")
    ax.set_xlabel("Context (tokens)")

    ax = axes[1]
    style_axes(ax, t)
    cold = [r["ttft_cold_s"]["median"] for r in rows]
    warm = [r["ttft_warm_s"]["median"] for r in rows]
    for ys, label, color in ((cold, "cuDNN plans cold", t.series[1]), (warm, "plans warm", t.series[0])):
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylim(bottom=0)
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 1.3)
    ax.set_title("Time to first token, s", color=t.ink, fontsize=10, loc="left")
    ax.set_xlabel("Context (tokens)")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    n = a["contexts"][0]["decode_wall_median_s"]["n_runs"]
    title(fig, t, "Full-GPU-KV baseline, Llama-3.2-1B bf16", f"Median across {n} independent runs; chunked prefill {a['config']['chunk_size']} tokens; attention `{a['attention']['strategy']}`")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.8, bottom=0.14, wspace=0.28)
    save(fig, "p1_baseline_latency", t)


def fig_p1_layer_split(a: dict[str, Any], t: Theme) -> None:
    rows = a["contexts"]
    parts = [
        ("attention_kernel_s", "Attention kernel"),
        ("attention_other_s", "Attention: projections, RoPE, cache write"),
        ("mlp_s", "MLP"),
        ("other_s", "Norms + residuals"),
    ]
    fig, ax = new_fig(t, 9.0, 4.2)
    labels = [_k(r["ctx_len"]) for r in rows]
    ys = list(range(len(rows)))
    left = [0.0] * len(rows)
    gap = 0.004  # ms of surface gap between stacked segments
    for (key, label), color in zip(parts, t.series):
        widths = [r["per_layer_mean_s"][key]["median"] * 1e3 for r in rows]
        ax.barh(ys, [max(0.0, w - gap) for w in widths], left=left, height=0.6, color=color, label=label)
        left = [l + w for l, w in zip(left, widths)]
    for y, total in zip(ys, left):
        ax.annotate(f"{total:.2f} ms", (total, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, color=t.ink2)
    ax.set_yticks(ys, labels)
    ax.set_xlabel("Milliseconds per decoder layer, per decode step (GPU timeline)")
    ax.set_xlim(0, max(left) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, "Where a decode layer's time goes", "At batch size 1 most of the layer span is host-side kernel launching, not GPU compute")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.3)
    save(fig, "p1_layer_split", t)


def fig_p1_window(a: dict[str, Any], t: Theme) -> None:
    w = a.get("window", {})
    if not w.get("available"):
        return
    rows = w["rows"]
    xs = [r["ctx_len"] for r in rows]
    fig, ax = new_fig(t)
    series = [
        ([r["fetch_all_one_layer_s"] * 1e3 for r in rows], xs, "Fetch the whole layer's KV (PCIe)"),
        ([r["layer_window_s"] * 1e3 for r in rows], xs, "Measured layer window (Phase 1)"),
    ]
    p0 = [(r["ctx_len"], r["phase0_bound_attention_s"] * 1e3) for r in rows if "phase0_bound_attention_s" in r]
    for (ys, xx, label), color in zip(series, t.series):
        ax.plot(xx, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label=label)
        ax.annotate(label, (xx[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    if p0:
        px, py = zip(*p0)
        ax.plot(px, py, color=t.series[2], linewidth=2, linestyle=(0, (4, 3)), marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label="Phase 0 bound: attention op alone (measured contexts only)")
        ax.annotate("Phase 0 bound (attention only)", (px[-1], py[-1]), xytext=(7, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 6)
    ax.set_xlabel("Context (tokens), Llama-3.2-1B")
    ax.set_ylabel("Milliseconds per layer")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, "The prefetch window, measured", "A copy on its own stream overlaps the layer's span; anything that fits under the window is free in wall time")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12)
    save(fig, "p1_window", t)


def fig_p1_regimes(lr: dict[str, Any], t: Theme) -> None:
    """Every timed token of the latency-regime experiment on one timeline, by core class."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    p_mask = int(lr["core_class_masks"][str(max(int(k) for k in lr["core_class_masks"]))], 16)
    fig, ax = new_fig(t, 9.0, 4.6)
    blocks = sorted(lr["blocks"], key=lambda b: b["t_end_s"][0])
    t_start = blocks[0]["t_end_s"][0]
    for b in blocks:
        x0, x1 = b["t_end_s"][0] - t_start, b["t_end_s"][-1] - t_start
        if b["throttling"] == "opted_out":
            ax.axvspan(x0, x1, color=t.grid, linewidth=0, zorder=0)
    pts = {True: ([], []), False: ([], [])}
    for b in blocks:
        for w, ts, c in zip(b["wall_s"], b["t_end_s"], b["processor"]):
            xs, ys = pts[bool(p_mask >> c & 1)]
            xs.append(ts - t_start)
            ys.append(w * 1e3)
    # Shape as well as color, so core class never rides on color alone.
    for on_p, color, marker, label in ((True, t.series[0], "o", "P-core"), (False, t.series[1], "^", "E-core")):
        xs, ys = pts[on_p]
        ax.scatter(xs, ys, s=7, color=color, marker=marker, linewidths=0, alpha=0.8, zorder=2)
    ax.axhline(lr["slow_threshold_s"] * 1e3, color=t.muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("slow-token threshold", (0, lr["slow_threshold_s"] * 1e3), xytext=(4, 4), textcoords="offset points", color=t.ink2, fontsize=8.5)
    ax.set_ylim(0, max(max(pts[True][1], default=0), max(pts[False][1], default=0)) * 1.08)
    ax.set_xlim(left=0)
    ax.set_xlabel("Seconds since the first timed token")
    ax.set_ylabel("Decode wall time per token, ms")
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=6, color=t.series[0], label="Thread on a P-core"),
        Line2D([], [], linestyle="none", marker="^", markersize=6, color=t.series[1], label="Thread on an E-core"),
        Patch(facecolor=t.grid, edgecolor="none", label="Power throttling opted out"),
        Patch(facecolor=t.surface, edgecolor=t.axis, linewidth=0.6, label="Power throttling OS-managed"),
    ]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=4, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, "Windows moved the benchmark onto E-cores", f"Llama-3.2-1B decode at {lr['config']['ctx']:,} tokens; {len(blocks)} interleaved blocks; nvidia-smi polling on in half of each kind")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.25)
    save(fig, "p1_latency_regimes", t)


P2_POLICIES = (("window_sink", "Window + sink (rung 2)"), ("window", "Window, no sink (control)"), ("lru", "LRU (rung 3)"))
# Ordinal budget ramp, one hue, validated per mode: small budget recedes, large budget stands out.
BUDGET_RAMP = {
    "light": ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"),
    "dark": ("#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
}


def _same_series(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], key: str, rel: float = 1e-9) -> bool:
    """True when two policies measured identical values at every budget (then one hides the other)."""
    va = {r["budget"]: r[key] for r in rows_a}
    vb = {r["budget"]: r[key] for r in rows_b}
    return bool(va) and va.keys() == vb.keys() and all(abs(va[b] - vb[b]) <= rel * max(abs(va[b]), abs(vb[b]), 1e-12) for b in va)


def _budget_axis(ax: plt.Axes, budgets: list[float]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlim(min(budgets) / 1.4, 1.4)


def fig_p2_pareto(a: dict[str, Any], t: Theme) -> None:
    """NIAH accuracy and decode speed against GPU KV budget: two plots, never a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=160)
    fig.patch.set_facecolor(t.surface)
    niah = {n["label"]: n for n in a["niah"]}
    speed = {s["label"]: s for s in a["speed"]}
    budgets = sorted({n["budget"] for n in a["niah"]} | {1.0})

    ax = axes[0]
    style_axes(ax, t)
    full = niah["full@1"]
    ax.axhline(100 * full["accuracy"], color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("full cache", (1.0, 100 * full["accuracy"]), xytext=(-4, 5), textcoords="offset points", ha="right", color=t.ink2, fontsize=8.5)
    niah_rows = {pol: sorted((n for n in a["niah"] if n["policy"] == pol), key=lambda n: n["budget"]) for pol, _ in P2_POLICIES}
    lru_is_window = _same_series(niah_rows["lru"], niah_rows["window"], "accuracy")
    for (policy, name), color in zip(P2_POLICIES, t.series):
        rows = niah_rows[policy]
        xs = [n["budget"] for n in rows]
        ys = [100 * n["accuracy"] for n in rows]
        lo = [100 * (n["accuracy"] - n["accuracy_ci95"][0]) for n in rows]
        hi = [100 * (n["accuracy_ci95"][1] - n["accuracy"]) for n in rows]
        # An identical series would be invisible under the next one; draw it wider underneath.
        wide = policy == "window" and lru_is_window
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linewidth=5 if wide else 2, marker="o", markersize=8 if wide else 5, markeredgecolor=t.surface, markeredgewidth=1.2, capsize=0, elinewidth=1, label=name, zorder=1 if wide else 2)
    if lru_is_window:
        ax.text(0.98, 0.03, "LRU and window, no sink: identical at every budget", transform=ax.transAxes, ha="right", va="bottom", color=t.ink2, fontsize=8.5)
    bf = niah.get("block_full@1")
    if bf:
        ax.plot([1.0], [100 * bf["accuracy"]], linestyle="none", marker="D", markersize=6, color=t.series[3], markeredgecolor=t.surface, markeredgewidth=1.2, label="Block pool, 100%")
    _budget_axis(ax, budgets)
    ax.set_ylim(-3, 103)
    ax.set_title("NIAH accuracy, %", color=t.ink, fontsize=10, loc="left")
    ax.set_xlabel("GPU KV budget (share of the sequence's KV)")

    ax = axes[1]
    style_axes(ax, t)
    if speed:
        fs = speed["full@1"]["tokens_per_s"]["median"]
        ax.axhline(fs, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("full cache", (1.0, fs), xytext=(-4, 5), textcoords="offset points", ha="right", color=t.ink2, fontsize=8.5)
        tops = [fs]
        for (policy, name), color in zip(P2_POLICIES, t.series):
            rows = sorted((r for r in a["speed"] if r["policy"] == policy), key=lambda r: r["budget"])
            ys = [r["tokens_per_s"]["median"] for r in rows]
            tops += ys
            ax.plot([r["budget"] for r in rows], ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
        if "block_full@1" in speed:
            ax.plot([1.0], [speed["block_full@1"]["tokens_per_s"]["median"]], linestyle="none", marker="D", markersize=6, color=t.series[3], markeredgecolor=t.surface, markeredgewidth=1.2)
        ax.set_ylim(0, max(tops) * 1.18)
    _budget_axis(ax, budgets)
    ax.set_title("Decode tokens/s (fast kernel)", color=t.ink, fontsize=10, loc="left")
    ax.set_xlabel("GPU KV budget (share of the sequence's KV)")

    leg = axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=4, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, f"GPU-only residency policies at {a['context']:,} tokens", f"{a['niah_prompts_per_condition']} NIAH prompts per point (95% bootstrap CI); speed is the median of {a['speed_runs']} independent runs")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.27, wspace=0.25)
    save(fig, "p2_pareto", t)


def fig_p2_depth(
    a: dict[str, Any],
    t: Theme,
    policies: tuple[tuple[str, str], ...] = P2_POLICIES,
    name: str = "p2_depth",
    heading: str = "Where the needle is decides whether it survives",
) -> None:
    """Accuracy by needle depth: evicting policies only find needles their window still holds."""
    fig, axes = plt.subplots(1, len(policies), figsize=(9.6, 3.9), dpi=160, sharey=True)
    fig.patch.set_facecolor(t.surface)
    budgets = sorted({n["budget"] for n in a["niah"] if n["policy"] in dict(policies)})
    ramp = BUDGET_RAMP[t.name]
    full = next(n for n in a["niah"] if n["label"] == "full@1")
    depths = [float(d) for d in full["by_depth"]]
    for ax, (policy, pretty) in zip(axes, policies):
        style_axes(ax, t)
        ax.plot([100 * d for d in depths], [100 * full["by_depth"][f"{d:g}"] for d in depths], color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        for budget, color in zip(budgets, ramp):
            row = next(n for n in a["niah"] if n["policy"] == policy and n["budget"] == budget)
            ax.plot([100 * d for d in depths], [100 * row["by_depth"][f"{d:g}"] for d in depths], color=color, linewidth=2, marker="o", markersize=4, markeredgecolor=t.surface, markeredgewidth=1, label=f"{100 * budget:g}%")
        ax.set_title(pretty, color=t.ink, fontsize=10, loc="left")
        ax.set_xticks([100 * d for d in depths])
        ax.set_xlabel("Needle depth in the prompt, %")
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("NIAH accuracy, %")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], color=t.muted, linewidth=1, linestyle=(0, (4, 3))))
    labels.append("full cache")
    leg = fig.legend(handles, [f"budget {l}" if l != "full cache" else l for l in labels], loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, heading, "Accuracy by needle depth, one line per budget; darker is a larger budget")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.26, wspace=0.12)
    save(fig, name, t)


def fig_p2_kl(a: dict[str, Any], t: Theme, policies: tuple[tuple[str, str], ...] = P2_POLICIES, name: str = "p2_kl") -> None:
    fig, ax = new_fig(t, 9.0, 4.2)
    budgets = sorted({r["budget"] for r in a["teacher_forced"] if r["policy"] in dict(policies)})
    floor = min((r["mean_kl"] for r in a["teacher_forced"] if r["mean_kl"] > 0), default=1e-6)
    tf_rows = {pol: sorted((r for r in a["teacher_forced"] if r["policy"] == pol), key=lambda r: r["budget"]) for pol, _ in policies}
    # KL of two no-sink policies can differ in the last digits; "overlap" here means visually indistinguishable.
    lru_is_window = "lru" in tf_rows and "window" in tf_rows and _same_series(tf_rows["lru"], tf_rows["window"], "mean_kl", rel=0.02)
    for (policy, pretty), color in zip(policies, t.series):
        rows = tf_rows[policy]
        ys = [r["mean_kl"] for r in rows]
        wide = policy == "window" and lru_is_window
        ax.plot([r["budget"] for r in rows], ys, color=color, linewidth=5 if wide else 2, marker="o", markersize=8 if wide else 5, markeredgecolor=t.surface, markeredgewidth=1.2, label=pretty, zorder=1 if wide else 2)
        if lru_is_window and policy == "lru":
            continue
        text = "Window, no sink, and LRU (within 2%)" if (lru_is_window and policy == "window") else pretty
        ax.annotate(text, (rows[-1]["budget"], ys[-1]), xytext=(9, 0), textcoords="offset points", color=t.ink2, fontsize=8.5, va="center")
    ax.set_yscale("log")
    _budget_axis(ax, budgets)
    ax.set_xlim(min(budgets) / 1.4, max(budgets) * 3.2)
    ax.set_ylim(bottom=floor / 3)
    ax.set_xlabel("GPU KV budget (share of the sequence's KV)")
    ax.set_ylabel("Mean KL from the full cache, nats (log)")
    title(fig, t, "Teacher-forced divergence from the full cache", "Mean over documents of per-position KL; the full cache and the 100% block pool are exactly zero, so they cannot appear on a log axis")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.13)
    save(fig, name, t)


P3_POLICIES = (("window_sink", "Window + sink (rung 2)"), ("h2o", "H2O-style (rung 4)"), ("quest", "Quest-style (rung 5)"))


def fig_p3_pareto(a: dict[str, Any], t: Theme) -> None:
    """Accuracy against the budget each policy varies, against resident GPU KV, and decode speed.

    Separate panels, never a dual axis. The middle panel is the brief's x-axis (GPU KV bytes
    resident): without a CPU tier, Quest-style selection attends to less but frees nothing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), dpi=160)
    fig.patch.set_facecolor(t.surface)
    niah = {n["label"]: n for n in a["niah"]}
    # No policy has a point at 100% here (the full cache is the dashed reference), so the
    # axis stops at the largest swept budget instead of crowding a 100% tick against 75%.
    budgets = sorted({n["budget"] for n in a["niah"] if n["policy"] in dict(P3_POLICIES)})
    full = niah["full@1"]
    full_gib = full["gpu_resident_kv_bytes_median"] / 2**30
    rows = {pol: sorted((n for n in a["niah"] if n["policy"] == pol), key=lambda n: n["budget"]) for pol, _ in P3_POLICIES}

    for panel, ax in enumerate(axes[:2]):
        style_axes(ax, t)
        xfull = 1.0 if panel == 0 else full_gib
        ax.axhline(100 * full["accuracy"], color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("full cache", (xfull, 100 * full["accuracy"]), xytext=(-4, 5), textcoords="offset points", ha="right", color=t.ink2, fontsize=8.5)
        for (policy, pretty), color in zip(P3_POLICIES, t.series):
            r = rows[policy]
            xs = [n["budget"] for n in r] if panel == 0 else [n["gpu_resident_kv_bytes_median"] / 2**30 for n in r]
            ys = [100 * n["accuracy"] for n in r]
            lo = [100 * (n["accuracy"] - n["accuracy_ci95"][0]) for n in r]
            hi = [100 * (n["accuracy_ci95"][1] - n["accuracy"]) for n in r]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, capsize=0, elinewidth=1, label=pretty if panel == 0 else None, alpha=0.9)
        ax.set_ylim(-3, 103)
    _budget_axis(axes[0], budgets)
    axes[0].set_title("NIAH accuracy, %", color=t.ink, fontsize=10, loc="left")
    axes[0].set_xlabel("Budget: resident (evicting) or attended (Quest)")
    axes[1].set_title("NIAH accuracy, % (same data)", color=t.ink, fontsize=10, loc="left")
    axes[1].set_xlabel("GPU-resident KV at the answer, GiB")
    axes[1].set_xlim(0, full_gib * 1.12)
    quest_gib = rows["quest"][0]["gpu_resident_kv_bytes_median"] / 2**30
    axes[1].annotate("Quest attends to less\nbut frees nothing", (quest_gib, 50), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", color=t.ink2, fontsize=8.5)

    ax = axes[2]
    style_axes(ax, t)
    if a["speed"]:
        speed = {s["label"]: s for s in a["speed"]}
        fs = speed["full@1"]["tokens_per_s"]["median"]
        ax.axhline(fs, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        # Below the line: policies can be faster than the full cache here, so a label above it collides.
        ax.annotate("full cache", (max(budgets), fs), xytext=(-4, -6), textcoords="offset points", ha="right", va="top", color=t.ink2, fontsize=8.5)
        tops = [fs]
        for (policy, _), color in zip(P3_POLICIES, t.series):
            r = sorted((s for s in a["speed"] if s["policy"] == policy), key=lambda s: s["budget"])
            ys = [s["tokens_per_s"]["median"] for s in r]
            tops += ys
            ax.plot([s["budget"] for s in r], ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
        ax.set_ylim(0, max(tops) * 1.18)
    _budget_axis(ax, budgets)
    ax.set_title("Decode tokens/s (fast kernel)", color=t.ink, fontsize=10, loc="left")
    ax.set_xlabel("Budget: resident (evicting) or attended (Quest)")

    leg = axes[0].legend(loc="upper center", bbox_to_anchor=(1.75, -0.2), ncol=3, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, f"Eviction versus query-aware selection at {a['context']:,} tokens", f"{a['niah_prompts_per_condition']} NIAH prompts per point (95% bootstrap CI); speed is the median of {a['speed_runs']} independent run{'s' if a['speed_runs'] != 1 else ''}")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.26, wspace=0.22)
    save(fig, "p3_pareto", t)


P4_SERIES = (
    ("quest", None, "Quest-style, all resident (rung 5)"),
    ("tiered_sync", "spare0", "CPU tier, sync fetch, VRAM = attended (rung 6)"),
    ("tiered_sync", "spare1", "CPU tier, sync fetch, 2x slots (rung 6)"),
    ("tiered_prefetch", "spare1", "CPU tier + layer-ahead prefetch, 2x slots (rung 7)"),
)


def _p4_speed_rows(a: dict[str, Any], policy: str, variant: str | None) -> list[dict[str, Any]]:
    src = a["spare0"]["speed"] if variant == "spare0" else a["speed"]
    return sorted((s for s in src if s["policy"] == policy), key=lambda s: s["budget"])


def fig_p4_tradeoff(a: dict[str, Any], t: Theme) -> None:
    """Decode speed against GPU-resident KV, and the host time behind it.

    Rungs 5-7 attend to identical key sets (bit-identical outputs), so accuracy is one curve and is
    not repeated here: what the tier changes is memory and latency, so those are the two axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=160)
    fig.patch.set_facecolor(t.surface)
    full = next(s for s in a["speed"] if s["label"] == "full@1")
    full_gib = full["gpu_resident_kv_bytes"]["median"] / 2**30
    fs = full["tokens_per_s"]["median"]
    colors = (t.muted,) + t.series[:3]
    ax = axes[0]
    style_axes(ax, t)
    ax.axhline(fs, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("full cache", (full_gib, fs), xytext=(-4, 5), textcoords="offset points", ha="right", color=t.ink2, fontsize=8.5)
    tops = [fs]
    for (policy, variant, pretty), color in zip(P4_SERIES, (t.series[3],) + t.series[:3]):
        rows = _p4_speed_rows(a, policy, variant)
        if not rows:
            continue
        xs = [r["gpu_resident_kv_bytes"]["median"] / 2**30 for r in rows]
        ys = [r["tokens_per_s"]["median"] for r in rows]
        tops += ys
        if policy == "quest":
            # Every rung 5 budget sits at the same x: it attends to less and frees nothing. A line
            # through those points would be a vertical stripe, so draw the span it covers.
            ax.plot([xs[0], xs[0]], [min(ys), max(ys)], color=color, linewidth=2.5, solid_capstyle="butt", label=pretty)
            ax.scatter(xs, ys, color=color, s=26, edgecolor=t.surface, linewidth=1.2, zorder=4)
            ax.annotate("every budget,\nsame VRAM", (xs[0], min(ys)), xytext=(-8, -2), textcoords="offset points", ha="right", va="top", color=t.ink2, fontsize=8)
            continue
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label=pretty)
        for r, x, y in zip(rows, xs, ys):
            # Only the extremes are labelled: adjacent budgets can land within a few MiB of each other.
            if r["budget"] in (min(q["budget"] for q in rows), max(q["budget"] for q in rows)):
                ax.annotate(f"{100 * r['budget']:g}%", (x, y), xytext=(0, -12), textcoords="offset points", ha="center", color=t.muted, fontsize=7.5)
    ax.set_xlim(0, full_gib * 1.12)
    ax.set_ylim(0, max(tops) * 1.15)
    ax.set_xlabel("GPU-resident KV during decode, GiB (labels: attended budget)")
    ax.set_title("Decode tokens/s", color=t.ink, fontsize=10, loc="left")

    # Right: where a token's time goes, at each budget, for rung 6 with 2x slots.
    ax = axes[1]
    style_axes(ax, t)
    budgets = sorted({s["budget"] for s in a["speed"] if s["policy"] == "tiered_sync"}, reverse=True)
    quest = {s["budget"]: s for s in a["speed"] if s["policy"] == "quest"}
    tier = a["tier"].get("tiered_sync", {})
    parts = (("Rank + host sync", "host_rank_ms_per_token"), ("Fetch launch", "host_fetch_ms_per_token"), ("Other select + gather", None), ("Seal", "host_seal_ms_per_token"))
    ys = list(range(len(budgets)))
    # Each bar is the measured rung 6 latency, split into the tier's own host time and everything
    # else (model forward, attention, and any GPU wait): the segments sum to the measurement by
    # construction, so nothing is implied that was not timed.
    walls = [tier.get(f"{b:g}", {}).get("decode_wall_median_s", 0.0) * 1e3 for b in budgets]
    base = [0.0] * len(budgets)
    for (name, key), color in zip(parts, t.series):
        vals = []
        for b in budgets:
            tr = tier.get(f"{b:g}", {})
            if key is None:
                v = tr.get("host_select_ms_per_token", 0) - (tr.get("host_rank_ms_per_token") or 0) - tr.get("host_fetch_ms_per_token", 0)
            else:
                v = tr.get(key) or 0.0
            vals.append(max(0.0, v))
        ax.barh(ys, vals, left=list(base), color=color, height=0.62, label=name)
        base = [base[i] + vals[i] for i in range(len(budgets))]
    rest = [max(0.0, walls[i] - base[i]) for i in range(len(budgets))]
    ax.barh(ys, rest, left=list(base), color=t.band, edgecolor=t.axis, linewidth=0.6, height=0.62, label="Model forward and everything else")
    for i, b in enumerate(budgets):
        q = quest.get(b)
        if q:
            ax.scatter([q["decode_wall_median_s"]["median"] * 1e3], [i], marker="|", s=300, color=t.ink, linewidths=2, zorder=5, label="Rung 5 decode, same run" if i == 0 else None)
    ax.set_yticks(ys, [f"{100 * b:g}%" for b in budgets])
    ax.invert_yaxis()
    ax.set_xlabel("ms per decoded token (bar: measured rung 6, split by what was timed)")
    ax.set_title("Where rung 6's token goes (2x slots)", color=t.ink, fontsize=10, loc="left")
    for axis in axes:
        leg = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8)
        for text in leg.get_texts():
            text.set_color(t.ink2)
    title(fig, t, f"What a CPU tier costs under query-aware selection, at {a['context']:,} tokens", f"Median of {a['speed_runs']} independent runs; rungs 5-7 produce identical outputs, so accuracy does not change along these curves")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.33, wspace=0.25)
    save(fig, "p4_tradeoff", t)


def fig_p4_fetch(a: dict[str, Any], t: Theme) -> None:
    """On-demand fetches per token and the share that are thrash, with and without spare slots."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=160)
    fig.patch.set_facecolor(t.surface)
    variants = (("spare0", a["spare0"]["tier"], "VRAM = attended set"), ("spare1", a["tier"], "2x slots"))
    for ax in axes:
        style_axes(ax, t)
    i = 0
    for policy, pname in (("tiered_sync", "sync"), ("tiered_prefetch", "prefetch")):
        for _, src, vname in variants:
            rows = src.get(policy, {})
            if not rows:
                continue
            bs = sorted(float(b) for b in rows)
            color = t.series[i % 4]
            dash = "-" if policy == "tiered_sync" else (0, (4, 2))
            axes[0].plot(bs, [rows[f"{b:g}"]["fetched_pairs_per_token"] for b in bs], color=color, linewidth=2, linestyle=dash, marker="o", markersize=4, label=f"{pname}, {vname}")
            axes[1].plot(bs, [100 * rows[f"{b:g}"]["thrash_share_of_fetches"] for b in bs], color=color, linewidth=2, linestyle=dash, marker="o", markersize=4)
            i += 1
    _budget_axis(axes[0], sorted(float(b) for b in a["tier"]["tiered_sync"]))
    _budget_axis(axes[1], sorted(float(b) for b in a["tier"]["tiered_sync"]))
    axes[0].set_yscale("log")
    axes[0].set_title("(head, block) pairs fetched on demand per token, all layers", color=t.ink, fontsize=10, loc="left")
    axes[1].set_title("Share of fetches evicted within the last 16 steps, %", color=t.ink, fontsize=10, loc="left")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("Attended budget")
    leg = axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=4, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, "Selection churn: what VRAM has to bring back every token", f"Median over every repeat of {a['speed_runs']} runs per variant")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.28, wspace=0.2)
    save(fig, "p4_fetch", t)


def fig_p4_blocksize(a: dict[str, Any], t: Theme) -> None:
    """Accuracy and decode speed against block size, at one budget, for rung 5 and rung 6.

    Phase 0 measured the PCIe efficiency knee; this is the other half of the block-size question,
    because a smaller block is a finer selection granularity *and* a smaller transfer.
    """
    rows = (a.get("blocksize") or {}).get("rows") or []
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=160)
    fig.patch.set_facecolor(t.surface)
    sizes = sorted({r["block_size"] for r in rows})
    series = (("quest", "Quest-style, all resident (rung 5)"), ("tiered_sync", "CPU tier, sync fetch (rung 6)"))
    for ax in axes:
        style_axes(ax, t)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.set_xlabel("Block size, tokens")
    for (policy, pretty), color in zip(series, t.series):
        r = sorted((x for x in rows if x["policy"] == policy), key=lambda x: x["block_size"])
        xs = [x["block_size"] for x in r]
        acc = [(x["block_size"], 100 * x["niah_accuracy"]) for x in r if x["niah_accuracy"] is not None]
        if acc:
            axes[0].plot([x for x, _ in acc], [y for _, y in acc], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
        # Labelled on the speed panel, which always has data: the accuracy panel is empty until the
        # per-block-size quality runs finish.
        axes[1].plot(xs, [x["tokens_per_s"]["median"] for x in r], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label=pretty)
        fetched = [(x["block_size"], x["tier"]["fetch_mean_transfer_kib"]) for x in r if x.get("tier")]
        if fetched:
            axes[2].plot([x for x, _ in fetched], [y for _, y in fetched], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
    axes[0].set_title("NIAH accuracy, %", color=t.ink, fontsize=10, loc="left")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Decode tokens/s", color=t.ink, fontsize=10, loc="left")
    axes[1].set_ylim(bottom=0)
    axes[2].set_title("Mean H2D transfer, KiB (rung 6)", color=t.ink, fontsize=10, loc="left")
    knee = (a.get("blocksize") or {}).get("pcie_knee_kib")
    if knee:
        axes[2].axhline(knee, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        axes[2].annotate("Phase 0 80% knee", (max(sizes), knee), xytext=(-4, 4), textcoords="offset points", ha="right", color=t.ink2, fontsize=8.5)
    leg = axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    budget = rows[0]["budget"]
    title(fig, t, f"Block size at a {100 * budget:g}% budget, {a['context']:,} tokens", "Smaller blocks are a finer selection granularity and a smaller transfer; both effects are measured here")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.28, wspace=0.22)
    save(fig, "p4_blocksize", t)


def fig_p5_capacity(a: dict[str, Any], t: Theme) -> None:
    """What the int8 warm/cold tier saves off-GPU, and what it charges in accuracy and latency.

    Three panels because rung 8 has three separable effects and conflating them is exactly how a
    capacity result gets mis-sold as a speed one.
    """
    cap = a.get("capacity") or []
    if not cap:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=160)
    fig.patch.set_facecolor(t.surface)
    budgets = [c["budget"] for c in cap]
    for ax in axes:
        style_axes(ax, t)
        _budget_axis(ax, budgets)
        ax.set_xlabel("Attended budget")
    for (key, pretty), color in zip((("exact", "bf16 host pool (rung 6)"), ("int8", "int8 host pool (rung 8)")), t.series):
        axes[0].plot(budgets, [c[f"host_kv_mib_{key}"] for c in cap], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2, label=pretty)
    axes[0].set_title("KV held off-GPU, MiB", color=t.ink, fontsize=10, loc="left")
    axes[0].set_ylim(bottom=0)

    # Drawn on the same budget axis as the other two panels rather than as categorical bars, so the
    # three panels line up and a reader can follow one budget across all of them.
    niah = {(r["policy"], r["budget"]): r for r in a["niah"]}
    sig = {d["budget"]: d["sign_test_p"] for d in a.get("divergence_vs_exact", [])}
    deltas = [100 * (niah[("tiered_int8", b)]["accuracy"] - niah[("tiered_sync", b)]["accuracy"]) for b in budgets]
    axes[1].axhline(0, color=t.muted, linewidth=1)
    axes[1].plot(budgets, deltas, color=t.series[2], linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
    # Every point carries its paired p-value: without it a few flipped prompts read as a real change.
    for b, d in zip(budgets, deltas):
        pv = sig.get(b)
        if pv is None:
            continue
        axes[1].annotate(f"p={pv:.2f}", (b, d), xytext=(0, 8 if d >= 0 else -14), textcoords="offset points", ha="center", color=t.ink2, fontsize=8)
    axes[1].set_title("NIAH accuracy against the exact tier, percentage points", color=t.ink, fontsize=10, loc="left")
    lo, hi = min(deltas), max(deltas)
    pad = max(1.0, 0.35 * (hi - lo))
    axes[1].set_ylim(lo - pad, hi + pad)

    lat = {r["budget"]: r["ratio_median"] for r in a["latency_vs_exact"]}
    ys = [lat[b] for b in budgets if lat.get(b) is not None]
    if ys:
        axes[2].axhline(1.0, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
        axes[2].plot([b for b in budgets if lat.get(b) is not None], ys, color=t.series[3], linewidth=2, marker="o", markersize=5, markeredgecolor=t.surface, markeredgewidth=1.2)
        axes[2].annotate("slower than rung 6 above this line", (budgets[0], 1.0), xytext=(4, 6), textcoords="offset points", color=t.ink2, fontsize=8.5)
    axes[2].set_title("Decode latency, x the exact tier", color=t.ink, fontsize=10, loc="left")
    leg = axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    title(fig, t, f"int8 warm/cold tier at {a['context']:,} tokens: a capacity result", f"Paired within repeats of {a['speed_runs']} speed runs; accuracy over {a['niah_prompts_per_condition']} NIAH prompts per condition")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.28, wspace=0.24)
    save(fig, "p5_capacity", t)


def fig_ladder_pareto(a: dict[str, Any], t: Theme) -> None:
    """Brief §B8's headline experiment: every rung on one memory axis, accuracy and speed.

    x is GPU-resident KV bytes, which is the resource the whole project is spending. Each rung is a
    curve across the same budget sweep, so a rung that sits up and to the left is strictly better.
    Where a rung was measured at two VRAM-slot counts, the curve uses the one that actually frees
    memory, because otherwise the x axis would mean different things in different curves.
    """
    rows = a.get("rows") or []
    if not rows:
        return
    # One row per (rung, budget), preferring the memory-honest variant.
    picked: dict[tuple[int, float], dict[str, Any]] = {}
    for r in rows:
        key = (r["rung"], r["budget"])
        if key not in picked or r["slots"] == "= attended":
            picked[key] = r
    by_rung: dict[int, list[dict[str, Any]]] = {}
    for r in picked.values():
        by_rung.setdefault(r["rung"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=160)
    fig.patch.set_facecolor(t.surface)
    for ax in axes:
        style_axes(ax, t)
        ax.set_xlabel("GPU-resident KV, MiB")
    target = 100 * 0.99 * a["summary"]["full_niah_accuracy"]
    axes[0].axhline(target, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].annotate("99% of full-cache accuracy", (0, target), xytext=(4, 5), textcoords="offset points", color=t.ink2, fontsize=8.5)
    for i, rung in enumerate(sorted(by_rung)):
        rs = sorted((r for r in by_rung[rung] if r["resident_mib"] is not None), key=lambda r: r["resident_mib"])
        if not rs:
            continue
        color = t.series[i % len(t.series)]
        dash = "-" if i < len(t.series) else (0, (4, 2))
        marker = "o" if rung != 1 else "*"
        name = rs[0]["name"]
        # The slot variant belongs in the legend: rungs 6 and 7 are shown with VRAM holding exactly
        # the attended set, rung 8 was only ever run at twice it, and without the label the memory
        # axis would quietly mean two different things.
        variant = rs[0]["slots"]
        axes[0].plot([r["resident_mib"] for r in rs], [100 * r["accuracy"] for r in rs], color=color, linestyle=dash,
                     linewidth=2, marker=marker, markersize=6 if rung != 1 else 12, markeredgecolor=t.surface,
                     markeredgewidth=1.2, label=f"{rung}. {name}" + (f" (slots {variant})" if variant else ""))
        # Rung 5 is a vertical line, and that is the finding rather than a drawing error.
        if rung == 5 and len({round(r["resident_mib"]) for r in rs}) == 1:
            top = max(rs, key=lambda r: r["accuracy"])
            axes[0].annotate(
                "rung 5 frees nothing:\nits budget buys accuracy,\nnot memory",
                (top["resident_mib"], 100 * top["accuracy"]), xytext=(-12, -58), textcoords="offset points",
                ha="right", color=t.ink2, fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": t.muted, "linewidth": 0.8},
            )
        sp = [r for r in rs if r["tokens_per_s"] is not None]
        if sp:
            axes[1].plot([r["resident_mib"] for r in sp], [r["tokens_per_s"] for r in sp], color=color, linestyle=dash,
                         linewidth=2, marker=marker, markersize=6 if rung != 1 else 12, markeredgecolor=t.surface, markeredgewidth=1.2)
    axes[0].set_title("NIAH accuracy, %", color=t.ink, fontsize=10, loc="left")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Decode tokens/s", color=t.ink, fontsize=10, loc="left")
    axes[1].set_ylim(bottom=0)
    leg = axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.13), ncol=3, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t.ink2)
    sm = a["summary"]
    title(fig, t, f"The policy ladder at {sm['context']:,} tokens: what a GPU KV budget buys",
          f"{sm['prompts']} NIAH prompts per condition, {sm['block_size']}-token blocks; "
          f"conditions measured in more than one phase agree to {sm['repeatability_accuracy_max_gap_pp']:.2f} pp")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.30, wspace=0.16)
    save(fig, "ladder_pareto", t)


def main() -> None:
    a = json.loads((RESULTS_DIR / "phase0" / "analysis" / "metrics.json").read_text(encoding="utf-8"))
    p1_path = RESULTS_DIR / "phase1" / "analysis" / "metrics.json"
    p1 = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else None
    p2_path = RESULTS_DIR / "phase2" / "analysis" / "metrics.json"
    p2 = json.loads(p2_path.read_text(encoding="utf-8")) if p2_path.exists() else None
    p3_path = RESULTS_DIR / "phase3" / "analysis" / "metrics.json"
    p3 = json.loads(p3_path.read_text(encoding="utf-8")) if p3_path.exists() else None
    p4_path = RESULTS_DIR / "phase4" / "analysis" / "metrics.json"
    p4 = json.loads(p4_path.read_text(encoding="utf-8")) if p4_path.exists() else None
    p5_path = RESULTS_DIR / "phase5" / "analysis" / "metrics.json"
    p5 = json.loads(p5_path.read_text(encoding="utf-8")) if p5_path.exists() else None
    ladder_path = RESULTS_DIR / "ladder" / "metrics.json"
    ladder = json.loads(ladder_path.read_text(encoding="utf-8")) if ladder_path.exists() else None
    lr_path = RESULTS_DIR / "phase1" / "latency_regimes" / "metrics.json"
    lr = json.loads(lr_path.read_text(encoding="utf-8")) if lr_path.exists() else None
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    for t in (LIGHT, DARK):
        fig_pcie(a, t)
        fig_ladder(a, t)
        fig_design(a, t)
        fig_telemetry(RESULTS_DIR / "phase0" / "feasibility" / "run_1", t)
        if p1 is not None:
            fig_p1_latency(p1, t)
            fig_p1_layer_split(p1, t)
            fig_p1_window(p1, t)
        if lr is not None:
            fig_p1_regimes(lr, t)
        if p2 is not None:
            fig_p2_pareto(p2, t)
            fig_p2_depth(p2, t)
            fig_p2_kl(p2, t)
        if p3 is not None:
            fig_p3_pareto(p3, t)
            fig_p2_depth(p3, t, P3_POLICIES, "p3_depth", "Needle depth under eviction and under selection")
            fig_p2_kl(p3, t, P3_POLICIES, "p3_kl")
        if p4 is not None:
            fig_p4_tradeoff(p4, t)
            fig_p4_fetch(p4, t)
            fig_p4_blocksize(p4, t)
        if p5 is not None:
            fig_p5_capacity(p5, t)
        if ladder is not None:
            fig_ladder_pareto(ladder, t)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
