"""Phase 0 figures, rendered in light and dark variants for the README's <picture> tags.

Reads results/phase0/analysis/metrics.json and run_1 telemetry. Writes docs/figures/*.png.
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


def main() -> None:
    a = json.loads((RESULTS_DIR / "phase0" / "analysis" / "metrics.json").read_text(encoding="utf-8"))
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    for t in (LIGHT, DARK):
        fig_pcie(a, t)
        fig_ladder(a, t)
        fig_design(a, t)
        fig_telemetry(RESULTS_DIR / "phase0" / "feasibility" / "run_1", t)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
