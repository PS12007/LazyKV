"""Derive the Phase 0 decision quantities from raw feasibility + KV-model metrics.

Reads   results/phase0/feasibility/run_*/metrics.json
        results/phase0/kv_memory_model/metrics.json   (optional)
Writes  results/phase0/analysis/metrics.json

Every derived number used by docs (ratios, knees, per-token costs for designs (a) and
(c)) is computed here, so the arithmetic in IMPLEMENTATION_PLAN.md is reproducible code,
not prose. Across independent runs we report the median of per-run medians plus the
min/max, because between-run spread on this laptop exceeded within-run IQR.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import fit_overhead, knee_size, overlap_fraction  # noqa: E402

KiB = 1024
MiB = 1024**2


def across(values: Iterable[float]) -> dict[str, float | int]:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if not xs:
        return {"n_runs": 0}
    return {"n_runs": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def load_runs() -> list[dict[str, Any]]:
    runs = []
    for path in sorted((RESULTS_DIR / "phase0" / "feasibility").glob("run_*/metrics.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit("no feasibility runs found; run scripts/00_feasibility.py first")
    return runs


def pcie_curve(run: dict[str, Any], mode: str, direction: str) -> tuple[list[int], list[float], list[float]]:
    rows = sorted(
        (r for r in run["pcie"]["rows"] if r["mode"] == mode and r["direction"] == direction),
        key=lambda r: r["size_bytes"],
    )
    return (
        [r["size_bytes"] for r in rows],
        [r["wall_bps"]["median"] for r in rows],
        [r["seconds_per_copy"]["median"] for r in rows],
    )


def analyze_pcie(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ("pinned", "pageable"):
        for direction in ("h2d", "d2h"):
            per_run = []
            for run in runs:
                sizes, bps, spc = pcie_curve(run, mode, direction)
                # Asymptote = mean of the two largest sizes: robust to one noisy top point,
                # and the curve is flat there (checked via `still_rising`).
                asym = (bps[-1] + bps[-2]) / 2
                fit = fit_overhead(sizes, spc)
                per_run.append(
                    {
                        "asymptote_bps": asym,
                        "still_rising": bps[-1] > 1.05 * bps[-2],
                        "knee50_bytes": knee_size(sizes, bps, 0.5, asym),
                        "knee80_bytes": knee_size(sizes, bps, 0.8, asym),
                        "fit_t0_s": fit.t0_s,
                        "fit_bandwidth_bps": fit.bandwidth_bps,
                        "fit_knee80_bytes": fit.knee_bytes(0.8),
                        "smallest_size_bytes": sizes[0],
                        "smallest_size_bps": bps[0],
                    }
                )
            key = f"{mode}_{direction}"
            out[key] = {
                "per_run": per_run,
                **{
                    f: across(r[f] for r in per_run)
                    for f in (
                        "asymptote_bps",
                        "knee50_bytes",
                        "knee80_bytes",
                        "fit_t0_s",
                        "fit_bandwidth_bps",
                        "fit_knee80_bytes",
                        "smallest_size_bps",
                    )
                },
                "smallest_size_bytes": per_run[0]["smallest_size_bytes"],
            }
    # Per-size curve across runs, for tables and plots.
    curve = []
    sizes = runs[0]["pcie"]["sizes_bytes"]
    for s in sizes:
        row: dict[str, Any] = {"size_bytes": s}
        for mode in ("pinned", "pageable"):
            for direction in ("h2d", "d2h"):
                vals = [
                    r["wall_bps"]["median"]
                    for run in runs
                    for r in run["pcie"]["rows"]
                    if r["size_bytes"] == s and r["mode"] == mode and r["direction"] == direction
                ]
                row[f"{mode}_{direction}"] = across(vals)
        curve.append(row)
    out["curve"] = curve
    out["pinned_over_pageable_h2d_asymptote"] = across(
        p["asymptote_bps"] / q["asymptote_bps"]
        for p, q in zip(out["pinned_h2d"]["per_run"], out["pageable_h2d"]["per_run"])
    )
    return out


def analyze_vram(runs: list[dict[str, Any]], pinned_h2d: list[float]) -> dict[str, Any]:
    d2d = []
    for run in runs:
        rows = run["vram"]["d2d_copy"]
        d2d.append(max(rows, key=lambda r: r["size_bytes"])["wall_bps"]["median"])
    largest = max(runs[0]["vram"]["d2d_copy"], key=lambda r: r["size_bytes"])["size_bytes"]

    attn: dict[int, dict[str, Any]] = {}
    for run in runs:
        tol = run["vram"]["decode_attention_scan"]["correctness_tolerance_rel"]
        for r in run["vram"]["decode_attention_scan"]["rows"]:
            if not r.get("available"):
                continue
            cell = attn.setdefault(r["ctx_len"], {"kv_bytes_per_layer": r["kv_bytes_per_layer"], "backends": {}})
            b = cell["backends"].setdefault(r["backend"], {"seconds": [], "errs": []})
            b["seconds"].append(r["seconds_per_call"]["median"])
            b["errs"].append(r["max_rel_err_vs_math"])
            b["tol"] = tol
    attn_out = []
    for ctx, cell in sorted(attn.items()):
        backends = {}
        for name, b in cell["backends"].items():
            backends[name] = {
                "seconds_per_call": across(b["seconds"]),
                "kv_scan_bps": across(cell["kv_bytes_per_layer"] / s for s in b["seconds"]),
                "max_rel_err_vs_math": max(b["errs"]),
                "correct": max(b["errs"]) <= b["tol"],
            }
        correct = {k: v for k, v in backends.items() if v["correct"]}
        best = min(correct, key=lambda k: correct[k]["seconds_per_call"]["median"])
        attn_out.append(
            {
                "ctx_len": ctx,
                "kv_bytes_per_layer": cell["kv_bytes_per_layer"],
                "backends": backends,
                "best_backend": best,
                "best_seconds_per_call": correct[best]["seconds_per_call"]["median"],
                "default_over_best": backends["default_dispatch_gqa"]["seconds_per_call"]["median"]
                / correct[best]["seconds_per_call"]["median"],
            }
        )
    return {
        "d2d_copy_size_bytes": largest,
        "d2d_copy_bps": across(d2d),
        "ratio_d2d_over_pinned_h2d": across(v / p for v, p in zip(d2d, pinned_h2d)),
        "decode_attention": attn_out,
    }


def analyze_overlap(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def med(run: dict[str, Any], k: str) -> float:
        return run["overlap"]["wall_s"][k]["median"]

    return {
        "async_engine_count": [run["overlap"]["driver_attributes"].get("ASYNC_ENGINE_COUNT") for run in runs],
        "driver_attributes_run1": runs[0]["overlap"]["driver_attributes"],
        "compute_h2d_overlap": across(
            overlap_fraction(med(r, "compute_only"), med(r, "h2d_only"), med(r, "compute_plus_h2d")) for r in runs
        ),
        "compute_d2h_overlap": across(
            overlap_fraction(med(r, "compute_only"), med(r, "d2h_only"), med(r, "compute_plus_d2h")) for r in runs
        ),
        "h2d_d2h_overlap": across(
            overlap_fraction(med(r, "h2d_only"), med(r, "d2h_only"), med(r, "h2d_plus_d2h")) for r in runs
        ),
        "wall_s_run1": {k: v["median"] for k, v in runs[0]["overlap"]["wall_s"].items()},
    }


def analyze_cpu(runs: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sorted({r["tokens"] for r in runs[0]["cpu_attention"]["rows"]})
    threads = sorted({r["threads"] for r in runs[0]["cpu_attention"]["rows"]})

    def sec(run: dict[str, Any], kind: str, n: int, t: int) -> float:
        return next(
            r["seconds_per_call"]["median"]
            for r in run["cpu_attention"]["rows"]
            if r["kind"] == kind and r["tokens"] == n and r["threads"] == t
        )

    grid = []
    best = []
    for n in tokens:
        row: dict[str, Any] = {"tokens": n}
        for kind in ("gemv", "partial_attn"):
            for t in threads:
                row[f"{kind}_t{t}"] = across(sec(run, kind, n, t) for run in runs)
        per_run_best = [min((sec(run, "partial_attn", n, t), t) for t in threads) for run in runs]
        gpu = [
            next(g["seconds_per_call"]["median"] for g in run["cpu_attention"]["gpu_partial_attn_fp32"] if g["tokens"] == n)
            for run in runs
        ]
        best_row = {
            "tokens": n,
            "best_seconds": across(s for s, _ in per_run_best),
            "best_threads": [t for _, t in per_run_best],
            "gpu_fp32_seconds": across(gpu),
            "cpu_over_gpu": across(s / g for (s, _), g in zip(per_run_best, gpu)),
        }
        grid.append(row)
        best.append(best_row)
    return {
        "logical_cpus": runs[0]["cpu_attention"]["logical_cpus"],
        "threads": threads,
        "host_memcpy_bps_1thread": across(r["cpu_attention"]["host_memcpy_bps_1thread"]["median"] for r in runs),
        "grid": grid,
        "partial_attn_best": best,
    }


def analyze_nvme(runs: list[dict[str, Any]], pinned_h2d: list[float]) -> dict[str, Any]:
    ok = [r for r in runs if r.get("nvme", {}).get("measured")]
    if not ok:
        return {"measured": False}
    seq = [r["nvme"]["sequential_1mib_bps"]["median"] for r in ok]
    rnd = {}
    for io in (4 * KiB, 128 * KiB):
        vals = [next(x for x in r["nvme"]["random"] if x["io_bytes"] == io) for r in ok]
        rnd[str(io)] = {
            "iops": across(v["iops"]["median"] for v in vals),
            "bps": across(v["bps"]["median"] for v in vals),
        }
    return {
        "measured": True,
        "sequential_bps": across(seq),
        "random": rnd,
        "pinned_h2d_over_nvme_seq": across(p / s for p, s in zip(pinned_h2d, seq)),
        "pinned_h2d_over_nvme_rand128k": across(
            p / next(x for x in r["nvme"]["random"] if x["io_bytes"] == 128 * KiB)["bps"]["median"]
            for p, r in zip(pinned_h2d, ok)
        ),
    }


def _effective_fraction(size: float, t0: float, bandwidth: float) -> float:
    """Fraction of asymptotic bandwidth a single transfer of `size` achieves under the fit."""
    return (size / (t0 + size / bandwidth)) / bandwidth


def design_arithmetic(
    pcie: dict[str, Any], vram: dict[str, Any], cpu: dict[str, Any], kv: dict[str, Any] | None
) -> dict[str, Any]:
    """Per-layer and per-token costs of designs (a) and (c), from measured components.

    Geometry is Llama-3.2-1B (the brief's primary model). Taken from the KV model when
    available so the numbers track the real config.json, else marked unavailable.
    """
    if kv is None:
        return {"available": False, "reason": "kv_memory_model not run"}
    model = next((m for m in kv["models"] if m["key"] == "llama-3.2-1b"), None)
    if model is None:
        return {"available": False, "reason": "llama-3.2-1b missing from kv model"}
    n_layers = model["num_hidden_layers"]
    bytes_per_token_layer_bf16 = model["kv_bytes_per_token_bf16"] // n_layers

    B = pcie["pinned_h2d"]["asymptote_bps"]["median"]
    t0 = pcie["pinned_h2d"]["fit_t0_s"]["median"]
    # A single batched transfer assumes the selected blocks are already contiguous in
    # pinned memory. If they are scattered, someone gathers them first, and a 1-thread
    # host memcpy is slower than the link itself on this machine. Cost it explicitly.
    memcpy = cpu["host_memcpy_bps_1thread"]["median"]
    rows = []
    attn_by_ctx = {a["ctx_len"]: a for a in vram["decode_attention"]}
    cpu_by_tokens = {c["tokens"]: c for c in cpu["partial_attn_best"]}
    for n in sorted(set(attn_by_ctx) | set(cpu_by_tokens)):
        nbytes = n * bytes_per_token_layer_bf16
        # (a) best case: all non-resident KV of a layer gathered into ONE pinned transfer.
        fetch_one = t0 + nbytes / B
        # (a) block-wise: one transfer per 64-token block, the naive layout.
        n_blocks = math.ceil(n / 64)
        fetch_blocks = n_blocks * t0 + nbytes / B
        row: dict[str, Any] = {
            "tokens_per_layer": n,
            "kv_bytes_per_layer_bf16": nbytes,
            "fetch_one_transfer_s": fetch_one,
            "fetch_64tok_blocks_s": fetch_blocks,
            "fetch_with_1thread_gather_s": nbytes / memcpy + fetch_one,
            "fetch_all_layers_one_transfer_s": n_layers * fetch_one,
        }
        if n in attn_by_ctx:
            g = attn_by_ctx[n]["best_seconds_per_call"]
            row["gpu_attention_s"] = g
            row["gpu_attention_all_layers_s"] = n_layers * g
            row["fetch_over_gpu_attention"] = fetch_one / g
        if n in cpu_by_tokens:
            c = cpu_by_tokens[n]["best_seconds"]["median"]
            row["cpu_partial_attn_s"] = c
            row["cpu_partial_attn_all_layers_s"] = n_layers * c
            row["cpu_over_fetch_one"] = c / fetch_one
            row["cpu_over_fetch_blocks"] = c / fetch_blocks
            row["cpu_over_fetch_with_gather"] = c / row["fetch_with_1thread_gather_s"]
        rows.append(row)
    knees = pcie["pinned_h2d"]
    return {
        "available": True,
        "model": model["key"],
        "n_layers": n_layers,
        "kv_bytes_per_token_per_layer_bf16": bytes_per_token_layer_bf16,
        "pinned_h2d_bps_used": B,
        "per_transfer_overhead_s_used": t0,
        "host_memcpy_bps_used": memcpy,
        "block_64tok_bytes_per_layer": 64 * bytes_per_token_layer_bf16,
        "knee50_tokens_per_layer": knees["knee50_bytes"]["median"] / bytes_per_token_layer_bf16,
        "knee80_tokens_per_layer": knees["knee80_bytes"]["median"] / bytes_per_token_layer_bf16,
        "block_64tok_fraction_of_asymptote": _effective_fraction(64 * bytes_per_token_layer_bf16, t0, B),
        "pcie_over_host_memcpy": B / memcpy,
        "rows": rows,
        "caveats": [
            "CPU partial attention measured in float32 with random KV; bf16 storage would add a cast.",
            "GPU attention is the attention op alone; the real overlap window per layer also includes MLP and "
            "projections, which Phase 0 did not measure (no model loaded).",
        ],
    }


def main() -> None:
    runs = load_runs()
    kv_path = RESULTS_DIR / "phase0" / "kv_memory_model" / "metrics.json"
    kv = json.loads(kv_path.read_text(encoding="utf-8")) if kv_path.exists() else None

    pcie = analyze_pcie(runs)
    pinned_h2d = [r["asymptote_bps"] for r in pcie["pinned_h2d"]["per_run"]]
    vram = analyze_vram(runs, pinned_h2d)
    cpu = analyze_cpu(runs)
    payload = {
        "run_ids": [r["config"]["run_id"] for r in runs],
        "run_provenance": [r["provenance"] for r in runs],
        "telemetry": [
            {k: r["telemetry"][k] for k in ("temperature_c", "power_w", "graphics_clock_mhz", "pcie_link_gen_seen", "pcie_link_width_seen")}
            for r in runs
        ],
        "pcie": pcie,
        "vram": vram,
        "overlap": analyze_overlap(runs),
        "cpu": cpu,
        "nvme": analyze_nvme(runs, pinned_h2d),
        "design": design_arithmetic(pcie, vram, cpu, kv),
    }
    path = write_metrics(RESULTS_DIR / "phase0" / "analysis", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
