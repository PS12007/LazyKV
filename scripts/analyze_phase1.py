"""Derive Phase 1 decision quantities from baseline runs, affinity, and quality metrics.

Reads   results/phase1/baseline/run_*/metrics.json
        results/phase1/affinity/metrics.json          (optional)
        results/phase1/quality_reference/metrics.json  (optional)
        results/phase0/analysis/metrics.json           (PCIe numbers for the window arithmetic)
Writes  results/phase1/analysis/metrics.json

Aggregation: within a run, each (context, repeat) condition gives a median; the run's value
for a context is the median over its repeats; the reported value is the median over runs,
with min/max across runs. TTFT is split into repeat 0 (cuDNN plans cold for that shape) and
later repeats (plans warm), because the difference is a real cost and not noise.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402

COMPONENTS = ("layer_s", "attention_kernel_s", "attention_other_s", "mlp_s", "other_s")


def across(values: list[float]) -> dict[str, float | int]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n_runs": 0}
    return {"n_runs": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS_DIR / "phase1" / "baseline").glob("run_*/metrics.json"))]
    if not runs:
        raise SystemExit("no baseline runs")
    p0 = load_json(RESULTS_DIR / "phase0" / "analysis" / "metrics.json")
    contexts = sorted({r["ctx_len"] for r in runs[0]["results"]})
    kv_per_token = runs[0]["model"]["kv_bytes_per_token"]
    n_layers = runs[0]["model"]["num_layers"]
    kv_per_token_layer = kv_per_token // n_layers

    per_ctx = []
    for ctx in contexts:
        def per_run(metric: str, reps: str = "all") -> list[float]:
            vals = []
            for run in runs:
                rows = [r for r in run["results"] if r["ctx_len"] == ctx]
                if reps == "cold":
                    rows = [r for r in rows if r["repeat"] == 0]
                elif reps == "warm":
                    rows = [r for r in rows if r["repeat"] > 0]
                if not rows:
                    continue
                vals.append(statistics.median(_get(r, metric) for r in rows))
            return vals

        decode_median = per_run("decode_wall_s.median")
        row: dict[str, Any] = {
            "ctx_len": ctx,
            "ttft_cold_s": across(per_run("ttft_wall_s", "cold")),
            "ttft_warm_s": across(per_run("ttft_wall_s", "warm")),
            "prefill_tokens_per_s_warm": across(per_run("prefill_tokens_per_s", "warm")),
            "decode_wall_median_s": across(decode_median),
            "decode_wall_p10_s": across(per_run("decode_wall_p10_s")),
            "decode_wall_p90_s": across(per_run("decode_wall_p90_s")),
            "decode_wall_iqr_s": across(per_run("decode_wall_s.iqr")),
            "decode_tokens_per_s": across([1.0 / d for d in decode_median]),
            "decode_event_median_s": across(per_run("decode_event_s.median")),
            "host_overhead_median_s": across(per_run("host_overhead_median_s")),
            "profiler_overhead_ratio": across(per_run("profiler_overhead_ratio")),
            "per_layer_mean_s": {c: across(per_run(f"per_layer_mean_s.{c}")) for c in COMPONENTS},
            "all_layers_sum_s": {c: across(per_run(f"all_layers_sum_s.{c}")) for c in COMPONENTS},
            "peak_allocated_bytes": across(per_run("memory.max_memory_allocated_bytes")),
            "peak_reserved_bytes": across(per_run("memory.max_memory_reserved_bytes")),
            "gpu_resident_kv_bytes": across(per_run("memory.gpu_resident_kv_bytes")),
            "nvidia_smi_process_used_mib": sorted({str(_get(r, "memory.nvidia_smi_process_used_mib")) for run in runs for r in run["results"] if r["ctx_len"] == ctx}),
        }
        layer = row["per_layer_mean_s"]["layer_s"]["median"]
        kernel = row["per_layer_mean_s"]["attention_kernel_s"]["median"]
        row["attention_kernel_share_of_layer"] = kernel / layer if layer else None
        # Share of a profiled step's time spent inside decoder layers (the rest: embedding,
        # final norm, lm_head, argmax, Python outside layers).
        prof_step = across(per_run("profiled_decode_wall_median_s"))["median"]
        row["layers_share_of_profiled_step"] = row["all_layers_sum_s"]["layer_s"]["median"] / prof_step if prof_step else None
        per_ctx.append(row)

    window: dict[str, Any] = {"available": False}
    if p0 is not None:
        B = p0["pcie"]["pinned_h2d"]["asymptote_bps"]["median"]
        t0 = p0["pcie"]["pinned_h2d"]["fit_t0_s"]["median"]
        attn_rows = {a["ctx_len"]: a for a in p0["vram"]["decode_attention"]}
        wrows = []
        for row in per_ctx:
            ctx = row["ctx_len"]
            layer = row["per_layer_mean_s"]["layer_s"]["median"]
            kernel = row["per_layer_mean_s"]["attention_kernel_s"]["median"]
            fetchable = max(0.0, layer - t0) * B / kv_per_token_layer
            wr = {
                "ctx_len": ctx,
                "layer_window_s": layer,
                "attention_kernel_in_model_s": kernel,
                "tokens_fetchable_within_layer": fetchable,
                "fetchable_fraction_within_layer": fetchable / ctx,
                "fetch_all_one_layer_s": t0 + ctx * kv_per_token_layer / B,
            }
            wr["fetch_all_over_layer_window"] = wr["fetch_all_one_layer_s"] / layer
            if ctx in attn_rows:
                wr["phase0_bound_attention_s"] = attn_rows[ctx]["best_seconds_per_call"]
                wr["phase0_fetchable_fraction"] = next(
                    (d["fetchable_fraction_within_gpu_attention"] for d in p0["design"]["rows"] if d["tokens_per_layer"] == ctx and "fetchable_fraction_within_gpu_attention" in d),
                    None,
                )
                wr["window_over_phase0_bound"] = layer / attn_rows[ctx]["best_seconds_per_call"]
            wrows.append(wr)
        window = {
            "available": True,
            "pinned_h2d_bps": B,
            "per_transfer_overhead_s": t0,
            "kv_bytes_per_token_per_layer": kv_per_token_layer,
            "rows": wrows,
            "caveat": (
                "The layer window is its span on the GPU timeline, which at batch size 1 is mostly host "
                "kernel-launch time. A copy on a separate stream overlaps it (Phase 0 overlap measurement), "
                "but shrinking host overhead (e.g. CUDA graphs) would shrink the window too."
            ),
        }

    first = runs[0]
    payload: dict[str, Any] = {
        "run_ids": [r["config"]["run_id"] for r in runs],
        "run_provenance": [r["provenance"] for r in runs],
        "model": first["model"],
        "attention": {"strategy": first["attention"]["strategy"], "bucket": first["attention"]["bucket"], "checks": first["attention"]["checks"]},
        "config": {k: first["config"][k] for k in ("contexts", "repeats", "decode_tokens", "warmup_steps", "profile_tokens", "chunk_size", "book")},
        "telemetry": [{k: r["telemetry"][k] for k in ("temperature_c", "power_w", "graphics_clock_mhz", "pcie_link_gen_seen")} for r in runs],
        "contexts": per_ctx,
        "window": window,
        "affinity": load_json(RESULTS_DIR / "phase1" / "affinity" / "metrics.json"),
        "quality_reference": load_json(RESULTS_DIR / "phase1" / "quality_reference" / "metrics.json"),
    }
    aff = payload["affinity"]
    if aff:
        c = aff["conditions"]
        payload["affinity_summary"] = {
            "e_over_default_median": c["e_cores_only"]["decode_wall_s"]["median"] / c["default"]["decode_wall_s"]["median"],
            "p_over_default_median": c["p_cores_only"]["decode_wall_s"]["median"] / c["default"]["decode_wall_s"]["median"],
        }
    path = write_metrics(RESULTS_DIR / "phase1" / "analysis", payload)
    print("wrote", path)


def _get(row: dict[str, Any], dotted: str) -> Any:
    node: Any = row
    for part in dotted.split("."):
        node = node[part]
    return node


if __name__ == "__main__":
    main()
