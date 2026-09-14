"""Derive Phase 1 decision quantities from baseline runs, affinity, and quality metrics.

Reads   results/phase1/baseline/run_*/metrics.json
        results/phase1/affinity/metrics.json                (optional)
        results/phase1/quality_reference/metrics.json        (optional)
        results/phase1/kernel_repeatability/metrics.json     (optional)
        results/phase1/latency_regimes/metrics.json          (optional)
        results/phase1/baseline_os_throttling/run_*/         (optional: first baseline, before the throttling fix)
        results/phase1/affinity_os_throttling/               (optional: first affinity run, same)
        results/phase1/quality_reference_first_run/          (optional: first quality run, cuDNN reference)
        results/phase0/analysis/metrics.json                 (PCIe numbers for the window arithmetic)
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


def load_runs(name: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS_DIR / "phase1" / name).glob("run_*/metrics.json"))]


def per_run_median(runs: list[dict[str, Any]], ctx: int, metric: str, reps: str = "all") -> list[float]:
    """One value per run: the median over that run's repeats at this context."""
    vals = []
    for run in runs:
        rows = [r for r in run["results"] if r["ctx_len"] == ctx]
        if reps == "cold":
            rows = [r for r in rows if r["repeat"] == 0]
        elif reps == "warm":
            rows = [r for r in rows if r["repeat"] > 0]
        if rows:
            vals.append(statistics.median(_get(r, metric) for r in rows))
    return vals


def main() -> None:
    runs = load_runs("baseline")
    if not runs:
        raise SystemExit("no baseline runs")
    p0 = load_json(RESULTS_DIR / "phase0" / "analysis" / "metrics.json")
    contexts = sorted({r["ctx_len"] for r in runs[0]["results"]})
    kv_per_token = runs[0]["model"]["kv_bytes_per_token"]
    n_layers = runs[0]["model"]["num_layers"]
    kv_per_token_layer = kv_per_token // n_layers

    per_ctx = []
    for ctx in contexts:
        def per_run(metric: str, reps: str = "all", ctx: int = ctx) -> list[float]:
            return per_run_median(runs, ctx, metric, reps)

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
    for key, name in (("affinity_summary", "affinity"), ("affinity_os_throttling_summary", "affinity_os_throttling")):
        aff = load_json(RESULTS_DIR / "phase1" / name / "metrics.json")
        if aff:
            c = aff["conditions"]
            payload[key] = {
                "conditions": {k: {"median_s": v["decode_wall_s"]["median"], "p10_s": v["p10_s"], "p90_s": v["p90_s"]} for k, v in c.items()},
                "e_over_default_median": c["e_cores_only"]["decode_wall_s"]["median"] / c["default"]["decode_wall_s"]["median"],
                "p_over_default_median": c["p_cores_only"]["decode_wall_s"]["median"] / c["default"]["decode_wall_s"]["median"],
            }

    # The first baseline ran under Windows' default power throttling, same code otherwise, so
    # the per-context difference is what leaving host scheduling to the OS cost.
    throttled = load_runs("baseline_os_throttling")
    if throttled:
        rows_t = []
        for ctx in contexts:
            before = across(per_run_median(throttled, ctx, "decode_wall_s.median"))
            after = across(per_run_median(runs, ctx, "decode_wall_s.median"))
            before_p90 = across(per_run_median(throttled, ctx, "decode_wall_p90_s"))
            after_p90 = across(per_run_median(runs, ctx, "decode_wall_p90_s"))
            rows_t.append({
                "ctx_len": ctx,
                "decode_median_os_throttling_s": before,
                "decode_median_opted_out_s": after,
                "decode_p90_os_throttling_s": before_p90,
                "decode_p90_opted_out_s": after_p90,
                "median_ratio_before_over_after": before["median"] / after["median"] if before.get("n_runs") and after.get("n_runs") else None,
                "p90_ratio_before_over_after": before_p90["median"] / after_p90["median"] if before_p90.get("n_runs") and after_p90.get("n_runs") else None,
            })
        payload["throttling_before_after"] = {"run_provenance_before": [r["provenance"] for r in throttled], "rows": rows_t}

    kr = load_json(RESULTS_DIR / "phase1" / "kernel_repeatability" / "metrics.json")
    if kr:
        strategies = list(dict.fromkeys(r["strategy"] for r in kr["rows"]))
        payload["kernel_repeatability"] = {
            "config": kr["config"],
            "torch": kr["torch"],
            "cudnn": kr["cudnn"],
            "rows": [{k: v for k, v in r.items() if k != "unrepeatable_calls"} for r in kr["rows"]],
            "by_strategy": {
                st: {
                    "calls": sum(r["decode_calls_checked"] for r in kr["rows"] if r["strategy"] == st),
                    "not_repeatable": sum(r["calls_not_repeatable"] for r in kr["rows"] if r["strategy"] == st),
                }
                for st in strategies
            },
        }
        # Accuracy of the variants: every distinct output's error against float64, pooled.
        errs = [e for r in kr["rows"] for u in r["unrepeatable_calls"] for e in u["rel_err_vs_float64"]]
        gaps = [u["max_abs_between_variants_rel"] for r in kr["rows"] for u in r["unrepeatable_calls"]]
        payload["kernel_repeatability"]["variants"] = {
            "n_variant_outputs": len(errs),
            "rel_err_vs_float64_max": max(errs) if errs else None,
            "rel_err_vs_float64_min": min(errs) if errs else None,
            "gap_between_variants_rel_max": max(gaps) if gaps else None,
            "gap_between_variants_rel_median": statistics.median(gaps) if gaps else None,
        }

    lr = load_json(RESULTS_DIR / "phase1" / "latency_regimes" / "metrics.json")
    if lr:
        keep = ("tokens", "decode_wall_s", "p90_s", "slow_fraction", "per_block_slow_fraction", "p_core_fraction", "p_core_fraction_slow_tokens", "p_core_fraction_fast_tokens", "slow_episode_tokens")

        def trim(d: dict[str, Any]) -> dict[str, Any]:
            return {k: {f: v[f] for f in keep} for k, v in d.items()}

        ft, fs = lr["factor_throttling"], lr["factor_nvidia_smi"]
        payload["latency_regimes"] = {
            "config": lr["config"],
            "provenance": lr["provenance"],
            "slow_threshold_s": lr["slow_threshold_s"],
            "fast_reference_p10_s": lr["fast_reference_p10_s"],
            "slow_threshold_rule": lr["slow_threshold_rule"],
            "conditions": trim(lr["conditions"]),
            "factor_throttling": trim(ft),
            "factor_nvidia_smi": trim(fs),
            "cpu_performance_pct": {k: v for k, v in lr["cpu_performance_pct"].items() if k != "trace"},
            "throttling_median_ratio": ft["os_default"]["decode_wall_s"]["median"] / ft["opted_out"]["decode_wall_s"]["median"],
            "throttling_p90_ratio": ft["os_default"]["p90_s"] / ft["opted_out"]["p90_s"],
            "nvidia_smi_median_ratio": fs["on"]["decode_wall_s"]["median"] / fs["off"]["decode_wall_s"]["median"],
        }

    fq = load_json(RESULTS_DIR / "phase1" / "quality_reference_first_run" / "metrics.json")
    payload["quality_first_run"] = fq
    q = payload["quality_reference"]
    floors = ("self_repeat", "fast_kernel_self_repeat")
    if q:
        def span(name: str, field: str) -> dict[str, float] | None:
            vals = [r["comparisons"][name][field] for r in q["rows"] if name in r["comparisons"]]
            return {"min": min(vals), "max": max(vals)} if vals else None

        cross = [c for r in q["rows"] for n, c in r["comparisons"].items() if n not in floors]
        payload["quality_summary"] = {
            "self_repeat_all_exact": all(r["comparisons"]["self_repeat"]["exact_match"] for r in q["rows"]),
            "contexts": [r["ctx_len"] for r in q["rows"]],
            "fast_kernel_self_repeat_top1": span("fast_kernel_self_repeat", "top1_agreement"),
            "fast_kernel_self_repeat_kl": span("fast_kernel_self_repeat", "mean_kl"),
            "cross_path_top1": {"min": min(c["top1_agreement"] for c in cross), "max": max(c["top1_agreement"] for c in cross)},
            "cross_path_kl": {"min": min(c["mean_kl"] for c in cross), "max": max(c["mean_kl"] for c in cross)},
        }
    if fq:
        sr = [r["comparisons"]["self_repeat"] for r in fq["rows"]]
        payload["quality_first_run_summary"] = {
            "self_repeat_top1": {"min": min(c["top1_agreement"] for c in sr), "max": max(c["top1_agreement"] for c in sr)},
            "self_repeat_kl": {"min": min(c["mean_kl"] for c in sr), "max": max(c["mean_kl"] for c in sr)},
            "self_repeat_any_exact": any(c["exact_match"] for c in sr),
        }

    meds = [(r["ctx_len"], r["decode_wall_median_s"]["median"]) for r in per_ctx]
    payload["decode_growth"] = {"first_ctx": meds[0][0], "last_ctx": meds[-1][0], "last_over_first": meds[-1][1] / meds[0][1]}
    payload["window_summary"] = _window_summary(window)
    path = write_metrics(RESULTS_DIR / "phase1" / "analysis", payload)
    print("wrote", path)


def _window_summary(window: dict[str, Any]) -> dict[str, Any] | None:
    if not window.get("available"):
        return None
    rows = window["rows"]
    last = rows[-1]
    with_bound = [r for r in rows if "window_over_phase0_bound" in r]
    return {
        "layer_window_min_s": min(r["layer_window_s"] for r in rows),
        "layer_window_max_s": max(r["layer_window_s"] for r in rows),
        "tokens_fetchable_min": min(r["tokens_fetchable_within_layer"] for r in rows),
        "tokens_fetchable_max": max(r["tokens_fetchable_within_layer"] for r in rows),
        "longest_ctx": last["ctx_len"],
        "longest_ctx_fetchable_fraction": last["fetchable_fraction_within_layer"],
        "longest_ctx_phase0_fraction": last.get("phase0_fetchable_fraction"),
        "window_over_phase0_bound_min": min((r["window_over_phase0_bound"] for r in with_bound), default=None),
        "window_over_phase0_bound_max": max((r["window_over_phase0_bound"] for r in with_bound), default=None),
    }


def _get(row: dict[str, Any], dotted: str) -> Any:
    node: Any = row
    for part in dotted.split("."):
        node = node[part]
    return node


if __name__ == "__main__":
    main()
