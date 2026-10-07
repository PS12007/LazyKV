"""Derive Phase 17 quantities: capacity, cost, validity and quality of layer-major tiered prefill.

Reads   results/phase17/capacity/metrics.json           (questions 1 and 2)
        results/phase17/quality_ctx*/metrics.json       (questions 3 and 4)
        results/phase15/policy_quality/metrics.json     (question 3's reference: rung 6 at 16K)
Writes  results/phase17/analysis/metrics.json

Spill check. WDDM reports pinned host memory as the process's shared GPU memory, so a tiered cell's
shared bytes include its own host store. A cell counts as inside dedicated VRAM when its shared bytes
minus its pinned bytes stay within `SPILL_SLACK` of the same model's full-cache cells, which pin
nothing and so show the process's baseline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.selection import QUEST_DENSE_LAYERS  # noqa: E402
from lazykv.stats import bootstrap_mean_ci, paired_ratio_ci  # noqa: E402

SPILL_SLACK = 64 * 2**20
GIB = 2**30


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def capacity(cap: dict[str, Any]) -> dict[str, Any]:
    cells = cap["cells"]
    models = sorted({c["model"] for c in cells})
    out: dict[str, Any] = {"rows": [], "models": {}}
    for m in models:
        mine = [c for c in cells if c["model"] == m]
        base = [c["process_gpu_memory"]["shared_bytes"] for c in mine if c["method"] == "full" and c.get("status") == "ok" and c.get("process_gpu_memory", {}).get("shared_bytes") is not None]
        baseline = min(base) if base else None
        for c in sorted(mine, key=lambda c: (c["method"], c["context"])):
            shared = (c.get("process_gpu_memory") or {}).get("shared_bytes")
            excess = None if shared is None or baseline is None else shared - c.get("host_pinned_bytes", 0) - baseline
            out["rows"].append({
                "model": m, "method": c["method"], "context": c["context"], "status": c["status"],
                "failed_stage": c.get("failed_stage"),
                "prefill_peak_gib": c.get("prefill_peak_allocated_bytes", 0) / GIB if c.get("prefill_peak_allocated_bytes") else None,
                "decode_peak_gib": c.get("decode_peak_allocated_bytes", 0) / GIB if c.get("decode_peak_allocated_bytes") else None,
                "weights_gib": c.get("weights_allocated_bytes", 0) / GIB if c.get("weights_allocated_bytes") else None,
                "full_kv_gib": c["context"] * c["kv_bytes_per_token"] / GIB if c.get("kv_bytes_per_token") else None,
                "host_pinned_gib": c["host_pinned_bytes"] / GIB if c.get("host_pinned_bytes") else None,
                "prefill_wall_s": c.get("prefill_wall_s"), "boundary_s": c.get("boundary_s"),
                "decode_ms": c.get("decode_wall_ms_median"), "score": c.get("score"), "answer": c.get("answer"),
                "shared_excess_mib": None if excess is None else excess / 2**20,
                "inside_dedicated": None if excess is None else excess <= SPILL_SLACK,
                "alloc_retries": (c.get("allocator") or {}).get("num_alloc_retries"),
            })
        mrows = [r for r in out["rows"] if r["model"] == m]
        ok = lambda meth: [r["context"] for r in mrows if r["method"] == meth and r["status"] == "ok" and r["inside_dedicated"] is not False]  # noqa: E731
        full_ok, tier_ok = ok("full"), ok("tiered")
        both = sorted(set(full_ok) & set(tier_ok))
        pf = {(r["method"], r["context"]): r for r in mrows}
        ratios = {str(x): pf[("tiered", x)]["prefill_wall_s"] / pf[("full", x)]["prefill_wall_s"] for x in both}
        peak_ratio = {str(x): pf[("tiered", x)]["prefill_peak_gib"] / pf[("full", x)]["prefill_peak_gib"] for x in both}
        first_fail = {meth: min([r["context"] for r in mrows if r["method"] == meth and r["status"] not in ("ok", "skipped")], default=None) for meth in ("full", "tiered")}
        dec_ratio = {str(x): pf[("tiered", x)]["decode_ms"] / pf[("full", x)]["decode_ms"] for x in both}
        top = pf.get(("tiered", max(tier_ok))) if tier_ok else None
        out["models"][m] = {
            "decode_time_ratio_range": [min(dec_ratio.values()), max(dec_ratio.values())] if dec_ratio else None,
            "tiered_max_full_kv_gib": None if top is None else top["full_kv_gib"],
            "tiered_max_host_pinned_gib": None if top is None else top["host_pinned_gib"],
            "tiered_max_prefill_peak_gib": None if top is None else top["prefill_peak_gib"],
            "tiered_max_prefill_s": None if top is None else top["prefill_wall_s"],
            "full_max_context": max(full_ok, default=None), "tiered_max_context": max(tier_ok, default=None),
            "full_first_failure": first_fail["full"], "tiered_first_failure": first_fail["tiered"],
            "context_gain": (max(tier_ok) / max(full_ok)) if full_ok and tier_ok else None,
            "tier_only_contexts": sorted(set(tier_ok) - set(full_ok)),
            "prefill_time_ratio": ratios, "prefill_peak_ratio": peak_ratio,
            "prefill_time_ratio_range": [min(ratios.values()), max(ratios.values())] if ratios else None,
            "prefill_peak_ratio_range": [min(peak_ratio.values()), max(peak_ratio.values())] if peak_ratio else None,
            "shared_baseline_mib": None if baseline is None else baseline / 2**20,
        }
    return out


def validity(q16: dict[str, Any], p15: dict[str, Any]) -> dict[str, Any]:
    """Question 3: tiered-prefill rung 6 against Phase 15's rung 6, prompt for prompt."""
    ref = {(r["kind"], r["depth"], r["sample"], r["budget"]): r for r in p15["niah"] if r["policy"] == "tiered_sync"}
    out: dict[str, Any] = {"budgets": {}}
    for b in sorted({r["budget"] for r in q16["niah"] if r["method"] == "tiered"}, reverse=True):
        rows = [r for r in q16["niah"] if r["method"] == "tiered" and r["budget"] == b]
        paired = [(r, ref[(r["kind"], r["depth"], r["sample"], b)]) for r in rows if (r["kind"], r["depth"], r["sample"], b) in ref]
        out["budgets"][f"{b:g}"] = {
            "prompts": len(rows), "paired": len(paired),
            "same_values": sum(a["values"] == p["values"] for a, p in paired),
            "same_answer": sum(a["answer"] == p["answer"] for a, p in paired),
            "same_score": sum(a["score"] == p["score"] for a, p in paired),
            "mismatches": [{"kind": a["kind"], "depth": a["depth"], "sample": a["sample"], "got": a["answer"], "phase15": p["answer"]} for a, p in paired if a["answer"] != p["answer"]],
            "mean_score": sum(a["score"] for a in rows) / len(rows) if rows else None,
        }
    v = out["budgets"].values()
    out["all_identical"] = all(x["paired"] == x["prompts"] and x["same_answer"] == x["paired"] for x in v) if v else None
    out["paired_total"] = sum(x["paired"] for x in v)
    full = [r["score"] for r in p15["niah"] if r["policy"] == "full"]
    out["phase15_full_mean"] = sum(full) / len(full) if full else None
    out["phase15_full_prompts"] = len(full)
    out["identical_total"] = sum(x["same_answer"] for x in v)
    return out


def _model_config_path(m: dict[str, Any]) -> Path:
    """The model's config.json: the local copy the run loaded, or else its source revision in the HF cache.

    The 3B run loaded a 4-bit copy from the external data drive, quantized from `source_revision`.
    Only the geometry is read here, which quantization does not change, so the source's config gives
    the same bytes when the drive is not mounted (Phase 18's audit could not rerun this analysis
    without it).
    """
    local = Path(m["repo"]) / "config.json"
    if local.exists() or not m.get("source_repo"):
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(m["source_repo"], "config.json", revision=m["source_revision"], local_files_only=True))


def host_store_bytes(run: dict[str, Any]) -> int:
    """Bytes the quality driver pinned for its host store, from the model's own config.

    The quality driver did not record them, and WDDM counts them as shared GPU memory, so the spill
    check has to subtract them. Same formula as `allocate_host_pools`: every selecting layer, every
    block of the capacity, keys and values in the model's dtype (2 bytes).
    """
    if "tiered" not in run["methods"]:
        return 0
    cfg = run["config"]
    model = json.loads(_model_config_path(cfg["models"][cfg["quality"]["model"]]).read_text(encoding="utf-8"))
    head_dim = model.get("head_dim") or model["hidden_size"] // model["num_attention_heads"]
    bs = cfg["block_size"]
    blocks = -(-run["capacity_tokens"] // bs)
    return (model["num_hidden_layers"] - QUEST_DENSE_LAYERS) * blocks * model["num_key_value_heads"] * 2 * bs * head_dim * 2


def quality(runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Question 4: absolute NIAH accuracy per context, method and budget, with a bootstrap CI over prompts."""
    out: dict[str, Any] = {"rows": []}
    for ctx in sorted(runs):
        rows = runs[ctx]["niah"]
        guard = runs[ctx]["memory_guard"]
        pinned = host_store_bytes(runs[ctx])
        growth = guard["shared_bytes_after"] - guard["shared_bytes_baseline"]
        for method, budget in sorted({(r["method"], r["budget"]) for r in rows}, key=lambda x: (x[0] != "full", -x[1])):
            s = [r["score"] for r in rows if r["method"] == method and r["budget"] == budget]
            lo, hi = bootstrap_mean_ci(s, iters=10000, seed=0)
            peaks = [r["peak_allocated_bytes"] for r in rows if r["method"] == method and r["budget"] == budget]
            pf = sorted(r["prefill_wall_s"] for r in rows if r["method"] == method and r["budget"] == budget)
            out["rows"].append({"context": ctx, "method": method, "budget": budget, "prompts": len(s), "mean": sum(s) / len(s), "ci95": [lo, hi],
                                "peak_gib": max(peaks) / GIB, "prefill_s_median": pf[len(pf) // 2],
                                "host_store_bytes": pinned, "shared_growth_bytes": growth,
                                # What remains is small pinned staging through the rounding allocator, or a spill.
                                "shared_beyond_host_store_mib": (growth - pinned) / 2**20,
                                "inside_dedicated": growth - pinned <= SPILL_SLACK})
    return out


def _key(method: str, ctx: int, budget: float) -> str:
    """Dot-free, so templates can address it: tiered_65536_b625 is the tier at 64K and 6.25%."""
    return f"{method}_{ctx}" if method == "full" else f"{method}_{ctx}_b{round(budget * 1e4)}"


def quality_summary(runs: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {_key(r["method"], r["context"], r["budget"]): {"mean": r["mean"], "lo": r["ci95"][0], "hi": r["ci95"][1], "peak_gib": r["peak_gib"], "prefill_s_median": r["prefill_s_median"]} for r in rows}
    # Retention where both ran: the tier's score over the full cache's on the same prompts.
    for ctx, run in runs.items():
        full = {(r["kind"], r["depth"], r["sample"]): r["score"] for r in run["niah"] if r["method"] == "full"}
        if not full:
            continue
        for b in sorted({r["budget"] for r in run["niah"] if r["method"] == "tiered"}):
            pairs = [(r["score"], full[(r["kind"], r["depth"], r["sample"])]) for r in run["niah"] if r["method"] == "tiered" and r["budget"] == b and (r["kind"], r["depth"], r["sample"]) in full]
            num, den = [p[0] for p in pairs], [p[1] for p in pairs]
            lo, hi = paired_ratio_ci(num, den, iters=10000, seed=0)
            out[f"retention_{ctx}_b{round(b * 1e4)}"] = {"value": sum(num) / sum(den), "lo": lo, "hi": hi, "prompts": len(pairs),
                                                         "tier_worse": sum(a < f for a, f in pairs), "tier_better": sum(a > f for a, f in pairs)}
    return out


def main() -> None:
    base = RESULTS_DIR / "phase17"
    payload: dict[str, Any] = {"sources": {}}
    cap = _load(base / "capacity" / "metrics.json")
    if cap is not None:
        payload["capacity"] = capacity(cap)
        payload["sources"]["capacity"] = cap["provenance"]
    runs: dict[int, dict[str, Any]] = {}
    for d in sorted(base.glob("quality_ctx*")):
        m = re.fullmatch(r"quality_ctx(\d+)", d.name)
        doc = _load(d / "metrics.json")
        if m and doc is not None:
            runs[int(m.group(1))] = doc
            payload["sources"][d.name] = doc["provenance"]
    p15 = _load(RESULTS_DIR / "phase15" / "policy_quality" / "metrics.json")
    if 16384 in runs and p15 is not None:
        payload["validity"] = validity(runs[16384], p15)
    if runs:
        payload["quality"] = quality(runs)
        payload["quality"]["summary"] = quality_summary(runs, payload["quality"]["rows"])
    path = write_metrics(base / "analysis", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
