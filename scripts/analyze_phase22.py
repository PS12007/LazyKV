"""Derive Phase 22 quantities: the "store" restore against Phase 20's "boundary" restore.

Reads   results/phase22/snapshot/metrics.json
Writes  results/phase22/analysis/metrics.json

Per context and rebuild path: medians and ranges of the restore and of its read and non-read parts,
the read rate, the transient VRAM, and the speedup over the tiered prefill measured in the same run.
Per context: the store path's speedup over the boundary path, from medians and from the worst
pairing (slowest store restore against the fastest boundary restore), and the share of a store
restore not spent reading (question 2). The exactness verdict is the state hash, per path; decoded
tokens are reported beside it (cuDNN single-query decode is not bit-repeatable, Phase 1).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402


def spread(xs: list[float]) -> dict[str, float]:
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="snapshot")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    src = json.loads((RESULTS_DIR / "phase22" / args.run / "metrics.json").read_text(encoding="utf-8"))
    rebuilds = src["config"]["rebuilds"]
    contexts: dict[str, Any] = {}
    for r in src["rows"]:
        e: dict[str, Any] = {"context": r["context"], "status": r["status"], "error": r.get("error"), "k_blocks": r["k_blocks"]}
        if r["status"] == "ok":
            e["prefill_s"] = spread(r["prefill_s"])
            e["kv_bytes"] = r["kv_bytes"]
            e["paths"] = {}
            for rb in rebuilds:
                idx = [j for j, x in enumerate(r["restore"]) if x["rebuild"] == rb]
                xs = [r["restore"][j] for j in idx]
                wall = [x["wall_s"] for x in xs]
                e["paths"][rb] = {
                    "restore_s": spread(wall),
                    "read_s": spread([x["read_s"] for x in xs]),
                    "rebuild_s": spread([x["rebuild_s"] for x in xs]),
                    "not_read_share": spread([x["rebuild_s"] / x["wall_s"] for x in xs]),
                    "read_gbps": spread([x["bytes_read"] / x["read_s"] / 1e9 for x in xs]),
                    # End to end: the snapshot's bytes over the whole restore.
                    "effective_gbps": spread([x["bytes_read"] / x["wall_s"] / 1e9 for x in xs]),
                    "transient_peak_mib": spread([x["transient_peak_bytes"] / 2**20 for x in xs]),
                    "speedup_over_prefill": statistics.median(r["prefill_s"]) / statistics.median(wall),
                    "restores": len(xs),
                    "state_match": sum(r["restore_state_match"][j] for j in idx),
                    "tokens_match": sum(r["restore_tokens_match"][j] for j in idx),
                }
            if {"store", "boundary"} <= set(rebuilds):
                s, b = e["paths"]["store"], e["paths"]["boundary"]
                e["store_over_boundary"] = b["restore_s"]["median"] / s["restore_s"]["median"]
                e["store_over_boundary_worst"] = b["restore_s"]["min"] / s["restore_s"]["max"]
        contexts[str(r["context"])] = e
    ok = [e for e in contexts.values() if e["status"] == "ok"]

    def over(key: str, fn: Any) -> Any:
        vals = [e[key] for e in ok if key in e]
        return fn(vals) if vals else None

    def path_over(rb: str, key: str, fn: Any) -> Any:
        vals = [e["paths"][rb][key]["median"] if isinstance(e["paths"][rb][key], dict) else e["paths"][rb][key] for e in ok]
        return fn(vals) if vals else None

    summary: dict[str, Any] = {
        "contexts": len(contexts),
        "contexts_ok": len(ok),
        "repeats": src["config"]["repeats"],
        "budget": src["config"]["budget"],
        "rebuilds": rebuilds,
        "restores": sum(p["restores"] for e in ok for p in e["paths"].values()),
        "state_match": sum(p["state_match"] for e in ok for p in e["paths"].values()),
        "tokens_match": sum(p["tokens_match"] for e in ok for p in e["paths"].values()),
        "store_over_boundary_min": over("store_over_boundary", min),
        "store_over_boundary_max": over("store_over_boundary", max),
        "store_over_boundary_worst_min": over("store_over_boundary_worst", min),
    }
    summary["all_states_match"] = bool(ok) and summary["state_match"] == summary["restores"]
    for rb in rebuilds:
        summary[rb] = {
            "speedup_over_prefill_min": path_over(rb, "speedup_over_prefill", min),
            "speedup_over_prefill_max": path_over(rb, "speedup_over_prefill", max),
            "not_read_share_min": path_over(rb, "not_read_share", min),
            "not_read_share_max": path_over(rb, "not_read_share", max),
            "transient_peak_mib_max": path_over(rb, "transient_peak_mib", max),
        }
    write_metrics(RESULTS_DIR / "phase22" / args.out, {"source": src["provenance"], "contexts": contexts, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
