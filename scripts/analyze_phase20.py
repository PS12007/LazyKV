"""Derive Phase 20 quantities: resuming a session from a snapshot against prefilling it again.

Reads   results/phase20/snapshot/metrics.json
Writes  results/phase20/analysis/metrics.json

Per context: medians and ranges of the prefill and of the restore (and the restore's read and
rebuild parts), the speedup, the snapshot's size over the session's KV bytes, and the two checks.
The exactness verdict is the state hash only; decoded tokens are reported beside it because cuDNN's
single-query decode is not bit-repeatable (Phase 1), so a token mismatch alone proves nothing.
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
    src = json.loads((RESULTS_DIR / "phase20" / args.run / "metrics.json").read_text(encoding="utf-8"))
    contexts: dict[str, Any] = {}
    for r in src["rows"]:
        e: dict[str, Any] = {"context": r["context"], "status": r["status"], "error": r.get("error"), "k_blocks": r["k_blocks"]}
        if r["status"] == "ok":
            rs = r["restore"]
            e.update({
                "prefill_s": spread(r["prefill_s"]),
                "restore_s": spread([x["wall_s"] for x in rs]),
                "read_s": spread([x["read_s"] for x in rs]),
                "rebuild_s": spread([x["rebuild_s"] for x in rs]),
                "read_gbps": spread([x["bytes_read"] / x["read_s"] / 1e9 for x in rs]),
                "speedup": r["speedup"],
                # The worst pairing: the slowest restore against the fastest prefill.
                "speedup_worst": min(r["prefill_s"]) / max(x["wall_s"] for x in rs),
                "save_s": r["save_s"],
                "save_gbps": r["snapshot_bytes"] / r["save_s"] / 1e9,
                "snapshot_bytes": r["snapshot_bytes"],
                "kv_bytes": r["kv_bytes"],
                "snapshot_over_kv": r["snapshot_bytes"] / r["kv_bytes"],
                "host_pinned_bytes": r["host_pinned_bytes"],
                "restores": len(rs),
                "state_match": sum(r["restore_state_match"]),
                "tokens_match": sum(r["restore_tokens_match"]),
                "niah_score": r["score"],
            })
        contexts[str(r["context"])] = e
    ok = [e for e in contexts.values() if e["status"] == "ok"]
    summary = {
        "contexts": len(contexts),
        "contexts_ok": len(ok),
        "max_context": max((e["context"] for e in ok), default=None),
        "restores": sum(e["restores"] for e in ok),
        "state_match": sum(e["state_match"] for e in ok),
        "tokens_match": sum(e["tokens_match"] for e in ok),
        "all_states_match": bool(ok) and all(e["state_match"] == e["restores"] for e in ok),
        "speedup_min": min((e["speedup"] for e in ok), default=None),
        "speedup_max": max((e["speedup"] for e in ok), default=None),
        "speedup_worst_min": min((e["speedup_worst"] for e in ok), default=None),
        "snapshot_over_kv_max": max((e["snapshot_over_kv"] for e in ok), default=None),
        "read_gbps_min": min((e["read_gbps"]["median"] for e in ok), default=None),
        "read_gbps_max": max((e["read_gbps"]["median"] for e in ok), default=None),
        "repeats": src["config"]["repeats"],
        "budget": src["config"]["budget"],
    }
    write_metrics(RESULTS_DIR / "phase20" / args.out, {"source": src["provenance"], "contexts": contexts, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
