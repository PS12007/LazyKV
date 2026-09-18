"""What removing the per-layer host round trip would buy, from the replay ablation.

Reads   results/phase5/host_ceiling/run_*/metrics.json
Writes  results/phase5/ceiling/metrics.json

Pass one decodes normally and records each layer's selection; pass two replays it, so the bound and
the host sync never run while the blocks chosen, the pairs fetched and the gathers stay identical.
The latency difference is the ceiling on any device-side residency decision.

Three guards are computed alongside it, because a ceiling is only worth as much as the evidence that
the ablation held everything else fixed:

- **Fetched pairs must match.** If replay moved a different number of pairs it changed the work, and
  the difference would not be the decision cost.
- **Token agreement should be high but need not be 1.0.** The fast kernel is not bit-repeatable for
  single-query decode (Phase 1), so two identical passes can part on a near-tie; a *collapse* in
  agreement would mean something structural, a few late flips would not.
- **The inter-layer gap** is reported against the full cache's, as the evidence that the stall sits
  inside a layer rather than between layers -- which is why this ablation was needed at all.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.sweep_analysis import span  # noqa: E402


def med(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    vals = []
    for r in rows:
        node: Any = r
        for k in path:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        if node is not None:
            vals.append(node)
    return statistics.median(vals) if vals else None


def main() -> None:
    base = RESULTS_DIR / "phase5"
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((base / "host_ceiling").glob("run_*/metrics.json"))]
    if not runs:
        raise SystemExit("no host_ceiling runs found")
    rows = [r for run in runs for r in run["results"]]
    conds = sorted({(r["policy"], r["budget"]) for r in rows}, key=lambda c: (c[0], -c[1]))
    full_gap = med([r for r in rows if r["policy"] == "full"], ("inter_layer_gap_s", "median"))

    out = []
    for policy, budget in conds:
        rs = [r for r in rows if r["policy"] == policy and r["budget"] == budget]
        normal = med(rs, ("decode_wall_s", "median"))
        replay = med(rs, ("replay_wall_s", "median"))
        gap = med(rs, ("inter_layer_gap_s", "median"))
        fetched = med(rs, ("fetched_pairs",))
        refetched = med(rs, ("replay_fetched_pairs",))
        out.append({
            "policy": policy,
            "budget": budget,
            "decode_ms": None if normal is None else 1e3 * normal,
            "replay_ms": None if replay is None else 1e3 * replay,
            "saved_ms": None if replay is None or normal is None else 1e3 * (normal - replay),
            # The ceiling, as a speedup a perfect device-side decision could reach and not exceed.
            "ceiling_speedup": None if replay is None or not replay else normal / replay,
            "inter_layer_gap_ms": None if gap is None else 1e3 * gap,
            # Excess over the full cache's gap under the same hooks: the part the tier owns.
            "inter_layer_gap_excess_ms": None if gap is None or full_gap is None else 1e3 * (gap - full_gap),
            "token_agreement": med(rs, ("replay_token_agreement",)),
            "fetched_pairs": fetched,
            "replay_fetched_pairs": refetched,
            "fetched_pairs_match": None if fetched is None or refetched is None else fetched == refetched,
            "repeats": len(rs),
        })

    tiered = [r for r in out if r["ceiling_speedup"] is not None]
    summary = {
        "runs": len(runs),
        "context": runs[0]["config"]["context"],
        "block_size": runs[0]["config"]["block_size"],
        "ceiling_speedup": span([r["ceiling_speedup"] for r in tiered]),
        "saved_ms": span([r["saved_ms"] for r in tiered]),
        "decode_ms": span([r["decode_ms"] for r in tiered]),
        "token_agreement": span([r["token_agreement"] for r in tiered if r["token_agreement"] is not None]),
        "all_fetched_pairs_match": all(r["fetched_pairs_match"] for r in tiered),
        "full_inter_layer_gap_ms": None if full_gap is None else 1e3 * full_gap,
        "inter_layer_gap_excess_ms": span([r["inter_layer_gap_excess_ms"] for r in tiered if r["inter_layer_gap_excess_ms"] is not None]),
        "ceiling_speedup_at": {f"{r['policy']}@{r['budget']:g}": r["ceiling_speedup"] for r in tiered},
    }
    write_metrics(base / "ceiling", {
        "sources": [r["provenance"] for r in runs],
        "conditions": out,
        "summary": summary,
    })
    print("wrote", base / "ceiling" / "metrics.json")


if __name__ == "__main__":
    main()
