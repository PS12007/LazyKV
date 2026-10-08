"""Derive Phase 21 quantities: dense turn ingestion for the tier (configs/phase21.yaml, questions 1-4).

Reads   results/phase21/multiturn/metrics.json       (rung 5 and the tier, dense turns, this phase)
        results/phase21/ingest_cost/metrics.json     (question 4, if present)
        results/phase19/multiturn_dense_turns/...    (question 2: Phase 19's post hoc rung 5 run)
        results/phase19/multiturn/...                (question 3: Phase 19's tier fed token by token)
Writes  results/phase21/analysis/metrics.json

Retentions are paired against this run's full cache (fed the same way) and resample whole sessions,
as in Phase 19.
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
from harness.stats import bootstrap_mean_ci, paired_ratio_ci, sign_test_p  # noqa: E402

KINDS = {
    "chat": lambda r: r["kind"] == "chat",
    "doc_turn0": lambda r: r["kind"] == "doc" and r["turn"] == 0,
    "doc_later": lambda r: r["kind"] == "doc" and r["turn"] > 0 and r["asks"] != "d0",
    "doc_reask": lambda r: r["kind"] == "doc" and r["turn"] > 0 and r["asks"] == "d0",
    "after_turn0": lambda r: r["turn"] > 0,
    "all": lambda r: True,
}


def label(policy: str, budget: float) -> str:
    return "full" if policy == "full" else f"{policy}_{100 * budget:g}".replace(".", "_")


def keyed(rows: list[dict[str, Any]], policy: str, budget: float, field: str = "score") -> dict[tuple[int, int], Any]:
    return {(r["session"], r["turn"]): r[field] for r in rows if r["policy"] == policy and r["budget"] == budget}


def same_answers(a: dict[tuple[int, int], str], b: dict[tuple[int, int], str]) -> dict[str, int]:
    keys = sorted(set(a) & set(b))
    return {"paired": len(keys), "same": sum(1 for k in keys if a[k] == b[k]), "only_one_side": len(set(a) ^ set(b))}


def paired(new: dict[tuple[int, int], float], old: dict[tuple[int, int], float], rows: list[dict[str, Any]], select: Any) -> dict[str, Any]:
    """`new` against `old` on the same (session, turn) where `select` holds: accuracies and a sign test."""
    keep = {(r["session"], r["turn"]) for r in rows if select(r)}
    keys = sorted(k for k in keep if k in new and k in old)
    better = sum(1 for k in keys if new[k] > old[k])
    worse = sum(1 for k in keys if new[k] < old[k])
    return {"turns": len(keys), "new_accuracy": sum(new[k] for k in keys) / len(keys) if keys else None,
            "old_accuracy": sum(old[k] for k in keys) / len(keys) if keys else None,
            "better": better, "worse": worse, "sign_p": sign_test_p(worse, better)}


def retention(rows: list[dict[str, Any]], policy: str, budget: float, select: Any) -> dict[str, Any]:
    ref, got = keyed([r for r in rows if select(r)], "full", 1.0), keyed([r for r in rows if select(r)], policy, budget)
    keys = sorted(k for k in ref if k in got)
    sessions = sorted({s for s, _ in keys})
    num = [sum(got[k] for k in keys if k[0] == s) for s in sessions]
    den = [sum(ref[k] for k in keys if k[0] == s) for s in sessions]
    lo, hi = paired_ratio_ci(num, den) if sum(den) else (None, None)
    return {"turns": len(keys), "value": sum(num) / sum(den) if sum(den) else None, "lo": lo, "hi": hi}


def cost(src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in src["rows"]:
        e: dict[str, Any] = {"context": row["context"], "status": row["status"], "error": row.get("error"), "kv_bytes": row["kv_bytes"]}
        for cell in row.get("cells", []):
            e[f"{cell['mode']}_{cell['turn_tokens']}"] = {
                "wall_s": {"median": cell["wall_s_median"], "min": cell["wall_s_min"], "max": cell["wall_s_max"], "n": len(cell["runs"])},
                "ms_per_token": cell["ms_per_token_median"], "h2d_bytes": cell["h2d_bytes"], "peak_extra_bytes": cell["peak_extra_bytes_max"],
                "h2d_gbps": cell["h2d_bytes"] / cell["wall_s_median"] / 1e9 if cell["h2d_bytes"] else None,
            }
        for n in (256,):
            d, p = e.get(f"dense_{n}"), e.get(f"per_token_{n}")
            if d and p:
                e[f"speedup_{n}"] = p["wall_s"]["median"] / d["wall_s"]["median"]
        out[str(row["context"])] = e
    ok = [e for e in out.values() if e["status"] == "ok" and "speedup_256" in e]
    return {"contexts": out, "summary": {
        "speedup_256_min": min((e["speedup_256"] for e in ok), default=None),
        "speedup_256_max": max((e["speedup_256"] for e in ok), default=None),
        "max_context": max((e["context"] for e in ok), default=None),
        "dense_1024_max_s": max((e["dense_1024"]["wall_s"]["median"] for e in ok if "dense_1024" in e), default=None),
        "peak_extra_max_bytes": max((c["peak_extra_bytes"] for e in ok for k, c in e.items() if k.startswith("dense_")), default=None),
    }}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    src = json.loads((RESULTS_DIR / "phase21" / "multiturn" / "metrics.json").read_text(encoding="utf-8"))
    rows = src["turns"]
    p19_dense = json.loads((RESULTS_DIR / "phase19" / "multiturn_dense_turns" / "metrics.json").read_text(encoding="utf-8"))
    p19 = json.loads((RESULTS_DIR / "phase19" / "multiturn" / "metrics.json").read_text(encoding="utf-8"))
    budgets = sorted({r["budget"] for r in rows if r["policy"] != "full"}, reverse=True)
    bkey = {b: f"{100 * b:g}".replace(".", "_") for b in budgets}

    conds: dict[str, Any] = {}
    for pol, b in [("full", 1.0)] + [(p, b) for p in ("quest", "tiered_sync") for b in budgets]:
        e: dict[str, Any] = {"policy": pol, "budget": b}
        for name, sel in KINDS.items():
            s = [r["score"] for r in rows if r["policy"] == pol and r["budget"] == b and sel(r)]
            e[name] = {"n": len(s), "mean": sum(s) / len(s) if s else None, "ci95": list(bootstrap_mean_ci(s)),
                       **({} if pol == "full" else {"retention": retention(rows, pol, b, sel)})}
        conds[label(pol, b)] = e

    # Q1: the tier against rung 5 in this run. Q2: rung 5 (and the full cache) against Phase 19's run.
    q1 = {bkey[b]: same_answers(keyed(rows, "tiered_sync", b, "answer"), keyed(rows, "quest", b, "answer")) for b in budgets}
    old = p19_dense["turns"]
    q2 = {bkey[b]: same_answers(keyed(rows, "quest", b, "answer"), keyed(old, "quest", b, "answer")) for b in budgets}
    q2["full"] = same_answers(keyed(rows, "full", 1.0, "answer"), keyed(old, "full", 1.0, "answer"))
    # Q3: the tier with dense turns against Phase 19's tier fed token by token, where that ran.
    q3: dict[str, Any] = {}
    for b in sorted({r["budget"] for r in p19["turns"] if r["policy"] == "tiered_sync"}, reverse=True):
        new, before = keyed(rows, "tiered_sync", b), keyed(p19["turns"], "tiered_sync", b)
        q3[bkey.get(b, str(b))] = {name: paired(new, before, rows, sel) for name, sel in KINDS.items() if name != "doc_turn0"}

    summary = {
        "sessions": len({r["session"] for r in rows}),
        "turn_chunk": src["config"].get("turn_chunk"),
        "q1_all_same": all(v["same"] == v["paired"] and v["only_one_side"] == 0 for v in q1.values()),
        "q1_same": sum(v["same"] for v in q1.values()), "q1_paired": sum(v["paired"] for v in q1.values()),
        "q2_all_same": all(v["same"] == v["paired"] and v["only_one_side"] == 0 for v in q2.values()),
        "q2_same": sum(v["same"] for v in q2.values()), "q2_paired": sum(v["paired"] for v in q2.values()),
        "session_s_median": {label(f["policy"], f["budget"]): statistics.median(g["session_s"] for g in src["cache_facts"] if (g["policy"], g["budget"]) == (f["policy"], f["budget"])) for f in src["cache_facts"]},
        "p19_tier_session_s_median": statistics.median(g["session_s"] for g in p19["cache_facts"] if g["policy"] == "tiered_sync"),
    }
    cpath = RESULTS_DIR / "phase21" / "ingest_cost" / "metrics.json"
    csrc = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else None
    write_metrics(RESULTS_DIR / "phase21" / args.out, {
        "source": src["provenance"], "cost_source": None if csrc is None else csrc["provenance"],
        "conditions": conds, "q1_tier_vs_rung5": q1, "q2_reproduction": q2, "q3_tier_gain": q3,
        "cost": None if csrc is None else cost(csrc), "summary": summary,
    })
    print(json.dumps({"summary": summary, "q1": q1, "q2": q2, "q3": q3}, indent=1))


if __name__ == "__main__":
    main()
