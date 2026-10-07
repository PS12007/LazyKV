"""Derive Phase 19 quantities: multi-turn retrieval under each policy, against the full cache.

Reads   results/phase19/multiturn/metrics.json
Writes  results/phase19/analysis/metrics.json

Every retention is paired: a policy's score over the full cache's on the same (session, turn). CIs
resample whole sessions (sums over a session's turns), because turns of one session share a
document, a conversation and the model's earlier answers, so they are not independent draws.

Question 1's decision rule is the config's, applied here verbatim: the failure A5 is meant to fix
counts as observed at a budget only if rung 5's chat-needle retention has a 95% CI upper bound below
100% and is below window_sink's on the same turns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.stats import bootstrap_mean_ci, paired_ratio_ci, sign_test_p  # noqa: E402

VIABLE = 0.5  # configs/phase19.yaml: the full cache must answer at least half the chat-needle turns


def label(policy: str, budget: float) -> str:
    return "full" if policy == "full" else f"{policy}_{100 * budget:g}".replace(".", "_")


def retention(rows: list[dict[str, Any]], cond: tuple[str, float], select: Any) -> dict[str, Any]:
    """Paired retention of `cond` against the full cache over the turns `select` keeps."""
    by: dict[tuple[str, float], dict[tuple[int, int], float]] = defaultdict(dict)
    for r in rows:
        if select(r):
            by[(r["policy"], r["budget"])][(r["session"], r["turn"])] = r["score"]
    ref, got = by[("full", 1.0)], by[cond]
    keys = sorted(k for k in ref if k in got)
    sessions = sorted({s for s, _ in keys})
    num = [sum(got[k] for k in keys if k[0] == s) for s in sessions]
    den = [sum(ref[k] for k in keys if k[0] == s) for s in sessions]
    worse = sum(1 for k in keys if got[k] < ref[k])
    better = sum(1 for k in keys if got[k] > ref[k])
    lo, hi = paired_ratio_ci(num, den) if sum(den) else (None, None)
    return {
        "turns": len(keys), "sessions": len(sessions),
        "accuracy": sum(got[k] for k in keys) / len(keys) if keys else None,
        "full_accuracy": sum(ref[k] for k in keys) / len(keys) if keys else None,
        "value": sum(num) / sum(den) if sum(den) else None, "lo": lo, "hi": hi,
        "worse": worse, "better": better, "sign_p": sign_test_p(worse, better),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multiturn", help="results/phase19/<run> to analyze")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    src = json.loads((RESULTS_DIR / "phase19" / args.run / "metrics.json").read_text(encoding="utf-8"))
    rows = src["turns"]
    conds = sorted({(r["policy"], r["budget"]) for r in rows}, key=lambda c: (c[0] != "full", c[0], -c[1]))
    selectors = {
        "chat": lambda r: r["kind"] == "chat",
        "doc_turn0": lambda r: r["kind"] == "doc" and r["turn"] == 0,
        # First asks after turn 0; the last turn re-asks d0 and is reported on its own.
        "doc_later": lambda r: r["kind"] == "doc" and r["turn"] > 0 and r["asks"] != "d0",
        "doc_reask": lambda r: r["kind"] == "doc" and r["turn"] > 0 and r["asks"] == "d0",
        "chat_2_turns": lambda r: r["kind"] == "chat" and r["turns_since_given"] == 2,
        "chat_3_turns": lambda r: r["kind"] == "chat" and r["turns_since_given"] == 3,
        "all": lambda r: True,
    }
    per: dict[str, Any] = {}
    for c in conds:
        entry: dict[str, Any] = {"policy": c[0], "budget": c[1]}
        for name, sel in selectors.items():
            scores = [r["score"] for r in rows if (r["policy"], r["budget"]) == c and sel(r)]
            ci = bootstrap_mean_ci(scores)
            entry[name] = {"n": len(scores), "mean": sum(scores) / len(scores) if scores else None, "ci95": list(ci),
                           **({} if c[0] == "full" else {"retention": retention(rows, c, sel)})}
        per[label(*c)] = entry

    # Question 1's decision rule, per budget.
    budgets = sorted({b for p, b in conds if p == "quest"}, reverse=True)
    decision = {}
    for b in budgets:
        q = per[label("quest", b)]["chat"]["retention"]
        w = per.get(label("window_sink", b), {}).get("chat", {}).get("retention")
        decision[f"{100 * b:g}"] = {
            "quest_retention": q["value"], "quest_hi": q["hi"], "window_retention": None if w is None else w["value"],
            "observed": bool(q["hi"] is not None and q["hi"] < 1.0 and w is not None and q["value"] < w["value"]),
        }

    # Question 4: the tier against rung 5, answer text and score, on every turn they both ran.
    tier_rows = [r for r in rows if r["policy"].startswith("tiered")]
    validity = None
    if tier_rows:
        b = tier_rows[0]["budget"]
        quest = {(r["session"], r["turn"]): r for r in rows if r["policy"] == "quest" and r["budget"] == b}
        paired = [(r, quest[(r["session"], r["turn"])]) for r in tier_rows if (r["session"], r["turn"]) in quest]
        validity = {"budget": b, "turns": len(tier_rows), "paired": len(paired),
                    "same_answer": sum(1 for t, q in paired if t["answer"] == q["answer"]),
                    "same_score": sum(1 for t, q in paired if t["score"] == q["score"])}

    full_chat = per["full"]["chat"]["mean"]
    facts = src.get("cache_facts", [])
    lengths = [f["final_length"] for f in facts if f["policy"] == "full"]
    summary = {
        "sessions": len({r["session"] for r in rows}),
        "turns_per_session": len(src["schedule"]),
        "context": src["config"]["context"],
        "final_length_min": min(lengths) if lengths else None,
        "final_length_max": max(lengths) if lengths else None,
        "conversation_tokens_median": sorted(f - src["config"]["context"] for f in lengths)[len(lengths) // 2] if lengths else None,
        "full_chat_accuracy": full_chat,
        "viable": full_chat is not None and full_chat >= VIABLE,
        "decision": decision,
        "a5_failure_observed_any_budget": any(d["observed"] for d in decision.values()),
        "validity": validity,
        "k_blocks": {label(f["policy"], f["budget"]): f.get("k_blocks") for f in facts if f.get("k_blocks") is not None},
    }
    write_metrics(RESULTS_DIR / "phase19" / args.out, {"source": src["provenance"], "conditions": per, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
