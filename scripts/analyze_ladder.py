"""The whole policy ladder on one axis: the brief's headline experiment (§B8).

Reads   results/phase{2,3,4,5}/analysis/metrics.json
Writes  results/ladder/metrics.json

No rung is re-run here. Phases 2-5 each swept the same budgets at the same context and block size
over the same 45 NIAH prompts, so the rungs can in principle be put on one frontier -- but "in
principle" is not a measurement, so this script tests it before doing it.

**The repeatability check comes first.** Adjacent phases deliberately overlap: `window_sink` was
measured in Phases 2 and 3, `quest` in 3 and 4, `tiered_sync` in 4 and 5, and the full cache in all
four. Every overlapping condition is compared across the phases that measured it, and the largest
disagreement is reported next to the frontier. If those disagreements were large, the combined
frontier would be an artifact of when each rung happened to be run, and the right response would be
to say so rather than to publish the picture.

**VRAM slots are not uniform across rungs and the table says which.** Phase 4 ran the tier twice,
with slots for exactly the attended set and for twice it; only the first actually frees VRAM at the
larger budgets. Rung 8 was run only at the 2x setting. Mixing them silently would make the memory
axis mean different things in different rows, so the variant is carried as a column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from harness.sweep_analysis import label, span  # noqa: E402

MIB = 2**20

# Rung -> (policy, the phase whose measurement is authoritative, human name).
# The authoritative phase is the latest one that measured the rung, so each rung is quoted from the
# run that also carried its own gate report.
LADDER: list[tuple[int, str, str, str]] = [
    (1, "full", "phase5", "Full GPU KV (reference, exact)"),
    (2, "window_sink", "phase2", "Sliding window + attention sink"),
    (3, "lru", "phase2", "LRU block eviction"),
    (4, "h2o", "phase3", "Attention-score eviction (H2O style)"),
    (5, "quest", "phase4", "Query-aware block selection (Quest style)"),
    (6, "tiered_sync", "phase5", "CPU tier, synchronous fetch"),
    (7, "tiered_prefetch", "phase4", "CPU tier + layer-ahead prefetch"),
    (8, "tiered_int8", "phase5", "CPU tier, int8 warm/cold"),
]
# Which VRAM-slot variant each phase's headline tier numbers come from.
SLOT_VARIANT = {"phase4": "2x attended", "phase5": "2x attended"}


def load(phase: str) -> dict[str, Any] | None:
    p = RESULTS_DIR / phase / "analysis" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def repeatability(phases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every condition measured in more than one phase, and how far the phases disagree."""
    seen: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for name, a in phases.items():
        for r in a["niah"]:
            seen.setdefault((r["policy"], r["budget"]), {})[name] = r
    out = []
    for (policy, budget), by_phase in sorted(seen.items()):
        if len(by_phase) < 2:
            continue
        accs = {n: r["accuracy"] for n, r in by_phase.items()}
        speeds = {}
        for n, a in phases.items():
            if n not in by_phase:
                continue
            row = next((s for s in a["speed"] if s["label"] == label(policy, budget)), None)
            if row:
                speeds[n] = row["tokens_per_s"]["median"]
        out.append({
            "policy": policy,
            "budget": budget,
            "phases": sorted(by_phase),
            "accuracy_by_phase": accs,
            "accuracy_max_gap_pp": 100 * (max(accs.values()) - min(accs.values())),
            "tokens_per_s_by_phase": speeds,
            "tokens_per_s_max_ratio": (max(speeds.values()) / min(speeds.values())) if len(speeds) > 1 else None,
        })
    return out


def main() -> None:
    phases = {n: a for n in ("phase2", "phase3", "phase4", "phase5") if (a := load(n)) is not None}
    if not phases:
        raise SystemExit("no phase analyses found")
    ref = phases.get("phase5") or next(iter(phases.values()))

    # Comparability: the frontier is only meaningful if every phase swept the same thing.
    shape = {
        n: {"context": a["context"], "block_size": a["block_size"], "prompts": a["niah_prompts_per_condition"], "full_niah_accuracy": a["full_niah_accuracy"]}
        for n, a in phases.items()
    }
    comparable = len({(v["context"], v["block_size"], v["prompts"], round(v["full_niah_accuracy"], 6)) for v in shape.values()}) == 1

    rows = []
    for rung, policy, phase, pretty in LADDER:
        a = phases.get(phase)
        if a is None:
            continue
        niah = {r["budget"]: r for r in a["niah"] if r["policy"] == policy}
        speed = {s["budget"]: s for s in a["speed"] if s["policy"] == policy}
        for budget in sorted(niah, reverse=True):
            n, s = niah[budget], speed.get(budget)
            rows.append({
                "rung": rung,
                "policy": policy,
                "name": pretty,
                "phase": phase,
                "slots": SLOT_VARIANT.get(phase) if policy.startswith("tiered") else None,
                "budget": budget,
                "accuracy": n["accuracy"],
                "accuracy_ci95": n["accuracy_ci95"],
                "retention": n["retention"],
                "resident_mib": s["gpu_resident_kv_bytes"]["median"] / MIB if s else None,
                "tokens_per_s": s["tokens_per_s"]["median"] if s else None,
            })

    # Phase 4 also ran rungs 6 and 7 with VRAM holding exactly the attended set, which is the only
    # tier configuration that actually frees memory at the larger budgets. Those rows carry the
    # memory axis; the 2x rows above carry the latency the gate reported. Accuracy is not re-stated
    # from a separate measurement: Phase 4 established that the tier is exact and that slot count
    # changes what is *resident*, never what is attended, so the accuracy is the same number.
    p4 = phases.get("phase4")
    if p4 and p4.get("spare0", {}).get("speed"):
        by_label = {r["label"]: r for r in p4["spare0"]["speed"]}
        for rung, policy in ((6, "tiered_sync"), (7, "tiered_prefetch")):
            niah = {r["budget"]: r for r in p4["niah"] if r["policy"] == policy}
            name = next(n for rg, _, _, n in LADDER if rg == rung)
            for budget in sorted(niah, reverse=True):
                sp = by_label.get(label(policy, budget))
                if not sp:
                    continue
                n = niah[budget]
                rows.append({
                    "rung": rung,
                    "policy": policy,
                    "name": name,
                    "phase": "phase4",
                    "slots": "= attended",
                    "budget": budget,
                    "accuracy": n["accuracy"],
                    "accuracy_ci95": n["accuracy_ci95"],
                    "retention": n["retention"],
                    "resident_mib": sp["gpu_resident_kv_bytes"]["median"] / MIB,
                    "tokens_per_s": sp["tokens_per_s"]["median"],
                })

    rep = repeatability(phases)
    best = {}
    for rung, policy, _, pretty in LADDER:
        rs = [r for r in rows if r["rung"] == rung and r["budget"] < 1.0] or [r for r in rows if r["rung"] == rung]
        # Where a rung was measured at two slot counts, the frontier quotes the one that frees VRAM.
        if any(r["slots"] == "= attended" for r in rs):
            rs = [r for r in rs if r["slots"] == "= attended"]
        if not rs:
            continue
        top = max(rs, key=lambda r: r["retention"] or 0)
        # Brief §B8: smallest budget retaining >= 99% of the full cache, and the speed there.
        passing = sorted((r for r in rs if (r["retention"] or 0) >= 0.99), key=lambda r: r["budget"])
        best[str(rung)] = {
            "policy": policy,
            "name": pretty,
            "best_retention": top["retention"],
            "best_retention_budget": top["budget"],
            "resident_mib_at_best": top["resident_mib"],
            "tokens_per_s_at_best": top["tokens_per_s"],
            "min_budget_meeting_target": passing[0]["budget"] if passing else None,
        }

    summary = {
        "context": ref["context"],
        "block_size": ref["block_size"],
        "prompts": ref["niah_prompts_per_condition"],
        "full_niah_accuracy": ref["full_niah_accuracy"],
        "phases_combined": sorted(phases),
        "sweeps_comparable": comparable,
        "rungs": len(best),
        "overlapping_conditions": len(rep),
        # The number that licenses the combined frontier, or refuses it.
        "repeatability_accuracy_max_gap_pp": max((r["accuracy_max_gap_pp"] for r in rep), default=None),
        "repeatability_tokens_per_s_max_ratio": max((r["tokens_per_s_max_ratio"] for r in rep if r["tokens_per_s_max_ratio"]), default=None),
        # Brief §B8's headline question, and the answer is no: the full cache trivially meets its own
        # target at a 100% budget, so rung 1 is excluded rather than allowed to answer it.
        "rungs_meeting_target_below_full": [k for k, v in best.items() if k != "1" and v["min_budget_meeting_target"] is not None],
        "any_rung_meeting_target_below_full": any(k != "1" and v["min_budget_meeting_target"] is not None for k, v in best.items()),
        "best_retention_overall": span([v["best_retention"] for v in best.values() if v["best_retention"] is not None]),
        # Rung 1 is the reference and retains itself perfectly; quoting it as "the best a policy
        # achieved" would overstate the ladder by the exact amount the study is trying to measure.
        "best_retention_excluding_full": span([v["best_retention"] for k, v in best.items() if k != "1" and v["best_retention"] is not None]),
        "best_retention_budget_excluding_full": max(((v["best_retention"], v["best_retention_budget"]) for k, v in best.items() if k != "1" and v["best_retention"] is not None), default=(None, None))[1],
    }
    write_metrics(RESULTS_DIR / "ladder", {"shape": shape, "rows": rows, "repeatability": rep, "best": best, "summary": summary})
    print("wrote", RESULTS_DIR / "ladder" / "metrics.json")


if __name__ == "__main__":
    main()
