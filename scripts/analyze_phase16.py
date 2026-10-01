"""Phase 16 analysis: selectors against the true-mass oracle, and slot replacement against Belady.

Reads the traces scripts/06_record_traces.py listed in results/phase16/traces/metrics.json and
replays them with lazykv/residency_sim.py. CPU only. Three questions:

1. **Selection.** At each token budget, what share of the true attention mass does each selector
   capture, against the oracle (the K blocks of highest true mass)?
   - The stale oracle (last step's true top-K) is compared on steps after the first only.
   - Over the NIAH traces, step 0 is reported on its own: the first answer token's query is the same
     under every policy, so step 0 is the one step whose trace is the policy's own. Later steps follow
     the full cache's trajectory.
2. **Does captured mass predict quality?** Phase 8 ran rung 5 on these exact prompts. Per budget,
   prompts where Quest-style selection lost the full cache's answer are compared with prompts where
   it kept it, on the mass Quest captured at step 0. The full cache's answers here are checked
   against Phase 8's first, so the join is on the same decode.
3. **Replacement.** For Quest's selection on the teacher-forced traces (policy-independent, 256
   steps), fetches per token under the tier's LRU rule and under Belady, for slot pools larger than
   K. NIAH decodes are 15-47 steps, mostly the cold start, so they are left out of this part.

    .venv\\Scripts\\python.exe scripts\\analyze_phase16.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402
from lazykv.residency_sim import SELECTORS, mass_captured, replay  # noqa: E402
from lazykv.selection import blocks_for_budget  # noqa: E402
from lazykv.stats import bootstrap_mean_ci  # noqa: E402
from lazykv.trace import AttentionTrace  # noqa: E402

log = logging.getLogger("analyze_phase16")


def read_metrics(d: Path) -> dict[str, Any]:
    return json.loads((d / "metrics.json").read_text(encoding="utf-8"))


def k_for(budget: float, row: dict[str, Any], bs: int) -> int:
    return blocks_for_budget(budget, row["total_tokens"], bs)


def tie_share(tr: AttentionTrace, k: int) -> float:
    """Share of (step, layer, head) cells whose K-th and (K+1)-th highest Quest bounds are equal.

    The runtime ranks a bf16 bound, so at a tie the simulator and `torch.topk` may admit different
    blocks. This is how often that can happen at all.
    """
    tied = total = 0
    for t in range(tr.steps):
        b = tr.bound[t, :, :, 1 : tr.n_full[t]]
        if b.shape[-1] <= k:
            continue
        part = -np.partition(-b, (k - 1, k), axis=-1)
        tied += int((part[..., k - 1] == part[..., k]).sum())
        total += part[..., 0].size
    return tied / total if total else float("nan")


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random `pos` value exceeds a random `neg` value), ties counted half: Mann-Whitney AUC."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="phase16/traces")
    p.add_argument("--out", default="phase16/analysis")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    src = read_metrics(RESULTS_DIR / args.traces)
    cfg = src["config"]
    bs, budgets, selectors, spares = cfg["block_size"], cfg["budgets"], cfg["selectors"], cfg["spares"]
    trace_dir = Path(src["trace_dir"])
    t0 = time.perf_counter()

    per_trace: list[dict[str, Any]] = []
    replacement: list[dict[str, Any]] = []
    for row in src["traces"]:
        tr = AttentionTrace.load(trace_dir / row["file"])
        cands = int(tr.n_full.min()) - 1
        out: dict[str, Any] = {k: row[k] for k in ("source", "kind", "depth", "sample", "book", "offset", "full_score") if k in row}
        out["steps"], out["budgets"] = tr.steps, {}
        for b in budgets:
            k = k_for(b, row, bs)
            if k > cands:
                continue
            cell: dict[str, Any] = {"k": k, "quest_tie_share": tie_share(tr, k)}
            chosen = {}
            for name in selectors:
                chosen[name] = SELECTORS[name](tr, k)
                m = mass_captured(tr, chosen[name])
                # "later" excludes step 0, where the stale oracle has no previous step and falls back to Quest.
                cell[name] = {"mean": float(m.mean()), "step0": float(m[0].mean()), "later": float(m[1:].mean()), "p05": float(np.quantile(m, 0.05)),
                              "min_over_heads_step0": float(m[0].min())}
            out["budgets"][f"{b:g}"] = cell
            if row["source"] == "teacher_forced":
                for spare in spares:
                    n_slots = int(round(k * (1 + spare)))
                    rec: dict[str, Any] = {"book": row["book"], "offset": row["offset"], "budget": b, "k": k, "spare": spare, "n_slots": n_slots}
                    if n_slots >= cands:
                        rec["pool_holds_every_candidate"] = True
                    else:
                        for rule in ("lru", "belady"):
                            f = replay(tr, chosen["quest"], n_slots, rule).sum(axis=(1, 2))
                            # Step 0 is the cold start after the boundary seed; steady state is the rest.
                            rec[rule] = {"step0": int(f[0]), "steady_per_token": float(f[1:].mean()), "total": int(f.sum())}
                        rec["belady_over_lru_steady"] = rec["belady"]["steady_per_token"] / rec["lru"]["steady_per_token"] if rec["lru"]["steady_per_token"] else None
                    replacement.append(rec)
                    log.info("replacement %s@%d b=%g spare=%g: %s", row["book"], row["offset"], b, spare,
                             {r: rec[r]["steady_per_token"] for r in ("lru", "belady") if r in rec})
        per_trace.append(out)
        log.info("%s (%.0fs)", row["file"], time.perf_counter() - t0)

    niah = [t for t in per_trace if t["source"] == "niah"]
    tf = [t for t in per_trace if t["source"] == "teacher_forced"]

    # -- 1. selection ------------------------------------------------------------------------------
    selection: dict[str, Any] = {}
    for b in budgets:
        key = f"{b:g}"
        rows = [t["budgets"][key] for t in niah if key in t["budgets"]]
        if not rows:
            continue
        s: dict[str, Any] = {"k": rows[0]["k"], "prompts": len(rows), "quest_tie_share": float(np.mean([r["quest_tie_share"] for r in rows]))}
        for name in selectors:
            step0 = [r[name]["step0"] for r in rows]
            s[name] = {
                "niah_step0": float(np.mean(step0)), "niah_step0_ci95": list(bootstrap_mean_ci(step0)),
                "niah_later": float(np.mean([r[name]["later"] for r in rows])),
                "niah_p05": float(np.mean([r[name]["p05"] for r in rows])),
                "tf_mean": float(np.mean([t["budgets"][key][name]["mean"] for t in tf if key in t["budgets"]])) if tf else None,
            }
        s["quest_share_of_oracle_step0"] = s["quest"]["niah_step0"] / s["oracle"]["niah_step0"]
        s["stale_minus_quest_later"] = s["stale_oracle"]["niah_later"] - s["quest"]["niah_later"]
        if tf:
            s["stale_minus_quest_tf"] = s["stale_oracle"]["tf_mean"] - s["quest"]["tf_mean"]
        selection[key] = s

    # -- 2. captured mass against Phase 8's answers ------------------------------------------------
    quality: dict[str, Any] = {}
    p8 = read_metrics(RESULTS_DIR / cfg["quality_source"])
    by = {(r["policy"], r["budget"], r["kind"], r["depth"], r["sample"]): r["score"] for r in p8["niah"]}
    full_match = [by.get(("full", 1.0, t["kind"], t["depth"], t["sample"])) == t["full_score"] for t in niah]
    quality["full_scores_match_phase8"] = int(sum(full_match))
    quality["prompts"] = len(niah)
    for b in budgets:
        key = f"{b:g}"
        kept, lost, oracle_kept, oracle_lost = [], [], [], []
        for t in niah:
            if key not in t["budgets"] or t["full_score"] == 0:
                continue  # a prompt the full cache misses has nothing to retain
            q = by.get(("quest", b, t["kind"], t["depth"], t["sample"]))
            if q is None:
                continue
            cell = t["budgets"][key]
            (kept if q >= t["full_score"] else lost).append(cell["quest"]["step0"])
            (oracle_kept if q >= t["full_score"] else oracle_lost).append(cell["oracle"]["step0"])
        quality[key] = {
            "kept": len(kept), "lost": len(lost),
            "quest_step0_kept_mean": float(np.mean(kept)) if kept else None,
            "quest_step0_lost_mean": float(np.mean(lost)) if lost else None,
            "oracle_step0_lost_mean": float(np.mean(oracle_lost)) if oracle_lost else None,
            # Probability a kept prompt captured more mass than a lost one; 0.5 is no signal.
            "auc_step0": auc(kept, lost),
        }

    # -- 3. replacement summary ----------------------------------------------------------------------
    rep_summary: dict[str, Any] = {}
    for spare in spares:
        ratios = [r["belady_over_lru_steady"] for r in replacement if r["spare"] == spare and r.get("belady_over_lru_steady") is not None]
        rep_summary[f"{spare:g}"] = {"conditions": len(ratios), "belady_over_lru_min": min(ratios) if ratios else None,
                                     "belady_over_lru_max": max(ratios) if ratios else None}
    forced = [r for r in replacement if r["spare"] == 0 and "lru" in r]
    rep_summary["forced_case_identical"] = all(r["lru"]["total"] == r["belady"]["total"] for r in forced)

    # -- summary: dot-free keys for the doc templates (a budget like "0.75" cannot be a template path) --
    def span(xs: list[float]) -> dict[str, float] | None:
        xs = [x for x in xs if x is not None]
        return {"min": min(xs), "max": max(xs)} if xs else None

    sel = list(selection.values())
    tight = selection[f"{min(budgets):g}"] if f"{min(budgets):g}" in selection else None
    summary = {
        "niah_prompts": len(niah),
        "tf_windows": len(tf),
        "full_scores_match_phase8": quality["full_scores_match_phase8"],
        "quest_share_of_oracle_step0": span([s["quest_share_of_oracle_step0"] for s in sel]),
        "stale_minus_quest_later": span([s["stale_minus_quest_later"] for s in sel]),
        "stale_minus_quest_tf": span([s.get("stale_minus_quest_tf") for s in sel]),
        "quest_tie_share": span([s["quest_tie_share"] for s in sel]),
        "auc_step0": span([quality[f"{b:g}"]["auc_step0"] for b in budgets if f"{b:g}" in quality]),
        "lost_at_tightest": quality.get(f"{min(budgets):g}", {}).get("lost"),
        "tightest_budget": min(budgets),
        "oracle_step0_tightest": tight["oracle"]["niah_step0"] if tight else None,
        "quest_step0_tightest": tight["quest"]["niah_step0"] if tight else None,
        "window_step0_tightest": tight["window"]["niah_step0"] if tight else None,
        "belady_over_lru_with_spare": span([r.get("belady_over_lru_steady") for r in replacement if r["spare"] > 0]),
        "forced_case_identical": rep_summary["forced_case_identical"],
    }

    write_metrics(RESULTS_DIR / args.out, {
        "summary": summary,
        "source": args.traces,
        "block_size": bs,
        "context": cfg["context"],
        "budgets": budgets,
        "selection": selection,
        "quality_join": quality,
        "replacement": replacement,
        "replacement_summary": rep_summary,
        "per_trace": per_trace,
        "wall_s": time.perf_counter() - t0,
    })
    log.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
