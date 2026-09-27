"""Phase 14 analysis on synthetic LongBench rows: viability, pairing, and the multi-hop comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("analyze_phase14", ROOT / "scripts" / "analyze_phase14.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(task: str, index: int, policy: str, budget: float, score: float) -> dict:
    return {"task": task, "index": index, "policy": policy, "budget": budget, "score": score, "truncated": False, "prompt_len": 1000}


def test_paired_uses_only_prompts_both_conditions_answered() -> None:
    m = _mod()
    rows = [_row("hotpotqa", 0, "full", 1.0, 0.8), _row("hotpotqa", 1, "full", 1.0, 0.4), _row("hotpotqa", 0, "quest", 0.5, 0.4)]
    num, den = m.paired(rows, ["hotpotqa"], "quest", 0.5)
    assert num == [0.4] and den == [0.8]


def test_a_task_below_the_floor_is_excluded_from_the_comparison(tmp_path: Path) -> None:
    m = _mod()
    rows = []
    for i in range(20):
        rows += [_row("hotpotqa", i, "full", 1.0, 0.5), _row("hotpotqa", i, "quest", 0.5, 0.1),
                 _row("musique", i, "full", 1.0, 0.0), _row("musique", i, "quest", 0.5, 0.0),
                 _row("multifieldqa_en", i, "full", 1.0, 0.5), _row("multifieldqa_en", i, "quest", 0.5, 0.5)]
    (tmp_path / "longbench").mkdir()
    (tmp_path / "longbench" / "metrics.json").write_text(json.dumps({"rows": rows, "wall_s": 1.0, "provenance": {"git_commit": "x", "git_dirty": False}}))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("budgets: [0.5]\nviability_floor: 0.15\nhard_tasks: [hotpotqa, musique]\ncontrol_task: multifieldqa_en\nbootstrap: {iters: 300, seed: 0}\n")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--base", str(tmp_path), "--config", str(cfg)]
    try:
        m.main()
    finally:
        sys.argv = argv
    out = json.loads((tmp_path / "analysis" / "metrics.json").read_text())
    s = out["summary"]
    assert s["excluded_tasks"] == ["musique"] and s["hard_tasks_used"] == ["hotpotqa"]
    c = out["comparison"]["quest_50"]
    assert abs(c["hard"]["retention"] - 0.2) < 1e-12 and c["easy"]["retention"] == 1.0 and c["gap_ci95"][1] < 0
