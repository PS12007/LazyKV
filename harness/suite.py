"""The benchmark suite as recorded: which command produced each result, and in what order to analyze.

Gap table item A1 asks for one entry point over the per-phase scripts. The commands are not kept in
a list here, because every metrics.json already records the argv and commit that produced it
(harness/results.py), and a hand-kept list drifts from what actually ran. configs/suite.yaml holds
only what provenance cannot say: the phase of each non-phaseN results directory, and the analysis
order, since several analyses read another phase's analysis output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.results import REPO_ROOT, RESULTS_DIR

SUITE_PATH = REPO_ROOT / "configs" / "suite.yaml"


@dataclass(frozen=True)
class Recorded:
    """One metrics.json and the command that wrote it."""

    result: str  # path under results/, posix, e.g. "phase4/budget_speed/run_1"
    phase: int
    argv: tuple[str, ...]  # normalized: scripts/<name>.py first, forward slashes
    commit: str | None
    dirty: bool | None

    @property
    def script(self) -> str:
        return self.argv[0] if self.argv else ""

    @property
    def is_analysis(self) -> bool:
        name = Path(self.script).name
        return name.startswith("analyze_")

    @property
    def is_smoke(self) -> bool:
        # *_quick directories are smoke tests; .gitignore keeps them out and no doc reports them.
        return any(part.endswith("_quick") for part in self.result.split("/"))

    def not_reproducible_because(self) -> str | None:
        """Why this command cannot be replayed from a clone, or None if it can."""
        if not self.argv:
            return "no argv recorded"
        if not (REPO_ROOT / self.script).exists():
            return f"script {self.script} no longer exists"
        for a in self.argv[1:]:
            # A config that lived in a session scratchpad is gone; the run cannot be replayed exactly.
            if a.endswith((".yaml", ".yml")) and not (REPO_ROOT / a).exists():
                return f"config {a} is not in the repo"
        return None


def load_suite(path: Path = SUITE_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_argv(argv: list[str]) -> tuple[str, ...]:
    """Recorded argv[0] is whatever the shell passed: 'scripts\\x.py', 'x.py' or an absolute path."""
    if not argv:
        return ()
    out = [str(a).replace("\\", "/") for a in argv]
    out[0] = "scripts/" + out[0].rsplit("/", 1)[-1]
    return tuple(out)


def phase_of(top: str, groups: dict[str, int]) -> int:
    if top in groups:
        return int(groups[top])
    m = re.fullmatch(r"phase(\d+)", top)
    if not m:
        raise KeyError(f"results/{top} has no phase: add it to groups in {SUITE_PATH.name}")
    return int(m.group(1))


def recorded(results_dir: Path = RESULTS_DIR, suite: dict[str, Any] | None = None) -> list[Recorded]:
    suite = suite or load_suite()
    groups = suite.get("groups", {})
    out = []
    for path in sorted(results_dir.rglob("metrics.json")):
        rel = path.parent.relative_to(results_dir).as_posix()
        prov = json.loads(path.read_text(encoding="utf-8")).get("provenance", {})
        out.append(
            Recorded(
                result=rel,
                phase=phase_of(rel.split("/")[0], groups),
                argv=normalize_argv(prov.get("argv") or []),
                commit=prov.get("git_commit"),
                dirty=prov.get("git_dirty"),
            )
        )
    return out


def analysis_steps(suite: dict[str, Any], phases: set[int] | None = None) -> list[tuple[int, tuple[str, ...]]]:
    steps = [(int(s["phase"]), tuple(s["argv"])) for s in suite["analysis_order"]]
    return [s for s in steps if phases is None or s[0] in phases]


def diff_paths(old: Any, new: Any, prefix: str = "", limit: int = 20) -> list[str]:
    """Key paths where two JSON values differ, at most `limit` of them; exact comparison."""
    out: list[str] = []

    def walk(a: Any, b: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a or k not in b:
                    out.append(f"{p}.{k}" if p else str(k))
                else:
                    walk(a[k], b[k], f"{p}.{k}" if p else str(k))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{p}[len {len(a)} -> {len(b)}]")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
        elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
            out.append(p)

    walk(old, new, prefix)
    return out


def without_provenance(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "provenance"}
