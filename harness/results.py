"""Writing and loading ``results/**/metrics.json``, the only source of truth for numbers."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SCHEMA_VERSION = 1


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # rstrip only: porcelain status lines start with a significant space (" M path"), and a full
    # strip() cut the first dirty file's name short by one character.
    return out.stdout.rstrip()


def provenance() -> dict[str, Any]:
    # Untracked files (e.g. the results being written right now) do not change the code
    # that produced the numbers, so only tracked modifications count as dirty.
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # A dirty tree means the numbers may come from uncommitted code; record it rather
        # than refusing to run, since Phase 0 scripts are often edited between runs.
        "git_dirty": bool(status) if status is not None else None,
        # Which tracked files were modified, so a dirty flag can be judged (e.g. plotting code
        # edited mid-run vs the benchmark itself) instead of only rerun on suspicion.
        "git_dirty_files": [line[3:] for line in status.splitlines()] if status else [],
        "argv": sys.argv,
        "python": sys.version.split()[0],
    }


def write_metrics(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    doc = {"provenance": provenance(), **payload}
    path.write_text(json.dumps(doc, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(obj: object) -> object:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def load_all(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load every metrics.json into a nested dict keyed by its directory path.

    ``results/phase0/feasibility/metrics.json`` becomes ``ctx["phase0"]["feasibility"]``,
    which is what the doc templates address as ``phase0.feasibility.<key>``.
    """
    ctx: dict[str, Any] = {}
    for path in sorted(results_dir.rglob("metrics.json")):
        rel = path.parent.relative_to(results_dir).parts
        node = ctx
        for part in rel[:-1]:
            node = node.setdefault(part, {})
        node[rel[-1]] = json.loads(path.read_text(encoding="utf-8"))
    return ctx
