"""One entry point over the benchmark suite (gap table item A1).

    reproduce.py list                     every phase: results, drivers recorded, analyses
    reproduce.py plan --phase 17          the exact commands behind a phase, from provenance
    reproduce.py chain --phase 17 --out x.cmd
                                          the same as a detachable .cmd chain with a status file
    reproduce.py analyze [--phase N ...]  rerun analyses (all by default), figures, docs
    reproduce.py audit                    rerun every analysis on the committed raw results and
                                          check each output equals the committed one

Driver commands come from the argv each metrics.json recorded (harness/suite.py), so `plan` prints
what produced the published numbers, not what a README says did. Drivers are never run in the
foreground here: the GPU ones take minutes to hours and must be detached (CLAUDE.md rule 4), so
`plan` and `chain` print them and `chain` writes them as a .cmd to launch detached.

The audit is the suite's self-check and Phase 18's experiment. It needs a clean tree, rewrites
analysis outputs while it runs, and restores every one of them byte for byte before it exits, so the
only file it leaves changed is its own report.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import REPO_ROOT, RESULTS_DIR, write_metrics  # noqa: E402
from harness.suite import Recorded, analysis_steps, diff_paths, load_suite, recorded, without_provenance  # noqa: E402

log = logging.getLogger("reproduce")
PY = sys.executable


def _phases(args: argparse.Namespace) -> set[int] | None:
    return set(args.phase) if args.phase else None


def cmd_list(_: argparse.Namespace) -> int:
    recs = recorded()
    suite = load_suite()
    for ph in sorted({r.phase for r in recs} | {p for p, _ in analysis_steps(suite)}):
        mine = [r for r in recs if r.phase == ph]
        drivers = sorted({r.script for r in mine if not r.is_analysis and not r.is_smoke})
        broken = [r for r in mine if not r.is_analysis and not r.is_smoke and r.not_reproducible_because()]
        analyses = [" ".join(a) for p, a in analysis_steps(suite, {ph})]
        print(f"Phase {ph}: {len([r for r in mine if not r.is_analysis])} raw results")
        print(f"  drivers:  {', '.join(Path(d).name for d in drivers) or '-'}")
        print(f"  analyses: {'; '.join(analyses) or '-'}")
        if broken:
            print(f"  not replayable from a clone: {len(broken)} (see plan)")
    return 0


def _driver_lines(recs: list[Recorded]) -> list[str]:
    lines = []
    for r in recs:
        why = r.not_reproducible_because()
        cmd = " ".join(r.argv)
        note = f"   :: {r.result} @ {(r.commit or '?')[:7]}{' dirty' if r.dirty else ''}"
        lines.append(("REM not replayable (" + why + "): " if why else "") + cmd + note)
    return lines


def cmd_plan(args: argparse.Namespace) -> int:
    phases = _phases(args)
    suite = load_suite()
    recs = [r for r in recorded() if (phases is None or r.phase in phases) and not r.is_analysis]
    if not args.include_smoke:
        recs = [r for r in recs if not r.is_smoke]
    print("# Drivers, as recorded (detach every one of these; see README 'Measurement discipline')")
    for line in _driver_lines(recs):
        print(line)
    print("# Then")
    for _, a in analysis_steps(suite, phases):
        print(" ".join(a))
    for a in suite["finish"]:
        print(" ".join(a))
    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    """A .cmd that runs a phase's drivers then its analyses, appending each exit code to a status
    file and touching <out>.done at the end, so it can be launched outside the session's process tree
    (memory: detached runs must outlive the session) and watched by polling the status file."""
    phases = _phases(args)
    suite = load_suite()
    recs = [r for r in recorded() if (phases is None or r.phase in phases) and not r.is_analysis and not r.is_smoke]
    out = Path(args.out).resolve()
    status = out.with_suffix(".status")
    logs = REPO_ROOT / "logs"
    lines = ["@echo off", f'cd /d "{REPO_ROOT}"', f'if not exist "{logs}" mkdir "{logs}"', f'type nul > "{status}"']
    steps: list[tuple[str, tuple[str, ...]]] = []
    for r in recs:
        why = r.not_reproducible_because()
        if why:
            lines.append(f"REM skipped {r.result}: {why}")
            continue
        steps.append((r.result.replace("/", "_"), r.argv))
    steps += [(f"analyze_{i}", a) for i, (_, a) in enumerate(analysis_steps(suite, phases))]
    steps += [(f"finish_{i}", tuple(a)) for i, a in enumerate(suite["finish"])]
    for name, argv in steps:
        win = " ".join(a.replace("/", "\\") for a in argv)
        lines.append(f'"{PY}" {win} > "{logs / ("suite_" + name + ".log")}" 2>&1')
        lines.append(f'echo {name} exit %errorlevel% >> "{status}"')
    lines.append(f'type nul > "{out.with_suffix(".done")}"')
    out.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    print(f"wrote {out} ({len(steps)} steps); status -> {status}")
    return 0


def _run(argv: tuple[str, ...] | list[str]) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    p = subprocess.run([PY, *argv], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    return p.returncode, time.perf_counter() - t0, " | ".join(tail)


def cmd_analyze(args: argparse.Namespace) -> int:
    suite = load_suite()
    failed = 0
    for _, a in analysis_steps(suite, _phases(args)) + [(None, tuple(f)) for f in suite["finish"]]:
        rc, dt, tail = _run(a)
        log.info("%s -> exit %d in %.1fs", " ".join(a), rc, dt)
        if rc:
            log.error("  %s", tail)
            failed += 1
    return 1 if failed else 0


def _tracked_dirty() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return [line[3:] for line in out.stdout.splitlines()]


def cmd_audit(args: argparse.Namespace) -> int:
    dirty = _tracked_dirty()
    if dirty and not args.allow_dirty:
        log.error("tracked files modified, refusing to audit: %s", dirty)
        return 2
    suite = load_suite()
    outputs = {r.result: RESULTS_DIR / r.result / "metrics.json" for r in recorded() if r.is_analysis}
    snapshot = {k: p.read_bytes() for k, p in outputs.items()}

    import importlib.util

    spec = importlib.util.spec_from_file_location("render_docs", REPO_ROOT / "scripts" / "render_docs.py")
    assert spec and spec.loader
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    committed_docs = {p: p.read_text(encoding="utf-8") for p in renderer.render_all()}

    steps = []
    try:
        for ph, a in analysis_steps(suite):
            rc, dt, tail = _run(a)
            log.info("phase %d: %s -> exit %d in %.1fs", ph, " ".join(a), rc, dt)
            steps.append({"phase": ph, "argv": list(a), "exit": rc, "seconds": round(dt, 2), "tail": tail if rc else ""})
        compared = []
        for key, path in sorted(outputs.items()):
            raw = path.read_bytes()
            if raw == snapshot[key]:
                # Every write stamps a fresh generated_at_utc, so unchanged bytes mean the analysis
                # never wrote: it failed. Counting that as "identical" would pass a broken step.
                compared.append({"result": key, "regenerated": False, "identical": None, "differing_paths": []})
                log.info("%s: NOT REGENERATED", key)
                continue
            old_doc, new_doc = json.loads(snapshot[key]), json.loads(raw)
            d = diff_paths(without_provenance(old_doc), without_provenance(new_doc))
            compared.append({"result": key, "regenerated": True, "identical": not d, "differing_paths": d})
            log.info("%s: %s", key, "identical" if not d else f"DIFFERS at {d[:5]}")
            # Docs are re-rendered with the committed provenance put back, so a changed doc means a
            # changed number; the run stamps (time, commit, dirty flag) differ by construction.
            new_doc["provenance"] = old_doc.get("provenance")
            path.write_text(json.dumps(new_doc, indent=2) + "\n", encoding="utf-8")
        rerendered = renderer.render_all()
        docs = []
        for p, text in sorted(rerendered.items()):
            old = committed_docs.get(p, "")
            changed = [ln for ln in text.splitlines() if ln not in set(old.splitlines())]
            docs.append({"doc": p.relative_to(REPO_ROOT).as_posix(), "identical": text == old, "changed_lines": changed[:10]})
    finally:
        for key, path in outputs.items():
            path.write_bytes(snapshot[key])
    left = _tracked_dirty()

    regen = [c for c in compared if c["regenerated"]]
    summary = {
        "analyses_run": len(steps),
        "analyses_failed": sum(1 for s in steps if s["exit"]),
        "outputs": len(compared),
        "outputs_not_regenerated": len(compared) - len(regen),
        "outputs_identical": sum(1 for c in regen if c["identical"]),
        "outputs_differing": sum(1 for c in regen if not c["identical"]),
        "docs_rendered": len(docs),
        "docs_identical": sum(d["identical"] for d in docs),
        "docs_differing": sum(not d["identical"] for d in docs),
        "tree_restored": left == dirty,
        "seconds_total": round(sum(s["seconds"] for s in steps), 1),
    }
    path = write_metrics(RESULTS_DIR / args.out, {"steps": steps, "outputs": compared, "docs": docs, "summary": summary})
    print(json.dumps(summary, indent=2))
    print("wrote", path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("plan", "chain", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--phase", type=int, nargs="+")
        if name == "plan":
            p.add_argument("--include-smoke", action="store_true")
        if name == "chain":
            p.add_argument("--out", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--out", default="phase18/audit")
    p.add_argument("--allow-dirty", action="store_true", help="for development only; the report records it")
    args = ap.parse_args()
    return {"list": cmd_list, "plan": cmd_plan, "chain": cmd_chain, "analyze": cmd_analyze, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
