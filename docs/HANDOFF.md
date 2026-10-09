# Handoff

Where the work stands, for the next session. No numbers here (rule 1): every result is in the phase
reports and `docs/RESEARCH_LOG.md`, rendered from `results/**/metrics.json`.

## State

- `main` is clean and pushed; no worktrees, nothing running.
- Latest phases: [22](phases/PHASE_22.md) (snapshot restore near disk speed),
  [23](phases/PHASE_23.md) (zero-copy tier: rung 5's speed at 6.25%, loses to rung 6 at 50%),
  [24](phases/PHASE_24.md) (float32 ranking bound: negative result, default stays bf16).
- Status and gap table: [STATUS.md](STATUS.md), [GAP_TABLE.md](GAP_TABLE.md).

## Next options (owner picks; rule 8)

1. **16-token blocks under the zero-copy tier.** Phase 4 found smaller blocks keep more NIAH answers,
   but rung 6 paid per-pair host work for them. The zero-copy tier has no per-pair host work, so the
   quality gain may come nearly free. Needs a speed run on an idle machine (the owner asked to be
   told before timing runs), plus a NIAH run.
2. **Zero-copy tier at 64K and 130K** through tiered prefill (Phase 17). Its cost grows with K, so
   this locates where it stops paying off at long contexts.
3. **3B multi-turn** (recommended after Phase 21, deferred by the owner). Needs the D: drive
   mounted: the 3B weights are only at `D:\CompSci\lazykv-data\models`.

## How to run things (see also the README reproduce table)

- Everything with `.venv\Scripts\python.exe`. Long runs: from a pinned worktree
  (`git worktree add ../LazyKV-gateN HEAD`, copy `data/corpus/*.txt` in), launched through WMI so
  they outlive the session, with a `.cmd` chain that writes a status file.
- Each phase: config with questions fixed before any run → smoke test (not reported) → run →
  `scripts/analyze_phaseN.py` → template in `docs/templates/` → `scripts/render_docs.py` →
  add the analysis to `configs/suite.yaml` → commit in small steps → push.
- `pytest` prints no summary line (addopts `-q` twice); check for `F`/`E` or pass `-p no:cacheprovider`.
