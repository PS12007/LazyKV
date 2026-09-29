# Contributing to LazyKV

LazyKV is a measurement study first and a runtime second. Contributions are welcome, and the
most useful ones are the ones that make the study harder to fool: a reproduction on different
hardware, a policy measured on the same terms as the existing ladder, or a bug in how something
is measured.

## Ground rules

These are the project's rules (see [`CLAUDE.md`](CLAUDE.md)), and they apply to contributions too.

1. **No hand-typed numbers in docs.** Prose lives in `docs/templates/*.md.tmpl`; every number is a
   placeholder resolved from `results/**/metrics.json` by `scripts/render_docs.py`. Edit the
   template, re-render, and commit both. `tests/test_docs_fresh.py` fails if a rendered doc drifts
   from its data.
2. **No unverified citations.** A paper cited anywhere must be listed in
   [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) with its arXiv ID or venue.
3. **Negative results are results.** If a change makes a number worse, report it; do not tune an
   experiment until it looks good.
4. **No new runtime dependencies without discussion.** The environment targets a Blackwell
   (sm_120) GPU and is easy to break. Open an issue first.
5. **Small commits, conventional-commit messages** (`feat(tier): ...`, `fix(quality): ...`,
   `docs(findings): ...`), one logical change each.

## Setting up

Python 3.12 and [uv](https://github.com/astral-sh/uv). On an NVIDIA GPU:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock `
  --index-url https://download.pytorch.org/whl/cu130 `
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.venv\Scripts\python.exe -m pytest
```

Without a GPU the CUDA tests skip and the rest (analysis, statistics, docs freshness, CPU paths)
still run; that is what CI checks. The 3B stress model additionally needs the `quant` extra
(`bitsandbytes`, `accelerate`).

## Running experiments

Each phase has a YAML under `configs/` and a reproduce section in the [README](README.md).
Benchmarks that take more than a minute or two should be launched detached with output to a log
file under `logs/`, and must not run while you edit tracked files: the drivers record the commit
they ran at. `--quick` gives a smoke run whose output is never reported.

## Where contributions help most

- **Reproductions on other GPUs.** Every result here comes from one RTX 5060 Laptop GPU. A
  Phase 0 run (`scripts/00_feasibility.py`) plus one ladder sweep on different hardware, submitted
  as a new `results/` directory with its `metrics.json`, is the single most valuable addition.
- **A new policy.** GPU-only policies subclass `Policy` in `lazykv/policies.py`; tiered ones build
  on `lazykv/tiered.py`. Wire it into `lazykv/sweep.py` so it gets the same budget sweep, prompts
  and quality metrics as every other rung.
- **Measurement bugs.** The [research log](docs/RESEARCH_LOG.md) lists the ones already caught.
  Another one is worth an issue even without a fix.

## Out of scope

Batching and prefix caching, multi-GPU, trained predictors, custom CUDA or Triton kernels, and
throughput races against serving engines. See the README's limitations section.
