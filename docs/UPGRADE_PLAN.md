> **Status of this file.** This is the owner's upgrade brief, kept verbatim as written on 2026-09-30.
> Its quantities are the competitors' published claims or planning targets, not LazyKV measurements.
> Step 0's audit, including where this plan's premises do not match the repository, is in
> [GAP_TABLE.md](GAP_TABLE.md) and [STATUS.md](STATUS.md).

# LazyKV upgrade plan (brief for Claude Code)

How to use: put this file in the root of the LazyKV repo as `docs/UPGRADE_PLAN.md`, then tell Claude Code:

> Read docs/UPGRADE_PLAN.md and do Step 0 only. Stop and show me the gap table.

After that, pick features from the menu one at a time and tell Claude Code which ID to build (for example: "Build A3 next").

---

## 1. Context for Claude Code

LazyKV is my existing project: a runtime that treats the KV cache of a long-context LLM as tiered virtual memory across GPU VRAM, CPU RAM and optionally SSD. A lot of it is already built. This plan is about adding features that make it clearly stronger and more distinctive than the projects that have appeared in the same space, not rebuilding it.

### Competitors you must know about

**TierKV** (https://github.com/arczhi/llama-tierkv), a llama.cpp fork, Linux, reference machine RTX 5060 Ti 16 GB with a 27B model:
- Three tiers: VRAM window (32 to 49K tokens), host RAM store, SSD snapshots for session resume.
- Pages are 64 tokens. Pages move only at turn boundaries, never per token, which keeps decode flat (about 22.7 tok/s from 8K to 262K, no speculation).
- Recall selectors: IDF-weighted word overlap, and Quest-style page-sparse attention using per-page per-channel K min/max summaries scored against the captured query. A hybrid of both.
- Head protection (first 2,048 positions never evicted), rate-limited staging, batched block gather, lazy host allocation, optional Q4_0 host store with Q5_0 window.

**TierKV's own stated limitations** (these are our openings):
1. MTP speculative decoding cannot be combined with 256K context on their implementation, because the draft context must cover the full position range. Windowed draft context is only on their roadmap.
2. Retrieval coverage: when the VRAM window is smaller than the working set, their selectors do not restore enough of the agent's own recent history, and agents can lose the thread.
3. Their sparse attention is a gather into ordinary KV cells, not a custom paged attention kernel.
4. The host store preallocates rows for the full max context.
5. Benchmarks are single-run. Linux only, scripts are shell scripts.
6. Their design moves pages only at turn boundaries, which suits multi-turn agents but does not help a single long-document request where the needed pages change mid-generation.

**KVMem** (https://github.com/kvmem/kvmem-llama.cpp, llama.cpp discussion #28894): another fork with a 262K logical context and a 32K retrieved window, and a windowed MTP pool with state replay.

**Adaptive KV Cache Streaming** (llama.cpp discussion #28216): evicts part of each layer's KV from VRAM and streams it back from RAM through a ring buffer, aiming for upstream.

### LazyKV's positioning (what every feature should push toward)

> LazyKV is virtual memory for the KV cache, designed and measured on an 8 GB laptop GPU, Windows-native, with principled, benchmarked page replacement policies.

Three pillars:
1. **Small-GPU first:** 8 GB laptop cards, not 16 GB desktop cards.
2. **Research rigor:** policies compared against an optimal oracle, multi-run statistics, reproducible numbers.
3. **Works where the others are weak:** single long-document requests, mid-generation recall, and coverage of the model's own recent output.

## 2. Ground rules

- **Hardware:** Windows 11, RTX 5060 Laptop GPU (8 GB VRAM, Blackwell), my laptop's RAM and NVMe. No extra GPUs, no paid cloud.
- **Windows-native only. No WSL, no Docker.** Builds use CMake + MSVC + the CUDA toolkit, or native Python on Windows. Scripts are Python or PowerShell, never bash-only.
- **No large-model training.** Small learned components are allowed only if they train on this laptop in under an hour.
- **No invented numbers.** Every number in docs or README comes from a script in this repo I can rerun. If something was estimated rather than measured, label it as an estimate.
- **Every new feature is behind a flag** so it can be ablated, and the default behavior of existing features does not change.
- **Tests for every feature** (correctness first: page bookkeeping, sinks never evicted, round-trip error bounds, bit-identical output when a feature is enabled but never triggers).
- **Small commits**, each message stating what changed and the measured effect.
- Keep `docs/LOG.md` updated with decisions, measurements and dead ends.
- Ask me before: big new dependencies, architecture changes, deleting code, or any run that takes more than about 2 hours.

## 3. Step 0: gap audit (do this first, then stop)

1. Read the entire repo. Identify the runtime LazyKV is built on (llama.cpp fork, PyTorch, custom), the page/block size, how tiers are implemented, what the eviction and recall policies are, how data moves between tiers (sync or async, pinned memory or not), what benchmarks and tests exist, and what is half-finished.
2. Write `docs/STATUS.md` describing the current system accurately.
3. Fill in this gap table in `docs/GAP_TABLE.md`, one row per feature ID in section 4:

| ID | Feature | Already in LazyKV? (yes / partial / no) | Evidence (file, function) | Effort estimate | Notes |
|----|---------|-----------------------------------------|---------------------------|-----------------|-------|

4. Write a short "honest comparison" section: where LazyKV is already better than TierKV, where it is behind, where it is equal.
5. Stop and show me STATUS.md and GAP_TABLE.md. Do not start building anything yet.

## 4. Feature menu

Priority: **A** = core differentiators, build first. **B** = strong upgrades. **C** = polish and usability. Skip anything the gap table marks as already done.

---

### A1. Reproducible benchmark suite (Windows-native)

**Why:** Competitors publish single-run numbers on 16 GB cards. A rigorous suite on an 8 GB laptop is a differentiator by itself, and every other feature needs it to prove its value.

**Build:**
- One entry point: `python bench/run.py --config bench/configs/<name>.yaml`
- Systems: stock llama.cpp (full KV on GPU), stock llama.cpp with quantized KV (`--cache-type-k/-v q8_0` and `q4_0`), stock llama.cpp with `--no-kv-offload`, and LazyKV with any combination of feature flags. Use official prebuilt Windows CUDA binaries of llama.cpp for the stock baselines where possible.
- Models that fit 8 GB with room for a KV window: one ~7 to 8B instruct model and one ~3 to 4B instruct model, both GGUF Q4_K_M (or the format LazyKV uses). Pin exact files and hashes in the config.
- Context sweep: 4K, 8K, 16K, 32K, 64K, 128K, and beyond if it runs.
- Performance metrics: max context before OOM or unusable speed (define unusable, for example below 3 tok/s), prefill tok/s, time to first token, decode tok/s median and p95, peak VRAM, peak host RAM, PCIe bytes per decoded token (instrumented in LazyKV).
- Quality metrics: needle-in-a-haystack at multiple depths and multiple needles, perplexity on a fixed long document, a small RULER or LongBench subset (3 to 4 tasks), and a multi-turn conversation trace where later questions depend on early turns.
- Statistics: warm-up runs, fixed seeds, at least 3 repetitions, report median and min/max or 95% interval.
- Environment logging: GPU driver, CUDA version, Windows power mode, plugged in or on battery, GPU temperature and clocks during the run (laptops throttle, so record it).
- Output: `results/<date>/<run>.json`, plots in `results/<date>/plots/`, and an auto-generated `results/<date>/SUMMARY.md` table.

**Done when:** one command reproduces the full baseline table on my laptop, and rerunning it gives numbers within the reported spread.

---

### A2. Policy plug-in interface + access trace recorder

**Why:** Turns LazyKV from "one clever policy" into a platform where replacement policies can be compared fairly. This is what makes it read as research, and it is how contributors add value.

**Build:**
- A clean interface, for example `Policy.on_insert(page)`, `on_access(page, score)`, `choose_victims(n)`, `choose_recalls(query_summary, budget)`.
- Port existing LazyKV logic into this interface without changing behavior (verify with a bit-identical or near-identical output test).
- Built-in policies: LRU, LFU, CLOCK, sink+recent window (StreamingLLM), H2O-style accumulated attention, Quest-style min/max upper bound (TierKV's approach, reimplemented from the idea, not copied), and LazyKV's current policy.
- A trace recorder that logs, per decode step and per layer (or sampled layers), which pages were needed and their attention mass. Store compactly (for example numpy arrays, chunked). Keep overhead low and make it opt-in.

**Done when:** I can switch policies with a flag, and a recorded trace from a real run can be saved and replayed.

---

### A3. Offline trace-replay simulator with an optimal oracle

**Why:** This is the most impressive research piece and costs almost no GPU time. Classic OS research measures page replacement against Belady's optimal algorithm (evict the page used farthest in the future). Nobody in this space reports how close their policy is to optimal. LazyKV would.

**Build:**
- `python sim/replay.py --trace traces/<name> --policies lru,quest,h2o,lazykv,oracle --budget 4096,8192,16384`
- Belady/OPT oracle using future knowledge from the trace.
- Metrics per policy and budget: page fault rate, attention mass captured (fraction of true attention mass that lands on resident pages), PCIe bytes moved, and a correlation check between simulated mass captured and real end-task quality from A1.
- Plots: fault rate vs VRAM budget for all policies, with the oracle as the lower bound.
- A write-up `docs/POLICY_STUDY.md` with findings, including where simple policies are surprisingly good.

**Done when:** the plot exists, is reproducible from saved traces, and the gap between LazyKV's policy and the oracle is quantified.

---

### A4. Asynchronous mid-generation prefetch

**Why:** TierKV moves pages only at turn boundaries. For a single long-document request (summarize this 100K-token PDF, answer a question about it), the needed pages change during generation, and turn-boundary staging does not help. Overlapping transfers with compute is a clear technical win and a clear differentiator.

**Build:**
- Dedicated CUDA copy stream(s), pinned (page-locked) host buffers, double buffering.
- While layer L computes, prefetch the predicted pages for layer L+1 (or for the next decode step) using the previous step's query and page summaries.
- Rate limiting and a PCIe budget so prefetch never starves compute.
- Windows specifics: verify pinned memory allocation limits and behavior under WDDM, document any quirks found. Note that laptop GPUs may run PCIe at reduced width or speed; measure actual host-to-device bandwidth with a microbenchmark and record it.
- Instrumentation: prefetch hit rate, wasted prefetch bytes, stall time waiting on transfers.

**Done when:** on single long-document tasks, decode tok/s improves over synchronous recall with no quality loss, shown in an A1 ablation.

---

### A5. Working-set-aware selector (fix the coverage problem)

**Why:** TierKV admits that when the window is smaller than the working set, the model loses track of its own recent history. A selector that explicitly models the working set fixes the exact weakness they document.

**Build:**
- Track per-page signals: recency of last high-attention access, exponential moving average of attention mass, Quest-style upper bound for the current query, whether the page contains the model's own generated output or tool output, and whether it was recently recalled and then used.
- Combine them into one score (start with a weighted sum, tune weights with the A3 simulator, not by hand-waving).
- Protect a "recent self-history" budget so the model never loses what it just said.
- Compare against Quest-only and IDF-only selectors in A3 and A1.

**Done when:** at a window smaller than the working set, the multi-turn trace in A1 keeps answering correctly more often than the Quest-only and IDF-only baselines.

---

### B1. Progressive precision tiers

**Why:** TierKV already stores host pages at Q4_0. LazyKV can go further and make precision a function of temperature, like a memory hierarchy: hot pages high precision, warm pages lower, cold pages lowest, with upgrade on recall.

**Build:**
- Example ladder: VRAM window at fp16 or q8, RAM tier at int4 with per-block scales, SSD tier at 2 to 3 bit. Treat keys and values separately (keys are often more sensitive).
- When a page is recalled, dequantize into the window; optionally keep a higher-precision copy if it is recalled repeatedly.
- Measure quality vs memory curves (NIAH, perplexity) for each ladder configuration.
- If my Triton quantized-KV kernel work is relevant, reuse its quantization scheme and cite it in docs.

**Done when:** a table shows memory saved per tier and quality impact, and one recommended ladder for 8 GB.

---

### B2. 8 GB VRAM auto-tuner

**Why:** The biggest practical pain on small GPUs is configuration. A tool that picks settings automatically makes LazyKV actually usable by other people.

**Build:**
- `lazykv tune --model <path> --target-context 128k` reads free VRAM, model size, layer count, KV head dims, and computes: VRAM window size, KV precision per tier, number of GPU layers, host RAM budget, and prefetch budget.
- Optional live adaptation: shrink the window if VRAM pressure appears (other apps, Windows desktop compositor), grow it when free.
- Print the reasoning step by step: weights size, CUDA and runtime overhead, remaining VRAM, bytes per token of KV for this model (layers x KV heads x head dim x 2 for K and V x bytes per element), and the resulting window in tokens.

**Done when:** on my laptop it produces a working config for both benchmark models without manual tweaking, and it never OOMs at startup.

---

### B3. Live observability view

**Why:** Makes the system understandable and gives the README its best visual. A virtual memory system should show its page table.

**Build:**
- A terminal UI (or a small local web page) showing: a strip map of all pages colored by tier (VRAM, RAM, SSD), page faults per second, prefetch hit rate, PCIe MB/s, VRAM and RAM usage, current decode tok/s.
- A recording mode that exports a short GIF or MP4 for the README.

**Done when:** I can watch a 100K-token request move pages between tiers live and record it.

---

### B4. Page-sparse attention kernel (stretch, high impact)

**Why:** TierKV gathers selected pages into ordinary KV cells and runs standard attention. A kernel that attends directly over a page table (selected pages plus recent window) skips the gather copy. This ties into my Triton KV kernel project.

**Build:**
- Decide based on the runtime: if LazyKV is PyTorch-based, write it in Triton (check that Triton works natively on Windows with the community Windows build before committing; if it does not, use CUDA C++ via a PyTorch extension). If LazyKV is llama.cpp-based, write a CUDA kernel in the ggml backend.
- Inputs: query, page table of selected page indices, paged K/V storage, recent window. Support my KV precision formats.
- Validate numerically against a reference attention implementation, then benchmark vs gather + standard attention.

**Done when:** correctness tests pass and an ablation shows the effect on decode latency at long context. If it is slower, document why; that is still a valid result.

---

### B5. Long context + speculative decoding together (stretch)

**Why:** TierKV cannot combine speculative decoding (MTP) with very long context because the draft context must cover the full position range. Making these compatible would be a headline feature.

**Build:**
- First check what speculative decoding support LazyKV's runtime has. If none, add a small draft model approach (same tokenizer family) before attempting MTP.
- Windowed draft context: the draft model sees only the recent window plus pinned head, with positions handled correctly; verify outputs stay identical to target-only decoding in greedy mode.
- Measure tok/s and acceptance rate at 32K, 64K, 128K.

**Done when:** speculative decoding runs at long context on 8 GB with measured speedup and identical greedy outputs.

---

### C1. Session snapshots to SSD

Parity feature (TierKV has it). Save the RAM tier and metadata to disk so a restarted process resumes a long conversation without reprocessing. Include versioning, integrity checks, and model-hash validation. Skip if already present.

### C2. OpenAI-compatible server + CLI

`lazykv serve --model <path>` exposing an OpenAI-compatible chat endpoint, so people can point existing tools at it. A simple CLI chat mode for demos. This is what turns a research repo into something people actually use.

### C3. Tiny learned page scorer (optional, Locret-inspired)

A very small scoring head (trains on this laptop in under an hour) that predicts page importance from cheap features, plugged in as a policy via A2 and evaluated in A3. Only build if A3 shows a meaningful gap between heuristics and the oracle.

### C4. Dynamic host store growth

Allocate host RAM in chunks as the context grows instead of preallocating for the maximum context (a stated TierKV limitation). Measure RAM saved for typical conversation lengths.

## 5. Comparing against TierKV without WSL

- First, time-box an attempt (about 1 hour) to build TierKV natively on Windows with CMake + MSVC + CUDA. It is a llama.cpp fork, so a native build may work. Its scripts are bash, so translate the run commands to PowerShell or Python.
- If the native build works: run it inside the A1 suite with clearly documented settings.
- If it does not: do not use WSL. Compare on design and on their published numbers only, in a table clearly labeled "different hardware, not a head-to-head" (their reference is an RTX 5060 Ti 16 GB with a 27B model; ours is an RTX 5060 Laptop 8 GB with 3 to 8B models). Never present cross-hardware numbers as if they were measured side by side.
- Either way, the strongest comparisons are the ones fully under our control: stock llama.cpp baselines, LazyKV feature ablations, and the A3 policy study, which reimplements competing policies from their published ideas.

## 6. README and presentation (after A1 to A5 are done)

README structure:
1. One-line pitch (the positioning statement).
2. Hero chart: decode tok/s and NIAH accuracy vs context length on an 8 GB laptop GPU, LazyKV vs stock llama.cpp variants.
3. Three bullets on what makes it different (small-GPU first, oracle-benchmarked policies, mid-generation prefetch).
4. Live tier-map GIF from B3.
5. Windows quickstart that works in under 10 minutes, plus `lazykv tune`.
6. Full results table, ablation table, policy study plot.
7. Related work: TierKV, KVMem, the llama.cpp streaming discussion, StreamingLLM, H2O, Quest, SnapKV, Locret, with a fair one-line description each.
8. Limitations, stated honestly.
9. How to reproduce every number.

Also write `docs/UPSTREAM.md`: which parts could be proposed to llama.cpp, how they relate to discussions #28216 and #28894, and a draft comment I could post there with our measured results.

## 7. Suggested order

1. Step 0 audit, then A1 (benchmarks first, always).
2. A2 then A3 (policy platform and oracle study, cheap and impressive).
3. A5 (uses A3 to tune), then A4 (prefetch).
4. B2 and B3 (usability and visuals).
5. README pass and a first public release.
6. B1, B4, B5, C-items as time allows.

After each item: run the benchmark, update GAP_TABLE.md, update LOG.md, commit, and stop to show me results before starting the next item.