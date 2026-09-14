# LazyKV project brief (v2)

This is the reference brief. `CLAUDE.md` holds the hard rules; this file holds the
reasoning and the plan. Re-read it at the start of every phase.

> **Status of numbers in this file.** Every quantity below (bandwidths, ratios) is a
> *pre-measurement hypothesis* written before Phase 0, kept verbatim so later phases can
> be checked against it. It is not a result. The original brief's hand-written KV-size
> table was removed from B3 on purpose: those values are now generated from real
> `config.json` files in `docs/KV_MEMORY_MODEL.md`, and measured values live in
> `SYSTEM_INFO.md`.

## B0. What this project is, and what it is not

LazyKV is a middleware layer that treats GPU VRAM as a cache over a larger logical KV
cache backed by CPU RAM. The contribution is the memory management layer and a fair,
reproducible empirical comparison of residency policies under a hard consumer-GPU
budget. It is not a new model, not a new quantization algorithm, and not a claim of
novelty over existing systems.

**Positioning, stated honestly from day one:** several published systems already do
pieces of this, some of them well. InfiniGen, Quest, ShadowKV, H2O, SnapKV,
StreamingLLM, FlexGen and LMCache all overlap. What is genuinely underserved is a
careful, open, apples-to-apples study of these policies on one constrained consumer
GPU, with honest quality measurement and full ablations. That is the contribution.
Frame the README and paper that way. Do not write first, novel, or state-of-the-art
anywhere unless a measurement supports it.

## B1. The design decision that determines whether this project works

This is the most important section. Read it before writing any architecture.

Standard dense attention at decode step t needs **every** key and value for the layer.
If a block lives in CPU RAM, one of three things must be true:

- **(a) Fetch it back before the layer runs.** Dense attention touches all blocks every
  step, so there is no locality to exploit. You would stream the entire non-resident KV
  over PCIe on every single token. Do the arithmetic before you build anything: VRAM
  bandwidth on this GPU is in the hundreds of GB/s, PCIe on a laptop dGPU is typically
  x8 Gen4, roughly 12 to 14 GB/s achievable with pinned memory. That is a 20x to 40x
  gap. Decode attention is memory-bandwidth bound with arithmetic intensity near 1, so
  option (a) is not a policy problem, it is an arithmetic impossibility. Prefetching
  cannot hide a transfer that is 30x longer than the compute it overlaps.

- **(b) Do not attend to it at all.** This is eviction. It is cheap and fast and it
  changes model output. Quality must then be measured, not asserted.

- **(c) Compute partial attention where the data is, and merge.** Softmax attention is
  associative under online rescaling (the FlashAttention / ring-attention merge). You
  can compute attention over GPU-resident blocks on the GPU, compute attention over
  CPU-resident blocks on the CPU, and merge the two partial outputs with their
  log-sum-exp normalizers. This is exact, zero quality loss, and moves only O(head_dim)
  bytes back instead of O(block_bytes). The cost moves to CPU compute and becomes a
  genuinely interesting tradeoff.

**Therefore the viable architecture is (b) and (c) combined, plus block-level sparsity:**

> Per layer, per decode step, cheaply estimate which blocks could matter for the
> current query, keep only those GPU-resident, and handle the rest either by skipping
> them (approximate) or by partial-attention-and-merge (exact).

The cheap estimator that makes this work is block-level query-aware scoring: store a
per-block elementwise min and max over its keys, compute an upper bound on q·k for the
block in O(head_dim) instead of O(block_len · head_dim), and rank blocks by that bound.
This is Quest's criterion and it is cheap and effective. Prefetching then becomes
meaningful, because the estimator for layer L+1 can be evaluated using the hidden state
before layer L+1's attention runs, giving you a real prefetch window. That is
InfiniGen's core insight.

**Phase 1 deliverable is to validate or refute this.** If the microbenchmarks say the
gap is worse than assumed, change the research question rather than building a system
on top of a false premise.

## B2. Phase 0: go/no-go microbenchmarks, before any system code

Write `scripts/00_feasibility.py`. It must measure, on the actual machine:

1. **PCIe H2D and D2H bandwidth**, pageable vs pinned, at transfer sizes from 64 KB to
   256 MB. Report achieved GB/s per size. Find the size below which per-transfer
   overhead dominates. This number sets the minimum sane block size.
2. **VRAM bandwidth**, via a large device-to-device copy or a bandwidth-bound kernel.
3. **The ratio of 2 to 1.** This is the single number the whole project lives under.
4. **Async overlap reality check.** Launch a compute kernel of known duration on one
   stream and a pinned H2D copy on another. Measure with CUDA events whether wall time
   is max(compute, copy) or compute+copy. Laptop GPUs sometimes have one copy engine;
   verify `asyncEngineCount` from device properties rather than assuming.
5. **CPU-side attention throughput.** Time a batched float32 GEMV of shape
   (n_kv_heads, block_len, head_dim) against a query, using numpy or torch on CPU with
   threads set explicitly. This determines whether option (c) above is affordable.
6. **NVMe sequential and random read bandwidth**, only if you still want the SSD tier
   after seeing 1 and 5.

Write the results into `SYSTEM_INFO.md` alongside OS, CPU model, RAM size and speed,
GPU name, VRAM, driver, CUDA, Python, torch, transformers versions, and
`torch.cuda.get_device_properties()` in full.

Then write a short paragraph in `docs/RESEARCH_LOG.md` stating which of the three
designs in B1 the measurements support. Stop and report.

## B3. Model selection: choose by KV geometry, not parameter count

Parameter count is almost irrelevant here. What matters is KV bytes per token:

```
kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
```

Aggressive GQA makes the KV cache small and makes the entire project uninteresting.
Compute this from `config.json` for every candidate before downloading weights, and put
the table in `docs/KV_MEMORY_MODEL.md`. Verify the candidate geometries against the real
configs rather than trusting any table written from memory. Candidates:

- Llama-3.2-1B-Instruct
- Llama-3.2-3B-Instruct
- Qwen2.5-1.5B-Instruct
- Phi-3-mini (no GQA)

Recommended pair:

- **Primary: Llama-3.2-1B-Instruct in bf16.** Its weights leave most of the 8 GB free,
  so you can sweep GPU KV budgets from 100% down to 6% at 32K to 64K context without
  fighting OOM. Sweeping the budget is the experiment, so headroom is the resource you
  need most.
- **Stress: Llama-3.2-3B-Instruct with NF4 weights** where KV dominates aggressively.
  Note in the writeup that 4-bit weights change the decode compute profile, which
  shifts the compute/transfer overlap balance. That is a caveat, not a problem.

Llama 3.2 repos are gated on Hugging Face. If access is a hassle, Qwen2.5-1.5B or
SmolLM2-1.7B are fine substitutes; just recompute the KV table.

**Practical trap:** at 64K prefill, computing logits for all positions is a vocab-sized
tensor per token and will OOM instantly. Chunked prefill with logits computed only for
the final position is mandatory, not optional. Implement it in Phase 1 of the baseline.

## B4. Integration point with HuggingFace Transformers

Do not rewrite a transformer. Do not monkeypatch broadly. The clean seam is:

- Subclass `transformers.Cache` as `LazyKVCache`, which owns block storage, tiering and
  bookkeeping, and exposes `update(key_states, value_states, layer_idx, ...)`.
- Override the attention forward for exactly one model class, isolated in
  `lazykv/models/patch_llama.py`, so that it asks the cache which blocks to use and
  performs gathered or partial attention instead of a single dense SDPA call.
- Everything else in Transformers stays untouched.

Pin the transformers version in the lockfile. This internal API moves between releases
and an unpinned upgrade will silently break the patch.

`ModelBackend` abstraction is fine, but write it after the second backend exists, not
before. One implementation behind an interface is not modularity, it is overhead.

## B5. Related work to survey (verify each with a real search)

Produce `docs/RELATED_WORK.md`. For each entry record: problem solved, KV storage
layout, movement mechanism, paging yes/no, CPU tier yes/no, SSD tier yes/no, quantized
KV yes/no, eviction policy, prefetch policy, main tradeoff, and one sentence on how
LazyKV differs. Candidate list, not exhaustive, and some names may be misremembered so
verify before citing:

- **Serving systems:** vLLM (PagedAttention), SGLang (RadixAttention), TensorRT-LLM,
  llama.cpp, LMDeploy, NVIDIA Dynamo, LMCache, Mooncake, AttentionStore / CachedAttention
- **Offloading:** FlexGen, HeadInfer, DeepSpeed-Inference / ZeRO-Inference
- **Eviction and sparsity:** H2O, StreamingLLM (attention sinks), Scissorhands, TOVA,
  SnapKV, PyramidKV, FastGen, Locret, ArkVale
- **Query-aware block sparsity and prefetch:** Quest, InfiniGen, ShadowKV, MagicPIG,
  RetrievalAttention, PQCache
- **KV quantization:** KIVI, KVQuant, Atom, GEAR
- **Memory mechanism:** vAttention (CUDA virtual memory for KV)
- **Evaluation:** RULER, LongBench, InfiniteBench, needle-in-a-haystack

Then write a short section titled Delta, stating plainly what LazyKV adds. If, after
reading, the honest answer is that it adds a reproducible cross-policy study on
constrained hardware rather than a new mechanism, write exactly that. It is a defensible
and interesting contribution and pretending otherwise is worse.

## B6. Metrics, measured properly

Per-experiment JSON must record memory (peak and mean VRAM via `torch.cuda`
max_memory_allocated and max_memory_reserved AND nvidia-smi process memory, since they
differ and the difference is itself informative; KV bytes GPU-resident; KV bytes in
host RAM; GPU KV residency fraction), performance (TTFT, per-token decode latency
distribution not just mean, tokens/sec, transfer time, GPU idle time), cache behavior
(hit rate, miss rate, prefetch hit rate, wasted prefetch rate, eviction rate, migration
rate, per-block residency duration), PCIe (bytes each direction, transfer count, mean
size, measured overlap fraction), quality (see B7), and overhead (manager wall time as
a fraction of decode, Python bookkeeping cost).

Measurement discipline, which matters more than any policy detail:

- Time GPU work with **CUDA events**, not `time.time()`, and synchronize deliberately.
- **Warm up** (at least 3 iterations discarded), then report **median and IQR over >=5
  repeats**, never a single run.
- **This is a laptop.** It thermally throttles. Log GPU clock, temperature and power
  throughout every benchmark. **Interleave conditions** (A,B,C,A,B,C) instead of running
  all of A then all of B, and randomize order across repeats. Otherwise the last
  condition in the matrix will look slower for thermal reasons and you will publish a
  thermal artifact as a finding. Include a plot of clock vs time for one long run as
  evidence you controlled for this.
- Record `PYTORCH_CUDA_ALLOC_CONF` and test `expandable_segments:True`, since
  fragmentation interacts strongly with block allocation.
- Note explicitly that determinism is not guaranteed across different attention paths;
  use greedy decoding and fixed seeds, and report agreement rather than assuming
  bitwise equality.

## B7. Quality measurement

Weak quality evaluation is the fastest way for this project to stop being credible.
Required, in order of cost:

1. **Teacher-forced divergence from the full-cache baseline.** On identical long
   prompts, compare per-token next-token distributions: top-1 agreement rate and mean
   KL divergence. This is cheap, sensitive, and catches degradation that greedy output
   comparison hides.
2. **Retrieval accuracy:** needle-in-a-haystack across needle depths (0%, 25%, 50%,
   75%, 100% of context) and context lengths. Use RULER's task definitions if feasible,
   at minimum single-needle and multi-needle variants.
3. **Perplexity** on long documents (PG-19 or similar), reported per context length.
4. **Task accuracy** on a couple of LongBench subsets if time allows.

Never write quality preserved. Write the delta with a confidence interval.

## B8. Experiments

The headline experiment is a **Pareto frontier**: GPU KV bytes resident on the x-axis,
with two y-axes across separate plots, tokens/sec and NIAH accuracy, one curve per
policy. Every policy gets the same budget sweep (100%, 75%, 50%, 25%, 12.5%, 6.25%) at
a fixed context length.

Single headline number, better than context-per-GB because it is not gameable:

> Minimum GPU KV budget at which policy P retains >= 99% of baseline NIAH accuracy at
> 32K context, and the tokens/sec at that point.

Policy ladder, each added only once the previous is measured:

1. Full GPU KV (reference, exact)
2. Sliding window + attention sink (the strong cheap baseline; many papers skip it and
   it often wins, so include it honestly)
3. LRU block eviction
4. Attention-score eviction (H2O style, using observed scores)
5. Query-aware block selection (Quest style min/max bounds), no offload
6. 5 + CPU tier with synchronous fetch
7. 6 + async prefetch with double buffering and pinned memory
8. 7 + mixed precision by tier (fp16 hot, int8 warm/cold)
9. 8 + exact partial-attention merge for non-resident blocks, if B2 says CPU compute
   affords it

Ablations strip one component at a time from the best configuration. Block sizes 16, 32,
64, 128, 256 get swept once, at one policy, and the chosen size is justified from the
PCIe efficiency curve in Phase 0 rather than assumed. 8-token blocks are almost
certainly below the PCIe efficiency knee; include them only to show that.

Context lengths: 4K, 8K, 16K, 32K, and 64K if it runs. Do not list a context length you
did not execute.

## B9. Scope kill list

Explicitly out of scope for v1. Write this in the README under Limitations rather than
leaving it ambiguous:

- Multi-request batching, continuous batching, cross-request prefix caching
- Multi-GPU anything
- Training or fine-tuning any predictor
- Custom CUDA or Triton kernels
- Beating vLLM on throughput
- **SSD tier.** On a single-stream workload, NVMe is slower than the PCIe path you have
  already shown to be the bottleneck, so it can only add capacity, not speed. Either cut
  it or implement it as a clearly-labeled capacity-only demonstration with one honest
  measurement showing it is not useful for latency. Do not spend a week on it.

## B10. Suggested timeline

Six weeks of part-time work, with a shippable artifact at week three:

- Week 1: Phase 0 microbenchmarks, SYSTEM_INFO, RELATED_WORK, KV_MEMORY_MODEL, baseline
  generation with chunked prefill and trustworthy timing.
- Week 2: block abstraction, GPU-only managed cache, CPU backing store with pinned
  pools, LRU, first real budget sweep.
- Week 3: query-aware selection, NIAH harness, first Pareto plot. **Repo is now
  presentable.** Write the README here, not at the end.
- Week 4: async prefetch, streams, overlap instrumentation, thrash detection.
- Week 5: mixed precision, adaptive controller, ablations.
- Week 6: failure analysis, profiling evidence, paper draft.
