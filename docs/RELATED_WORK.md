# Related work

Survey for LazyKV Phase 0 (`docs/BRIEF.md` §B5). **Every entry below was verified to exist
by a live lookup on 2026-09-14**: arXiv entries by fetching their `arxiv.org/abs/<id>`
page and reading the title and abstract, and non-arXiv systems through their official
documentation or repository. Claims about each system come from its abstract or
official docs. Where a detail was not verifiable that way, the cell says so instead of
guessing.

**Venues are deliberately omitted.** Several of these papers appeared at conferences, but
this pass did not verify a venue for each one. The arXiv ID is the verified identifier.
Reported speedups from the papers are also omitted. They were measured on other
hardware and are not comparable to anything measured here.

## Verification notes on the candidate list

Every name in the brief's list resolved to a real system or paper. Five names needed
clarification:

| Name in brief | What it actually is |
|---|---|
| AttentionStore / CachedAttention | One paper, arXiv 2403.19708. v1 was titled *AttentionStore*; the current title uses *CachedAttention*. |
| TOVA | The policy name inside *Transformers are Multi-State RNNs* (arXiv 2401.06104). |
| FastGen | The method in *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs* (arXiv 2310.01801). |
| DeepSpeed-Inference / ZeRO-Inference | Two related artifacts: the DeepSpeed Inference paper (arXiv 2207.00032) and the ZeRO-Inference blog release that added KV-cache offloading (deepspeed.ai, 2023-09-12). |
| Needle-in-a-haystack | A GitHub test harness (`gkamradt/LLMTest_NeedleInAHaystack`), not a paper. RULER (below) formalizes and extends it. |

No entry had to be removed.

## At a glance

✔ = a core mechanism of the system · ◐ = supported in a limited or optional form · — = not a mechanism of the system

| System | Paging / blocks | CPU tier | SSD tier | Quantized KV | Evicts / skips KV | Query-aware selection | Prefetch / overlap | Attention on CPU |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| vLLM (PagedAttention) | ✔ | ◐ swap on preemption | — | ◐ | — | — | — | — |
| SGLang (RadixAttention) | ✔ | — | — | — | ◐ LRU of cached prefixes | — | — | — |
| TensorRT-LLM | ✔ | ◐ offload of reusable blocks | — | ✔ | ◐ reuse-cache eviction | — | — | — |
| llama.cpp | — | ◐ all-or-nothing (`--no-kv-offload`) | — | ✔ | — | — | — | ◐ |
| LMDeploy (TurboMind) | ✔ | — | — | ✔ | — | — | — | — |
| NVIDIA Dynamo (KVBM) | ✔ | ✔ | ✔ | — | ◐ tiered demotion | — | — | — |
| LMCache | ✔ | ✔ | ✔ | — | ◐ | — | — | — |
| Mooncake | ✔ | ✔ | ✔ | — | ◐ | — | — | — |
| CachedAttention (AttentionStore) | ✔ | ✔ | ✔ | — | ✔ scheduler-aware | — | ✔ layer-wise pre-load | — |
| FlexGen | — | ✔ | ✔ | ✔ | — | — | ◐ | ✔ |
| HeadInfer | — | ✔ | — | — | — | — | — | — |
| DeepSpeed / ZeRO-Inference | — | ✔ | ✔ | — | — | — | — | — |
| H2O | — | — | — | — | ✔ | — | — | — |
| StreamingLLM | — | — | — | — | ✔ | — | — | — |
| Scissorhands | — | — | — | — | ✔ | — | — | — |
| TOVA | — | — | — | — | ✔ | ◐ current query | — | — |
| SnapKV | — | — | — | — | ✔ prompt only | ◐ observation window | — | — |
| PyramidKV | — | — | — | — | ✔ | — | — | — |
| FastGen | — | — | — | — | ✔ per-head | — | — | — |
| Locret | ◐ cache units | — | — | — | ✔ learned | — | — | — |
| ArkVale | ✔ | ✔ backup | — | — | ✔ recallable | ✔ page digests | ◐ async backup | — |
| Quest | ✔ | — | — | — | ◐ skipped, not freed | ✔ min/max bounds | — | — |
| InfiniGen | — | ✔ | — | — | ◐ | ✔ speculated | ✔ next-layer prefetch | — |
| ShadowKV | ◐ chunks | ✔ values | — | ◐ low-rank keys | ◐ | ✔ | ◐ | — |
| MagicPIG | — | ✔ | — | — | — | ✔ LSH sampling | — | ✔ |
| RetrievalAttention | — | ✔ | — | — | — | ✔ ANN search | — | ✔ |
| PQCache | — | ✔ | — | ◐ PQ codes for keys | — | ✔ PQ retrieval | ✔ | — |
| KIVI / KVQuant / Atom / GEAR | — | — | — | ✔ | — | — | — | — |
| vAttention | — | — | — | — | — | — | — | — |

LazyKV's planned mechanisms touch almost every column. This table is the reason the
contribution is framed as a **study**, not a mechanism (see [Delta](#delta)).

---

## 1. Serving systems

### vLLM: PagedAttention
*arXiv [2309.06180](https://arxiv.org/abs/2309.06180): Efficient Memory Management for Large Language Model Serving with PagedAttention*

- **Problem:** KV-cache fragmentation and duplication limit batch size in high-throughput serving.
- **KV layout:** fixed-size blocks in non-contiguous GPU memory, indexed per sequence through a block table, following OS paging.
- **Movement:** whole-sequence preemption, either swapped to CPU or recomputed later.
- **Paging** yes · **CPU tier** limited (preemption swap) · **SSD** no · **Quantized KV** engine option, not part of the paper.
- **Eviction:** none at token level. Whole requests are preempted.
- **Prefetch:** none.
- **Tradeoff:** near-zero waste and cross-request sharing, but every running sequence's KV must sit on the GPU.
- **LazyKV differs:** it keeps a *single* sequence partially resident instead of preempting whole requests. The block-table idea carries over directly.

### SGLang: RadixAttention
*arXiv [2312.07104](https://arxiv.org/abs/2312.07104): SGLang: Efficient Execution of Structured Language Model Programs*

- **Problem:** KV reuse across the many related calls made by LLM programs.
- **KV layout:** a radix tree of token prefixes over paged GPU KV.
- **Movement:** none across tiers in the paper.
- **Paging** yes · **CPU tier** not in the paper · **SSD** no · **Quantized KV** not in the paper.
- **Eviction:** cached prefixes are evicted from the tree when memory is needed.
- **Prefetch:** none.
- **Tradeoff:** large wins when prefixes are shared; no help for one long, unshared context.
- **LazyKV differs:** it targets a single long context with no cross-request sharing, which is out of scope for v1 (§B9).

### TensorRT-LLM
*Official docs: [KV cache system](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html), [KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)*

- **Problem:** production-grade optimized inference on NVIDIA GPUs.
- **KV layout:** paged blocks, reusable across requests.
- **Movement:** before a reusable block is evicted from GPU memory it can be offloaded to host memory (`host_cache_size`, default 0), and it is copied back to the GPU before reuse.
- **Paging** yes · **CPU tier** yes, for reusable blocks · **SSD** no · **Quantized KV** yes.
- **Eviction:** prioritized eviction of reuse-cache blocks.
- **Prefetch:** none documented. Blocks are copied back on reuse.
- **Tradeoff:** the docs note the copy cost is small on Grace-Hopper and "small enough" on x86 Hopper systems. Both are data-center interconnects, not a laptop x8 link.
- **LazyKV differs:** it offloads *live* KV of the running sequence, not idle reusable prefixes. Every copy therefore sits on the decode path.

### llama.cpp
*Repository: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)*

- **Problem:** portable local inference on consumer hardware.
- **KV layout:** per-layer contiguous KV buffers.
- **Movement:** static placement. KV buffers live on the GPU by default, and `--no-kv-offload` keeps them in CPU RAM for the whole run.
- **Paging** no · **CPU tier** all-or-nothing · **SSD** no · **Quantized KV** yes (`f16`, `q8_0`, `q4_0` cache types).
- **Eviction:** none by default.
- **Prefetch:** none.
- **Tradeoff:** simple and robust, but residency is a launch-time switch rather than a policy.
- **LazyKV differs:** it studies *partial, dynamic* residency. llama.cpp's two static extremes are effectively the endpoints of LazyKV's budget sweep.

### LMDeploy: TurboMind
*Docs: [INT4/INT8 KV cache](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) · repository [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)*

- **Problem:** compressing and serving LLMs efficiently.
- **KV layout:** a managed KV block pool with persistent batching.
- **Movement:** none across tiers.
- **Paging** yes · **CPU tier** no · **SSD** no · **Quantized KV** yes, online INT4 and INT8, asymmetric per-head per-token (since v0.4.0).
- **Eviction:** none documented.
- **Prefetch:** none.
- **Tradeoff:** more KV blocks at a fixed budget, bought with quantization error.
- **LazyKV differs:** quantization is only one rung of its ladder (rung 8), measured against residency policies under the same budget.

### NVIDIA Dynamo: KV Block Manager (KVBM)
*Docs: [KVBM guide](https://docs.dynamo.nvidia.com/dynamo/dev/user-guides/kv-cache-offloading) · [NVIDIA technical blog](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)*

- **Problem:** distributed, disaggregated inference at data-center scale.
- **KV layout:** logical KV blocks managed across GPU, CPU host memory, local disk, and remote storage.
- **Movement:** NIXL transport between tiers. It integrates with vLLM and TensorRT-LLM and with KV-aware routing.
- **Paging** yes · **CPU tier** yes · **SSD** yes · **Quantized KV** not a KVBM feature.
- **Eviction:** demotion down the tiers. By default, blocks go from CPU to disk only if accessed at least twice, to spare SSD wear.
- **Prefetch:** not documented as query-aware.
- **Tradeoff:** capacity and sharing across a cluster rather than decode latency on one GPU.
- **LazyKV differs:** it studies single-stream decode on one consumer GPU, where every tier crossing costs latency.

### LMCache
*arXiv [2510.09665](https://arxiv.org/abs/2510.09665): LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*

- **Problem:** extract, store, and share KV caches outside GPU memory, across engines and queries.
- **KV layout:** KV chunks held by a caching layer spanning GPU, CPU, storage, and network.
- **Movement:** batched data movement with I/O pipelining, through a connector to vLLM and SGLang.
- **Paging** yes · **CPU tier** yes · **SSD** yes · **Quantized KV** not a stated focus.
- **Eviction:** cache-layer management across tiers.
- **Prefetch:** oriented to prefix reuse and prefill-decode disaggregation.
- **Tradeoff:** avoids recomputing reused prefixes, but does not reduce KV touched per decode step.
- **LazyKV differs:** LMCache wins by *not recomputing* reused KV. LazyKV asks how little live KV the GPU needs to hold *during* decode.

### Mooncake
*arXiv [2407.00079](https://arxiv.org/abs/2407.00079): Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*

- **Problem:** serving Kimi at scale under SLOs with overloaded clusters.
- **KV layout:** a disaggregated KVCache pool built from the cluster's CPU, DRAM, and SSD.
- **Movement:** KV transfer between separate prefill and decode clusters.
- **Paging** yes · **CPU tier** yes · **SSD** yes · **Quantized KV** not a stated focus.
- **Eviction:** cluster-level cache management plus prediction-based early rejection of requests.
- **Prefetch:** scheduler-driven.
- **Tradeoff:** cluster throughput under SLOs, which requires fast network fabric.
- **LazyKV differs:** it has one machine and one GPU, so the problem is residency rather than scheduling.

### CachedAttention (AttentionStore)
*arXiv [2403.19708](https://arxiv.org/abs/2403.19708): Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention (v1 title: AttentionStore)*

- **Problem:** recomputing the KV of conversation history on every turn.
- **KV layout:** hierarchical KV caching across GPU memory, host DRAM, and disk.
- **Movement:** **layer-wise pre-loading and asynchronous saving that overlap KV access with GPU computation.**
- **Paging** yes · **CPU tier** yes · **SSD** yes · **Quantized KV** no.
- **Eviction:** scheduler-aware fetching and eviction place KV in the right tier.
- **Prefetch:** yes, driven by scheduler hints.
- **Tradeoff:** large TTFT gains for multi-turn sessions. The KV it saves is between turns, not inside one decode.
- **LazyKV differs:** it applies the same compute/transfer overlap idea *within* a single decode step. Phase 0's overlap and copy-engine measurements test whether that is possible on a laptop.

## 2. Offloading

### FlexGen
*arXiv [2303.06865](https://arxiv.org/abs/2303.06865): FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*

- **Problem:** throughput-oriented, latency-insensitive batched inference with limited GPU memory.
- **KV layout:** tensors placed across GPU, CPU, and disk according to a linear-programming search.
- **Movement:** offloaded KV lives in host memory and storage. The search space covers schedule, placement, and **computation delegation, including running attention on the CPU**.
- **Paging** no · **CPU tier** yes · **SSD** yes · **Quantized KV** yes (4-bit weights and KV).
- **Eviction:** none; everything is kept, in some tier.
- **Prefetch:** a zig-zag block schedule over large batches.
- **Tradeoff:** maximum throughput at very high per-token latency.
- **LazyKV differs:** it is latency-oriented at batch size 1. FlexGen's CPU delegation is design (c) in the brief, and Phase 0 measured whether this CPU and this RAM can afford it.

### HeadInfer
*arXiv [2502.12574](https://arxiv.org/abs/2502.12574): HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading*

- **Problem:** million-token contexts on a single consumer GPU.
- **KV layout:** head-wise split. KV for selected attention heads stays on the GPU; the rest goes to CPU RAM.
- **Movement:** fine-grained per-head offload, with attention computed dynamically and backed by a roofline analysis.
- **Paging** no · **CPU tier** yes · **SSD** no · **Quantized KV** no.
- **Eviction:** none. It is exact, with no approximation.
- **Prefetch:** not a stated mechanism.
- **Tradeoff:** exactness and very long contexts, paid for with transfer or compute on offloaded heads.
- **LazyKV differs:** it splits along the *token/block* axis and chooses blocks per query. HeadInfer splits along the *head* axis. The two are complementary, and a head axis is a candidate Phase 5 ablation.

### DeepSpeed Inference / ZeRO-Inference
*arXiv [2207.00032](https://arxiv.org/abs/2207.00032): DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale · [ZeRO-Inference blog, 2023-09-12](https://www.deepspeed.ai/2023/09/12/ZeRO-Inference.html)*

- **Problem:** running models too large for aggregate GPU memory.
- **KV layout:** a heterogeneous hierarchy of GPU, CPU, and NVMe. The 2023 ZeRO-Inference release added KV-cache offload to CPU alongside 4-bit weights.
- **Movement:** fetches weights (and, since 2023, KV) over PCIe.
- **Paging** no · **CPU tier** yes · **SSD** yes (weights) · **Quantized KV** no (weights are quantized).
- **Eviction:** none.
- **Prefetch:** layer-wise fetching of weights.
- **Tradeoff:** throughput-oriented; the docs openly accept PCIe fetch latency.
- **LazyKV differs:** its model fits on the GPU, so only the KV is tiered and decode latency is the metric.

## 3. Eviction and sparsity

### H2O: Heavy-Hitter Oracle
*arXiv [2306.14048](https://arxiv.org/abs/2306.14048)*

- **Problem:** KV memory footprint in long generation.
- **KV layout:** a fixed budget on the GPU.
- **Movement:** none; evicted KV is gone.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** keeps a balance of recent tokens and "heavy hitters" ranked by accumulated attention. Framed as dynamic submodular optimization.
- **Prefetch:** none.
- **Tradeoff:** cheap, but irreversible. A token that becomes important later is already gone.
- **LazyKV differs:** H2O is rung 4 of LazyKV's policy ladder, and it gets the same budget sweep and quality metrics as everything else.

### StreamingLLM: attention sinks
*arXiv [2309.17453](https://arxiv.org/abs/2309.17453): Efficient Streaming Language Models with Attention Sinks*

- **Problem:** stable generation past the training window with bounded KV.
- **KV layout:** the first few "sink" tokens plus a sliding window of recent tokens.
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** everything outside sinks and window.
- **Prefetch:** none.
- **Tradeoff:** very cheap and stable, but cannot retrieve facts that fall out of the window.
- **LazyKV differs:** this is rung 2, the strong cheap baseline the brief requires. NIAH is expected to expose exactly its retrieval weakness, and it will be reported either way.

### Scissorhands
*arXiv [2305.17118](https://arxiv.org/abs/2305.17118)*

- **Problem:** fixed-budget KV compression at test time.
- **KV layout:** a fixed GPU budget.
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** composable with 4-bit.
- **Eviction:** keeps "pivotal" tokens, based on the persistence-of-importance hypothesis.
- **Prefetch:** none.
- **Tradeoff:** relies on importance persisting across steps.
- **LazyKV differs:** it tests persistence empirically through block residency durations (a §B6 metric) instead of assuming it.

### TOVA: Token Omission Via Attention
*arXiv [2401.06104](https://arxiv.org/abs/2401.06104): Transformers are Multi-State RNNs*

- **Problem:** bounding KV size by treating the transformer as a bounded multi-state RNN.
- **KV layout:** a fixed-size state.
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** drops the token with the lowest attention from the current query.
- **Prefetch:** none.
- **Tradeoff:** simple and training-free, but irreversible.
- **LazyKV differs:** TOVA's current-query criterion is a token-level cousin of Quest's block bound. LazyKV compares the two at block granularity.

### SnapKV
*arXiv [2404.14469](https://arxiv.org/abs/2404.14469): SnapKV: LLM Knows What You are Looking for Before Generation*

- **Problem:** long-prompt KV size.
- **KV layout:** a compressed prompt KV per attention head.
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** once, after prefill. KV positions are selected using an observation window at the end of the prompt.
- **Prefetch:** none.
- **Tradeoff:** very effective when the question sits at the end of the prompt; the choice is fixed before generation.
- **LazyKV differs:** its selection is per decode step and reversible, since non-resident blocks still exist on the CPU.

### PyramidKV
*arXiv [2406.02069](https://arxiv.org/abs/2406.02069)*

- **Problem:** allocating KV budget sensibly across layers.
- **KV layout:** per-layer budgets that shrink with depth ("pyramidal information funneling").
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** attention-based within each layer's budget.
- **Prefetch:** none.
- **Tradeoff:** better quality than uniform budgets at the same total.
- **LazyKV differs:** its budget is uniform per layer in v1. Per-layer budgets are a natural ablation for the adaptive controller (week 5).

### FastGen
*arXiv [2310.01801](https://arxiv.org/abs/2310.01801): Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*

- **Problem:** one eviction rule does not fit every attention head.
- **KV layout:** a per-head choice between local window, special tokens, or full cache.
- **Movement:** none.
- **Paging** no · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** chosen per head by lightweight attention profiling.
- **Prefetch:** none.
- **Tradeoff:** needs a profiling pass, then compression that adapts to each head's structure.
- **LazyKV differs:** it applies one policy to all heads in v1.

### Locret
*arXiv [2410.01805](https://arxiv.org/abs/2410.01805)*

- **Problem:** long-context eviction on consumer-grade GPUs that stays compatible with chunked prefill.
- **KV layout:** cache units scored by trained "retaining heads".
- **Movement:** none.
- **Paging** cache units · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** learned causal-importance scores.
- **Prefetch:** none.
- **Tradeoff:** better eviction decisions, at the cost of a small training step.
- **LazyKV differs:** training any predictor is out of scope (§B9). Locret is the closest prior work on the *consumer-GPU plus chunked-prefill* setting.

### ArkVale
*NeurIPS 2024 proceedings: [ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cd4b49379efac6e84186a3ffce108c37-Abstract-Conference.html) (venue verified; no arXiv ID found)*

- **Problem:** tokens evicted early can become important again later.
- **KV layout:** pages. Each filled page is **backed up asynchronously to CPU memory** and summarized as a small bounding-volume digest of its keys.
- **Movement:** before attention, page importance is estimated from the digests, and important pages are recalled from the CPU backup.
- **Paging** yes · **CPU tier** yes (backup) · **SSD** no · **Quantized KV** no.
- **Eviction:** recallable eviction of unimportant pages.
- **Prefetch:** recall happens before attention, not predicted a layer ahead.
- **Tradeoff:** reversibility, paid for with CPU-to-GPU recall.
- **LazyKV differs:** ArkVale is the **closest existing mechanism** to LazyKV rungs 5–6: bounding-box digests, CPU backup, and recall. LazyKV's added value is the measured cost of that recall on an x8 laptop link, compared on equal terms against the cheap baselines.

## 4. Query-aware block sparsity and prefetch

### Quest
*arXiv [2406.10774](https://arxiv.org/abs/2406.10774): Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*

- **Problem:** decode attention slows as the KV cache grows.
- **KV layout:** pages with per-page element-wise min and max of the keys.
- **Movement:** none. **The full KV cache is retained in memory**, and only the top-K pages are *loaded into attention* each step.
- **Paging** yes · **CPU tier** no · **SSD** no · **Quantized KV** no.
- **Eviction:** none; non-selected pages are skipped, not freed.
- **Prefetch:** none.
- **Tradeoff:** speeds up attention without saving memory.
- **LazyKV differs:** it uses Quest's bound (rung 5) and asks the question Quest does not: what happens when the skipped pages are *not in VRAM*?

### InfiniGen
*arXiv [2406.19707](https://arxiv.org/abs/2406.19707): InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*

- **Problem:** the cost of fetching KV from host memory in offloading-based systems.
- **KV layout:** full KV in host memory.
- **Movement:** **speculates the next layer's important tokens** with a minimal rehearsal (current-layer input, part of the next layer's query weights, and part of its key cache), then prefetches only those entries.
- **Paging** no · **CPU tier** yes · **SSD** no · **Quantized KV** no.
- **Eviction:** implicit, since tokens that are not fetched are not attended.
- **Prefetch:** yes, one layer ahead.
- **Tradeoff:** prefetch accuracy against a smaller fetch.
- **LazyKV differs:** its prefetch (rung 7) follows this idea. Phase 0's contribution is measuring the overlap window it needs on hardware with one copy engine.

### ShadowKV
*arXiv [2410.21465](https://arxiv.org/abs/2410.21465): ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference*

- **Problem:** high-throughput long-context serving.
- **KV layout:** a **low-rank key cache on the GPU**, with the value cache offloaded to CPU.
- **Movement:** a minimal sparse set of KV pairs is reconstructed on the fly per step.
- **Paging** chunks · **CPU tier** yes (values) · **SSD** no · **Quantized KV** low-rank keys rather than bit-width quantization.
- **Eviction:** none; the design is sparse selection.
- **Prefetch:** selection of the sparse set each step.
- **Tradeoff:** larger batches on data-center GPUs, with low-rank approximation of the keys.
- **LazyKV differs:** it runs batch 1 on a consumer laptop and does not compress keys in v1.

### MagicPIG
*arXiv [2410.16179](https://arxiv.org/abs/2410.16179): MagicPIG: LSH Sampling for Efficient LLM Generation*

- **Problem:** top-K attention loses quality when attention is not as sparse as assumed.
- **KV layout:** LSH hash tables and KV on the CPU.
- **Movement:** **attention runs on the CPU** from LSH-sampled keys and values.
- **Paging** no · **CPU tier** yes · **SSD** no · **Quantized KV** no.
- **Eviction:** none; sampling replaces top-K.
- **Prefetch:** none.
- **Tradeoff:** a sampling estimator with guarantees, but CPU compute is on the decode path.
- **LazyKV differs:** MagicPIG shows CPU-side attention can pay off on desktop-class hosts. LazyKV's Phase 0 measures that same CPU path on a laptop CPU with **single-channel** RAM.

### RetrievalAttention
*arXiv [2409.10516](https://arxiv.org/abs/2409.10516)*

- **Problem:** long contexts with limited GPU memory.
- **KV layout:** approximate-nearest-neighbour indexes over KV vectors in CPU memory.
- **Movement:** attention-aware vector search retrieves the relevant KV during generation.
- **Paging** no · **CPU tier** yes · **SSD** no · **Quantized KV** no.
- **Eviction:** none.
- **Prefetch:** retrieval per step.
- **Tradeoff:** needs index construction and an attention-aware search, because queries and keys are out of distribution for standard indexes.
- **LazyKV differs:** it uses a far cheaper block-level bound, and measures whether a block bound is *enough* before reaching for ANN search.

### PQCache
*arXiv [2407.12820](https://arxiv.org/abs/2407.12820)*

- **Problem:** long-context KV size with low serving latency.
- **KV layout:** product-quantized key codes and centroids; full KV stored off the GPU.
- **Movement:** PQ codes identify important tokens, and only those KV pairs are fetched.
- **Paging** no · **CPU tier** yes · **SSD** no · **Quantized KV** PQ codes used for retrieval.
- **Eviction:** none.
- **Prefetch:** overlap and caching are designed to cut communication overhead.
- **Tradeoff:** better selection quality than coarse bounds, at the cost of PQ training during prefill.
- **LazyKV differs:** it uses Quest-style bounds, which need no training. PQ-based selection is a possible later comparison.

## 5. KV quantization

| Paper | arXiv | Scheme | Relevance to LazyKV |
|---|---|---|---|
| **KIVI** | [2402.02750](https://arxiv.org/abs/2402.02750) | Tuning-free asymmetric 2-bit; keys per-channel, values per-token | Reference design for rung 8 (int8 warm/cold tier) |
| **KVQuant** | [2401.18079](https://arxiv.org/abs/2401.18079) | Per-channel keys, pre-RoPE keys, non-uniform datatypes, per-vector dense-and-sparse | Shows why naive key quantization fails |
| **Atom** | [2310.19102](https://arxiv.org/abs/2310.19102) | 4-bit weight-activation quantization for serving; the KV cache is quantized asymmetrically and dequantized before FP16 attention (per the paper and the [MLSys'24 slides](https://mlsys.org/media/mlsys-2024/Slides/2655.pdf)) | KV quantization as one part of a whole-model scheme |
| **GEAR** | [2403.05527](https://arxiv.org/abs/2403.05527) | Ultra-low-bit quantization plus a low-rank error term plus a sparse outlier matrix | Error-correction option if int8 degrades quality |

All four trade KV bytes for approximation error and never move KV between tiers.
LazyKV uses quantization only as a per-tier choice and measures its quality delta with the
same teacher-forced KL as eviction (§B7).

## 6. Memory mechanism

### vAttention
*arXiv [2405.04437](https://arxiv.org/abs/2405.04437): vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*

- **Problem:** PagedAttention makes KV virtual memory non-contiguous, which forces custom kernels.
- **Mechanism:** CUDA virtual-memory-management APIs keep KV contiguous in virtual memory while physical memory is allocated on demand.
- **Tradeoff:** unmodified attention kernels; depends on platform VMM support.
- **LazyKV note:** PyTorch's `expandable_segments` allocator is built on the same CUDA VMM facility. Phase 0 recorded that it **is not supported on this Windows platform** (see `SYSTEM_INFO.md`). A vAttention-style design is therefore unlikely to be available here, and LazyKV should plan for explicit block tables with preallocated pools.

## 7. Evaluation

| Benchmark | Source | What LazyKV uses it for |
|---|---|---|
| **Needle-in-a-haystack** | [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) | Retrieval sweep over needle depths and context lengths |
| **RULER** | arXiv [2404.06654](https://arxiv.org/abs/2404.06654) | Single- and multi-needle task definitions. RULER's own finding is that vanilla NIAH is only a superficial long-context test, so LazyKV will not rely on NIAH alone |
| **LongBench** | arXiv [2308.14508](https://arxiv.org/abs/2308.14508) | Task accuracy on a few subsets, if time allows (§B7.4) |
| **InfiniteBench (∞Bench)** | arXiv [2402.13718](https://arxiv.org/abs/2402.13718) | 100K+ tasks. Likely beyond this 8 GB GPU's reach at bf16; used only where the context fits |

---

## Delta

**LazyKV does not add a new mechanism.** After reading the prior work:

- **Block-level query-aware selection** already exists: Quest's min/max bounds, ArkVale's bounding-volume digests.
- **CPU backup with recall** already exists: ArkVale, InfiniGen, ShadowKV, PQCache.
- **Next-layer speculative prefetch** already exists: InfiniGen. **Layer-wise pre-loading that overlaps transfer with compute** already exists too: CachedAttention.
- **Attention on the CPU** already exists: FlexGen's computation delegation, MagicPIG, RetrievalAttention.
- **Mixed-precision KV** already exists: KIVI, KVQuant, Atom, GEAR, LMDeploy, llama.cpp.
- **Tiered KV offload** is shipped in production: TensorRT-LLM, NVIDIA Dynamo, LMCache, Mooncake.

What LazyKV adds is **a reproducible, apples-to-apples study of these residency policies
on one constrained consumer GPU**:

1. **One harness, one hardware point.** Every policy, from sliding window plus sinks up to query-aware selection with CPU tier, prefetch, and mixed precision, runs on the same 8 GB laptop GPU. They share the model, the budget sweep, and the prompts.
2. **Hardware-grounded costs.** Tier crossings are costed with bandwidths and a copy-engine count measured on that machine (`SYSTEM_INFO.md`), not with data-center interconnect figures.
3. **Honest quality measurement.** Every policy gets teacher-forced KL, top-1 agreement, and depth-swept NIAH, with confidence intervals.
4. **Strong baselines included.** The sliding-window-plus-sink baseline and full ablations are in, and negative results are reported.

That is a narrower claim than a new system, and a defensible one. The abstracts above
report results on A100-class data-center GPUs or 24 GB RTX 4090 hosts. There is less
evidence about which of these ideas still pay off on an 8 GB laptop GPU whose CPU tier
sits behind an x8 link, with single-channel host RAM (both measured in
`SYSTEM_INFO.md`).
