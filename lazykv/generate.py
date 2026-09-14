"""Model loading, chunked prefill, timed greedy decode, and teacher-forced scoring.

All GPU timing uses CUDA events plus deliberately synchronized wall clocks. Decode
synchronizes once per token on purpose: a real generator must bring each token to the
host (to detokenize or check stop conditions), so per-token wall latency is the honest
user-facing number.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache

from lazykv.attention import IMPLEMENTATION_NAME, KernelCheck, install

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    repo: str
    revision: str | None
    attention_strategy: str
    kernel_checks: list[KernelCheck]

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def kv_bytes_per_token(self) -> int:
        c = self.model.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * torch.finfo(self.model.dtype).bits // 8


def load(repo: str, revision: str | None = None, strategy: str | None = None, dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=dtype, attn_implementation="sdpa")
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    chosen, checks = install(strategy, c.num_attention_heads, c.num_key_value_heads, head_dim)
    # Switch only after install() registered the implementation name.
    model.set_attn_implementation(IMPLEMENTATION_NAME)
    model.to("cuda").eval()
    return LoadedModel(model, tokenizer, repo, revision, chosen, checks)


def cache_model_kwargs(cache: Cache) -> dict[str, object]:
    """Extra model() kwargs a cache asks for (e.g. an attention observer); none for the full cache."""
    fn = getattr(cache, "model_kwargs", None)
    return fn() if fn is not None else {}


@dataclass
class PrefillResult:
    last_logits: torch.Tensor  # [vocab], float32, on GPU
    wall_s: float
    event_s: float
    chunks: int


@torch.inference_mode()
def prefill(model: PreTrainedModel, cache: Cache, input_ids: torch.Tensor, chunk_size: int) -> PrefillResult:
    """Chunked prefill computing logits for the final position only.

    Full-sequence logits at 64K would be a vocab x 64K tensor (docs/KV_MEMORY_MODEL.md),
    larger than the GPU. Chunking also bounds attention and MLP activation memory.
    """
    ids = input_ids.to("cuda").view(1, -1)
    n = ids.shape[1]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    logits = None
    chunks = 0
    for s in range(0, n, chunk_size):
        out = model(input_ids=ids[:, s : s + chunk_size], past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits
        chunks += 1
    assert logits is not None
    last = logits[0, -1].float()
    end.record()
    torch.cuda.synchronize()
    return PrefillResult(last, time.perf_counter() - t0, start.elapsed_time(end) / 1e3, chunks)


@dataclass
class DecodeResult:
    tokens: list[int]
    wall_s: list[float] = field(default_factory=list)  # per token, includes host sync
    event_s: list[float] = field(default_factory=list)  # per token, CUDA events


@torch.inference_mode()
def greedy_decode(model: PreTrainedModel, cache: Cache, first_logits: torch.Tensor, n_tokens: int) -> DecodeResult:
    """Greedy decode. Token 0 comes from the prefill logits; each step feeds the last token."""
    tok = int(torch.argmax(first_logits).item())
    res = DecodeResult(tokens=[tok])
    ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
    extra = cache_model_kwargs(cache)
    for _ in range(n_tokens):
        ids.fill_(tok)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
        nxt = torch.argmax(out.logits[0, -1])
        end.record()
        tok = int(nxt.item())  # host sync: the token must reach the host in real generation
        res.wall_s.append(time.perf_counter() - t0)
        res.event_s.append(start.elapsed_time(end) / 1e3)
        res.tokens.append(tok)
    return res


@torch.inference_mode()
def teacher_forced_steps(model: PreTrainedModel, cache: Cache, context_ids: torch.Tensor, continuation_ids: torch.Tensor, chunk_size: int) -> Iterator[torch.Tensor]:
    """Yield next-token log-probs (float32, on GPU) at each continuation position.

    Row i is the distribution over continuation token i given context + continuation[:i].
    Scoring through single-token decode steps (not a batched forward) matters: residency
    policies act on the decode path, so that is where quality must be measured.
    A generator, so callers can compare position by position without holding
    positions x vocab tensors in host RAM (a full-vocab float32 row is ~0.5 MB for Llama 3).
    """
    pre = prefill(model, cache, context_ids, chunk_size)
    yield from teacher_forced_decode(model, cache, pre.last_logits, continuation_ids)


@torch.inference_mode()
def teacher_forced_decode(model: PreTrainedModel, cache: Cache, first_logits: torch.Tensor, continuation_ids: torch.Tensor) -> Iterator[torch.Tensor]:
    """The decode half of `teacher_forced_steps`, on a cache that is already prefilled.

    Split out so one exact prefill can be scored under many residency policies: a policy's
    cache is built from the prefilled KV, and row 0 (from the prefill logits) is shared.
    """
    yield torch.log_softmax(first_logits.float(), dim=-1)
    ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
    extra = cache_model_kwargs(cache)
    for tok in continuation_ids.view(-1).tolist()[:-1]:
        ids.fill_(tok)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
        yield torch.log_softmax(out.logits[0, -1].float(), dim=-1)


def teacher_forced_logprobs(model: PreTrainedModel, cache: Cache, context_ids: torch.Tensor, continuation_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """All rows of `teacher_forced_steps` as one float32 CPU tensor [positions, vocab].

    Preallocated and filled in place: stacking a list of rows would briefly hold two copies.
    """
    rows = None
    for i, row in enumerate(teacher_forced_steps(model, cache, context_ids, continuation_ids, chunk_size)):
        if rows is None:
            rows = torch.empty((continuation_ids.numel(), row.numel()), dtype=torch.float32)
        rows[i].copy_(row)
    assert rows is not None
    return rows
