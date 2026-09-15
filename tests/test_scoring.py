"""H2O-style prefill scoring: with stride 1 it must equal the exact accumulated attention mass."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_stride_one_equals_exact_column_sums_across_chunks() -> None:
    from lazykv.scoring import PrefillScorer

    torch.manual_seed(0)
    n_q, n_kv, d, L = 8, 2, 16, 70
    q = torch.randn(1, n_q, L, d, device="cuda")
    k = torch.randn(1, n_kv, L, d, device="cuda")
    # Exact: full causal attention over all L queries, summed over heads and queries.
    kk = k.repeat_interleave(n_q // n_kv, 1)
    scores = (q @ kk.transpose(-1, -2)) / d**0.5
    causal = torch.ones(L, L, dtype=torch.bool, device="cuda").tril()
    exact = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1).sum(dim=(0, 1, 2))

    scorer = PrefillScorer(num_layers=1, max_len=L, device=torch.device("cuda"), stride=1, query_batch=5)
    for s in range(0, L, 32):  # chunked, as prefill runs: each chunk's queries see all earlier keys
        e = min(L, s + 32)
        scorer.observe(0, q[:, :, s:e], k[:, :, :e])
    assert scorer.sampled_queries[0] == L
    assert torch.allclose(scorer.mass[0], exact, rtol=1e-4, atol=1e-4)
    blocks = scorer.block_scores(0, n_blocks=4, block_size=16)
    assert torch.allclose(blocks, exact[:64].view(4, 16).sum(-1), rtol=1e-4, atol=1e-4)


@cuda
def test_stride_samples_absolute_positions_across_chunks() -> None:
    from lazykv.scoring import PrefillScorer

    scorer = PrefillScorer(num_layers=1, max_len=100, device=torch.device("cuda"), stride=7)
    q = torch.randn(1, 4, 100, 8, device="cuda")
    k = torch.randn(1, 2, 100, 8, device="cuda")
    for s in range(0, 100, 30):
        e = min(100, s + 30)
        scorer.observe(0, q[:, :, s:e], k[:, :, :e])
    assert scorer.sampled_queries[0] == len(range(0, 100, 7))
