"""Load verified corpus text as model token ids."""

from __future__ import annotations

import hashlib
import json

import torch
from transformers import PreTrainedTokenizerBase

from harness.results import REPO_ROOT

CORPUS_DIR = REPO_ROOT / "data" / "corpus"


def manifest() -> dict[str, object]:
    return json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))


def load_tokens(book: str, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """BOS + book tokens. Refuses to run on text that differs from the manifest."""
    text_bytes = (CORPUS_DIR / f"{book}.txt").read_bytes()
    expected = manifest()["books"][book]["body_sha256"]  # type: ignore[index]
    got = hashlib.sha256(text_bytes).hexdigest()
    if got != expected:
        raise RuntimeError(f"{book}: sha256 {got} != manifest {expected}; rerun scripts/fetch_corpus.py")
    ids = tokenizer(text_bytes.decode("utf-8"), add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    return torch.tensor(([bos] if bos is not None else []) + ids, dtype=torch.long)
