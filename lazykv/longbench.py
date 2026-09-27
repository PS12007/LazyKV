"""LongBench (v1) question answering: prompts and scoring, following the reference implementation.

LongBench (arXiv 2308.14508) is real long-document QA rather than a synthetic probe: the contexts
are Wikipedia passages, papers and reports at their natural length. This module follows the
reference code at THUDM/LongBench commit 2e00731f8d0bff23dc4325161044d0ed8af94c1e:

- prompts are `config/dataset2prompt.json` verbatim, and generation lengths `dataset2maxlen.json`;
- a prompt longer than `max_length` tokens keeps its first and last `max_length // 2` tokens
  (`pred.py`: "truncate in the middle, since the left and right side may contain crucial
  instructions"), and only then is wrapped in the chat template;
- the score is `qa_f1_score` (`metrics.py`): answer normalization (lower case, no punctuation, no
  articles), token-level F1, maximum over the reference answers.

The data files are the official `data.zip` from the Hugging Face dataset THUDM/LongBench, revision
5e628be450b7e67fb7ae6e201bd6d8f7056f7672, unpacked outside the repository (they are not committed).
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from lazykv.niah import CHAT_DATE

_MULTIDOC = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)
PROMPTS = {
    "hotpotqa": _MULTIDOC,
    "2wikimqa": _MULTIDOC,
    "musique": _MULTIDOC,
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
}
MAX_NEW_TOKENS = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32, "multifieldqa_en": 64}


@dataclass(frozen=True)
class LongBenchPrompt:
    task: str
    index: int
    input_ids: torch.Tensor  # [L], long, CPU
    answers: tuple[str, ...]
    truncated: bool

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


def load_task(data_dir: Path, task: str) -> list[dict[str, object]]:
    if task not in PROMPTS:
        raise ValueError(f"unsupported LongBench task {task!r}")
    with open(data_dir / f"{task}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(tokenizer: PreTrainedTokenizerBase, row: dict[str, object], task: str, index: int, max_length: int) -> LongBenchPrompt:
    prompt = PROMPTS[task].format(context=row["context"], input=row["input"])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    truncated = len(ids) > max_length
    if truncated:
        half = max_length // 2
        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(ids[-half:], skip_special_tokens=True)
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE)
    out = tokenizer(text, add_special_tokens=False)["input_ids"]
    return LongBenchPrompt(task, index, torch.tensor(out, dtype=torch.long), tuple(str(a) for a in row["answers"]), truncated)


def normalize_answer(s: str) -> str:
    """LongBench's normalization: lower case, strip punctuation, drop articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(prediction: str, truth: str) -> float:
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    same = sum((Counter(p) & Counter(t)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def score(prompt: LongBenchPrompt, generated_text: str) -> float:
    """qa_f1_score, maximized over the reference answers, as LongBench's eval.py does."""
    return max(f1(generated_text, a) for a in prompt.answers)
