"""Needle-in-a-haystack prompts and scoring, adapted from RULER's NIAH task definitions.

RULER (arXiv 2404.06654) defines single-needle, multi-key and multi-value retrieval with
"special magic number" needles. This module follows its needle, question and answer-prefix
wording, but it is not RULER's harness: the haystack is a public-domain book from the
verified corpus, and prompts are assembled at the token level so their length is exact.

Kinds:
  single      one needle; the question asks for its value
  multikey    the target needle plus distractor needles with other keys
  multivalue  several needles share one key; the answer must list every value

Why token-level assembly: a policy's budget is a fraction of the sequence, so prompt length
must be controlled exactly. Needles are inserted after a sentence-ending token, never inside
a word.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

KINDS = ("single", "multikey", "multivalue")

# The chat template stamps "Today Date" from the wall clock unless told otherwise, which would
# make prompts (and their lengths) change from day to day. Pinned to the template's own default.
CHAT_DATE = "26 Jul 2024"

INTRO = "Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n"
NEEDLE = " One of the special magic numbers for {key} is: {value}."
QUESTION = {
    "single": "\nWhat is the special magic number for {key} mentioned in the provided text?",
    "multikey": "\nWhat is the special magic number for {key} mentioned in the provided text?",
    "multivalue": "\nWhat are all the special magic numbers for {key} mentioned in the provided text?",
}
ANSWER_PREFIX = {
    "single": "The special magic number for {key} mentioned in the provided text is",
    "multikey": "The special magic number for {key} mentioned in the provided text is",
    "multivalue": "The special magic numbers for {key} mentioned in the provided text are",
}

# Fixed key vocabulary so a seed fully determines every prompt.
KEYS = (
    "anchor apricot badger bamboo basalt beacon birch bramble canyon cedar cobalt comet coral cricket "
    "cypress dune ember falcon fern fjord garnet glacier granite harbor hazel heron indigo iris jasper "
    "juniper kestrel lagoon lantern lichen magnet maple marble meadow meteor nebula nectar oasis onyx "
    "orchid osprey otter pebble pepper pine plume quartz quill raven reef saffron sage sapphire "
    "sequoia sparrow spruce summit talon thistle thunder topaz tundra umber valley velvet willow zephyr"
).split()


@dataclass(frozen=True)
class NiahPrompt:
    kind: str
    depth: float
    sample: int
    input_ids: torch.Tensor  # [L], long, CPU
    key: str
    values: tuple[str, ...]  # values the answer must contain
    needle_token_positions: tuple[int, ...] = field(default_factory=tuple)  # first token of each needle; target first

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


def _ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _sentence_end_ids(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> set[int]:
    return {t for t in set(ids) if tokenizer.decode([t]).rstrip().endswith((".", "!", "?"))}


def build_prompt(tokenizer: PreTrainedTokenizerBase, book_ids: torch.Tensor, target_len: int, kind: str, depth: float, sample: int, seed: int = 0, n_needles: int = 4) -> NiahPrompt:
    """A prompt of exactly `target_len` tokens (chat template, haystack, needles, question, answer prefix)."""
    if kind not in KINDS:
        raise ValueError(f"unknown NIAH kind {kind!r}")
    rng = random.Random(f"{seed}:{kind}:{depth}:{sample}")
    keys = rng.sample(KEYS, n_needles)
    key = keys[0]
    n_values = n_needles if kind == "multivalue" else 1
    values = [str(rng.randrange(1_000_000, 10_000_000)) for _ in range(n_needles)]

    if kind == "single":
        needles = [(key, values[0])]
    elif kind == "multikey":
        needles = [(key, values[0])] + [(k, v) for k, v in zip(keys[1:], values[1:])]
    else:
        needles = [(key, v) for v in values]
    needle_ids = [_ids(tokenizer, NEEDLE.format(key=k, value=v)) for k, v in needles]

    # Split the rendered chat template around a placeholder so the haystack can be spliced in as ids.
    marker = "<<LAZYKV_CONTEXT>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": INTRO + marker + QUESTION[kind].format(key=key)}], tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE
    )
    head, tail = rendered.split(marker)
    head_ids = _ids(tokenizer, head)
    tail_ids = _ids(tokenizer, tail + ANSWER_PREFIX[kind].format(key=key))

    hay_len = target_len - len(head_ids) - len(tail_ids) - sum(len(n) for n in needle_ids)
    if hay_len <= 0:
        raise ValueError(f"target_len {target_len} too small for the template and needles")
    body = book_ids.tolist()
    if body and body[0] == tokenizer.bos_token_id:
        body = body[1:]
    # A different stretch of the book per sample, so samples are not the same haystack.
    offset = rng.randrange(0, max(1, len(body) - hay_len))
    hay = body[offset : offset + hay_len]
    if len(hay) < hay_len:
        raise ValueError("book too short for the requested context")
    ends = _sentence_end_ids(tokenizer, hay)

    def insertion_point(frac: float) -> int:
        pos = round(frac * len(hay))
        if pos <= 0:
            return 0
        if pos >= len(hay):
            return len(hay)
        while pos > 0 and hay[pos - 1] not in ends:
            pos -= 1
        return pos

    # Target at the requested depth; other needles at seeded random depths.
    points = [insertion_point(depth)] + [insertion_point(rng.random()) for _ in needle_ids[1:]]
    order = sorted(range(len(needle_ids)), key=lambda i: (points[i], i))
    out = list(head_ids)
    starts = [0] * len(needle_ids)
    cursor = 0
    for i in order:
        out += hay[cursor : points[i]]
        cursor = points[i]
        starts[i] = len(out)
        out += needle_ids[i]
    out += hay[cursor:]
    out += tail_ids
    assert len(out) == target_len, (len(out), target_len)
    return NiahPrompt(kind, depth, sample, torch.tensor(out, dtype=torch.long), key, tuple(values[:n_values]), tuple(starts))


def score(prompt: NiahPrompt, generated_text: str) -> float:
    """Fraction of required values present in the generated answer (0 or 1 except for multivalue)."""
    return sum(1 for v in prompt.values if v in generated_text) / len(prompt.values)
