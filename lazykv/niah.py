"""Needle-in-a-haystack prompts and scoring, adapted from RULER's NIAH task definitions.

RULER (arXiv 2404.06654) defines single-needle, multi-key and multi-value retrieval with
"special magic number" needles. This module follows its needle, question and answer-prefix
wording, but it is not RULER's harness: the haystack is a public-domain book from the
verified corpus, and prompts are assembled at the token level so their length is exact.

Kinds:
  single      one needle; the question asks for its value
  multikey    the target needle plus distractor needles with other keys
  multivalue  several needles share one key; the answer must list every value
  vt          RULER variable tracking: a chain of five assignments (VAR A = 12345, VAR B = VAR A,
              ...) hidden in the book; the answer must name every variable holding the value.
              Multi-hop: each hop's block is needed to follow the next, and none of the later
              statements contains the value the question asks about.
  cwe         RULER common words extraction: a numbered word list where ten words appear 30 times
              and the rest 3 times; the answer must name the ten. Aggregation: the evidence is
              spread over hundreds of positions rather than sitting in a few blocks.

The RULER kinds follow RULER's templates and answer prefixes verbatim (NVIDIA/RULER,
scripts/data/synthetic/constants.py). Two deviations, both forced: variable tracking uses the book
as its haystack (RULER's "essay" noise mode, the closest to this harness's other kinds), and common
words are drawn from the verified corpus's vocabulary instead of RULER's `wonderwords` package,
which this project cannot add (CLAUDE.md rule 5). `depth` places the needle for the NIAH kinds and
the chain's first statement for vt; cwe has no depth and uses it only to seed a sample.

Why token-level assembly: a policy's budget is a fraction of the sequence, so prompt length
must be controlled exactly. Needles are inserted after a sentence-ending token, never inside
a word.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

KINDS = ("single", "multikey", "multivalue", "vt", "cwe")

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

# RULER's templates (constants.py), split at the context. The answer prefix's leading space is
# dropped: here the prefix starts the assistant turn rather than following the question inline.
RULER_TEMPLATE = {
    "vt": ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n",
           "\nQuestion: Find all variables that are assigned the value {query} in the text above."),
    "cwe": ("Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n",
            "\nQuestion: What are the 10 most common words in the above list?"),
}
RULER_ANSWER_PREFIX = {
    "vt": "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: ",
    "cwe": "Answer: The top 10 words that appear most often in the list are:",
}
VT_HOPS = 4  # RULER default: one chain of four hops, five variables
CWE_COMMON, CWE_FREQ_COMMON, CWE_FREQ_UNCOMMON = 10, 30, 3  # RULER defaults

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
    if kind == "vt":
        return _build_vt(tokenizer, book_ids, target_len, depth, sample, seed)
    if kind == "cwe":
        return _build_cwe(tokenizer, book_ids, target_len, depth, sample, seed)
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


def _wrap(tokenizer: PreTrainedTokenizerBase, before: str, after: str, answer_prefix: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat template before and after the spliced-in context."""
    marker = "<<LAZYKV_CONTEXT>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": before + marker + after}], tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE
    )
    head, tail = rendered.split(marker)
    return _ids(tokenizer, head), _ids(tokenizer, tail + answer_prefix)


def _book_body(tokenizer: PreTrainedTokenizerBase, book_ids: torch.Tensor) -> list[int]:
    body = book_ids.tolist()
    return body[1:] if body and body[0] == tokenizer.bos_token_id else body


def _build_vt(tokenizer: PreTrainedTokenizerBase, book_ids: torch.Tensor, target_len: int, depth: float, sample: int, seed: int) -> NiahPrompt:
    rng = random.Random(f"{seed}:vt:{depth}:{sample}")
    names: list[str] = []
    while len(names) < VT_HOPS + 1:  # distinct, so one answer never stands in for another
        n = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        if n not in names:
            names.append(n)
    value = str(rng.randrange(10000, 99999))
    statements = [f" VAR {names[0]} = {value}."] + [f" VAR {names[j + 1]} = VAR {names[j]}." for j in range(VT_HOPS)]
    stmt_ids = [_ids(tokenizer, st) for st in statements]
    before, after = RULER_TEMPLATE["vt"]
    head_ids, tail_ids = _wrap(tokenizer, before, after.format(query=value), RULER_ANSWER_PREFIX["vt"].format(num_v=len(names), query=value))
    hay_len = target_len - len(head_ids) - len(tail_ids) - sum(len(x) for x in stmt_ids)
    body = _book_body(tokenizer, book_ids)
    offset = rng.randrange(0, max(1, len(body) - hay_len))
    hay = body[offset : offset + hay_len]
    if hay_len <= 0 or len(hay) < hay_len:
        raise ValueError(f"target_len {target_len} does not fit a vt prompt")
    ends = _sentence_end_ids(tokenizer, hay)

    def at_sentence_end(frac: float) -> int:
        pos = min(len(hay), max(0, round(frac * len(hay))))
        while 0 < pos < len(hay) and hay[pos - 1] not in ends:
            pos -= 1
        return pos

    # The chain starts at `depth` and the later hops fall after it, in order, so the text reads
    # forwards the way RULER's does; at depth 1.0 the whole chain sits at the end.
    fracs = [depth] + sorted(depth + (1.0 - depth) * rng.random() for _ in range(VT_HOPS))
    points = [at_sentence_end(f) for f in fracs]
    out: list[int] = list(head_ids)
    starts: list[int] = []
    cursor = 0
    for pt, ids in zip(points, stmt_ids):
        pt = max(pt, cursor)
        out += hay[cursor:pt]
        cursor = pt
        starts.append(len(out))
        out += ids
    out += hay[cursor:] + tail_ids
    assert len(out) == target_len, (len(out), target_len)
    return NiahPrompt("vt", depth, sample, torch.tensor(out, dtype=torch.long), value, tuple(names), tuple(starts))


def _vocabulary(tokenizer: PreTrainedTokenizerBase, book_ids: torch.Tensor) -> list[str]:
    """Distinct lowercase words of 4-10 letters from the book, in first-seen order (deterministic)."""
    seen: dict[str, None] = {}
    for w in tokenizer.decode(_book_body(tokenizer, book_ids)).split():
        w = w.strip(PUNCT).lower()
        if 4 <= len(w) <= 10 and w.isascii() and w.isalpha():
            seen.setdefault(w, None)
    return list(seen)


PUNCT = ".,;:!?\"'()[]-_"


def _build_cwe(tokenizer: PreTrainedTokenizerBase, book_ids: torch.Tensor, target_len: int, depth: float, sample: int, seed: int) -> NiahPrompt:
    rng = random.Random(f"{seed}:cwe:{depth}:{sample}")
    vocab = _vocabulary(tokenizer, book_ids)
    before, after = RULER_TEMPLATE["cwe"]
    head_ids, tail_ids = _wrap(tokenizer, before, after, RULER_ANSWER_PREFIX["cwe"])
    room = target_len - len(head_ids) - len(tail_ids)
    words = rng.sample(vocab, len(vocab))
    common, uncommon = words[:CWE_COMMON], words[CWE_COMMON:]

    def listing(n_uncommon: int) -> list[int]:
        pool = common * CWE_FREQ_COMMON + uncommon[:n_uncommon] * CWE_FREQ_UNCOMMON
        order = random.Random(f"{seed}:cwe-order:{depth}:{sample}:{n_uncommon}").sample(pool, len(pool))
        return _ids(tokenizer, "\n".join(f"{i + 1}. {w}" for i, w in enumerate(order)))

    # RULER sizes the list to the context by searching over the word count; so does this, by
    # bisection on the number of uncommon words. The few tokens left over become newlines before
    # the question, so the prompt is exactly target_len and the frequencies are exactly RULER's.
    lo, hi = 0, len(uncommon)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(listing(mid)) <= room:
            lo = mid
        else:
            hi = mid - 1
    ids = listing(lo)
    if len(ids) > room:
        raise ValueError(f"target_len {target_len} does not fit a cwe prompt")
    pad = _ids(tokenizer, "\n")
    if len(pad) != 1:
        raise ValueError("newline is not a single token; cwe padding assumes it is")
    out = head_ids + ids + pad * (room - len(ids)) + tail_ids
    assert len(out) == target_len, (len(out), target_len)
    return NiahPrompt("cwe", depth, sample, torch.tensor(out, dtype=torch.long), "", tuple(common), ())


def score(prompt: NiahPrompt, generated_text: str) -> float:
    """Fraction of required values present in the answer, case-insensitively: RULER's string_match_all.

    Case only matters for the RULER kinds (a model may capitalize a listed word); NIAH values are
    digits, so their scores are unchanged by it.
    """
    text = generated_text.lower()
    return sum(1 for v in prompt.values if v.lower() in text) / len(prompt.values)
