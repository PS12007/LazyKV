"""Multi-turn retrieval sessions: a long document, then a conversation that keeps adding to the KV.

Every quality test before Phase 19 was single-turn: one prompt, one short answer, and the KV the
policy manages is almost entirely the prompt. Gap table item A5 (a working-set-aware selector that
protects recent history) is motivated by a failure only a conversation can show: after the document,
the user and the model keep writing, and a policy that ranks blocks by the current query alone might
drop something said a few turns ago. This module builds sessions in which that can be observed.

A session is a document of exactly `ctx` tokens with four doc needles (RULER's multi-key NIAH needles
at spread depths), followed by a fixed schedule of turns. Each turn after the first is a user message
that pastes a passage from a *different* book (so the conversation grows by real text, as a document
chat does), may introduce a new **chat needle** ("Also note: one of the special magic numbers for X
is: N."), and asks one question, about a doc needle or about a chat needle given earlier. The model's
answers are its own greedy output, so the conversation is the policy's own trajectory.

How the session is fed: every token after the document goes through the decode path one token at a
time, for every policy. The tier (rungs 6-9) is decode-only, and rung 5 attends densely to a
multi-token forward, so feeding a user turn as one chunk would give rung 5 a dense attention the
tier cannot have. One token at a time is the decode-path semantics every policy shares, and it is
where residency acts. The cost is speed, which this module does not measure.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache

from lazykv.generate import cache_model_kwargs
from lazykv.niah import CHAT_DATE, INTRO, KEYS, NEEDLE, _ids, _sentence_end_ids

# Turn 0 is the NIAH multikey question, verbatim, so turn 0 is directly comparable to single-turn NIAH.
DOC_QUESTION_0 = "\nWhat is the special magic number for {key} mentioned in the provided text?"
DOC_ANSWER_0 = "The special magic number for {key} mentioned in the provided text is"
# Later turns have pasted passages in between, so "the provided text" would be ambiguous.
DOC_QUESTION = "What is the special magic number for {key} mentioned in the first text I gave you?"
DOC_ANSWER = "The special magic number for {key} mentioned in the first text is"
CHAT_NOTE = "Also note: one of the special magic numbers for {key} is: {value}."
CHAT_QUESTION = "What is the special magic number for {key} that I told you earlier in our conversation?"
CHAT_ANSWER = "The special magic number for {key} that you told me is"
PASSAGE_INTRO = "Here is a passage from another book, for later:\n"

# (asks, introduces). d<i> is doc needle i, c<j> chat needle j. Fixed before any run (configs/phase19.yaml
# may override it): chat needles are asked 2 and 3 turns after they are given, every doc needle is
# asked once, and the last turn re-asks doc needle 0, whose answer is now also in the conversation.
DEFAULT_SCHEDULE: tuple[tuple[str, str | None], ...] = (
    ("d0", None), ("d1", "c0"), ("d2", "c1"), ("c0", None), ("d3", "c2"), ("c1", None), ("c2", None), ("d0", None),
)
DOC_DEPTHS = (0.1, 0.35, 0.6, 0.85)


@dataclass(frozen=True)
class Turn:
    index: int
    asks: str  # "d0".."d3" or "c0".."c2"
    key: str
    value: str
    introduces: str | None
    feed_ids: tuple[int, ...]  # fed after the previous answer, before this turn's answer; turn 0's are the document

    @property
    def kind(self) -> str:
        return "doc" if self.asks.startswith("d") else "chat"


@dataclass(frozen=True)
class Session:
    sample: int
    ctx: int
    prefix_ids: torch.Tensor  # [ctx], the document turn, ending in turn 0's answer prefix
    turns: tuple[Turn, ...]
    doc_needles: dict[str, tuple[str, str, float, int]]  # id -> (key, value, depth, first token position)
    chat_needles: dict[str, tuple[str, str, int]]  # id -> (key, value, turn introduced)

    def turns_since_given(self, turn: Turn) -> int | None:
        if turn.kind != "chat":
            return None
        return turn.index - self.chat_needles[turn.asks][2]

    def planned_tokens(self, max_new: int) -> int:
        """An upper bound on the session's length: the document, every fed turn and every answer."""
        return self.ctx + sum(len(t.feed_ids) for t in self.turns[1:]) + max_new * len(self.turns)


def _split_template(tokenizer: PreTrainedTokenizerBase, message: str) -> tuple[list[int], list[int]]:
    """Ids of a follow-up user turn around a <<P>> placeholder in `message`, from the model's own template.

    Rendered as the continuation of a two-turn chat after an assistant answer, so the ids start with the
    end-of-turn token that closes the previous answer.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "<<ANS>>"}, {"role": "user", "content": message}],
        tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE,
    )
    after = rendered.split("<<ANS>>", 1)[1]
    head, tail = after.split("<<P>>")
    return _ids(tokenizer, head), _ids(tokenizer, tail)


def build_session(
    tokenizer: PreTrainedTokenizerBase,
    book_ids: torch.Tensor,
    passage_ids: torch.Tensor,
    ctx: int,
    sample: int,
    seed: int = 0,
    schedule: tuple[tuple[str, str | None], ...] = DEFAULT_SCHEDULE,
    passage_tokens: int = 192,
) -> Session:
    rng = random.Random(f"{seed}:multiturn:{sample}")
    n_doc = 1 + max(int(a[1:]) for a, _ in schedule if a.startswith("d"))
    chat_ids = sorted({x for a, i in schedule for x in (a, i) if x and x.startswith("c")}, key=lambda s: int(s[1:]))
    keys = rng.sample(KEYS, n_doc + len(chat_ids))
    values = [str(rng.randrange(1_000_000, 10_000_000)) for _ in keys]
    depths = list(DOC_DEPTHS[:n_doc])
    rng.shuffle(depths)  # which needle sits where varies by sample; the set of depths does not
    doc = {f"d{i}": (keys[i], values[i], depths[i]) for i in range(n_doc)}
    chat = {cid: (keys[n_doc + j], values[n_doc + j]) for j, cid in enumerate(chat_ids)}

    # ---- turn 0: the document, as build_prompt assembles it, with all doc needles ----------------
    first_key = doc[schedule[0][0]][0]
    marker = "<<LAZYKV_CONTEXT>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": INTRO + marker + DOC_QUESTION_0.format(key=first_key)}], tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE
    )
    head, tail = rendered.split(marker)
    head_ids = _ids(tokenizer, head)
    tail_ids = _ids(tokenizer, tail + DOC_ANSWER_0.format(key=first_key))
    needle_ids = {d: _ids(tokenizer, NEEDLE.format(key=k, value=v)) for d, (k, v, _) in doc.items()}
    hay_len = ctx - len(head_ids) - len(tail_ids) - sum(len(n) for n in needle_ids.values())
    body = book_ids.tolist()
    if body and body[0] == tokenizer.bos_token_id:
        body = body[1:]
    offset = rng.randrange(0, max(1, len(body) - hay_len))
    hay = body[offset : offset + hay_len]
    if len(hay) < hay_len:
        raise ValueError("book too short for the requested context")
    ends = _sentence_end_ids(tokenizer, hay)

    def at_sentence_end(frac: float) -> int:
        pos = round(frac * len(hay))
        while 0 < pos < len(hay) and hay[pos - 1] not in ends:
            pos -= 1
        return pos

    points = {d: at_sentence_end(doc[d][2]) for d in doc}
    out = list(head_ids)
    starts: dict[str, int] = {}
    cursor = 0
    for d in sorted(doc, key=lambda x: (points[x], x)):
        out += hay[cursor : points[d]]
        cursor = points[d]
        starts[d] = len(out)
        out += needle_ids[d]
    out += hay[cursor:] + tail_ids
    assert len(out) == ctx, (len(out), ctx)

    # ---- later turns: passage, optional chat needle, question ------------------------------------
    p_body = passage_ids.tolist()
    if p_body and p_body[0] == tokenizer.bos_token_id:
        p_body = p_body[1:]
    p_ends = _sentence_end_ids(tokenizer, p_body)
    turns = [Turn(0, schedule[0][0], doc[schedule[0][0]][0], doc[schedule[0][0]][1], schedule[0][1], ())]
    given: dict[str, int] = {}
    for idx, (asks, introduces) in enumerate(schedule):
        if introduces:
            given[introduces] = idx
        if idx == 0:
            if introduces:
                raise ValueError("turn 0 is the document turn; it cannot introduce a chat needle")
            continue
        if asks.startswith("c") and given.get(asks, idx) >= idx:
            raise ValueError(f"turn {idx} asks {asks} before it is given")
        key, value = (doc[asks][0], doc[asks][1]) if asks.startswith("d") else chat[asks]
        question = DOC_QUESTION if asks.startswith("d") else CHAT_QUESTION
        note = (CHAT_NOTE.format(key=chat[introduces][0], value=chat[introduces][1]) + "\n") if introduces else ""
        pre, post = _split_template(tokenizer, PASSAGE_INTRO + "<<P>>\n" + note + question.format(key=key))
        # A sentence-aligned stretch of the passage book, different per turn and sample.
        start = rng.randrange(0, len(p_body) - 4 * passage_tokens)
        while start > 0 and p_body[start - 1] not in p_ends:
            start -= 1
        passage = p_body[start : start + passage_tokens]
        answer = (DOC_ANSWER if asks.startswith("d") else CHAT_ANSWER).format(key=key)
        feed = pre + passage + post + _ids(tokenizer, answer)
        turns.append(Turn(idx, asks, key, value, introduces, tuple(feed)))
    return Session(
        sample, ctx, torch.tensor(out, dtype=torch.long), tuple(turns),
        {d: (k, v, dep, starts[d]) for d, (k, v, dep) in doc.items()},
        {c: (k, v, given[c]) for c, (k, v) in chat.items()},
    )


@dataclass
class TurnResult:
    index: int
    answer_ids: list[int]
    length_before: int  # KV length when the answer started


@torch.inference_mode()
def run_session(model: PreTrainedModel, cache: Cache, first_logits: torch.Tensor, session: Session, eot: int, max_new: int, start_len: int, turn_chunk: int | None = None) -> Iterator[TurnResult]:
    """Drive a prefilled cache through every turn, one token per forward. Yields each turn's answer.

    The answer stops at the end-of-turn token or after `max_new` tokens. The end-of-turn token itself is
    not fed: every later turn's feed begins with it (the template closes the previous answer), so a
    session's KV is the same whether or not the model ended its answer on its own.

    `turn_chunk` feeds each user turn in forwards of up to that many tokens instead of one at a time.
    Rung 5 attends densely to a multi-token forward, so this processes the user's turn (the question
    included) with exact attention and only the answer sparsely, as every single-turn test did. The
    tier cannot do this (it is decode-only); the option exists to measure what that costs.
    """
    ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
    extra = cache_model_kwargs(cache)
    logits = first_logits
    length = start_len

    def step(tok: int) -> torch.Tensor:
        ids.fill_(tok)
        return model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra).logits[0, -1]

    for turn in session.turns:
        if turn_chunk and turn.feed_ids:
            feed = torch.tensor(turn.feed_ids, dtype=torch.long, device="cuda").view(1, -1)
            for s in range(0, feed.shape[1], turn_chunk):
                logits = model(input_ids=feed[:, s : s + turn_chunk], past_key_values=cache, use_cache=True, logits_to_keep=1, **extra).logits[0, -1]
            length += feed.shape[1]
        else:
            for tok in turn.feed_ids:
                logits = step(tok)
                length += 1
        answer: list[int] = []
        before = length
        while len(answer) < max_new:
            tok = int(torch.argmax(logits).item())
            if tok == eot:
                break
            answer.append(tok)
            logits = step(tok)
            length += 1
        yield TurnResult(turn.index, answer, before)


def score_turn(turn: Turn, text: str) -> float:
    return 1.0 if turn.value in text else 0.0
