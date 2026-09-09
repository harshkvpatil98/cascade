"""Sentence-aligned chunking: 512 tokens, 64 overlap (spec §3.2).

Pure: no I/O, no clock, no RNG, and no model. The tokenizer is injected as a
callable so the same code can be property-tested with a trivial counter and
run in production against the embedding model's own tokenizer -- which is the
only counter whose 512 means what the model means by 512.

Two rules interact and the order matters:

*Sentence alignment* means a chunk never ends mid-sentence. A retrieved chunk
is shown to an agent as evidence, and half a sentence is evidence of nothing.

*The 512-token cap* is hard, because it is the model's context window for a
single embedding. When one sentence exceeds it on its own -- SEC filings and
legislative text do this routinely -- alignment has to yield, so the sentence
is split on a token boundary rather than dropped or truncated.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

__all__ = [
    "MAX_TOKENS",
    "OVERLAP_TOKENS",
    "ChunkPlan",
    "chunk_text",
    "split_sentences",
    "whitespace_token_count",
]

MAX_TOKENS = 512
OVERLAP_TOKENS = 64

TokenCounter = Callable[[str], int]
BatchTokenCounter = Callable[[Sequence[str]], list[int]]

# Sentence terminator followed by whitespace and something that can begin a
# sentence. The lookbehind excludes the common abbreviations that would
# otherwise split a sentence in the middle of a name or a citation.
_ABBREVIATIONS = (
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Sr",
    "Jr",
    "St",
    "Inc",
    "Ltd",
    "Co",
    "Corp",
    "vs",
    "etc",
    "al",
    "Fig",
    "No",
    "Rep",
    "Sen",
    "Gov",
    "Gen",
    "U.S",
    "U.K",
    "E.U",
    "a.m",
    "p.m",
    "i.e",
    "e.g",
    # Month abbreviations. "Filed Jan. 4" is a date, and splitting there cuts
    # a sentence in half -- which SEC filings and press releases do constantly.
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
)
_ABBREVIATION_SET = frozenset(_ABBREVIATIONS)
# A terminator followed by whitespace and something that can open a sentence.
# The preceding word is checked separately: Python's `re` only allows
# fixed-width lookbehind, and the abbreviation list is not fixed width.
_SENTENCE_END = re.compile(r"([.!?][\"'\u201d\u2019)\]]*)(\s+)(?=[\"'\u201c\u2018(\[]*[A-Z0-9])")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)[.!?][\"'\u201d\u2019)\]]*$")

_PARAGRAPH = re.compile(r"\n\s*\n+")


def whitespace_token_count(text: str) -> int:
    """A tokenizer-free counter, for tests and for a dry run without the model.

    Deliberately an *over*-estimate of wordpiece counts (roughly 1.3 pieces per
    whitespace word for English prose), so a chunk planned with it never
    overflows the real 512-token window. Production always injects the model's
    own tokenizer -- see ``cascade.corpus.embed``.
    """
    words = text.split()
    return max(1, int(len(words) * 1.3) + 1)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, preserving paragraph boundaries as hard breaks.

    Paragraph breaks are treated as sentence breaks even without terminal
    punctuation: headlines, list items and table rows frequently carry none,
    and gluing them onto the following paragraph produces chunks that read as
    nonsense.

    A candidate break is suppressed when the word before it is a known
    abbreviation, so "Dr. Smith" and "U.S. policy" stay whole.
    """
    out: list[str] = []
    for paragraph in _PARAGRAPH.split(text.strip()):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        start = 0
        for match in _SENTENCE_END.finditer(cleaned):
            candidate = cleaned[start : match.end(1)]
            trailing = _TRAILING_WORD.search(candidate)
            if trailing and trailing.group(1).rstrip(".") in _ABBREVIATION_SET:
                continue
            piece = candidate.strip()
            if piece:
                out.append(piece)
            start = match.end()
        remainder = cleaned[start:].strip()
        if remainder:
            out.append(remainder)
    return out


@dataclass(frozen=True)
class ChunkPlan:
    """One planned chunk: its text and the token count that produced it."""

    body: str
    token_count: int


def _special_token_overhead(count_tokens: TokenCounter) -> int:
    """Tokens a counter adds for [CLS]/[SEP] regardless of content.

    Measured once, because the per-piece costs below are summed: adding two
    special tokens per sentence would over-count a 40-sentence chunk by 80
    tokens and split it early for no reason.
    """
    return max(0, count_tokens(""))


def _hard_split(
    sentence: str, count_tokens: TokenCounter, limit: int, overhead: int
) -> Iterator[str]:
    """Split an over-long sentence on word boundaries under ``limit`` tokens.

    Word boundaries rather than characters: a fragment cut mid-word embeds
    badly and reads worse. This is the one place alignment yields to the cap.

    Each word is measured **once** and the costs are accumulated. Re-measuring
    the growing prefix after every word is quadratic in sentence length, and
    SEC filings and legislative text routinely produce single "sentences" of
    thousands of words -- which turned a whole-corpus ingest from minutes into
    something that never finished.
    """
    words = sentence.split()
    if not words:
        return
    budget = max(1, limit - overhead)
    current: list[str] = []
    running = 0
    for word in words:
        cost = max(1, count_tokens(word) - overhead)
        if current and running + cost > budget:
            yield " ".join(current)
            current = [word]
            running = cost
        else:
            current.append(word)
            running += cost
    if current:
        yield " ".join(current)


def _overlap_tail(
    sentences: Sequence[str], costs: Sequence[int], overlap: int
) -> tuple[list[str], int]:
    """Return the trailing sentences worth about ``overlap`` tokens.

    Overlap is what stops a fact that straddles a chunk boundary from being
    invisible to both chunks. Taken in whole sentences for the same reason
    chunks are: a partial sentence is not evidence.

    Costs are passed in rather than recomputed -- the caller has already
    measured every sentence exactly once.
    """
    tail: list[str] = []
    total = 0
    for index in range(len(sentences) - 1, -1, -1):
        # No "keep at least one" clause. A trailing sentence larger than the
        # overlap budget would be carried into the next chunk in full, and the
        # next chunk would then exceed max_tokens -- measured at 1,022 tokens
        # against a 512 cap on hard-split input. Overlap is a nicety; the cap
        # is the model's context window, so the cap wins and overlap is simply
        # omitted when it will not fit.
        if total + costs[index] > overlap:
            break
        tail.insert(0, sentences[index])
        total += costs[index]
    return tail, total


def chunk_text(
    text: str,
    *,
    count_tokens: TokenCounter = whitespace_token_count,
    count_tokens_batch: BatchTokenCounter | None = None,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[ChunkPlan]:
    """Split ``text`` into sentence-aligned chunks under ``max_tokens``.

    Preserves the two invariants the retrieval layer depends on: every chunk
    fits the embedding model's window, and no chunk ends mid-sentence unless a
    single sentence could not fit on its own.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive; got {max_tokens}")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            f"overlap_tokens must be in [0, max_tokens); got {overlap_tokens} of {max_tokens}"
        )

    overhead = _special_token_overhead(count_tokens)
    budget = max(1, max_tokens - overhead)

    raw_sentences = split_sentences(text)
    if count_tokens_batch is not None and raw_sentences:
        raw_costs = [max(1, size - overhead) for size in count_tokens_batch(raw_sentences)]
    else:
        raw_costs = [max(1, count_tokens(sentence) - overhead) for sentence in raw_sentences]

    sentences: list[str] = []
    for sentence, cost in zip(raw_sentences, raw_costs, strict=True):
        if cost > budget:
            sentences.extend(_hard_split(sentence, count_tokens, max_tokens, overhead))
        else:
            sentences.append(sentence)

    # Every sentence is measured exactly once; the packing loop and the
    # overlap tail both read from here. Measured in one batched call when the
    # caller supplies one -- per-sentence calls dominate ingest CPU.
    if count_tokens_batch is not None and sentences:
        costs = [max(1, size - overhead) for size in count_tokens_batch(sentences)]
    else:
        costs = [max(1, count_tokens(sentence) - overhead) for sentence in sentences]

    plans: list[ChunkPlan] = []
    current: list[str] = []
    current_costs: list[int] = []
    running = 0

    def emit() -> None:
        plans.extend(_verified(current, count_tokens, max_tokens))

    for sentence, cost in zip(sentences, costs, strict=True):
        if current and running + cost > budget:
            emit()
            current, running = _overlap_tail(current, current_costs, overlap_tokens)
            current_costs = current_costs[len(current_costs) - len(current) :] if current else []
        current.append(sentence)
        current_costs.append(cost)
        running += cost

    if current:
        emit()
    return plans


def _verified(
    sentences: Sequence[str], count_tokens: TokenCounter, max_tokens: int, depth: int = 0
) -> list[ChunkPlan]:
    """Measure a planned chunk exactly, splitting further if it overflows.

    The packing loop works from per-sentence costs, which are summed. Summing
    assumes the counter is additive, and no tokenizer is exactly additive --
    wordpiece merges across a join, and the whitespace fallback is off by its
    own constant per piece. Rather than trusting the estimate, every emitted
    chunk is measured once against the real counter and split if it is over.

    Splitting happens **at sentence boundaries** while more than one sentence
    remains. Splitting the joined text by words instead would silently break
    the alignment guarantee -- chunks would end mid-sentence for a reason that
    has nothing to do with the one documented exception. Word-level splitting
    is reached only for a single sentence that exceeds the cap on its own.

    This makes the 512-token cap a *verified* property rather than a hoped-for
    one, and makes the stored ``token_count`` exact instead of estimated --
    which matters because M3 reports retrieval quality per chunk length.
    """
    parts = [part for part in sentences if part.strip()]
    if not parts:
        return []
    body = " ".join(parts).strip()
    actual = count_tokens(body)
    # The depth guard cannot be reached by ordinary text: each level at least
    # halves the input, and character halving converges even without spaces.
    # It exists so a pathological counter cannot spin.
    if actual <= max_tokens or depth >= 24:
        return [ChunkPlan(body=body, token_count=actual)]

    if len(parts) == 1:
        words = parts[0].split()
        if len(words) >= 2:
            middle = len(words) // 2
            return [
                *_verified([" ".join(words[:middle])], count_tokens, max_tokens, depth + 1),
                *_verified([" ".join(words[middle:])], count_tokens, max_tokens, depth + 1),
            ]
        # No whitespace to split on. Japanese and Chinese prose has none, and
        # neither does a minified JSON blob that survived HTML extraction --
        # both were measured in the corpus at up to 2,125 tokens, four times
        # the cap, because word-splitting cannot divide a single "word".
        # Characters always can, so the cap holds for any input rather than
        # only for whitespace-delimited languages.
        if len(body) < 2:
            return [ChunkPlan(body=body, token_count=actual)]
        middle = len(body) // 2
        return [
            *_verified([body[:middle]], count_tokens, max_tokens, depth + 1),
            *_verified([body[middle:]], count_tokens, max_tokens, depth + 1),
        ]

    middle = len(parts) // 2
    return [
        *_verified(parts[:middle], count_tokens, max_tokens, depth + 1),
        *_verified(parts[middle:], count_tokens, max_tokens, depth + 1),
    ]
