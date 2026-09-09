"""Sentence-aligned chunking: 512 tokens, 64 overlap (spec §3.2).

The 512-token cap is the embedding model's context window. A chunk over it is
silently truncated at embed time, so its tail sits in the database as text
that is absent from its own vector -- retrievable by keyword, invisible to
retrieval. That makes the cap a correctness property, not a preference, and it
is asserted here for every input shape including the pathological ones.
"""

from __future__ import annotations

import pytest

from cascade.corpus.chunker import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    chunk_text,
    split_sentences,
    whitespace_token_count,
)

PROSE = " ".join(f"Sentence number {index} concerns the negotiation." for index in range(400))


def test_spec_constants() -> None:
    assert MAX_TOKENS == 512
    assert OVERLAP_TOKENS == 64


# ---------------------------------------------------------------------------
# Sentence boundaries
# ---------------------------------------------------------------------------


def test_sentences_split_on_terminators() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith met Mr. Jones today at noon.",
        "The U.S. and the E.U. signed the accord.",
        "Filed with the SEC on Jan. 4 by Acme Inc. and reviewed.",
    ],
)
def test_abbreviations_do_not_split_a_sentence(text: str) -> None:
    """ "Dr." and "U.S." end in a period but do not end a sentence."""
    assert split_sentences(text) == [text]


def test_paragraph_breaks_are_hard_boundaries() -> None:
    """Headlines and list items carry no terminator; gluing them is nonsense."""
    assert split_sentences("Headline here\n\nBody follows.") == [
        "Headline here",
        "Body follows.",
    ]


def test_blank_input_yields_no_sentences() -> None:
    assert split_sentences("   \n\n  ") == []


# ---------------------------------------------------------------------------
# The cap -- the property that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("empty", ""),
        ("one word", "hi"),
        ("prose", PROSE),
        ("one enormous sentence", "word " * 5000),
        ("no terminators", "alpha beta gamma " * 900),
        ("many paragraphs", "\n\n".join("Para body text here." for _ in range(500))),
        ("unicode", "Ünïcödé sentence with accents. " * 300),
    ],
)
def test_no_chunk_ever_exceeds_the_cap(name: str, text: str) -> None:
    """Verified, not estimated: every emitted chunk is measured.

    The packing loop sums per-sentence costs, and no tokenizer is exactly
    additive, so the emitted body is re-measured and split further if it is
    over. This test is what makes that guarantee real.
    """
    for plan in chunk_text(text):
        assert plan.token_count <= MAX_TOKENS, name
        assert whitespace_token_count(plan.body) <= MAX_TOKENS, name


def test_reported_token_count_is_exact() -> None:
    """M3 reports retrieval quality per chunk length; an estimate would skew it."""
    for plan in chunk_text(PROSE):
        assert plan.token_count == whitespace_token_count(plan.body)


def test_chunks_are_non_empty() -> None:
    assert all(plan.body.strip() for plan in chunk_text(PROSE))


def test_chunking_is_deterministic() -> None:
    assert [plan.body for plan in chunk_text(PROSE)] == [plan.body for plan in chunk_text(PROSE)]


def test_every_sentence_survives_somewhere() -> None:
    """Chunking must not drop content; overlap may repeat it."""
    joined = " ".join(plan.body for plan in chunk_text(PROSE))
    for index in (0, 200, 399):
        assert f"Sentence number {index} concerns the negotiation." in joined


def test_consecutive_chunks_overlap() -> None:
    """Overlap stops a fact straddling a boundary from being invisible to both."""
    plans = chunk_text(PROSE)
    assert len(plans) > 2
    first_words = set(plans[0].body.split())
    second_words = set(plans[1].body.split())
    assert first_words & second_words


def test_chunks_are_sentence_aligned_when_sentences_fit() -> None:
    """A chunk ends at a sentence end; half a sentence is evidence of nothing."""
    for plan in chunk_text(PROSE):
        assert plan.body.rstrip().endswith((".", "!", "?"))


def test_an_oversized_sentence_is_split_rather_than_dropped() -> None:
    """Alignment yields to the cap, but the content stays."""
    plans = chunk_text("word " * 3000)
    assert plans
    assert sum(plan.body.count("word") for plan in plans) >= 3000


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_a_non_positive_cap_is_refused(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_text(PROSE, max_tokens=max_tokens)


@pytest.mark.parametrize("overlap", [-1, 512, 900])
def test_overlap_must_be_below_the_cap(overlap: int) -> None:
    """Overlap at or above the cap cannot terminate."""
    with pytest.raises(ValueError, match="overlap_tokens must be in"):
        chunk_text(PROSE, overlap_tokens=overlap)


def test_zero_overlap_is_allowed() -> None:
    plans = chunk_text(PROSE, overlap_tokens=0)
    assert plans
    assert all(plan.token_count <= MAX_TOKENS for plan in plans)


def test_a_counter_that_overstates_still_respects_the_cap() -> None:
    """Guard against trusting additivity: a hostile counter must not break it."""
    plans = chunk_text(PROSE, count_tokens=lambda text: len(text.split()) * 3 + 5)
    assert plans
    assert all(plan.token_count <= MAX_TOKENS for plan in plans)


# ---------------------------------------------------------------------------
# Text without whitespace
#
# Regression: Japanese prose and a minified JSON blob both reached the corpus
# at up to 2,125 tokens -- four times the cap -- because word-splitting cannot
# divide a string with no spaces in it. Character splitting always can.
# ---------------------------------------------------------------------------

JAPANESE = (
    "即日キャッシングとは、口コミなどでも言われているとおり申込当日に、資金の入金をしてくれます。"
    * 40
)
CHINESE = "今天中央银行宣布提高利率这是市场普遍预期的举措" * 200
JSON_BLOB = (
    '{"pages":['
    + ",".join('{"title":"01-607","src":"http://example.com/a/b"}' for _ in range(300))
    + "]}"
)


@pytest.mark.parametrize(
    ("name", "text"),
    [("japanese", JAPANESE), ("chinese", CHINESE), ("json blob", JSON_BLOB)],
)
def test_text_without_whitespace_still_respects_the_cap(name: str, text: str) -> None:
    plans = chunk_text(text)
    assert plans, name
    for plan in plans:
        assert plan.token_count <= MAX_TOKENS, name


def test_no_whitespace_content_is_preserved() -> None:
    """Splitting on characters must not drop any of the text."""
    rebuilt = "".join(plan.body for plan in chunk_text(JAPANESE)).replace(" ", "")
    assert rebuilt == JAPANESE.replace(" ", "")
