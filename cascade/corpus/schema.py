"""Boundary types for the evidence corpus (spec §3.2).

The pipeline is ``fetch -> normalize -> date-validate -> dedupe -> chunk ->
embed -> index``, and each stage hands the next a different type. Keeping them
distinct is what stops a half-normalised record reaching the database: a
:class:`RawDocument` has a date that may be anything at all, a
:class:`DatedDocument` has one that has been *proved* usable, and only the
latter can be chunked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "WINDOW_END",
    "WINDOW_START",
    "Chunk",
    "CorpusSource",
    "DatedDocument",
    "DropReason",
    "RawDocument",
]

CorpusSource = Literal["gdelt", "ccnews", "wikipedia", "edgar", "govpr"]

# The temporal window the corpus covers, and therefore the range for which
# migration 003 creates partitions. These two must agree exactly: there is no
# DEFAULT partition (deliberately -- it could never be pruned), so a document
# outside the window has nowhere to go and would abort the COPY of its whole
# batch. `normalize.validate` drops such documents with `out_of_window`.
#
# The lower bound predates the earliest scenario cutoff; the upper bound is
# well past the latest resolution date. Sources legitimately return documents
# outside it -- CC-NEWS re-crawls archive pages, so a 2016 crawl can carry an
# article published in 2013 -- and those are out of scope, not errors.
WINDOW_START = datetime(2015, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2028, 1, 1, tzinfo=UTC)

DropReason = Literal[
    "missing_date",
    "naive_date",
    "future_date",
    "pre_epoch_date",
    "out_of_window",
    "empty_body",
    "too_short",
    "duplicate",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RawDocument(_Frozen):
    """What a fetcher produces: text plus whatever date the source gave.

    ``published_at`` is deliberately optional and deliberately allowed to be
    naive. The whole point of the date-validate stage is that it sees the
    unvalidated value and decides; a type that could not represent a bad date
    would push the decision back into the fetchers, where it would be made
    five different ways.
    """

    source: CorpusSource
    source_ref: str = Field(min_length=1)
    url: str = ""
    title: str = ""
    body: str
    published_at: datetime | None = None


class DatedDocument(_Frozen):
    """A document whose date has been proved usable, with its SimHash.

    Reaching this type is the guarantee that ``published_at`` is non-null,
    timezone-aware and not in the future -- the three ways a date becomes a
    leakage vector (spec §3.2).
    """

    document_id: str = Field(min_length=1)
    source: CorpusSource
    source_ref: str
    url: str
    title: str
    body: str
    published_at: datetime
    simhash: int

    @field_validator("published_at")
    @classmethod
    def _must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        return value


class Chunk(_Frozen):
    """One retrieval unit: 512 tokens with 64 of overlap, sentence-aligned.

    ``published_at`` is denormalised from the parent document so the partition
    key travels with the vector (spec §3.3). Without it the time-lock would
    need a join and would stop being structural.
    """

    chunk_id: str = Field(min_length=1)
    document_id: str
    ordinal: int = Field(ge=0)
    body: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    published_at: datetime
