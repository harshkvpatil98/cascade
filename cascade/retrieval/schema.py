"""Boundary types for Chronofence (spec §4.2).

Every type here carries ``as_of`` or is derived from something that did. That
is deliberate: a retrieval result with no record of the cutoff it was taken
under cannot be audited later, and M8 has to be able to walk from an outcome
back to the evidence that produced it and prove the evidence was admissible.

Pydantic v2 at the boundary, per the engineering standards -- a raw dict
crossing out of this package is a bug.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "IndexAction",
    "IndexPlan",
    "IndexReport",
    "LatencySummary",
    "PartitionIndex",
    "RetrievedChunk",
    "SearchResult",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievedChunk(_Frozen):
    """One chunk returned by ``chronofence_search``.

    ``published_at`` is carried out of the database rather than dropped so the
    caller can re-assert the time lock without trusting it. The leakage suite
    does exactly that: a property test over the full retrieval trace checks
    ``published_at < as_of`` on every row that ever reached an agent, which
    would catch a regression in the SQL that the SQL's own tests missed.
    """

    chunk_id: str
    document_id: str
    ordinal: int
    body: str
    published_at: datetime
    source: str
    url: str
    title: str
    distance: float


class SearchResult(_Frozen):
    """The chunks one query returned, with the cutoff they were taken under."""

    as_of: datetime
    k: int
    chunks: tuple[RetrievedChunk, ...]
    elapsed_ms: float

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    def latest_published_at(self) -> datetime | None:
        """The newest evidence returned, or None when nothing was."""
        if not self.chunks:
            return None
        return max(chunk.published_at for chunk in self.chunks)


class PartitionIndex(_Frozen):
    """Measured state of one ``chunks`` partition and its IVFFlat index."""

    partition: str
    rows: int
    index_name: str | None
    lists: int | None


IndexAction = str  # one of: "create", "rebuild", "keep", "skip-empty"


class IndexPlan(_Frozen):
    """What ``cascade retrieval index`` intends to do to one partition.

    Separated from execution so the sizing rule is a pure function of measured
    row counts and can be property-tested without a database.
    """

    partition: str
    rows: int
    current_lists: int | None
    target_lists: int
    action: IndexAction
    reason: str


class IndexReport(_Frozen):
    """What it actually did."""

    plans: tuple[IndexPlan, ...]
    created: int
    rebuilt: int
    kept: int
    skipped_empty: int
    elapsed_s: float


class LatencySummary(_Frozen):
    """Latency percentiles over a bench run, in milliseconds.

    Percentiles are reported, never a mean. A mean latency hides exactly the
    tail the 15 ms budget is about.
    """

    n: int = Field(ge=0)
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    histogram: tuple[tuple[float, float, int], ...]
    """(lower_ms, upper_ms, count) buckets, contiguous and ordered."""
