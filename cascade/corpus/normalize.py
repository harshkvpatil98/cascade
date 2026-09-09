"""Date validation and near-duplicate collapse (spec §3.2).

Pure: no I/O, no RNG, and **no clock** -- ``now`` is a required argument.
That is invariant 1's reasoning applied one layer down: a function that reads
the wall clock to decide whether a document is "in the future" gives a
different answer depending on when it runs, and a corpus built at 09:00 would
differ from the same corpus rebuilt at 17:00.

The two stages here are the corpus's leakage defences:

*date-validate* drops any document whose date is null, naive or in the future.
A document with an uncertain date cannot be placed relative to a cutoff, and
guessing is how post-cutoff evidence reaches an agent.

*dedupe* collapses near-identical documents and keeps the **earliest** date.
Wire copy is syndicated within minutes; keeping a later copy would move
evidence forward in time past a cutoff that should have excluded it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cascade.corpus.schema import (
    WINDOW_END,
    WINDOW_START,
    DatedDocument,
    DropReason,
    RawDocument,
)
from cascade.corpus.simhash import HASH_BITS, hamming, simhash

__all__ = [
    "BANDS",
    "BAND_BITS",
    "MAX_HAMMING",
    "MIN_BODY_CHARS",
    "DedupeIndex",
    "DedupeResult",
    "deduplicate",
    "sanitize",
    "validate",
]

# Spec §3.2: "SimHash 64-bit, Hamming distance <= 3 collapses to one row".
MAX_HAMMING = 3

# Pigeonhole banding. Two hashes within `MAX_HAMMING` bits must agree exactly
# on at least one of `MAX_HAMMING + 1` disjoint bands, so candidates can be
# found by dictionary lookup instead of comparing every pair. At 1.3M
# documents the pairwise scan would be ~8.5e11 comparisons; this is linear.
BANDS = MAX_HAMMING + 1
BAND_BITS = HASH_BITS // BANDS

# Shorter than this and a "document" is a headline stub or a navigation
# fragment: it produces one degenerate chunk and no usable evidence.
MIN_BODY_CHARS = 200

# Nothing in the study predates this, and a date below it signals a parse
# failure (a Unix epoch zero, a two-digit year read as 0019) rather than a
# genuinely old document.
_EARLIEST_PLAUSIBLE = datetime(1990, 1, 1)

# Clock skew allowance. A publisher timestamping a few minutes ahead of our
# clock is not a leakage risk; a document dated next week is.
_FUTURE_TOLERANCE = timedelta(hours=6)

# Control characters Postgres will not store in a text column. NUL is the one
# that actually occurs -- scraped HTML and SGML filings carry it routinely --
# and it aborts a COPY of the whole batch, not just the offending row. Tab,
# newline and carriage return are kept because the chunker needs paragraph
# structure to find sentence boundaries.
_ILLEGAL_CONTROL = {codepoint: None for codepoint in range(32) if codepoint not in (9, 10, 13)}
_ILLEGAL_CONTROL[0x7F] = None


def sanitize(text: str) -> str:
    """Strip control characters no text column can hold.

    Applied at the single funnel every document passes through rather than in
    each fetcher, so a new source cannot reintroduce the problem by forgetting
    to call it.
    """
    return text.translate(_ILLEGAL_CONTROL)


def validate(
    document: RawDocument,
    *,
    now: datetime,
    window: tuple[datetime, datetime] = (WINDOW_START, WINDOW_END),
) -> DatedDocument | DropReason:
    """Return a dated document, or the reason it cannot be used.

    Preserves the corpus's half of the time-lock: everything downstream may
    assume ``published_at`` is non-null, timezone-aware, and not in the
    future. ``now`` is required rather than defaulted for the same reason
    ``as_of`` is (invariant 1) -- a defaulted clock makes the result depend on
    when the pipeline ran.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware; a naive clock cannot order events")

    body = sanitize(document.body).strip()
    if not body:
        return "empty_body"
    if len(body) < MIN_BODY_CHARS:
        return "too_short"

    published = document.published_at
    if published is None:
        return "missing_date"
    if published.tzinfo is None or published.utcoffset() is None:
        # Never coerced to UTC: an assumed offset is an invented instant, and
        # a document an hour on the wrong side of a cutoff is a leak.
        return "naive_date"
    if published > now + _FUTURE_TOLERANCE:
        return "future_date"
    if published.replace(tzinfo=None) < _EARLIEST_PLAUSIBLE:
        return "pre_epoch_date"
    if not window[0] <= published < window[1]:
        # Real date, outside the corpus's stated scope. Dropping is correct
        # and is not a failure: there is no partition for it, and inventing
        # one would create a bucket the M3 time-lock cannot prune.
        return "out_of_window"

    return DatedDocument(
        document_id=f"{document.source}:{document.source_ref}",
        source=document.source,
        source_ref=document.source_ref,
        url=sanitize(document.url)[:2000],
        title=sanitize(document.title)[:1000],
        body=body,
        published_at=published,
        simhash=simhash(f"{document.title}\n{body}"),
    )


@dataclass
class DedupeResult:
    """What a dedupe pass kept, and what it collapsed."""

    kept: list[DatedDocument] = field(default_factory=list)
    collapsed: int = 0

    @property
    def seen(self) -> int:
        return len(self.kept) + self.collapsed

    @property
    def collapse_ratio(self) -> float:
        """Fraction of input documents absorbed into an earlier near-duplicate."""
        return self.collapsed / self.seen if self.seen else 0.0


def _bands(value: int) -> list[tuple[int, int]]:
    """Split a hash into ``BANDS`` disjoint segments, tagged by position."""
    mask = (1 << BAND_BITS) - 1
    return [(index, value >> (index * BAND_BITS) & mask) for index in range(BANDS)]


class DedupeIndex:
    """Incremental near-duplicate index over 64-bit SimHashes.

    Held across ingest units so a story syndicated into two different months
    still collapses. Seeded from the database on resume, because a corpus half
    built in a previous run would otherwise re-admit every document it already
    holds (invariant 8).
    """

    __slots__ = ("_bands", "_hashes")

    def __init__(self) -> None:
        self._bands: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._hashes: list[int] = []

    def __len__(self) -> int:
        return len(self._hashes)

    def find(self, value: int) -> int | None:
        """Return the position of a near-duplicate, or ``None``."""
        for band in _bands(value):
            for position in self._bands[band]:
                if hamming(self._hashes[position], value) <= MAX_HAMMING:
                    return position
        return None

    def add(self, value: int) -> int:
        """Register ``value`` and return its position."""
        position = len(self._hashes)
        self._hashes.append(value)
        for band in _bands(value):
            self._bands[band].append(position)
        return position

    def seed(self, values: Iterable[int]) -> None:
        """Load fingerprints already in the corpus."""
        for value in values:
            self.add(value)


def deduplicate(
    documents: Iterable[DatedDocument], *, index: DedupeIndex | None = None
) -> DedupeResult:
    """Collapse near-duplicates, keeping the earliest ``published_at``.

    Preserves the rule that a syndicated story enters the corpus at the moment
    it first became public. When a later copy collapses into an earlier one the
    earlier date survives; when an *earlier* copy arrives after a later one,
    the surviving record's date is moved back, so the result does not depend
    on arrival order.

    Passing an ``index`` carries duplicate detection across calls -- positions
    seeded from already-stored documents have no in-memory record, so a match
    against one collapses the incoming document without resurrecting the
    stored one.

    Deterministic: input is processed in the order given.
    """
    result = DedupeResult()
    shared = index if index is not None else DedupeIndex()
    offset = len(shared)
    canonical: list[DatedDocument] = []

    for document in documents:
        match = shared.find(document.simhash)
        if match is None:
            shared.add(document.simhash)
            canonical.append(document)
            continue

        result.collapsed += 1
        local = match - offset
        if not 0 <= local < len(canonical):
            # Matched a document stored by an earlier run. It is already in
            # the corpus with its own date; this copy is simply dropped.
            continue
        existing = canonical[local]
        if document.published_at < existing.published_at:
            # The earlier copy arrived second. Keep the earlier instant --
            # otherwise the corpus would date this evidence later than it
            # actually became public, which is a cutoff error in the unsafe
            # direction.
            canonical[local] = existing.model_copy(update={"published_at": document.published_at})

    result.kept = canonical
    return result
