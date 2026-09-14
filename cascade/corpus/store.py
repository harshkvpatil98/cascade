"""Corpus persistence: bulk writes, resume state, and verification queries.

Writes go through ``COPY`` rather than ``INSERT``. At 1.3M chunks the
difference is not a micro-optimisation -- row-at-a-time inserts turn a
minutes-long load into an hours-long one, and a load that takes hours is one
nobody re-runs, which quietly ends the reproducibility claim.

Documents and their chunks are written in **one transaction**. A document row
without its chunks is invisible to retrieval while still counting against the
corpus total, and the count is an acceptance criterion.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from cascade.config import Settings
from cascade.corpus.schema import Chunk, DatedDocument
from cascade.corpus.simhash import from_signed, to_signed

__all__ = [
    "CorpusStats",
    "ScenarioCoverage",
    "completed_units",
    "corpus_stats",
    "mark_unit",
    "scenario_coverage",
    "seen_simhashes",
    "write_batch",
]


def _connect(settings: Settings) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url("admin"), connect_timeout=30)


def _vector_literal(values: Sequence[float]) -> str:
    """Render a vector for pgvector's text input format."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def write_batch(
    settings: Settings,
    documents: Sequence[DatedDocument],
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> tuple[int, int]:
    """Write documents and their embedded chunks in one transaction.

    Returns ``(documents_written, chunks_written)``. Raises if the chunk and
    embedding counts disagree -- a chunk stored without its vector is
    unreachable by retrieval and would silently fail the 100% coverage
    criterion.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} vectors"
        )
    if not documents:
        return 0, 0

    counts = {document.document_id: 0 for document in documents}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1

    with _connect(settings) as conn, conn.cursor() as cur:
        with cur.copy(
            "COPY documents (document_id, source, source_ref, url, title, "
            "published_at, simhash, n_chunks) FROM STDIN"
        ) as copy:
            for document in documents:
                copy.write_row(
                    (
                        document.document_id,
                        document.source,
                        document.source_ref,
                        document.url,
                        document.title,
                        document.published_at,
                        to_signed(document.simhash),
                        counts.get(document.document_id, 0),
                    )
                )

        with cur.copy(
            "COPY chunks (chunk_id, document_id, ordinal, body, token_count, "
            "published_at, embedding) FROM STDIN"
        ) as copy:
            for chunk, vector in zip(chunks, embeddings, strict=True):
                copy.write_row(
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.body,
                        chunk.token_count,
                        chunk.published_at,
                        _vector_literal(vector),
                    )
                )
        conn.commit()
    return len(documents), len(chunks)


def completed_units(settings: Settings, source: str) -> set[str]:
    """Units already finished, so a restart can skip them (invariant 8)."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT unit_key FROM corpus_ingest_state WHERE source = %s AND state = 'done'",
            (source,),
        )
        return {str(row[0]) for row in cur.fetchall()}


def mark_unit(
    settings: Settings,
    *,
    source: str,
    unit_key: str,
    state: str,
    n_documents: int = 0,
    n_chunks: int = 0,
    detail: str = "",
) -> None:
    """Record a unit's outcome. Idempotent, so a re-run overwrites cleanly."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO corpus_ingest_state (source, unit_key, state, n_documents, "
            "n_chunks, detail, updated_at) VALUES (%s, %s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (source, unit_key) DO UPDATE SET state = EXCLUDED.state, "
            "n_documents = EXCLUDED.n_documents, n_chunks = EXCLUDED.n_chunks, "
            "detail = EXCLUDED.detail, updated_at = now()",
            (source, unit_key, state, n_documents, n_chunks, detail[:500]),
        )
        conn.commit()


def seen_simhashes(settings: Settings) -> list[int]:
    """Every fingerprint already in the corpus, for cross-run dedupe."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT simhash FROM documents")
        return [from_signed(int(row[0])) for row in cur.fetchall()]


def existing_document_ids(settings: Settings) -> set[str]:
    """Document ids already stored, so a re-run cannot violate the primary key."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT document_id FROM documents")
        return {str(row[0]) for row in cur.fetchall()}


@dataclass(frozen=True)
class CorpusStats:
    """Everything the M2 acceptance criteria are measured from."""

    n_documents: int
    n_chunks: int
    per_source: tuple[tuple[str, int, int], ...]
    null_dates: int
    future_dates: int
    missing_embeddings: int
    earliest: Any = None
    latest: Any = None

    @property
    def embedding_coverage(self) -> float:
        if not self.n_chunks:
            return 0.0
        return (self.n_chunks - self.missing_embeddings) / self.n_chunks


def corpus_stats(settings: Settings) -> CorpusStats:
    """Measure the corpus over the **full** table, never a sample.

    Spec §3.2's date criterion says "assert over the full table, not a
    sample", so every count here is an unqualified aggregate. A sampled check
    on a leakage invariant is a check that can miss the one row that matters.
    """
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        n_documents = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT count(*) FROM chunks")
        n_chunks = int((cur.fetchone() or [0])[0])

        # Reads the denormalised `documents.n_chunks` rather than joining
        # documents to chunks. The join is over two partitioned tables on a
        # non-partition key, so at corpus scale it degenerates into a full
        # scan of both -- minutes of work to produce a number that
        # `write_batch` already computed and stored.
        cur.execute(
            "SELECT source, count(*), coalesce(sum(n_chunks), 0) "
            "FROM documents GROUP BY source ORDER BY source"
        )
        per_source = tuple((str(r[0]), int(r[1]), int(r[2])) for r in cur.fetchall())

        cur.execute("SELECT count(*) FROM documents WHERE published_at IS NULL")
        null_dates = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT count(*) FROM documents WHERE published_at > now()")
        future_dates = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        missing = int((cur.fetchone() or [0])[0])

        cur.execute("SELECT min(published_at), max(published_at) FROM documents")
        row = cur.fetchone() or (None, None)

    return CorpusStats(
        n_documents=n_documents,
        n_chunks=n_chunks,
        per_source=per_source,
        null_dates=null_dates,
        future_dates=future_dates,
        missing_embeddings=missing,
        earliest=row[0],
        latest=row[1],
    )


@dataclass(frozen=True)
class ScenarioCoverage:
    """Admissible evidence for one scenario, measured at its own cutoff."""

    scenario_id: str
    cutoff_ts: Any
    chunks_in_window: int
    chunks_before_cutoff: int
    latest_admissible: Any
    staleness_days: int | None

    def under_covered(self, minimum: int) -> bool:
        return self.chunks_in_window < minimum


def scenario_coverage(settings: Settings, *, lookback_months: int) -> tuple[ScenarioCoverage, ...]:
    """Per-scenario evidence availability, measured at each scenario's cutoff.

    Answers the question the chunk count cannot: not "is the corpus big" but
    "does this scenario have anything recent to read". ``chunks_in_window``
    counts chunks published in the ``lookback_months`` before the cutoff;
    ``staleness_days`` is the gap between the cutoff and the most recent
    admissible chunk, which is the number that exposes a corpus concentrated
    in the wrong years.

    Reads ``scenarios`` and ``chunks``. No outcome is touched (invariant 2),
    and the cutoff comes from the registry rather than an argument, so this
    cannot be asked about a time a scenario was not sealed at.

    **Measured at day resolution, from one pass over the corpus.** A
    correlated count per scenario is 180 aggregates over a 1.8M-row
    partitioned table; a single day histogram is ~4,000 rows and the rest is
    arithmetic. Days are compared strictly before the cutoff's own day, so a
    chunk published earlier on the cutoff date is excluded -- the error is
    always in the direction of reporting *less* coverage than exists, which is
    the safe direction for a warning.
    """
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT published_at::date AS day, count(*) FROM chunks GROUP BY 1 ORDER BY 1")
        histogram = [(row[0], int(row[1])) for row in cur.fetchall()]
        cur.execute("SELECT scenario_id, cutoff_ts FROM scenarios ORDER BY scenario_id")
        scenarios = cur.fetchall()

    days = [day for day, _ in histogram]
    counts = [count for _, count in histogram]
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)

    coverage: list[ScenarioCoverage] = []
    for scenario_id, cutoff_ts in scenarios:
        cutoff_day = cutoff_ts.date()
        window_start = (cutoff_ts - timedelta(days=int(lookback_months * 30.436875))).date()
        end = bisect_left(days, cutoff_day)
        start = bisect_left(days, window_start)
        latest = days[end - 1] if end > 0 else None
        coverage.append(
            ScenarioCoverage(
                scenario_id=str(scenario_id),
                cutoff_ts=cutoff_ts,
                chunks_in_window=prefix[end] - prefix[start],
                chunks_before_cutoff=prefix[end],
                latest_admissible=latest,
                staleness_days=(cutoff_day - latest).days if latest is not None else None,
            )
        )
    return tuple(coverage)
