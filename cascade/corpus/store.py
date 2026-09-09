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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cascade.config import Settings
from cascade.corpus.schema import Chunk, DatedDocument
from cascade.corpus.simhash import from_signed, to_signed

__all__ = [
    "CorpusStats",
    "completed_units",
    "corpus_stats",
    "mark_unit",
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
