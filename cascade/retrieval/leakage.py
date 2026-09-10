"""Leakage instrumentation for Chronofence (spec §4.2).

The four probes in §4.2 answer four different questions, and only the first
three are engineering problems:

1. **Poison pill** -- can a document dated after resolution reach retrieval?
   Synthesises documents that state each scenario's outcome, dates them one day
   post-resolution, and asserts none is ever returned. Tests the mechanism with
   evidence that is *maximally* attractive to the retriever: the poison text is
   built from the scenario's own question, so it sits as close to the query
   vector as anything in the corpus. A time lock that survives this survives
   ordinary documents trivially.

2. **Signature scan** -- does anything already in the corpus read like
   resolution text? Flags, does not fail. A pre-cutoff document discussing the
   question is legitimate evidence, and dropping it would be curating the
   corpus toward a conclusion.

3. **Date monotonicity** -- did every chunk ever returned satisfy
   ``published_at < as_of``? A property over the whole retrieval trace rather
   than over the SQL, so a regression in the function is caught by the data it
   produced.

4. **Parametric probe** -- does the agent model already know the answer from
   pre-training? See :mod:`cascade.retrieval.memorization`. That one cannot be
   fixed by engineering; it is measured and disclosed.

The poison-pill helpers write to the corpus, so they take an explicit
``psycopg`` connection rather than opening their own: the caller owns the
transaction and is expected to roll it back, which is what keeps a leakage
probe from permanently contaminating the evidence corpus it is testing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from cascade.ledger.schema import ScenarioRecord
from cascade.retrieval.schema import RetrievedChunk

__all__ = [
    "POISON_MARKER",
    "PoisonDocument",
    "SignatureFinding",
    "build_poison_documents",
    "insert_poison",
    "poison_document_id",
    "resolution_signature",
    "scan_signatures",
    "scenario_of",
    "violations",
]

# Every synthesised document carries this in its ``document_id`` so a stray row
# can be identified and removed unambiguously later. A poison document that
# escaped its transaction and could not be told apart from real evidence would
# be worse than the leak it was written to detect.
POISON_MARKER = "poison-pill"


def poison_document_id(scenario_id: str, digest: str) -> str:
    """``poison-pill|<scenario_id>|<digest>``.

    The scenario is embedded rather than hashed away because the probe has to
    attribute a retrieved row back to the scenario it poisons. Without that
    attribution the only available assertion is "no poison anywhere", which is
    the wrong criterion: poison planted for a scenario that resolved in 2018 is
    *legitimate* pre-cutoff evidence for a scenario with a 2019 cutoff, and
    failing on it would be failing on correct behaviour.
    """
    return f"{POISON_MARKER}|{scenario_id}|{digest}"


def scenario_of(document_id: str) -> str | None:
    """The scenario a poison document belongs to, or None if it is not poison."""
    parts = document_id.split("|")
    if len(parts) != 3 or parts[0] != POISON_MARKER:
        return None
    return parts[1]


@dataclass(frozen=True)
class PoisonDocument:
    """A synthetic post-resolution document for one scenario."""

    document_id: str
    scenario_id: str
    body: str
    published_at: datetime
    cutoff_ts: datetime


def _outcome_phrase(outcome: int) -> str:
    return "resolved YES" if outcome == 1 else "resolved NO"


def build_poison_documents(
    records: Sequence[ScenarioRecord], *, count: int
) -> tuple[PoisonDocument, ...]:
    """Synthesise ``count`` post-resolution documents spread over ``records``.

    Each document is dated **one day after** its scenario's ``resolve_ts``, so
    it is post-cutoff by construction and by a wide margin -- a probe dated one
    second after the cutoff would test float rounding rather than the lock.

    The text deliberately restates the scenario's own question before stating
    the outcome. That maximises cosine similarity to any query derived from the
    same scenario, which is the point: the probe should be the *easiest*
    possible thing for the retriever to return.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if not records:
        raise ValueError("cannot build poison documents without scenarios")

    out: list[PoisonDocument] = []
    for index in range(count):
        record = records[index % len(records)]
        scenario = record.scenario
        published = record.label.resolved_at + timedelta(days=1)
        digest = hashlib.blake2b(
            f"{scenario.scenario_id}:{index}".encode(), digest_size=8
        ).hexdigest()
        body = (
            f"{scenario.question} The question {_outcome_phrase(record.label.outcome)}. "
            f"Final determination: {_outcome_phrase(record.label.outcome)}. "
            f"Resolution criterion: {scenario.resolution_criterion} "
            f"The outcome is now confirmed and settled beyond dispute. "
            f"Parties involved: {', '.join(scenario.party_names) or 'none recorded'}."
        )
        out.append(
            PoisonDocument(
                document_id=poison_document_id(scenario.scenario_id, digest),
                scenario_id=scenario.scenario_id,
                body=body,
                published_at=published,
                cutoff_ts=scenario.cutoff_ts,
            )
        )
    return tuple(out)


def insert_poison(
    conn: Any, documents: Sequence[PoisonDocument], embeddings: Sequence[Sequence[float]]
) -> int:
    """Insert poison documents and one chunk each. Returns rows written.

    Takes a live connection so the caller keeps the transaction. Nothing here
    commits: the intended usage is insert -> query -> assert -> **rollback**,
    which leaves the corpus byte-identical afterwards.
    """
    from cascade.retrieval.search import vector_literal

    if len(documents) != len(embeddings):
        raise ValueError(
            f"{len(documents)} documents but {len(embeddings)} embeddings; "
            "a poison document without its vector is unreachable and would "
            "make the probe pass by accident"
        )

    written = 0
    with conn.cursor() as cur:
        for document, vector in zip(documents, embeddings, strict=True):
            cur.execute(
                "INSERT INTO documents (document_id, source, source_ref, url, title, "
                "published_at, simhash, n_chunks) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    document.document_id,
                    "govpr",  # a valid CorpusSource; never read by the probe
                    document.document_id,
                    "",
                    f"{POISON_MARKER} for {document.scenario_id}",
                    document.published_at,
                    0,
                    1,
                ),
            )
            cur.execute(
                "INSERT INTO chunks (chunk_id, document_id, ordinal, body, token_count, "
                "published_at, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::halfvec)",
                (
                    f"{document.document_id}-0",
                    document.document_id,
                    0,
                    document.body,
                    len(document.body.split()),
                    document.published_at,
                    vector_literal(vector),
                ),
            )
            written += 1
    return written


def violations(chunks: Sequence[RetrievedChunk], *, as_of: datetime) -> tuple[RetrievedChunk, ...]:
    """Chunks that should never have been returned under ``as_of``. Pure.

    The comparison is strict ``>=``: a chunk published exactly at the cutoff is
    a violation, because ``chronofence_search`` promises *strictly* before and
    an equal timestamp means the document existed at the instant the forecast
    was made.
    """
    return tuple(chunk for chunk in chunks if chunk.published_at >= as_of)


# Phrases that only appear once an outcome is known. Deliberately narrow --
# every one of these is a *retrospective* construction. Words like "wins" or
# "approved" are excluded because they appear constantly in pre-cutoff
# reporting about a live question and would flag most of the corpus.
_RESOLUTION_PATTERNS: tuple[str, ...] = (
    r"\bresolved\s+(yes|no)\b",
    r"\bfinal\s+determination\b",
    r"\bthe\s+outcome\s+(is|was)\s+(now\s+)?confirmed\b",
    r"\bofficially\s+(concluded|settled)\b",
    r"\bhas\s+been\s+settled\b",
    r"\bbeyond\s+dispute\b",
)

_SIGNATURE_RE = re.compile("|".join(_RESOLUTION_PATTERNS), re.IGNORECASE)


def resolution_signature(text: str) -> tuple[str, ...]:
    """Retrospective phrases found in ``text``. Pure."""
    return tuple(match.group(0).lower() for match in _SIGNATURE_RE.finditer(text))


@dataclass(frozen=True)
class SignatureFinding:
    """One retrieved chunk that reads like resolution text."""

    chunk_id: str
    scenario_id: str
    published_at: datetime
    cutoff_ts: datetime
    phrases: tuple[str, ...]
    similarity: float


def scan_signatures(
    chunks: Sequence[RetrievedChunk],
    *,
    scenario_id: str,
    cutoff_ts: datetime,
    similarities: Sequence[float],
    threshold: float,
) -> tuple[SignatureFinding, ...]:
    """Flag retrieved chunks that resemble resolution text. Pure.

    Two independent signals, either of which flags: a retrospective phrase, or
    embedding similarity to the scenario's resolution criterion above
    ``threshold``. Regex alone misses paraphrase; similarity alone flags every
    on-topic document, because the resolution criterion *describes the topic*.

    Flagging is reporting, not filtering. A pre-cutoff document that discusses
    the question in resolution-like language is legitimate evidence, and the
    reviewed count is what §4.2 asks for.
    """
    if len(chunks) != len(similarities):
        raise ValueError(
            f"{len(chunks)} chunks but {len(similarities)} similarities; "
            "the scan cannot pair a chunk with another chunk's score"
        )
    out: list[SignatureFinding] = []
    for chunk, similarity in zip(chunks, similarities, strict=True):
        phrases = resolution_signature(chunk.body)
        if phrases or similarity >= threshold:
            out.append(
                SignatureFinding(
                    chunk_id=chunk.chunk_id,
                    scenario_id=scenario_id,
                    published_at=chunk.published_at,
                    cutoff_ts=cutoff_ts,
                    phrases=phrases,
                    similarity=float(similarity),
                )
            )
    return tuple(out)
