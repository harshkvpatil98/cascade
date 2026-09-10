"""The ingest pipeline: fetch -> normalize -> date-validate -> dedupe ->
chunk -> embed -> index (spec §3.2).

The thin shell around the pure stages. Everything that decides *whether* a
document is usable lives in ``normalize.py``, everything that decides how it is
split lives in ``chunker.py``, and neither can reach a network or a clock.

Resumability is per unit (invariant 8). A unit is an EDGAR quarter, a Federal
Register month, a CC-NEWS WARC file, a GDELT month-query, or one scenario's
Wikipedia snapshots. Units are marked done only after their rows are
committed, so a crash mid-unit re-does that unit and nothing else.

Rate limits are **per source**, because they are properties of the upstream:
GDELT rejects faster than one request per five seconds, SEC EDGAR allows ten,
and letting one source spend another's budget is how a source starts returning
nothing for reasons that look like an empty archive.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from cascade.config import Settings
from cascade.corpus.chunker import chunk_text
from cascade.corpus.embed import Embedder
from cascade.corpus.fetch import Fetcher
from cascade.corpus.normalize import DedupeIndex, deduplicate, validate
from cascade.corpus.schema import Chunk, DatedDocument, RawDocument
from cascade.corpus.sources import ccnews, edgar, gdelt, govpr, wikipedia
from cascade.corpus.store import (
    completed_units,
    existing_document_ids,
    mark_unit,
    seen_simhashes,
    write_batch,
)

__all__ = ["IngestReport", "SourceReport", "run_ingest"]

# GDELT documents a hard floor of one request per five seconds and answers
# faster clients with HTTP 429. This is not tuning; it is the published limit.
GDELT_REQUESTS_PER_SECOND = 0.2


@dataclass
class SourceReport:
    """What one source contributed, and what it discarded."""

    source: str
    units_done: int = 0
    units_skipped: int = 0
    units_failed: int = 0
    documents_seen: int = 0
    documents_written: int = 0
    chunks_written: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    collapsed: int = 0
    detail: str = ""

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


@dataclass
class IngestReport:
    """The whole run, in the shape the acceptance criteria are stated in."""

    sources: list[SourceReport] = field(default_factory=list)

    @property
    def documents_written(self) -> int:
        return sum(report.documents_written for report in self.sources)

    @property
    def chunks_written(self) -> int:
        return sum(report.chunks_written for report in self.sources)

    @property
    def collapsed(self) -> int:
        return sum(report.collapsed for report in self.sources)

    @property
    def documents_seen(self) -> int:
        return sum(report.documents_seen for report in self.sources)

    @property
    def collapse_ratio(self) -> float:
        """Fraction of validated documents absorbed as near-duplicates."""
        total = self.documents_written + self.collapsed
        return self.collapsed / total if total else 0.0


def _wikipedia_requests(settings: Settings) -> dict[str, list[wikipedia.SnapshotRequest]]:
    """Build one snapshot unit per scenario, anchored at that scenario's cutoff.

    This is the source's whole point: the article as it stood at the cutoff.
    Anchoring to the scenario that will be forecast from it is what makes the
    snapshot honest -- a single global "as of" would be too late for early
    scenarios and needlessly early for late ones.

    Reads ``scenarios`` only. The parties come from the registry's own
    ``party_names``, and no outcome is touched (invariant 2).
    """
    from cascade.ledger.store import load_scenarios

    limit = settings.corpus.wikipedia_max_articles_per_scenario
    units: dict[str, list[wikipedia.SnapshotRequest]] = {}
    for scenario in load_scenarios(settings, role="admin"):
        titles = [name for name in scenario.party_names if len(name) > 2][:limit]
        if not titles:
            continue
        units[scenario.scenario_id] = [
            wikipedia.SnapshotRequest(title, scenario.cutoff_ts) for title in titles
        ]
    return units


def _units_for(settings: Settings, source: str) -> list[str]:
    corpus = settings.corpus
    span = {"start_year": corpus.start_year, "end_year": corpus.end_year}
    if source == "govpr":
        return govpr.unit_keys(**span)
    if source == "edgar":
        return edgar.unit_keys(**span)
    if source == "gdelt":
        return gdelt.unit_keys(**span)
    if source == "ccnews":
        return ccnews.unit_keys(**span)
    if source == "wikipedia":
        return sorted(_wikipedia_requests(settings))
    raise ValueError(f"unknown corpus source: {source}")


def _documents_for(
    settings: Settings,
    source: str,
    unit_key: str,
    fetchers: dict[str, Fetcher],
    wiki_units: dict[str, list[wikipedia.SnapshotRequest]],
) -> Iterator[RawDocument]:
    corpus = settings.corpus
    limit = corpus.max_documents_per_unit
    if source == "govpr":
        return govpr.load_page(fetchers["govpr"], unit_key=unit_key, max_documents=limit)
    if source == "edgar":
        return edgar.load_quarter(fetchers["edgar"], unit_key=unit_key, max_documents=limit)
    if source == "gdelt":
        return gdelt.load_window(
            fetchers["gdelt"],
            fetchers["web"],
            unit_key=unit_key,
            max_records=corpus.gdelt_max_records,
        )
    if source == "wikipedia":
        return wikipedia.load_snapshots(fetchers["wikipedia"], wiki_units.get(unit_key, []))
    if source == "ccnews":
        return _ccnews_unit(settings, unit_key, fetchers["ccnews"])
    raise ValueError(f"unknown corpus source: {source}")


def _ccnews_unit(settings: Settings, unit_key: str, fetcher: Fetcher) -> Iterator[RawDocument]:
    """Stream a bounded number of WARC files for one month, in parallel.

    Measured single-stream, this source runs at ~26% CPU: almost all of the
    wall time is spent waiting on a ~1 GB download while the tokenizer and the
    GPU sit idle. WARC files are independent, so they are fetched concurrently
    and the CPU-bound stages downstream get a steady supply.

    Each worker gets its **own** ``Fetcher``. Sharing one would serialise the
    downloads behind that fetcher's rate limiter, which is exactly what this
    is trying to avoid, and ``httpx.Client`` connection reuse across threads is
    not what the limiter is for.
    """
    corpus = settings.corpus
    # A missing month propagates: `unit_keys` already excludes the months the
    # collection does not cover, so a failure here is a real outage and should
    # be recorded as a failed unit rather than an empty one.
    paths = ccnews.month_paths(fetcher, unit_key=unit_key)
    selected = paths[: corpus.ccnews_max_files]
    if not selected:
        return

    workers = max(1, min(corpus.fetch_workers, len(selected)))

    def drain(path: str) -> list[RawDocument]:
        worker = Fetcher(requests_per_second=corpus.requests_per_second, user_agent=corpus.contact)
        try:
            return list(
                ccnews.load_warc(worker, path=path, max_records=corpus.ccnews_max_records_per_file)
            )
        except Exception:  # noqa: BLE001 -- one bad WARC must not fail the month
            return []
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Results are consumed in submission order so the ingest stays
        # deterministic given the same WARC list.
        for documents in pool.map(drain, selected):
            yield from documents


def _chunks_for(document: DatedDocument, settings: Settings, embedder: Embedder) -> list[Chunk]:
    """Split one document, counting tokens the way the model does."""
    corpus = settings.corpus
    plans = chunk_text(
        document.body,
        count_tokens=embedder.count_tokens,
        count_tokens_batch=embedder.count_tokens_batch,
        max_tokens=corpus.chunk_max_tokens,
        overlap_tokens=corpus.chunk_overlap_tokens,
    )
    return [
        Chunk(
            chunk_id=f"{document.document_id}#{ordinal}",
            document_id=document.document_id,
            ordinal=ordinal,
            body=plan.body,
            token_count=plan.token_count,
            published_at=document.published_at,
        )
        for ordinal, plan in enumerate(plans)
    ]


def run_ingest(
    settings: Settings,
    *,
    now: datetime,
    sources: tuple[str, ...] | None = None,
    max_units_per_source: int | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Run the pipeline for each enabled source.

    ``now`` is required rather than read from the clock, for the same reason
    ``as_of`` is (invariant 1): a future-date check that consults the wall
    clock gives different answers on different runs.
    """
    corpus = settings.corpus
    selected = sources if sources is not None else corpus.enabled_sources
    report = IngestReport()

    model = embedder or Embedder(
        model_name=settings.models.embedding, batch_size=corpus.embed_batch_size
    )
    model.load()

    index = DedupeIndex()
    index.seed(seen_simhashes(settings))
    known_ids = existing_document_ids(settings)

    fetchers = {
        "govpr": Fetcher(requests_per_second=corpus.requests_per_second, user_agent=corpus.contact),
        "edgar": Fetcher(requests_per_second=corpus.requests_per_second, user_agent=corpus.contact),
        "wikipedia": Fetcher(
            requests_per_second=corpus.requests_per_second, user_agent=corpus.contact
        ),
        "ccnews": Fetcher(
            requests_per_second=corpus.requests_per_second, user_agent=corpus.contact
        ),
        "gdelt": Fetcher(
            requests_per_second=GDELT_REQUESTS_PER_SECOND,
            user_agent=corpus.contact,
            # GDELT serves refusals under 200 as readily as under 429. Without
            # this the notice reaches the JSON parser and the unit dies with a
            # parse error instead of backing off and retrying.
            classify_body=gdelt.classify_body,
        ),
        "web": Fetcher(
            requests_per_second=corpus.requests_per_second,
            user_agent=corpus.contact,
            max_retries=1,
        ),
    }
    wiki_units = _wikipedia_requests(settings) if "wikipedia" in selected else {}

    try:
        for source in selected:
            source_report = SourceReport(source=source)
            report.sources.append(source_report)

            try:
                units = _units_for(settings, source)
            except Exception as exc:  # noqa: BLE001 -- reported per source, never fatal
                source_report.detail = f"{type(exc).__name__}: {exc}"[:200]
                continue

            done = completed_units(settings, source)
            pending = [unit for unit in units if unit not in done]
            source_report.units_skipped = len(units) - len(pending)
            if max_units_per_source is not None:
                pending = pending[:max_units_per_source]

            for unit_key in pending:
                try:
                    raw_documents = list(
                        _documents_for(settings, source, unit_key, fetchers, wiki_units)
                    )
                except Exception as exc:  # noqa: BLE001 -- one unit must not kill a run
                    source_report.units_failed += 1
                    mark_unit(
                        settings,
                        source=source,
                        unit_key=unit_key,
                        state="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                source_report.documents_seen += len(raw_documents)
                validated: list[DatedDocument] = []
                for raw in raw_documents:
                    outcome = validate(raw, now=now)
                    if isinstance(outcome, str):
                        source_report.drop(outcome)
                        continue
                    if outcome.document_id in known_ids:
                        source_report.drop("duplicate")
                        continue
                    known_ids.add(outcome.document_id)
                    validated.append(outcome)

                collapsed = deduplicate(validated, index=index)
                source_report.collapsed += collapsed.collapsed

                documents = collapsed.kept
                if not documents:
                    mark_unit(
                        settings, source=source, unit_key=unit_key, state="done", detail="empty"
                    )
                    source_report.units_done += 1
                    continue

                # Flush in bounded batches rather than accumulating the whole
                # unit. A CC-NEWS month can hold ~100k chunks, and holding
                # their text and vectors before a single write would put
                # hundreds of megabytes on the heap and lose all of it to one
                # interruption. Batching bounds memory and commits progress as
                # it goes; the unit is still only marked done at the end, so a
                # crash mid-unit re-does the unit and never skips it.
                batch_size = max(1, settings.corpus.write_batch_size)
                for start in range(0, len(documents), batch_size):
                    batch = documents[start : start + batch_size]
                    chunks: list[Chunk] = []
                    for document in batch:
                        chunks.extend(_chunks_for(document, settings, model))
                    if not chunks:
                        continue
                    vectors = model.encode([chunk.body for chunk in chunks])
                    written_documents, written_chunks = write_batch(
                        settings, batch, chunks, vectors
                    )
                    source_report.documents_written += written_documents
                    source_report.chunks_written += written_chunks

                source_report.units_done += 1
                mark_unit(
                    settings,
                    source=source,
                    unit_key=unit_key,
                    state="done",
                    n_documents=len(documents),
                    n_chunks=source_report.chunks_written,
                )
    finally:
        # Sorted (invariant 7). Close order does not matter today, but the
        # rule is that no mapping is ever walked in insertion order -- and the
        # static check does not carve out exceptions for cleanup paths.
        for _, fetcher in sorted(fetchers.items()):
            fetcher.close()

    return report
