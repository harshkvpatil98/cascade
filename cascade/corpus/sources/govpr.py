"""US Federal Register loader -- government/IGO releases (spec §3.2, ~40k).

The Federal Register publishes every rule, proposed rule, notice and
presidential document with a legally exact ``publication_date`` and a plain
text rendering of the full document. That combination is unusually good for
this corpus: no date inference, no HTML extraction guesswork, and the content
is exactly the policy and sanction material §3.2 wants this source for.

The API is free and needs no credential.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime

from cascade.corpus.fetch import Fetcher, FetchError
from cascade.corpus.schema import RawDocument

__all__ = ["DOCUMENTS_URL", "load_page", "unit_keys"]

DOCUMENTS_URL = "https://www.federalregister.gov/api/v1/documents.json"

_FIELDS = (
    "document_number",
    "title",
    "publication_date",
    "abstract",
    "raw_text_url",
    "html_url",
    "type",
)


def unit_keys(*, start_year: int, end_year: int) -> list[str]:
    """One unit per month, so a restart resumes at month granularity."""
    return [
        f"{year:04d}-{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def _month_bounds(unit_key: str) -> tuple[str, str]:
    year, month = (int(part) for part in unit_key.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    return start.date().isoformat(), end.date().isoformat()


def _parse_date(value: object) -> datetime | None:
    """Parse ``YYYY-MM-DD`` as midnight UTC.

    The Federal Register publishes on a date, not at an instant. Midnight UTC
    is a stated convention, not an inference: a document published on day D is
    treated as knowable from the start of day D, which is the conservative
    direction for a cutoff comparison.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip()).replace(tzinfo=UTC)
    except ValueError:
        return None


def load_page(
    fetcher: Fetcher, *, unit_key: str, max_documents: int, fetch_bodies: bool = True
) -> Iterator[RawDocument]:
    """Yield documents published in the month named by ``unit_key``.

    ``fetch_bodies`` trades breadth for depth: with it off the abstract is used
    as the body, which is one extra request per *page* instead of one per
    document. The pipeline turns it on -- an abstract is a summary, and the
    corpus exists so agents can read the source.
    """
    start, end = _month_bounds(unit_key)
    fields = "&".join(f"fields[]={field}" for field in _FIELDS)
    url = (
        f"{DOCUMENTS_URL}?{fields}&per_page=1000&order=oldest"
        f"&conditions[publication_date][gte]={start}"
        f"&conditions[publication_date][lt]={end}"
    )

    payload = fetcher.get_json(url)
    if not isinstance(payload, dict):
        return
    results = payload.get("results")
    if not isinstance(results, list):
        return

    emitted = 0
    for entry in results:
        if emitted >= max_documents:
            return
        if not isinstance(entry, dict):
            continue
        number = str(entry.get("document_number") or "").strip()
        if not number:
            continue

        body = str(entry.get("abstract") or "")
        raw_url = entry.get("raw_text_url")
        if fetch_bodies and isinstance(raw_url, str) and raw_url:
            # Keep the abstract rather than dropping the document: a missing
            # body is thin evidence, not a leakage risk, and the length floor
            # in normalize.validate decides whether it stays.
            with suppress(FetchError):
                body = fetcher.get_text(raw_url)

        yield RawDocument(
            source="govpr",
            source_ref=number,
            url=str(entry.get("html_url") or ""),
            title=str(entry.get("title") or ""),
            body=body,
            published_at=_parse_date(entry.get("publication_date")),
        )
        emitted += 1
