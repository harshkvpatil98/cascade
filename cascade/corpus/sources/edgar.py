"""SEC EDGAR loader -- 8-K and DEF 14A filings (spec §3.2, ~70k chunks).

The value of this source is date precision. A filing's ``Date Filed`` is a
legal fact recorded by the regulator, not a publisher's guess, so nothing here
needs date inference -- which matters because an inferred date is the single
most direct route to cutoff leakage.

Form scope follows the spec: **8-K** (material events -- mergers, executive
departures, bankruptcies) and **DEF 14A** (proxy statements -- board contests,
compensation votes). Both are the corporate-event material the merger-review
and shareholder scenarios need.

EDGAR rejects clients that omit a contact address or exceed 10 requests per
second, so both are configured rather than assumed; see
``cascade/corpus/fetch.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime

from cascade.corpus.fetch import Fetcher, FetchError
from cascade.corpus.schema import RawDocument

__all__ = ["FORM_TYPES", "FULL_INDEX_URL", "load_quarter", "parse_form_index", "unit_keys"]

FULL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index"
ARCHIVES_URL = "https://www.sec.gov/Archives"

FORM_TYPES = ("8-K", "DEF 14A")

# A complete submission bundles exhibits, XBRL and sometimes whole annual
# reports. Past this many characters the extra is boilerplate that would
# swamp the corpus with low-information chunks.
_MAX_BODY_CHARS = 120_000

_SGML_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
_SGML_TYPE = re.compile(r"<TYPE>\s*([^\s<]+)", re.IGNORECASE)
_SGML_TEXT = re.compile(r"<TEXT>(.*?)(?:</TEXT>|$)", re.DOTALL | re.IGNORECASE)


def unit_keys(*, start_year: int, end_year: int) -> list[str]:
    """One unit per calendar quarter -- the granularity EDGAR indexes at."""
    return [
        f"{year:04d}Q{quarter}"
        for year in range(start_year, end_year + 1)
        for quarter in (1, 2, 3, 4)
    ]


def parse_form_index(text: str) -> list[tuple[str, str, str, str]]:
    """Parse ``form.idx`` into ``(form_type, company, date_filed, path)``.

    The file is column-aligned rather than delimited, and company names
    contain runs of spaces, so the split is anchored on the two fields with
    fixed shapes -- the ISO date and the ``edgar/data/...`` path -- rather than
    on whitespace.
    """
    rows: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        if not line.startswith(FORM_TYPES):
            continue
        match = re.search(r"\s(\d{4}-\d{2}-\d{2})\s+(edgar/data/\S+)\s*$", line)
        if match is None:
            continue
        date_filed, path = match.group(1), match.group(2)
        head = line[: match.start()].rstrip()
        form_type = next((form for form in FORM_TYPES if head.startswith(form)), "")
        if not form_type:
            continue
        company = head[len(form_type) :].strip()
        # Trailing CIK column, if it survived the head slice.
        company = re.sub(r"\s+\d{4,10}$", "", company).strip()
        rows.append((form_type, company, date_filed, path))
    return rows


def _extract_filing_text(raw: str, form_type: str) -> str:
    """Pull the primary document's text out of a complete SGML submission.

    Takes the first ``<DOCUMENT>`` whose ``<TYPE>`` matches the form, falling
    back to the first document. Exhibits and XBRL blocks are left behind: they
    are attachments, and indexing them would dilute the filing's own text.
    """
    from cascade.corpus.fetch import html_to_text

    chosen: str | None = None
    for block in _SGML_DOCUMENT.finditer(raw):
        body = block.group(1)
        type_match = _SGML_TYPE.search(body)
        text_match = _SGML_TEXT.search(body)
        if text_match is None:
            continue
        if chosen is None:
            chosen = text_match.group(1)
        if type_match and type_match.group(1).upper() == form_type.replace(" ", "").upper():
            chosen = text_match.group(1)
            break
    payload = chosen if chosen is not None else raw
    return html_to_text(payload)[:_MAX_BODY_CHARS]


def load_quarter(fetcher: Fetcher, *, unit_key: str, max_documents: int) -> Iterator[RawDocument]:
    """Yield filings indexed in the quarter named by ``unit_key`` (``2018Q2``).

    Filings are taken in index order, which is chronological within the
    quarter, so a truncated unit still covers a contiguous period rather than
    a biased sample of large companies.
    """
    year, quarter = unit_key.split("Q")
    index_text = fetcher.get_text(f"{FULL_INDEX_URL}/{year}/QTR{quarter}/form.idx")

    emitted = 0
    for form_type, company, date_filed, path in parse_form_index(index_text):
        if emitted >= max_documents:
            return
        try:
            published = datetime.fromisoformat(date_filed).replace(tzinfo=UTC)
        except ValueError:
            continue

        accession = path.rsplit("/", 1)[-1].removesuffix(".txt")
        try:
            raw = fetcher.get_text(f"{ARCHIVES_URL}/{path}")
        except FetchError:
            # One unavailable filing must not abort a quarter. The unit is
            # still marked done only if the index itself was readable.
            continue

        yield RawDocument(
            source="edgar",
            source_ref=accession,
            url=f"{ARCHIVES_URL}/{path}",
            title=f"{form_type} - {company}",
            body=_extract_filing_text(raw, form_type),
            published_at=published,
        )
        emitted += 1
