"""GDELT 2.0 loader (spec §3.2, ~600k chunks).

GDELT indexes worldwide news with exact timestamps and is free, which is why
§3.2 calls it the backbone. What it does **not** distribute is article text --
the Doc API returns metadata and a URL, so the body has to be fetched from the
publisher.

Two consequences are handled here rather than discovered later:

* GDELT enforces **one request per five seconds** and answers faster clients
  with an explanatory notice. The fetcher for this source is configured at
  0.2 requests/second. The notice arrives under HTTP 429 *and*, measured
  against the live service, under **HTTP 200 with a plain-text body** -- so
  the status code alone cannot be trusted to identify it, and a throttled
  source that is not identified looks exactly like a source with no articles.
  :func:`classify_body` closes that gap and is wired into this source's
  fetcher by the pipeline.
* Publisher fetches fail often -- paywalls, geo-blocks, dead links. A failed
  body is skipped, never substituted with the GDELT title, because a headline
  is not the evidence the corpus claims to hold.

``seendate`` is GDELT's first-observation timestamp, which is when the article
became publicly visible to the crawler. It is used as ``published_at``
directly and is never adjusted.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx

from cascade.corpus.fetch import Fetcher, FetchError, SoftFailure, html_to_text
from cascade.corpus.schema import RawDocument

__all__ = [
    "DOC_API_URL",
    "QUERIES",
    "classify_body",
    "load_window",
    "unit_keys",
]

DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Phrases from GDELT's refusal notices. Used only to *label* an unusable body
# as a throttle rather than to decide whether it is unusable -- see
# `classify_body` for why that distinction matters.
_THROTTLE_MARKERS = (
    "limit requests to one every",
    "your query is too broad",
    "rate limit",
)


def classify_body(response: httpx.Response) -> SoftFailure | None:
    """Classify a GDELT 2xx body: ``None`` when it is a real answer.

    Every request this module makes carries ``format=json``, so **a 200 whose
    body is not JSON is not an answer**, whatever the reason. That is the test
    applied here, and it is deliberately not a search for known error strings.

    A marker whitelist was tried first and proved too narrow within one ingest
    run: two of the three failing units were correctly identified as throttled,
    while the third came back with a 200 carrying a body that matched no known
    phrase and was recorded, once again, as ``returned non-JSON``. Enumerating
    a third party's error prose is a losing game -- the shape of a *valid*
    answer is the thing this client actually knows.

    The markers survive only to distinguish "throttled" (worth waiting for)
    from "unusable" (retried, but not evidence of impatience), because those
    two mean different things in an ingest report.
    """
    if response.status_code == 429:
        return "throttled"

    try:
        response.json()
    except ValueError:
        pass
    else:
        return None

    head = response.text[:512].lower()
    if any(marker in head for marker in _THROTTLE_MARKERS):
        return "throttled"
    return "unusable"


# Broad strategic-affairs queries. Chosen to span the domains the scenario set
# is stratified over (spec §3.1) rather than to match any individual question:
# a corpus assembled by searching for the scenarios' own subjects would be a
# retrieval benchmark built from its own answer key.
QUERIES: tuple[str, ...] = (
    "sanctions",
    "ceasefire negotiations",
    "merger antitrust",
    "labor strike union",
    "election results",
    "trade agreement",
    "central bank policy",
    "coalition government",
)


def unit_keys(*, start_year: int, end_year: int) -> list[str]:
    """One unit per month per query."""
    return [
        f"{year:04d}-{month:02d}:{index}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        for index in range(len(QUERIES))
    ]


def _window(unit_key: str) -> tuple[str, str, str]:
    period, index = unit_key.rsplit(":", 1)
    year, month = (int(part) for part in period.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    return (
        start.strftime("%Y%m%d%H%M%S"),
        end.strftime("%Y%m%d%H%M%S"),
        QUERIES[int(index)],
    )


def _parse_seendate(value: object) -> datetime | None:
    """Parse GDELT's ``YYYYMMDDTHHMMSSZ`` stamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def load_window(
    listing: Fetcher, bodies: Fetcher, *, unit_key: str, max_records: int
) -> Iterator[RawDocument]:
    """Yield documents for one ``month:query`` unit.

    Takes two fetchers because the two hosts have unrelated politeness
    budgets: GDELT's 5-second floor must not throttle publisher fetches, and
    publisher retries must not spend GDELT's allowance.
    """
    start, end, query = _window(unit_key)
    url = (
        f"{DOC_API_URL}?query={query.replace(' ', '%20')}%20sourcelang:eng"
        f"&mode=artlist&maxrecords={max_records}&format=json"
        f"&startdatetime={start}&enddatetime={end}&sort=hybridrel"
    )

    payload = listing.get_json(url)
    if not isinstance(payload, dict):
        return
    articles = payload.get("articles")
    if not isinstance(articles, list):
        return

    for article in articles:
        if not isinstance(article, dict):
            continue
        link = str(article.get("url") or "").strip()
        published = _parse_seendate(article.get("seendate"))
        if not link or published is None:
            continue
        try:
            body = html_to_text(bodies.get_text(link))
        except FetchError:
            continue
        if not body:
            continue
        yield RawDocument(
            source="gdelt",
            source_ref=link,
            url=link,
            title=str(article.get("title") or ""),
            body=body,
            published_at=published,
        )
