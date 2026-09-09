"""GDELT 2.0 loader (spec §3.2, ~600k chunks).

GDELT indexes worldwide news with exact timestamps and is free, which is why
§3.2 calls it the backbone. What it does **not** distribute is article text --
the Doc API returns metadata and a URL, so the body has to be fetched from the
publisher.

Two consequences are handled here rather than discovered later:

* GDELT enforces **one request per five seconds** and answers faster clients
  with HTTP 429 and an explanatory body. The fetcher for this source is
  configured at 0.2 requests/second; a source silently returning 429 looks
  identical to a source with no articles.
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

from cascade.corpus.fetch import Fetcher, FetchError, html_to_text
from cascade.corpus.schema import RawDocument

__all__ = ["DOC_API_URL", "QUERIES", "load_window", "unit_keys"]

DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

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
