"""Wikipedia revision snapshots (spec §3.2, ~180k chunks).

Spec §3.2 marks this source **critical**: "use the revision as of the cutoff,
never the current article". A current Wikipedia article about a 2016 election
states who won. Indexing it would put the answer to a scenario directly into
the evidence corpus, and no amount of partition pruning at M3 would help,
because the *document's own date* would be today rather than the date the
information became true.

So every fetch here is anchored: the API is asked for the last revision at or
before an ``as_of`` instant, and ``published_at`` is that revision's timestamp
-- not the article's, not today's. The anchor is a required argument for the
same reason ``as_of`` is required everywhere else (invariant 1).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from cascade.corpus.fetch import Fetcher, FetchError, html_to_text
from cascade.corpus.schema import RawDocument

__all__ = ["API_URL", "SnapshotRequest", "load_snapshots", "revision_before"]

API_URL = "https://en.wikipedia.org/w/api.php"


class SnapshotRequest:
    """One article, anchored to one instant."""

    __slots__ = ("as_of", "title")

    def __init__(self, title: str, as_of: datetime) -> None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError(f"as_of must be timezone-aware for {title!r}")
        self.title = title
        self.as_of = as_of


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed.astimezone(UTC)


def revision_before(
    fetcher: Fetcher, *, title: str, as_of: datetime
) -> tuple[int, datetime] | None:
    """Return ``(revision_id, timestamp)`` for the last revision at or before ``as_of``.

    ``rvdir=older`` with ``rvstart=as_of`` asks MediaWiki to walk backwards
    from the anchor, so the first result is by construction not newer than the
    cutoff. Returns ``None`` when the article did not exist yet -- which is
    itself correct behaviour, not an error: an article created after the
    cutoff is exactly what must not enter the corpus.
    """
    stamp = as_of.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{API_URL}?action=query&prop=revisions&titles={title.replace(' ', '_')}"
        f"&rvstart={stamp}&rvdir=older&rvlimit=1&rvprop=ids%7Ctimestamp"
        "&format=json&formatversion=2"
    )
    payload = fetcher.get_json(url)
    if not isinstance(payload, dict):
        return None
    pages = payload.get("query", {}).get("pages")
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    if not isinstance(page, dict) or page.get("missing"):
        return None
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return None
    revision = revisions[0]
    revision_id = revision.get("revid")
    timestamp = _parse_ts(revision.get("timestamp"))
    if not isinstance(revision_id, int) or timestamp is None:
        return None
    if timestamp > as_of:
        # Defensive: the anchor is meant to guarantee this, and a violation
        # would be a silent leak rather than a visible failure.
        return None
    return revision_id, timestamp


def _render(fetcher: Fetcher, revision_id: int) -> str:
    """Render one revision to text via the parse API.

    Rendered HTML rather than raw wikitext: templates, infoboxes and
    references expand to the prose a reader would actually have seen, whereas
    wikitext is full of markup an embedding model would treat as content.
    """
    url = (
        f"{API_URL}?action=parse&oldid={revision_id}&prop=text"
        "&format=json&formatversion=2&disablelimitreport=1&disableeditsection=1"
    )
    payload = fetcher.get_json(url)
    if not isinstance(payload, dict):
        return ""
    text = payload.get("parse", {}).get("text")
    return html_to_text(text) if isinstance(text, str) else ""


def load_snapshots(fetcher: Fetcher, requests: Iterable[SnapshotRequest]) -> Iterator[RawDocument]:
    """Yield one document per resolvable ``(article, as_of)`` pair.

    ``source_ref`` carries the revision id, so the same article snapshotted at
    two different cutoffs produces two distinct documents rather than one that
    silently overwrites the other.
    """
    for request in requests:
        try:
            found = revision_before(fetcher, title=request.title, as_of=request.as_of)
        except FetchError:
            continue
        if found is None:
            continue
        revision_id, timestamp = found
        try:
            body = _render(fetcher, revision_id)
        except FetchError:
            continue
        if not body:
            continue
        yield RawDocument(
            source="wikipedia",
            source_ref=str(revision_id),
            url=f"https://en.wikipedia.org/w/index.php?oldid={revision_id}",
            title=request.title,
            body=body,
            published_at=timestamp,
        )
