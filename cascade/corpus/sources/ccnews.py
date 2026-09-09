"""Common Crawl CC-NEWS loader (spec §3.2, ~420k chunks).

CC-NEWS is the one source that distributes article **text** in bulk rather
than links, which makes it the practical backbone of the corpus: GDELT hands
back URLs that then have to be scraped one at a time from publishers that
often refuse.

Two decisions here are load-bearing.

**The date is extracted from the page, never from the crawl.** A WARC record
carries ``WARC-Date``, which is when Common Crawl fetched the page -- typically
hours to days after publication, and occasionally much longer. Using it would
be exactly the "inferred date" §3.2 forbids, and it errs in the unsafe
direction: an article would be dated *later* than it was knowable, so a
cutoff that should have excluded it might not. Instead the publication date is
read from the document's own metadata, and a page that does not state one is
dropped.

**WARC parsing is stdlib-only.** The pinned stack (spec §2.3) has no WARC
library, and adding one would be a substitution requiring an ADR. The format
is a header block, a blank line and a length-delimited body, which is little
enough to read directly.
"""

from __future__ import annotations

import gzip
import re
import zlib
from collections.abc import Iterator
from datetime import UTC, datetime

from cascade.corpus.fetch import Fetcher, html_to_text
from cascade.corpus.schema import RawDocument

__all__ = ["PATHS_URL", "extract_published_at", "load_warc", "month_paths", "unit_keys"]

DATA_URL = "https://data.commoncrawl.org"
PATHS_URL = f"{DATA_URL}/crawl-data/CC-NEWS"

# Publication-date metadata, most explicit first. Every one of these is the
# publisher *stating* a date; none is inferred from position or crawl time.
_DATE_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(
        rb'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)', re.I
    ),
    re.compile(
        rb'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']',
        re.I,
    ),
    re.compile(
        rb'<meta[^>]+name=["\'](?:pubdate|publishdate|publish-date|date|DC\.date\.issued)["\'][^>]+content=["\']([^"\']+)',
        re.I,
    ),
    re.compile(rb'"datePublished"\s*:\s*"([^"]+)"', re.I),
    re.compile(rb'<time[^>]+datetime=["\']([^"\']+)["\'][^>]*>', re.I),
)

_WARC_TARGET = re.compile(rb"^WARC-Target-URI:\s*(.+?)\s*$", re.I | re.M)
_WARC_TYPE = re.compile(rb"^WARC-Type:\s*(.+?)\s*$", re.I | re.M)
_CONTENT_LENGTH = re.compile(rb"^Content-Length:\s*(\d+)\s*$", re.I | re.M)


# Common Crawl's CC-NEWS collection begins in August 2016. Months before that
# have no `warc.paths.gz` at all, and generating them would turn a known gap
# into a stream of fetch failures that look like an outage.
FIRST_YEAR = 2016
FIRST_MONTH = 8


def unit_keys(*, start_year: int, end_year: int) -> list[str]:
    """One unit per month; a month's WARC list is fetched as a whole.

    Clamped to the collection's actual start so a configured range that
    reaches further back does not manufacture missing months.
    """
    return [
        f"{year:04d}/{month:02d}"
        for year in range(max(start_year, FIRST_YEAR), end_year + 1)
        for month in range(1, 13)
        if (year, month) >= (FIRST_YEAR, FIRST_MONTH)
    ]


def month_paths(fetcher: Fetcher, *, unit_key: str) -> list[str]:
    """Return the WARC paths Common Crawl published for one month."""
    response = fetcher.get(f"{PATHS_URL}/{unit_key}/warc.paths.gz")
    return gzip.decompress(response.content).decode("utf-8", "replace").split()


def extract_published_at(html: bytes) -> datetime | None:
    """Read the publication date the document states, or ``None``.

    Returning ``None`` is a correct and common outcome. A page that does not
    declare when it was published cannot be placed relative to a cutoff, and
    ``normalize.validate`` drops it -- which is the rule §3.2 states for this
    source specifically.
    """
    for pattern in _DATE_PATTERNS:
        match = pattern.search(html)
        if match is None:
            continue
        raw = match.group(1).decode("utf-8", "replace").strip()
        # Trailing "Z" and offsets without a colon both appear in the wild.
        candidate = raw.replace("Z", "+00:00")
        candidate = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", candidate)
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            # A naive published_time is not usable: validate() will drop it,
            # and assuming UTC here would be the guess the spec forbids.
            return parsed
        return parsed.astimezone(UTC)
    return None


def _iter_warc_records(
    stream: Iterator[bytes], *, max_records: int
) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(header_block, content_block)`` from a multi-member gzip WARC.

    CC-NEWS compresses each record as its own gzip member so the file can be
    read incrementally. ``zlib`` is driven directly, carrying ``unused_data``
    across members, so records can be produced -- and the download abandoned --
    without holding a multi-gigabyte file in memory.
    """
    decompressor = zlib.decompressobj(wbits=31)
    buffer = bytearray()
    produced = 0

    def drain(buf: bytearray) -> Iterator[tuple[bytes, bytes]]:
        while True:
            marker = buf.find(b"WARC/1.0\r\n")
            if marker < 0:
                return
            header_end = buf.find(b"\r\n\r\n", marker)
            if header_end < 0:
                return
            header = bytes(buf[marker:header_end])
            length_match = _CONTENT_LENGTH.search(header)
            if length_match is None:
                del buf[: header_end + 4]
                continue
            length = int(length_match.group(1))
            body_start = header_end + 4
            if len(buf) < body_start + length:
                return
            content = bytes(buf[body_start : body_start + length])
            del buf[: body_start + length]
            yield header, content

    for compressed in stream:
        data = decompressor.decompress(compressed)
        buffer.extend(data)
        while decompressor.unused_data:
            leftover = decompressor.unused_data
            decompressor = zlib.decompressobj(wbits=31)
            buffer.extend(decompressor.decompress(leftover))
        for record in drain(buffer):
            yield record
            produced += 1
            if produced >= max_records:
                return


def load_warc(fetcher: Fetcher, *, path: str, max_records: int) -> Iterator[RawDocument]:
    """Stream one WARC file and yield the article responses it contains.

    The download is abandoned once ``max_records`` responses have been seen,
    so a single 1 GB file need not be fetched in full to contribute.
    """
    url = f"{DATA_URL}/{path}"
    client = fetcher._http()
    fetcher.requests += 1

    seen = 0
    with client.stream("GET", url) as response:
        if response.status_code != 200:
            return
        records = _iter_warc_records(
            response.iter_bytes(chunk_size=1 << 20), max_records=max_records * 3
        )
        for header, content in records:
            if seen >= max_records:
                return
            type_match = _WARC_TYPE.search(header)
            if type_match is None or type_match.group(1).lower() != b"response":
                continue
            target = _WARC_TARGET.search(header)
            if target is None:
                continue

            split = content.find(b"\r\n\r\n")
            payload = content[split + 4 :] if split >= 0 else content
            published = extract_published_at(payload)
            body = html_to_text(payload.decode("utf-8", "replace"))
            if not body:
                continue

            link = target.group(1).decode("utf-8", "replace")
            yield RawDocument(
                source="ccnews",
                source_ref=link,
                url=link,
                title="",
                body=body,
                published_at=published,
            )
            seen += 1
