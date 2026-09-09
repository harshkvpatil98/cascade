"""Corpus source adapters and the shared fetch layer (spec §3.2).

An adapter's only job is normalisation. It never decides whether a document is
usable -- it reports what the source said, including a missing or malformed
date, and one shared rule in ``normalize.validate`` applies to all five.
These tests pin that division: parsers are exercised on recorded payload
shapes, and the date rules are asserted to *not* be duplicated here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cascade.corpus.fetch import Fetcher, html_to_text
from cascade.corpus.sources import ccnews, edgar, govpr, wikipedia

# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def test_script_and_navigation_are_discarded() -> None:
    """Boilerplate would otherwise dominate short articles."""
    html = (
        "<html><head><title>t</title><script>var x=1;</script>"
        "<style>.a{}</style></head><body><nav>menu</nav>"
        "<p>Real body text.</p><footer>copyright</footer></body></html>"
    )
    text = html_to_text(html)
    assert "Real body text." in text
    for boilerplate in ("var x=1", "menu", "copyright", ".a{}"):
        assert boilerplate not in text


def test_block_elements_become_line_breaks() -> None:
    """Without breaks, paragraphs run together and sentences cannot be found."""
    assert html_to_text("<p>One.</p><p>Two.</p>") == "One.\nTwo."


def test_entities_are_decoded() -> None:
    assert html_to_text("<p>A &amp; B &lt;C&gt;</p>") == "A & B <C>"


def test_malformed_markup_does_not_raise() -> None:
    """One bad page must not abort a 100,000-document run."""
    assert html_to_text("<p>text<<>><div unclosed").startswith("text")


def test_empty_html_yields_empty_text() -> None:
    assert html_to_text("") == ""


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def test_user_agent_carries_a_contact_address() -> None:
    """SEC EDGAR answers 403 to any agent without one -- measured, not assumed.

    A blocked client looks exactly like a source with no documents, so this is
    pinned rather than left to configuration drift.
    """
    from cascade.corpus.fetch import USER_AGENT

    assert "@" in USER_AGENT


def test_fetcher_rate_limit_is_configurable_per_source() -> None:
    """GDELT allows one request per five seconds; EDGAR allows ten per second.

    One shared budget would either throttle EDGAR pointlessly or get the
    client blocked by GDELT.
    """
    from cascade.corpus.pipeline import GDELT_REQUESTS_PER_SECOND

    assert pytest.approx(0.2) == GDELT_REQUESTS_PER_SECOND
    assert Fetcher(requests_per_second=10.0).requests_per_second == 10.0


# ---------------------------------------------------------------------------
# Federal Register
# ---------------------------------------------------------------------------


def test_govpr_units_span_the_configured_years() -> None:
    keys = govpr.unit_keys(start_year=2016, end_year=2024)
    assert len(keys) == 108
    assert keys[0] == "2016-01"
    assert keys[-1] == "2024-12"


def test_govpr_month_bounds_are_half_open() -> None:
    """A closed upper bound would double-count documents on month boundaries."""
    start, end = govpr._month_bounds("2018-12")
    assert (start, end) == ("2018-12-01", "2019-01-01")


def test_govpr_parses_a_publication_date_as_midnight_utc() -> None:
    """A stated convention, not an inference: publication is dated, not timed."""
    assert govpr._parse_date("2018-06-01") == datetime(2018, 6, 1, tzinfo=UTC)


@pytest.mark.parametrize("value", [None, "", "not-a-date", 20180601])
def test_govpr_refuses_an_unparseable_date(value: object) -> None:
    assert govpr._parse_date(value) is None


# ---------------------------------------------------------------------------
# EDGAR
# ---------------------------------------------------------------------------


FORM_IDX = """Description:           Master Index
Form Type   Company Name                        CIK   Date Filed  File Name
---------------------------------------------------------------------------
8-K         1 800 FLOWERS COM INC             1084869 2018-05-01  edgar/data/1084869/0001157523-18-000898.txt
DEF 14A     ACME CORP  INC                     123456 2018-06-15  edgar/data/123456/0000123456-18-000001.txt
10-Q        IGNORED CO                         999999 2018-05-02  edgar/data/999999/0000999999-18-000002.txt
"""


def test_edgar_index_parses_column_aligned_rows() -> None:
    """Company names contain runs of spaces, so the split is anchored on the
    two fields with fixed shapes: the ISO date and the archive path."""
    rows = edgar.parse_form_index(FORM_IDX)
    assert len(rows) == 2
    assert rows[0][0] == "8-K"
    assert rows[0][2] == "2018-05-01"
    assert rows[0][3].endswith("0001157523-18-000898.txt")
    assert rows[1][0] == "DEF 14A"


def test_edgar_index_ignores_other_form_types() -> None:
    """Scope is 8-K and DEF 14A (spec §3.2); a 10-Q is not a material event."""
    assert all(row[0] in edgar.FORM_TYPES for row in edgar.parse_form_index(FORM_IDX))


def test_edgar_units_are_quarters() -> None:
    keys = edgar.unit_keys(start_year=2016, end_year=2024)
    assert len(keys) == 36
    assert keys[0] == "2016Q1"


def test_edgar_extracts_the_primary_document() -> None:
    """Exhibits and XBRL are attachments; indexing them dilutes the filing."""
    raw = (
        "<SEC-DOCUMENT>\n<DOCUMENT>\n<TYPE>8-K\n<TEXT><html><body>"
        "<p>The board approved the transaction.</p></body></html></TEXT>\n</DOCUMENT>\n"
        "<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT><html><body><p>Exhibit noise.</p>"
        "</body></html></TEXT>\n</DOCUMENT>\n"
    )
    text = edgar._extract_filing_text(raw, "8-K")
    assert "board approved" in text
    assert "Exhibit noise" not in text


# ---------------------------------------------------------------------------
# Wikipedia -- the leakage-critical source
# ---------------------------------------------------------------------------


def test_wikipedia_snapshot_requires_an_aware_anchor() -> None:
    """A naive anchor cannot be compared to a revision timestamp."""
    with pytest.raises(ValueError, match="timezone-aware"):
        wikipedia.SnapshotRequest("Brexit", datetime(2016, 4, 15))


def test_wikipedia_snapshot_keeps_its_anchor() -> None:
    anchor = datetime(2016, 4, 15, tzinfo=UTC)
    assert wikipedia.SnapshotRequest("Brexit", anchor).as_of == anchor


# ---------------------------------------------------------------------------
# CC-NEWS
# ---------------------------------------------------------------------------


def test_ccnews_units_start_where_the_collection_does() -> None:
    """Months before 2016-08 have no warc.paths.gz at all.

    Generating them would turn a known gap into a stream of fetch failures
    that look like an outage.
    """
    keys = ccnews.unit_keys(start_year=2015, end_year=2024)
    assert keys[0] == "2016/08"
    assert "2016/07" not in keys
    assert keys[-1] == "2024/12"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            b'<meta property="article:published_time" content="2018-06-01T10:00:00Z">',
            datetime(2018, 6, 1, 10, tzinfo=UTC),
        ),
        (
            b'<meta content="2018-06-01T10:00:00+00:00" property="article:published_time">',
            datetime(2018, 6, 1, 10, tzinfo=UTC),
        ),
        (b'{"datePublished":"2019-03-04T08:30:00Z"}', datetime(2019, 3, 4, 8, 30, tzinfo=UTC)),
        (b'<meta name="pubdate" content="2020-01-02T00:00:00Z">', datetime(2020, 1, 2, tzinfo=UTC)),
        (
            b'<time datetime="2021-11-05T12:00:00Z">Nov 5</time>',
            datetime(2021, 11, 5, 12, tzinfo=UTC),
        ),
    ],
)
def test_ccnews_reads_the_date_the_publisher_states(html: bytes, expected: datetime) -> None:
    """Never the crawl date.

    ``WARC-Date`` is when Common Crawl fetched the page -- hours to days after
    publication. Using it would date evidence *later* than it was knowable, so
    a cutoff that should have excluded it might not.
    """
    assert ccnews.extract_published_at(html) == expected


def test_ccnews_offsets_without_a_colon_are_handled() -> None:
    assert ccnews.extract_published_at(
        b'<meta property="article:published_time" content="2018-06-01T10:00:00+0200">'
    ) == datetime(2018, 6, 1, 8, tzinfo=UTC)


def test_ccnews_returns_none_when_no_date_is_stated() -> None:
    """A common and correct outcome: validate() then drops the document."""
    assert ccnews.extract_published_at(b"<html><body>no metadata</body></html>") is None


def test_ccnews_naive_date_is_returned_naive_for_validate_to_reject() -> None:
    """The adapter does not assume an offset; the shared rule decides."""
    parsed = ccnews.extract_published_at(
        b'<meta property="article:published_time" content="2018-06-01T10:00:00">'
    )
    assert parsed is not None
    assert parsed.tzinfo is None


# ---------------------------------------------------------------------------
# Retry backoff
#
# Regression: exponential backoff from one second is faster than GDELT's
# published floor of one request per five seconds, so retrying a 429 that way
# spends the very allowance the retry is waiting for. Measured as an IP-level
# block that outlasted the run.
# ---------------------------------------------------------------------------


def test_backoff_is_never_shorter_than_the_source_interval() -> None:
    """A GDELT retry must wait at least its 5-second interval, not 1 second."""
    gdelt = Fetcher(requests_per_second=0.2)
    assert gdelt._backoff(0, None) >= 5.0
    assert gdelt._backoff(1, None) >= 5.0


def test_backoff_grows_for_a_fast_source() -> None:
    fast = Fetcher(requests_per_second=10.0)
    assert fast._backoff(0, None) == pytest.approx(1.0)
    assert fast._backoff(2, None) == pytest.approx(4.0)


def test_retry_after_header_wins() -> None:
    """The server stating its own terms outranks any local heuristic."""
    import httpx

    response = httpx.Response(429, headers={"Retry-After": "30"})
    assert Fetcher(requests_per_second=10.0)._backoff(0, response) == pytest.approx(30.0)


def test_retry_after_is_capped() -> None:
    """An absurd Retry-After must not stall an ingest indefinitely."""
    import httpx

    response = httpx.Response(429, headers={"Retry-After": "99999"})
    assert Fetcher(requests_per_second=10.0)._backoff(0, response) == pytest.approx(120.0)
