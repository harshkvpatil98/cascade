"""Polite HTTP fetching and HTML-to-text extraction for the corpus.

Unlike the scenario registry, the corpus does **not** cache raw responses to
disk. The registry has to be byte-reproducible because the manifest hashes it;
the corpus is reproducible through the database instead, and caching 1.3M
article bodies twice -- once as HTTP payloads, once as rows -- would double an
already large footprint for no guarantee.

Resumability comes from ``corpus_ingest_state`` (invariant 8): a restart skips
units already marked done.

Politeness is a functional requirement, not manners. SEC EDGAR publishes a
10 requests/second limit and blocks clients that ignore it or omit a contact
User-Agent, and a blocked client looks exactly like an empty source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import httpx

__all__ = ["FetchError", "Fetcher", "html_to_text"]

# SEC EDGAR rejects requests whose User-Agent does not carry a contact address
# in "Name email@domain" form -- measured: a descriptive-but-address-free agent
# gets HTTP 403 on every Archives path, which looks exactly like an empty
# source. Configurable via `corpus.contact` so an operator can supply a real
# address, which is what the SEC actually asks for.
USER_AGENT = "Cascade Research cascade-research@example.com"

# Elements whose contents are markup, navigation or boilerplate rather than
# document text. Their contents are discarded entirely.
_SKIP_ELEMENTS = frozenset(
    {"script", "style", "noscript", "svg", "head", "nav", "footer", "header", "form", "aside"}
)

# Elements that imply a line break in the extracted text; without them
# paragraphs run together and the sentence splitter cannot find boundaries.
_BLOCK_ELEMENTS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "table",
    }
)


class FetchError(RuntimeError):
    """A source returned something unusable."""


class _TextExtractor(HTMLParser):
    """Collect visible text, preserving block structure as newlines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_ELEMENTS:
            self._skip_depth += 1
        elif tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_ELEMENTS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Extract readable text from an HTML document.

    Written against ``html.parser`` from the standard library rather than a
    third-party extractor: the pinned stack (spec §2.3) does not include one,
    and adding a dependency to strip tags would be a substitution requiring an
    ADR. Block elements become newlines so the sentence splitter has the
    paragraph boundaries it needs.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 -- malformed markup must not kill an ingest
        # Partial text is still usable evidence; a parse failure on one badly
        # formed page should not abort a 100,000-document run. The exception
        # type is preserved in the message for diagnosis.
        parser.parts.append(f"\n[extraction truncated: {type(exc).__name__}]")

    text = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


@dataclass
class Fetcher:
    """A rate-limited HTTP client with retries.

    One instance per source, so each source's politeness budget is its own and
    a slow source cannot spend another's.
    """

    requests_per_second: float = 5.0
    max_retries: int = 3
    timeout_s: float = 45.0
    user_agent: str = USER_AGENT
    client: httpx.Client | None = None
    requests: int = 0
    failures: int = 0
    _last_request: float = field(default=0.0, init=False)

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(
                timeout=self.timeout_s,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                follow_redirects=True,
            )
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _throttle(self) -> None:
        if self.requests_per_second <= 0:
            return
        interval = 1.0 / self.requests_per_second
        elapsed = time.monotonic() - self._last_request
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request = time.monotonic()

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float:
        """Seconds to wait before retrying.

        Never shorter than this source's own request interval. Exponential
        backoff from one second is *faster* than GDELT's published floor of one
        request per five seconds, so retrying a 429 that way spends the
        allowance the retry is waiting for and deepens the throttle -- measured
        as an IP-level block that outlasted the run. ``Retry-After`` wins when
        the server sends one, because that is the server stating its own terms.
        """
        if response is not None:
            header = response.headers.get("Retry-After", "")
            if header.strip().isdigit():
                return min(float(header.strip()), 120.0)
        interval = 1.0 / self.requests_per_second if self.requests_per_second > 0 else 0.0
        return max(min(2.0**attempt, 8.0), interval * 2.0)

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
        """Fetch ``url``, retrying transient failures with backoff.

        Retries only what is worth retrying: connection errors and 5xx/429.
        A 404 is a fact about the resource and retrying it wastes the
        politeness budget that the rest of the ingest needs.
        """
        last: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            self.requests += 1
            throttled: httpx.Response | None = None
            try:
                response = self._http().get(url, headers=headers)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code == 200:
                    return response
                if response.status_code in {429, 500, 502, 503, 504}:
                    last = FetchError(f"HTTP {response.status_code} for {url}")
                    throttled = response
                else:
                    self.failures += 1
                    raise FetchError(f"HTTP {response.status_code} for {url}")
            time.sleep(self._backoff(attempt, throttled))
        self.failures += 1
        raise FetchError(f"giving up on {url} after {self.max_retries} attempts: {last}")

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        response = self.get(url, headers=headers)
        try:
            return response.json()
        except ValueError as exc:
            raise FetchError(f"{url} returned non-JSON: {exc}") from exc

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        return self.get(url, headers=headers).text
