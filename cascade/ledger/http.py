"""Content-addressed cache for source API responses.

The registry is assembled once and sealed, but it has to be *rebuildable* --
by CI, by a reviewer, on another machine -- and the upstream pools change
every day. A build that re-queried the live APIs would produce a different
180 each time and the manifest hash would be meaningless.

So raw responses are recorded to disk on first fetch and every later build
reads them, exactly as ``llm/cache.py`` does for model calls and for the same
reason. ``--refresh`` is the only way to go back to the network, and it is a
deliberate act that changes the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from cascade.canonical import canonical_json

__all__ = ["SourceCache", "SourceFetchError", "SourceOffline"]

USER_AGENT = "cascade-ledger/0.1 (research backtest; contact via repository)"


class SourceFetchError(RuntimeError):
    """A source API returned something unusable."""


class SourceOffline(RuntimeError):
    """A response was needed that has not been recorded, and refresh is off."""


@dataclass
class SourceCache:
    """Records ``GET`` responses by URL, and replays them thereafter."""

    root: Path
    refresh: bool = False
    client: httpx.Client | None = None
    hits: int = 0
    misses: int = 0
    fetched: int = 0
    _owned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(
                timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
            )
            self._owned = True
        return self.client

    def close(self) -> None:
        if self.client is not None and self._owned:
            self.client.close()
            self.client = None
            self._owned = False

    def path_for(self, url: str) -> Path:
        key = hashlib.sha256(canonical_json({"method": "GET", "url": url}).encode()).hexdigest()
        return self.root / key[:2] / f"{key}.json"

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        """Return the parsed body for ``url``, from disk when it is recorded.

        Preserves rebuildability: the same cache directory yields the same
        registry, and therefore the same manifest hash, on any machine.
        """
        path = self.path_for(url)
        if path.is_file() and not self.refresh:
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        self.misses += 1
        if not self.refresh and not path.is_file():
            raise SourceOffline(
                f"no recorded response for {url}. Run `cascade ledger build --refresh` "
                "to fetch it; that changes the pool and therefore the manifest."
            )

        response = self._http().get(url, headers=headers)
        if response.status_code != 200:
            raise SourceFetchError(
                f"GET {url} returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceFetchError(f"GET {url} returned non-JSON: {exc}") from exc

        self._write(path, payload)
        self.fetched += 1
        return payload

    def put_json(self, url: str, payload: Any) -> None:
        """Record ``payload`` as the answer for ``url``.

        Used to record a *pagination boundary*: when an API rejects an offset
        past its ceiling, "there is no more data here" is the real answer and
        has to be recorded like any other, or the next cached build asks for a
        page that was never stored and drops the whole source.
        """
        self._write(self.path_for(url), payload)

    def _write(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            # Cleanup and re-raise: a truncated recording would be replayed as
            # though it were the real response.
            Path(temporary).unlink(missing_ok=True)
            raise

    def count(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.is_dir() else 0
