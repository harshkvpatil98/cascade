"""Content-addressed store for recorded model calls (spec §8.3).

The cache is the replay mechanism and the reason the study is affordable, so
its key derivation is a correctness surface, not an optimisation. Two rules
follow from that:

* The key is computed over a *canonical* serialisation. Dict ordering, float
  repr and unicode escaping are all nondeterminism vectors, and a key that
  varies with insertion order silently halves the hit rate.
* The key covers exactly what changes the response. ``cache_control`` markers
  do not (ADR-0007), so they are stripped before hashing and prompt-cache
  tuning stays a pure cost change.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cascade.canonical import canonical_json
from cascade.llm.types import CachedCall, LLMRequest

# Re-exported: the cache key is defined in terms of this serialisation, so a
# reader of this module should not have to go looking for it. There is exactly
# one definition, in cascade/canonical.py.
__all__ = ["CacheStats", "CallCache", "cache_key", "canonical_json"]


def cache_key(request: LLMRequest) -> str:
    """Return the spec §8.3 content address for ``request``.

    Preserves the invariant that the key domain is exactly the fields that
    determine the response -- see ``LLMRequest.cache_domain``.
    """
    return hashlib.sha256(canonical_json(request.cache_domain()).encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    """Hit/miss accounting for one process.

    The aggregate hit rate is the single biggest lever on total study cost
    (spec §12.2), so it is measured rather than assumed.
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from disk. Zero lookups reads as 0.0."""
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class CallCache:
    """A directory of recorded calls, addressed by content hash.

    Entries are sharded two levels deep on the key prefix: a single flat
    directory holding the study's recordings would put hundreds of thousands
    of entries in one inode, which is slow to enumerate on every filesystem
    that matters.
    """

    root: Path
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def path_for(self, key: str) -> Path:
        """Return the on-disk location for ``key`` without touching the disk."""
        if len(key) < 4:
            raise ValueError(f"cache key {key!r} is too short to shard")
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def __contains__(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def get(self, key: str) -> CachedCall | None:
        """Return the recorded call for ``key``, or ``None`` on a miss.

        A corrupt entry is reported, never skipped: silently treating an
        unreadable recording as a miss would let replay mode quietly reach for
        the network in ``record``, or fail with a misleading CacheMiss.
        """
        path = self.path_for(key)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"cache entry {path} is not valid JSON: {exc}") from exc
        call = CachedCall.model_validate(payload)
        if call.key != key:
            raise ValueError(
                f"cache entry {path} records key {call.key!r} but is filed under {key!r}; "
                "the store has been corrupted or hand-edited"
            )
        self.stats.hits += 1
        return call

    def put(self, call: CachedCall) -> None:
        """Persist ``call``, atomically.

        Written to a temporary file in the destination directory and then
        renamed, so a process killed mid-write leaves either the old entry or
        the new one -- never a half-written record that a later replay would
        fail to parse.
        """
        path = self.path_for(call.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{call.key[:8]}-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(call.model_dump_json(indent=None))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            # Cleanup and re-raise -- the failure propagates untouched. Leaving
            # a half-written .tmp behind would be picked up by count().
            Path(temporary).unlink(missing_ok=True)
            raise
        self.stats.writes += 1

    def count(self) -> int:
        """Number of recorded calls on disk. Used by the acceptance report."""
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.rglob("*.json"))
