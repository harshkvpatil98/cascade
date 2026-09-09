"""64-bit SimHash for near-duplicate collapse (spec §3.2).

Pure: no I/O, no clock, no RNG.

News corpora are heavily syndicated -- the same wire story appears under
dozens of mastheads within minutes. Left alone those copies would each
contribute chunks, so the corpus would look larger than it is while carrying
the same evidence many times, and retrieval would return k copies of one
article instead of k pieces of evidence.

The collapse keeps the **earliest** ``published_at`` of the group. That
direction is not arbitrary: the earliest copy is when the information actually
became public, and keeping a later one would move evidence *forward* in time
past a cutoff that should have excluded it.
"""

from __future__ import annotations

import re
from hashlib import blake2b

__all__ = ["HASH_BITS", "hamming", "shingles", "simhash"]

HASH_BITS = 64
_MASK = (1 << HASH_BITS) - 1
_TOKEN = re.compile(r"[a-z0-9]+")

# Word-level shingles. Single words collide across unrelated documents;
# 3-grams capture enough phrasing that an edited headline still collapses
# while a genuinely different article does not.
_SHINGLE_SIZE = 3


def shingles(text: str, size: int = _SHINGLE_SIZE) -> list[str]:
    """Return overlapping word n-grams, lowercased and punctuation-stripped.

    Normalising this way is what makes the hash robust to the reformatting a
    syndicated copy picks up -- different quote characters, a different
    dateline prefix -- while still separating documents that differ in words.
    """
    words = _TOKEN.findall(text.lower())
    if not words:
        return []
    if len(words) < size:
        return [" ".join(words)]
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def simhash(text: str) -> int:
    """Return the 64-bit SimHash of ``text``.

    Uses blake2b rather than the built-in ``hash()``: ``hash()`` is salted per
    process, so the same document would fingerprint differently on every run
    and dedupe would silently stop working across a resume.
    """
    features = shingles(text)
    if not features:
        return 0

    columns = [0] * HASH_BITS
    for feature in features:
        digest = int.from_bytes(blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(HASH_BITS):
            columns[bit] += 1 if digest >> bit & 1 else -1

    value = 0
    for bit in range(HASH_BITS):
        if columns[bit] > 0:
            value |= 1 << bit
    return value & _MASK


def hamming(left: int, right: int) -> int:
    """Number of differing bits. Distance <= 3 collapses (spec §3.2)."""
    return ((left ^ right) & _MASK).bit_count()


def to_signed(value: int) -> int:
    """Reinterpret an unsigned 64-bit hash as signed, for Postgres ``bigint``.

    Postgres has no unsigned 64-bit integer. The bit pattern is what matters
    for Hamming distance, and it round-trips exactly.
    """
    return value - (1 << HASH_BITS) if value >= 1 << (HASH_BITS - 1) else value


def from_signed(value: int) -> int:
    """Inverse of :func:`to_signed`."""
    return value + (1 << HASH_BITS) if value < 0 else value
