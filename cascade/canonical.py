"""The one canonical serialisation the whole system hashes against.

Two independent guarantees are computed from this function -- the LLM cache
key (spec §8.3) and the scenario manifest (§1.3, §3.1) -- and a third arrives
at M8 as the event-log hash. They must agree on bytes across machines,
processes and ``PYTHONHASHSEED`` values, so there is exactly one definition and
everything imports it rather than reimplementing "sorted keys, no spaces".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

__all__ = ["canonical_json", "canonical_timestamp"]


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to the one JSON string the whole system agrees on.

    Preserves the invariant that identical content hashes identically anywhere:
    keys sorted, separators fixed, non-ASCII escaped so the byte sequence does
    not depend on the filesystem encoding, and NaN rejected because it is not
    JSON and does not compare equal to itself.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_timestamp(value: datetime) -> str:
    """Render an aware datetime as a UTC ISO-8601 string.

    Preserves the invariant that one instant has one representation. The same
    moment written as ``2024-01-01T12:00:00+00:00`` and
    ``2024-01-01T07:00:00-05:00`` must produce the same manifest hash, so the
    offset is normalised away rather than formatted through.

    A naive datetime raises: it denotes no instant, and guessing one is how a
    cutoff silently moves.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"refusing to canonicalise a naive datetime: {value!r}")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
