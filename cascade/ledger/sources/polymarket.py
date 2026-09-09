"""Polymarket loader (spec §3.1, "Polymarket / Manifold ... target ~50").

Reads **events**, not markets. An event is one real-world question carrying a
mutually exclusive market per outcome -- one per candidate, one per team -- and
that structure is what this loader is for:

* it establishes ">= 3 distinguishable parties with non-identical objectives"
  from the shape of the event rather than from reading the title, which is the
  strongest party evidence available anywhere in the pipeline; and
* it makes the independence problem explicit. Sixty markets under
  "World Cup Winner" are sixty views of one event, and admitting more than one
  would inflate the effective sample the M7 bootstrap treats as independent.

One market per event survives, chosen by a keyed hash of its question so the
choice is reproducible and -- critically -- **independent of the outcome**.
Choosing the market that resolved YES would make every Polymarket scenario a
YES and destroy the base-rate control by construction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import blake2b
from typing import Any

from cascade.ledger.http import SourceCache
from cascade.ledger.schema import RawQuestion

__all__ = ["EVENTS_KEYSET_URL", "EVENTS_URL", "load_polymarket", "parse_event"]

EVENTS_URL = "https://gamma-api.polymarket.com/events"
# Keyset pagination. The offset endpoint refuses offsets past ~2,000, which
# caps it at roughly 2,000 events; this one walks the whole archive. The cursor
# parameter is `after_cursor` -- taken from the service's own OpenAPI document
# at /openapi.json rather than guessed.
EVENTS_KEYSET_URL = "https://gamma-api.polymarket.com/events/keyset"


def _parse_ts(value: Any) -> datetime | None:
    """Parse an upstream timestamp into an aware UTC datetime, or ``None``.

    Returns ``None`` rather than guessing. Polymarket mixes ISO-8601 with a
    Postgres-style ``2024-11-06 15:17:41+00``, so both are handled explicitly;
    anything else is treated as missing.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _as_list(value: Any) -> list[Any]:
    """Gamma returns ``outcomes``/``outcomePrices`` as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _outcome_of(market: dict[str, Any]) -> int | None:
    """Return 1/0 for a cleanly resolved binary market, else ``None``.

    Only an unambiguous ``["1","0"]`` or ``["0","1"]`` counts. A market that
    settled to a mid price never resolved to a fact, and spec §3.1 requires
    "resolvable to 0 or 1 with no ambiguity".
    """
    outcomes = [str(item).strip().lower() for item in _as_list(market.get("outcomes"))]
    if outcomes != ["yes", "no"]:
        return None
    prices = _as_list(market.get("outcomePrices"))
    if len(prices) != 2:
        return None
    try:
        yes_price = float(prices[0])
        no_price = float(prices[1])
    except (TypeError, ValueError):
        return None
    if yes_price == 1.0 and no_price == 0.0:
        return 1
    if yes_price == 0.0 and no_price == 1.0:
        return 0
    return None


def parse_event(event: dict[str, Any], *, salt: str) -> RawQuestion | None:
    """Normalise one event into a single candidate question, or ``None``.

    Returns ``None`` when the event carries no cleanly resolved binary market,
    rather than reaching for a partial one.
    """
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        return None

    resolved: list[tuple[dict[str, Any], int]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        if str(market.get("umaResolutionStatus") or "").lower() not in {"resolved", ""}:
            continue
        outcome = _outcome_of(market)
        if outcome is None:
            continue
        resolved.append((market, outcome))

    if not resolved:
        return None

    # Outcome-independent choice. Sorting by a keyed hash of the question makes
    # the pick reproducible without letting the resolution influence it.
    resolved.sort(
        key=lambda pair: blake2b(
            str(pair[0].get("question", "")).encode("utf-8"),
            key=salt.encode("utf-8"),
            digest_size=16,
        ).digest()
    )
    market, outcome = resolved[0]

    open_ts = _parse_ts(market.get("startDate")) or _parse_ts(event.get("startDate"))
    resolved_at = _parse_ts(market.get("closedTime")) or _parse_ts(market.get("endDate"))
    close_ts = _parse_ts(market.get("endDate")) or resolved_at
    if open_ts is None or resolved_at is None or close_ts is None:
        return None

    question = str(market.get("question") or "").strip()
    if not question:
        return None

    try:
        volume = float(market.get("volumeNum") or event.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0

    slug = str(event.get("slug") or event.get("ticker") or event.get("id") or "").strip()
    if not slug:
        return None

    return RawQuestion(
        source="polymarket",
        source_ref=f"polymarket:{slug}:{market.get('id')}",
        question=question,
        resolution_criterion=str(market.get("description") or "").strip(),
        open_ts=open_ts,
        close_ts=close_ts,
        resolved_at=resolved_at,
        outcome=outcome,
        volume=volume,
        event_group=f"polymarket:{slug}",
        # Sibling count over *all* markets in the event, not just the resolved
        # ones: the parties existed whether or not each leg settled cleanly.
        event_sibling_count=len(markets),
    )


def load_polymarket(
    cache: SourceCache, *, salt: str, pages: int, page_size: int
) -> list[RawQuestion]:
    """Fetch closed events, highest volume first, and normalise them.

    Ordered by volume because spec §3.1 asks for "non-trivial volume": a
    thinly traded market's resolution is less reliable, and its price carries
    little information about whether the outcome was genuinely unsettled.

    Paginates by keyset cursor rather than offset. The offset endpoint rejects
    offsets past ~2,000 with a 422, which capped the reachable archive at about
    2,000 events -- and since the >= 3-party rule is carried mostly by
    Polymarket's event structure, that cap was the binding constraint on the
    whole scenario set. The keyset endpoint has no such ceiling.
    """
    out: list[RawQuestion] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(pages):
        url = f"{EVENTS_KEYSET_URL}?limit={page_size}&closed=true&order=volume&ascending=false"
        if cursor is not None:
            url = f"{url}&after_cursor={cursor}"
        payload = cache.get_json(url)
        if not isinstance(payload, dict):
            break
        events = payload.get("events")
        if not isinstance(events, list) or not events:
            break

        for event in events:
            if not isinstance(event, dict):
                continue
            question = parse_event(event, salt=salt)
            if question is not None:
                out.append(question)

        next_cursor = payload.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        # A cursor that repeats means the server stopped advancing; continuing
        # would re-fetch the same page until `pages` is exhausted.
        if next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return out
