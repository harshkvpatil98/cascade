"""Manifold loader (spec §3.1, "Polymarket / Manifold ... target ~50").

Manifold binary markets are standalone: there is no event structure to
establish parties from, so every candidate from this source has to earn its
place through the ``named_parties`` rule on the question plus its description.
That is the weaker of the two party rules and it rejects most of what arrives
here, which is the intended direction of the error.

Resolution is read strictly. ``MKT`` (resolved to a probability) and ``CANCEL``
are not binary facts and are dropped: spec §3.1 requires an outcome
"resolvable to 0 or 1 with no ambiguity", and a market that settled at 0.35 is
precisely the case that rule excludes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cascade.ledger.http import SourceCache
from cascade.ledger.schema import RawQuestion

__all__ = ["SEARCH_URL", "load_manifold", "parse_market"]

SEARCH_URL = "https://api.manifold.markets/v0/search-markets"


def _parse_ms(value: Any) -> datetime | None:
    """Convert epoch milliseconds to an aware UTC datetime, or ``None``."""
    if not isinstance(value, int | float) or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def parse_market(market: dict[str, Any]) -> RawQuestion | None:
    """Normalise one Manifold market, or ``None`` if it is unusable."""
    if market.get("outcomeType") != "BINARY" or not market.get("isResolved"):
        return None

    resolution = str(market.get("resolution") or "").strip().upper()
    if resolution == "YES":
        outcome = 1
    elif resolution == "NO":
        outcome = 0
    else:
        # MKT / CANCEL / anything else: not an unambiguous binary fact.
        return None

    open_ts = _parse_ms(market.get("createdTime"))
    resolved_at = _parse_ms(market.get("resolutionTime"))
    close_ts = _parse_ms(market.get("closeTime")) or resolved_at
    if open_ts is None or resolved_at is None or close_ts is None:
        return None

    question = str(market.get("question") or "").strip()
    identifier = str(market.get("id") or "").strip()
    if not question or not identifier:
        return None

    try:
        volume = float(market.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0

    description = market.get("textDescription")
    return RawQuestion(
        source="manifold",
        source_ref=f"manifold:{identifier}",
        question=question,
        resolution_criterion=str(description or "").strip(),
        open_ts=open_ts,
        close_ts=close_ts,
        resolved_at=resolved_at,
        outcome=outcome,
        volume=volume,
        event_group=None,
        event_sibling_count=1,
    )


def load_manifold(cache: SourceCache, *, pages: int, page_size: int) -> list[RawQuestion]:
    """Fetch resolved binary markets and normalise them.

    Paginates with the ``beforeTime`` cursor rather than ``offset``: Manifold
    rejects offsets past 1,000, which caps offset paging at ~1,500 markets --
    far too few to survive the party screen. The cursor walks the full history.

    ``sort=newest`` is the only sort the cursor supports, so the sample is
    recency-ordered rather than volume-ordered; the volume floor in
    ``rules.screen`` does the "non-trivial activity" filtering that
    ``sort=most-popular`` would otherwise have contributed.
    """
    out: list[RawQuestion] = []
    cursor: int | None = None
    for _ in range(pages):
        url = (
            f"{SEARCH_URL}?term=&filter=resolved&contractType=BINARY"
            f"&limit={page_size}&sort=newest"
        )
        if cursor is not None:
            url = f"{url}&beforeTime={cursor}"
        payload = cache.get_json(url)
        if not isinstance(payload, list) or not payload:
            break
        for market in payload:
            if not isinstance(market, dict):
                continue
            question = parse_market(market)
            if question is not None:
                out.append(question)
        last = payload[-1]
        next_cursor = last.get("createdTime") if isinstance(last, dict) else None
        if not isinstance(next_cursor, int) or next_cursor == cursor:
            break
        cursor = next_cursor
    return out
