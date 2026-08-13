"""Metaculus loader (spec §3.1, primary source, target ~90 scenarios).

**This source requires a credential that the study does not currently have.**
As of this milestone the public API answers every unauthenticated request with:

    Permission Error: The API is only available to authenticated users.
    Please create an account and use your API token to access the API.

That is an upstream policy change since the spec was written -- §3.1 describes
Metaculus as "resolved binary questions via the public API". The loader is
written, tested against recorded fixtures, and wired into the registry; it
activates the moment ``CASCADE_METACULUS_TOKEN`` is set, and contributes
nothing until then.

It is *not* silently skipped. ``cascade ledger build`` reports the source as
unavailable and names the variable, because a primary source contributing zero
scenarios is exactly the kind of shortfall that must be visible in the report
rather than absorbed by the other loaders.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cascade.ledger.http import SourceCache
from cascade.ledger.schema import RawQuestion

__all__ = ["QUESTIONS_URL", "MetaculusUnavailable", "load_metaculus", "parse_question"]

QUESTIONS_URL = "https://www.metaculus.com/api2/questions/"


class MetaculusUnavailable(RuntimeError):
    """Metaculus needs an API token that is not configured."""


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def parse_question(question: dict[str, Any]) -> RawQuestion | None:
    """Normalise one Metaculus question, or ``None`` if it is unusable.

    Kept separate from the fetch so it is testable against recorded fixtures
    while the API itself is unreachable.
    """
    if question.get("type") not in {"binary", None} and question.get("possibilities", {}).get(
        "type"
    ) not in {"binary", None}:
        return None

    resolution = question.get("resolution")
    if resolution not in (0, 1, 0.0, 1.0):
        # Ambiguous or annulled resolutions carry no binary fact.
        return None
    outcome = int(resolution)

    open_ts = _parse_ts(question.get("publish_time")) or _parse_ts(question.get("created_time"))
    resolved_at = _parse_ts(question.get("resolve_time"))
    close_ts = _parse_ts(question.get("close_time")) or resolved_at
    if open_ts is None or resolved_at is None or close_ts is None:
        return None

    title = str(question.get("title") or "").strip()
    identifier = question.get("id")
    if not title or identifier is None:
        return None

    return RawQuestion(
        source="metaculus",
        source_ref=f"metaculus:{identifier}",
        question=title,
        resolution_criterion=str(question.get("resolution_criteria") or "").strip(),
        open_ts=open_ts,
        close_ts=close_ts,
        resolved_at=resolved_at,
        outcome=outcome,
        # Metaculus is not a market; there is no volume. The volume floor is
        # applied only to market sources (see rules.screen).
        volume=0.0,
        event_group=None,
        event_sibling_count=1,
    )


def load_metaculus(
    cache: SourceCache, *, token: str | None, pages: int, page_size: int
) -> list[RawQuestion]:
    """Fetch resolved binary questions. Raises when no token is configured.

    Preserves the requirement that a missing primary source is *reported*, not
    absorbed: the caller catches :class:`MetaculusUnavailable` and records it
    in the build report rather than quietly producing a set assembled from the
    remaining sources.
    """
    if not token:
        raise MetaculusUnavailable(
            "Metaculus requires an API token: the public API rejects "
            "unauthenticated requests. Set CASCADE_METACULUS_TOKEN to a token "
            "from https://www.metaculus.com/accounts/settings/ to enable the "
            "spec's primary source (~90 of 180 scenarios)."
        )

    headers = {"Authorization": f"Token {token}"}
    out: list[RawQuestion] = []
    for page in range(pages):
        url = (
            f"{QUESTIONS_URL}?limit={page_size}&offset={page * page_size}"
            "&status=resolved&type=binary&order_by=-activity"
        )
        payload = cache.get_json(url, headers=headers)
        if not isinstance(payload, dict):
            break
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            break
        for question in results:
            if not isinstance(question, dict):
                continue
            parsed = parse_question(question)
            if parsed is not None:
                out.append(parsed)
    return out
