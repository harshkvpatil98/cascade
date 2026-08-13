"""Curated historical loader (spec §3.1, target ~40 scenarios).

These are the scenarios that most stress causal decomposition, because they
have the most actors: labour negotiations, merger reviews, sanction regimes,
coalition formations. They are also the only source where the cutoff is chosen
by judgement rather than derived by rule, which is the point of curating them.

**Every entry must carry a ``provenance`` string and at least three named
parties, and the loader refuses the file otherwise.** A curated scenario is a
hand-written label on a forecasting benchmark: an entry nobody can check is
indistinguishable from an invented one, and a wrong label corrupts the
headline metric silently and permanently. Failing the load is the only safe
behaviour.

Entries in ``data/curated/`` are drawn from the public record and are marked
``review_status`` so an unreviewed set cannot be mistaken for a verified one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cascade.ledger.schema import Domain, RawQuestion

__all__ = ["CuratedTemplateError", "load_curated", "parse_entry"]

REQUIRED_FIELDS = (
    "id",
    "question",
    "resolution_criterion",
    "domain",
    "parties",
    "open_ts",
    "cutoff_ts",
    "resolved_at",
    "outcome",
    "provenance",
)

MIN_CURATED_PARTIES = 3


class CuratedTemplateError(ValueError):
    """A curated entry does not satisfy the fixed template."""


def _parse_ts(value: Any, *, field: str, entry_id: str) -> datetime:
    """Parse a template timestamp, requiring an explicit instant.

    A date alone (``2016-06-23``) is accepted and read as midnight UTC; that
    is a deliberate, documented convention rather than an inference, because
    the curated episodes are dated to the day in the public record.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise CuratedTemplateError(f"{entry_id}: {field} is not ISO-8601: {value!r}") from exc
    else:
        from datetime import date as _date

        if isinstance(value, _date):
            parsed = datetime(value.year, value.month, value.day)
        else:
            raise CuratedTemplateError(f"{entry_id}: {field} is not a date: {value!r}")

    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def parse_entry(entry: dict[str, Any]) -> RawQuestion:
    """Validate one entry against the fixed template and normalise it.

    Raises rather than skipping. A curated file with a malformed entry is a
    file someone is midway through editing; loading the rest of it would seal
    a set that does not match what its author believes it contains.
    """
    entry_id = str(entry.get("id") or "<missing id>")

    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise CuratedTemplateError(f"{entry_id}: missing required field(s): {', '.join(missing)}")

    parties = entry["parties"]
    if not isinstance(parties, list) or len(parties) < MIN_CURATED_PARTIES:
        raise CuratedTemplateError(
            f"{entry_id}: needs at least {MIN_CURATED_PARTIES} named parties "
            f"(spec §3.1); got {parties!r}"
        )
    if not all(isinstance(party, str) and party.strip() for party in parties):
        raise CuratedTemplateError(f"{entry_id}: every party must be a non-empty string")

    provenance = str(entry["provenance"] or "").strip()
    if not provenance:
        raise CuratedTemplateError(
            f"{entry_id}: provenance is required. A curated label nobody can check is "
            "indistinguishable from an invented one."
        )

    outcome = entry["outcome"]
    if outcome not in (0, 1):
        raise CuratedTemplateError(f"{entry_id}: outcome must be 0 or 1; got {outcome!r}")

    domain = str(entry["domain"])
    open_ts = _parse_ts(entry["open_ts"], field="open_ts", entry_id=entry_id)
    cutoff_ts = _parse_ts(entry["cutoff_ts"], field="cutoff_ts", entry_id=entry_id)
    resolved_at = _parse_ts(entry["resolved_at"], field="resolved_at", entry_id=entry_id)

    if not open_ts < cutoff_ts < resolved_at:
        raise CuratedTemplateError(
            f"{entry_id}: requires open_ts < cutoff_ts < resolved_at; got "
            f"{open_ts.date()} / {cutoff_ts.date()} / {resolved_at.date()}"
        )

    return RawQuestion(
        source="curated",
        source_ref=f"curated:{entry['id']}",
        question=str(entry["question"]).strip(),
        resolution_criterion=str(entry["resolution_criterion"]).strip(),
        open_ts=open_ts,
        close_ts=resolved_at,
        resolved_at=resolved_at,
        outcome=outcome,
        volume=0.0,
        event_group=None,
        event_sibling_count=1,
        curated_parties=tuple(str(party).strip() for party in parties),
        curated_domain=domain,
        explicit_cutoff=cutoff_ts,
        provenance=provenance,
    )


def load_curated(directory: Path) -> list[RawQuestion]:
    """Load every curated entry, sorted by file then by id.

    Sorted (invariant 7) so the pool order does not depend on the filesystem's
    directory ordering, which differs between machines.
    """
    if not directory.is_dir():
        return []
    out: list[RawQuestion] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            continue
        if not isinstance(loaded, list):
            raise CuratedTemplateError(f"{path.name}: expected a YAML list of entries")
        for entry in loaded:
            if not isinstance(entry, dict):
                raise CuratedTemplateError(f"{path.name}: every entry must be a mapping")
            question = parse_entry(entry)
            if question.source_ref in seen:
                raise CuratedTemplateError(f"{path.name}: duplicate id {question.source_ref}")
            seen.add(question.source_ref)
            out.append(question)
    return sorted(out, key=lambda item: item.source_ref)


def valid_domains() -> tuple[str, ...]:
    """The domain tags a curated entry may use."""
    return tuple(sorted(Domain.__args__))  # type: ignore[attr-defined]
