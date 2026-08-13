"""Source loaders: normalisation, and what they refuse.

A loader's only job is normalisation. It never invents a field: a missing or
unparseable timestamp drops the question rather than defaulting, which is the
same rule the corpus applies to ``published_at`` at M2 and for the same
reason -- a question with an uncertain date is a leakage vector.

Fixtures are trimmed real payloads, so a change in an upstream's shape shows
up here rather than as a smaller set three milestones later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cascade.ledger.sources.curated import CuratedTemplateError, load_curated, parse_entry
from cascade.ledger.sources.manifold import parse_market
from cascade.ledger.sources.metaculus import MetaculusUnavailable, load_metaculus, parse_question
from cascade.ledger.sources.polymarket import parse_event

SALT = "test-salt"


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


def event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "900",
        "slug": "presidential-election-winner-2024",
        "title": "Presidential Election Winner 2024",
        "startDate": "2024-01-04T22:58:00Z",
        "volume": "1531479284.5",
        "markets": [
            {
                "id": "1",
                "question": "Will Alpha win the 2024 election?",
                "description": "Resolves YES if Alpha wins.",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["1", "0"]',
                "volumeNum": "1531479284.5",
                "startDate": "2024-01-04T22:58:00Z",
                "endDate": "2024-11-05T12:00:00Z",
                "closedTime": "2024-11-06 15:17:41+00",
                "umaResolutionStatus": "resolved",
            },
            {
                "id": "2",
                "question": "Will Beta win the 2024 election?",
                "description": "Resolves YES if Beta wins.",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0", "1"]',
                "volumeNum": "900000.0",
                "startDate": "2024-01-04T22:58:00Z",
                "endDate": "2024-11-05T12:00:00Z",
                "closedTime": "2024-11-06 15:17:41+00",
                "umaResolutionStatus": "resolved",
            },
            {"id": "3", "question": "Will Gamma win?", "outcomes": '["Yes", "No"]'},
        ],
    }
    base.update(overrides)
    return base


def test_polymarket_event_becomes_one_question() -> None:
    """Sixty markets under one event are sixty views of one event."""
    parsed = parse_event(event(), salt=SALT)
    assert parsed is not None
    assert parsed.source == "polymarket"
    assert parsed.event_group == "polymarket:presidential-election-winner-2024"
    assert parsed.event_sibling_count == 3


def test_polymarket_parses_both_timestamp_dialects() -> None:
    """Gamma mixes ISO-8601 with a Postgres-style ``... +00``."""
    parsed = parse_event(event(), salt=SALT)
    assert parsed is not None
    assert parsed.resolved_at == datetime(2024, 11, 6, 15, 17, 41, tzinfo=UTC)
    assert parsed.open_ts == datetime(2024, 1, 4, 22, 58, tzinfo=UTC)


def test_polymarket_representative_choice_is_outcome_independent() -> None:
    """Always taking the YES leg would make every scenario a YES.

    That would destroy the base-rate control by construction, in a way the
    downstream base-rate check could not distinguish from a real pool.
    """
    parsed = parse_event(event(), salt=SALT)
    again = parse_event(event(), salt=SALT)
    assert parsed is not None and again is not None
    assert parsed.source_ref == again.source_ref  # deterministic

    flipped = event()
    flipped["markets"][0]["outcomePrices"] = '["0", "1"]'
    flipped["markets"][1]["outcomePrices"] = '["1", "0"]'
    swapped = parse_event(flipped, salt=SALT)
    assert swapped is not None
    # The same market is chosen regardless of how the legs resolved.
    assert swapped.source_ref == parsed.source_ref


@pytest.mark.parametrize("prices", ['["0.35", "0.65"]', '["0.5", "0.5"]', "[]", "not json"])
def test_polymarket_rejects_unresolved_prices(prices: str) -> None:
    """A market that settled mid never resolved to a fact (spec §3.1)."""
    payload = event()
    for market in payload["markets"]:
        market["outcomePrices"] = prices
    assert parse_event(payload, salt=SALT) is None


def test_polymarket_rejects_an_event_with_no_markets() -> None:
    assert parse_event(event(markets=[]), salt=SALT) is None


def test_polymarket_rejects_a_missing_timestamp() -> None:
    """Dropped rather than defaulted: an invented date is a leakage vector."""
    payload = event()
    for market in payload["markets"]:
        market.pop("closedTime", None)
        market.pop("endDate", None)
    assert parse_event(payload, salt=SALT) is None


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------


def market(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "abc123",
        "question": "Will the United States, Iran and Israel agree a ceasefire?",
        "textDescription": "Resolves YES if all three announce a ceasefire.",
        "outcomeType": "BINARY",
        "isResolved": True,
        "resolution": "YES",
        "createdTime": 1690318393221,
        "closeTime": 1735737881975,
        "resolutionTime": 1735737881975,
        "volume": 55179.5,
    }
    base.update(overrides)
    return base


def test_manifold_market_normalises() -> None:
    parsed = parse_market(market())
    assert parsed is not None
    assert parsed.source == "manifold"
    assert parsed.outcome == 1
    assert parsed.event_group is None
    assert parsed.event_sibling_count == 1
    assert parsed.open_ts.tzinfo is not None


@pytest.mark.parametrize("resolution", ["MKT", "CANCEL", "", "N/A"])
def test_manifold_rejects_non_binary_resolutions(resolution: str) -> None:
    """A market that settled at 0.35 is exactly what the "no ambiguity" rule excludes."""
    assert parse_market(market(resolution=resolution)) is None


def test_manifold_rejects_unresolved_and_non_binary() -> None:
    assert parse_market(market(isResolved=False)) is None
    assert parse_market(market(outcomeType="MULTIPLE_CHOICE")) is None


def test_manifold_rejects_a_missing_resolution_time() -> None:
    assert parse_market(market(resolutionTime=None)) is None


def test_manifold_no_resolves_to_zero() -> None:
    parsed = parse_market(market(resolution="NO"))
    assert parsed is not None
    assert parsed.outcome == 0


# ---------------------------------------------------------------------------
# Metaculus -- the source that is currently unavailable
# ---------------------------------------------------------------------------


def test_metaculus_without_a_token_reports_rather_than_returning_empty() -> None:
    """A primary source contributing zero must be visible, not absorbed.

    Returning ``[]`` would let the build look like a thin pool rather than a
    missing credential, and the report would carry no trace of it.
    """
    with pytest.raises(MetaculusUnavailable, match="CASCADE_METACULUS_TOKEN"):
        load_metaculus(None, token=None, pages=1, page_size=10)  # type: ignore[arg-type]


def test_metaculus_parser_works_against_a_recorded_payload() -> None:
    """The loader is ready the moment a token is supplied."""
    parsed = parse_question(
        {
            "id": 12345,
            "title": "Will the United States, Iran and Israel agree a ceasefire?",
            "resolution_criteria": "Resolves YES if all three announce a ceasefire.",
            "publish_time": "2024-01-01T00:00:00Z",
            "close_time": "2024-06-01T00:00:00Z",
            "resolve_time": "2024-06-15T00:00:00Z",
            "resolution": 1,
        }
    )
    assert parsed is not None
    assert parsed.source == "metaculus"
    assert parsed.outcome == 1
    assert parsed.source_ref == "metaculus:12345"


@pytest.mark.parametrize("resolution", [None, -1, "ambiguous", 0.5])
def test_metaculus_rejects_annulled_or_ambiguous_resolutions(resolution: Any) -> None:
    assert (
        parse_question(
            {
                "id": 1,
                "title": "t",
                "publish_time": "2024-01-01T00:00:00Z",
                "resolve_time": "2024-06-15T00:00:00Z",
                "resolution": resolution,
            }
        )
        is None
    )


# ---------------------------------------------------------------------------
# Curated
# ---------------------------------------------------------------------------


def curated_entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "example-episode",
        "question": "Will the parties reach an agreement?",
        "resolution_criterion": "YES if an agreement is announced.",
        "domain": "labor",
        "parties": ["Union", "Employer", "Mediator"],
        "open_ts": "2023-01-01",
        "cutoff_ts": "2023-03-01",
        "resolved_at": "2023-06-01",
        "outcome": 1,
        "provenance": "Public record, cited.",
    }
    base.update(overrides)
    return base


def test_curated_entry_parses() -> None:
    parsed = parse_entry(curated_entry())
    assert parsed.source == "curated"
    assert parsed.explicit_cutoff is not None
    assert parsed.curated_parties == ("Union", "Employer", "Mediator")


def test_curated_entry_requires_provenance() -> None:
    """A curated label nobody can check is indistinguishable from an invented one."""
    with pytest.raises(CuratedTemplateError, match="provenance is required"):
        parse_entry(curated_entry(provenance="  "))


def test_curated_entry_requires_three_parties() -> None:
    with pytest.raises(CuratedTemplateError, match="at least 3 named parties"):
        parse_entry(curated_entry(parties=["Union", "Employer"]))


@pytest.mark.parametrize("field", ["id", "question", "domain", "outcome", "cutoff_ts"])
def test_curated_entry_requires_every_template_field(field: str) -> None:
    entry = curated_entry()
    entry.pop(field)
    with pytest.raises(CuratedTemplateError):
        parse_entry(entry)


def test_curated_entry_requires_ordered_timestamps() -> None:
    with pytest.raises(CuratedTemplateError, match="open_ts < cutoff_ts < resolved_at"):
        parse_entry(curated_entry(cutoff_ts="2022-01-01"))


def test_curated_entry_rejects_a_non_binary_outcome() -> None:
    with pytest.raises(CuratedTemplateError, match="outcome must be 0 or 1"):
        parse_entry(curated_entry(outcome=2))


def test_a_malformed_entry_fails_the_whole_file(tmp_path: Path) -> None:
    """Loading the rest would seal a set that does not match its author's intent."""
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump([curated_entry(), curated_entry(id="x", provenance="")]),
        encoding="utf-8",
    )
    with pytest.raises(CuratedTemplateError):
        load_curated(tmp_path)


def test_a_missing_curated_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_curated(tmp_path / "absent") == []


def test_the_shipped_curated_set_loads_and_satisfies_the_template() -> None:
    """The real data file must satisfy its own rules."""
    from cascade.config import repo_root

    entries = load_curated(repo_root() / "data" / "curated")
    assert entries, "the shipped curated set is empty"
    for entry in entries:
        assert len(entry.curated_parties) >= 3
        assert entry.provenance
        assert entry.explicit_cutoff is not None
        assert entry.open_ts < entry.explicit_cutoff < entry.resolved_at
