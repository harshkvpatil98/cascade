"""The frozen split: manifest sealing and tamper detection (spec §1.3).

M1 acceptance criterion: ``manifest.sha256`` is written, and mutating one
label makes the manifest check fail. That second half is the whole point --
a manifest that survives an edited outcome would be decoration.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from cascade.ledger.climatology import brier_score, climatology_of
from cascade.ledger.manifest import (
    ManifestMismatch,
    compute_manifest,
    manifest_rows,
    verify_manifest,
)
from cascade.ledger.schema import Scenario, ScenarioLabel, ScenarioRecord

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def record(
    scenario_id: str,
    *,
    outcome: int = 1,
    cutoff: datetime | None = None,
    domain: str = "elections",
) -> ScenarioRecord:
    cutoff_ts = cutoff or BASE
    resolve_ts = cutoff_ts + timedelta(days=60)
    return ScenarioRecord(
        scenario=Scenario(
            scenario_id=scenario_id,
            question=f"Will {scenario_id} happen?",
            resolution_criterion="stated criterion",
            cutoff_ts=cutoff_ts,
            resolve_ts=resolve_ts,
            domain=domain,  # type: ignore[arg-type]
            source="curated",
            source_ref=scenario_id,
            party_rule="curated",
            party_names=("A", "B", "C"),
            event_group=None,
        ),
        label=ScenarioLabel(
            scenario_id=scenario_id,
            outcome=outcome,  # type: ignore[arg-type]
            resolved_at=resolve_ts,
        ),
    )


def registry(n: int = 8) -> tuple[ScenarioRecord, ...]:
    return tuple(record(f"s{index:03d}", outcome=index % 2) for index in range(n))


# ---------------------------------------------------------------------------
# The hash
# ---------------------------------------------------------------------------


def test_manifest_is_a_sha256_hex_digest() -> None:
    digest = compute_manifest(registry())
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_manifest_is_stable_across_input_ordering() -> None:
    """The seal must not depend on the order loaders happened to return."""
    records = registry()
    assert compute_manifest(records) == compute_manifest(tuple(reversed(records)))


def test_manifest_covers_exactly_four_fields() -> None:
    """Widening the projection silently invalidates every existing seal."""
    rows = manifest_rows(registry(3))
    assert [field for field in vars(rows[0])] == [
        "scenario_id",
        "cutoff_ts",
        "resolve_ts",
        "outcome",
    ]


def test_manifest_ignores_fields_it_does_not_cover() -> None:
    """Rewording a question must not break a seal; it changes no label."""
    original = record("s1")
    reworded = ScenarioRecord(
        scenario=original.scenario.model_copy(update={"question": "Totally different?"}),
        label=original.label,
    )
    assert compute_manifest((original,)) == compute_manifest((reworded,))


# ---------------------------------------------------------------------------
# Tamper detection -- the M1 acceptance criterion
# ---------------------------------------------------------------------------


def test_mutating_one_label_fails_the_manifest_check() -> None:
    """M1 acceptance: flip a single outcome and the seal must reject the set."""
    records = registry(12)
    sealed = compute_manifest(records)
    verify_manifest(records, sealed)  # unchanged set still verifies

    tampered = (
        ScenarioRecord(
            scenario=records[0].scenario,
            label=records[0].label.model_copy(update={"outcome": 1 - records[0].label.outcome}),
        ),
        *records[1:],
    )
    with pytest.raises(ManifestMismatch) as caught:
        verify_manifest(tampered, sealed)
    assert "Do not reseal" in str(caught.value)
    assert caught.value.expected == sealed


def test_moving_a_cutoff_fails_the_manifest_check() -> None:
    """A cutoff shifted later is a leakage edit, and is exactly what to catch."""
    records = registry(6)
    sealed = compute_manifest(records)
    moved = (
        ScenarioRecord(
            scenario=records[0].scenario.model_copy(
                update={"cutoff_ts": records[0].scenario.cutoff_ts + timedelta(days=1)}
            ),
            label=records[0].label,
        ),
        *records[1:],
    )
    with pytest.raises(ManifestMismatch):
        verify_manifest(moved, sealed)


def test_dropping_a_scenario_fails_the_manifest_check() -> None:
    records = registry(6)
    sealed = compute_manifest(records)
    with pytest.raises(ManifestMismatch):
        verify_manifest(records[:-1], sealed)


def test_timezone_representation_does_not_change_the_hash() -> None:
    """One instant has one hash, whatever offset it was written in."""
    utc = record("s1", cutoff=datetime(2024, 6, 1, 12, 0, tzinfo=UTC))
    shifted_zone = timezone_shifted(utc)
    assert compute_manifest((utc,)) == compute_manifest((shifted_zone,))


def timezone_shifted(source: ScenarioRecord) -> ScenarioRecord:
    """Re-express the same instants in a non-UTC offset."""
    other = timezone(timedelta(hours=-5))
    return ScenarioRecord(
        scenario=source.scenario.model_copy(
            update={
                "cutoff_ts": source.scenario.cutoff_ts.astimezone(other),
                "resolve_ts": source.scenario.resolve_ts.astimezone(other),
            }
        ),
        label=source.label.model_copy(
            update={"resolved_at": source.label.resolved_at.astimezone(other)}
        ),
    )


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------


def test_climatology_of_a_balanced_set() -> None:
    """A 50/50 set has base rate 0.5 and Brier 0.25 -- the p(1-p) identity."""
    result = climatology_of(registry(100))
    assert result.base_rate == pytest.approx(0.5)
    assert result.brier == pytest.approx(0.25)
    assert result.n == 100
    assert result.n_yes == 50


def test_climatology_matches_the_variance_identity() -> None:
    """Brier of a constant base-rate forecast equals p(1-p).

    Computed by the direct definition in ``climatology_of``; asserted against
    the identity here, so M7's Murphy decomposition has something to check
    rather than something it assumes.
    """
    records = tuple(record(f"s{i}", outcome=1 if i < 30 else 0) for i in range(100))
    result = climatology_of(records)
    assert result.base_rate == pytest.approx(0.30)
    assert result.brier == pytest.approx(0.30 * 0.70)


def test_brier_of_a_perfect_forecaster_is_zero() -> None:
    assert brier_score((1.0, 0.0, 1.0), (1, 0, 1)) == 0.0


def test_brier_of_an_inverted_forecaster_is_one() -> None:
    assert brier_score((0.0, 1.0), (1, 0)) == 1.0


def test_brier_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        brier_score((0.5, 0.5), (1,))


def test_climatology_of_an_empty_set_is_undefined() -> None:
    """Returning 0.0 would silently make an empty study look perfect."""
    with pytest.raises(ValueError, match="undefined"):
        climatology_of(())
