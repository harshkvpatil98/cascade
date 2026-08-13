"""Invariant 2, enforced by Postgres: the simulation cannot read outcomes.

M1 acceptance criterion: connecting as ``cascade_sim`` and selecting from
``scenario_labels`` raises ``InsufficientPrivilege``.

This is the test the whole role-separation design exists for, so it asserts
the mechanism rather than the intention: not that the code avoids the table,
but that the database refuses the read. Code review is not a mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cascade.config import Settings
from cascade.ledger.manifest import ManifestMismatch, compute_manifest, verify_manifest

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real settings against the running stack, with the project's own .env."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))
    settings = Settings()
    try:
        import psycopg

        with psycopg.connect(settings.database_url("admin"), connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001 -- a skip needs the reason, not a traceback
        pytest.skip(f"postgres not reachable: {type(exc).__name__}")
    return settings


def count(settings: Settings, role: str, table: str) -> int:
    import psycopg

    with (
        psycopg.connect(settings.database_url(role), connect_timeout=10) as conn,  # type: ignore[arg-type]
        conn.cursor() as cur,
    ):
        cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608 -- fixed literals below
        row = cur.fetchone()
    return int(row[0]) if row else 0


def require_registry(settings: Settings) -> int:
    loaded = count(settings, "admin", "scenarios")
    if loaded == 0:
        pytest.skip("no registry loaded; run `cascade ledger build --write`")
    return loaded


# ---------------------------------------------------------------------------
# The grant -- the M1 acceptance criterion
# ---------------------------------------------------------------------------


def test_sim_role_cannot_read_scenario_labels(live_settings: Settings) -> None:
    """M1 acceptance: the simulation role gets a permission error on outcomes."""
    require_registry(live_settings)
    from psycopg import errors

    with pytest.raises(errors.InsufficientPrivilege):
        count(live_settings, "sim", "scenario_labels")


def test_sim_role_can_read_scenarios(live_settings: Settings) -> None:
    """The separation must be surgical: questions yes, answers no.

    A sim role locked out of both tables would pass the test above while
    making the system unable to run.
    """
    assert count(live_settings, "sim", "scenarios") == require_registry(live_settings)


def test_eval_role_can_read_both(live_settings: Settings) -> None:
    """Only ``cascade_eval`` sees outcomes -- and it must actually see them."""
    loaded = require_registry(live_settings)
    assert count(live_settings, "eval", "scenarios") == loaded
    assert count(live_settings, "eval", "scenario_labels") == loaded


def test_every_scenario_has_exactly_one_label(live_settings: Settings) -> None:
    """A missing label would silently drop a scenario from every metric."""
    loaded = require_registry(live_settings)
    assert count(live_settings, "eval", "scenario_labels") == loaded


def test_load_scenarios_returns_no_outcome_field(live_settings: Settings) -> None:
    """The code boundary matches the grant boundary.

    ``load_scenarios`` performs no join, so even a role that *could* read
    labels does not receive an outcome through it.
    """
    require_registry(live_settings)
    from cascade.ledger.store import load_scenarios

    scenarios = load_scenarios(live_settings, role="sim")
    assert scenarios
    assert all("outcome" not in scenario.model_dump() for scenario in scenarios)


# ---------------------------------------------------------------------------
# The frozen split, against the real database
# ---------------------------------------------------------------------------


def test_the_loaded_registry_matches_its_seal(live_settings: Settings) -> None:
    from cascade.ledger.store import load_records, read_manifest

    require_registry(live_settings)
    sealed = read_manifest(live_settings, role="admin")
    if sealed is None:
        pytest.skip("registry is not sealed; run `cascade ledger seal`")
    records = load_records(live_settings, role="admin")
    verify_manifest(records, sealed.manifest_sha256)
    assert sealed.n_scenarios == len(records)


def test_mutating_a_stored_label_breaks_the_seal(live_settings: Settings) -> None:
    """M1 acceptance, against real rows rather than a fixture.

    The label is flipped in memory, not in the database: the point is that the
    *hash* notices, and mutating the stored registry would corrupt the very
    thing every later phase asserts against.
    """
    from cascade.ledger.store import load_records, read_manifest

    require_registry(live_settings)
    sealed = read_manifest(live_settings, role="admin")
    if sealed is None:
        pytest.skip("registry is not sealed; run `cascade ledger seal`")

    records = load_records(live_settings, role="admin")
    tampered = (
        records[0].model_copy(
            update={
                "label": records[0].label.model_copy(
                    update={"outcome": 1 - records[0].label.outcome}
                )
            }
        ),
        *records[1:],
    )
    assert compute_manifest(tampered) != sealed.manifest_sha256
    with pytest.raises(ManifestMismatch):
        verify_manifest(tampered, sealed.manifest_sha256)


def test_stored_scenarios_satisfy_the_horizon_check(live_settings: Settings) -> None:
    """The SQL CHECK and the Python rule must agree exactly.

    A row that passed the Python screen and failed the constraint would abort
    the load halfway through, leaving a registry the manifest does not
    describe.
    """
    require_registry(live_settings)
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT count(*) FROM scenarios " "WHERE resolve_ts <= cutoff_ts + interval '21 days'"
        )
        row: Any = cur.fetchone()
    assert row[0] == 0


def test_stored_base_rate_is_inside_the_configured_band(live_settings: Settings) -> None:
    """Base-rate control is enforced by construction, and this proves it stuck."""
    from cascade.ledger.store import load_records

    require_registry(live_settings)
    records = load_records(live_settings, role="eval")
    yes_rate = sum(record.label.outcome for record in records) / len(records)
    assert live_settings.ledger.yes_rate_min <= yes_rate <= live_settings.ledger.yes_rate_max
