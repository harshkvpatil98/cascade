"""Forecasts against the live database (migration 010, spec §9.1, PHASE 3).

The load-bearing check here is not the upsert -- it is the role. §2.2 puts
aggregation in PHASE 3 and scoring in PHASE 4, and the reason that boundary is
real rather than stylistic is that collapsing must not be able to see an
outcome. Running it as `cascade_sim` makes that a permission, and this asserts
the permission rather than the intention.
"""

from __future__ import annotations

from typing import Any

import pytest

from cascade.config import Settings
from cascade.ensemble.schema import Forecast
from cascade.ensemble.store import forecast_stats, load_forecasts, write_forecast

pytestmark = pytest.mark.integration

CONFIG = "test-m6"


@pytest.fixture(scope="module")
def scenario_id(live_settings: Settings) -> str:
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT scenario_id FROM scenarios ORDER BY scenario_id LIMIT 1")
        row = cur.fetchone()
    if row is None:
        pytest.skip("scenario registry is empty; run `cascade ledger build`")
    return str(row[0])


@pytest.fixture
def stored(live_settings: Settings, scenario_id: str) -> Any:
    forecast = Forecast(
        scenario_id=scenario_id,
        config_id=CONFIG,
        p_hat=0.42,
        sigma=0.31,
        ci_lo=0.35,
        ci_hi=0.49,
        modality="multi",
        dip_p=0.004,
        bimodality=0.71,
        n_replicates=200,
        mean_events=118.0,
        mean_steps=24.0,
        absorbed_runs=3,
    )
    write_forecast(live_settings, forecast, role="admin")
    yield forecast
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("DELETE FROM forecasts WHERE config_id = %s", (CONFIG,))
        conn.commit()


def test_a_forecast_round_trips_through_the_database(
    live_settings: Settings, stored: Forecast
) -> None:
    loaded = load_forecasts(live_settings, config_id=CONFIG)
    assert len(loaded) == 1
    assert loaded[0] == stored


def test_recollapsing_replaces_rather_than_duplicates(
    live_settings: Settings, stored: Forecast
) -> None:
    """A forecast is derived data: more replicates must replace it, not accumulate."""
    revised = stored.model_copy(update={"p_hat": 0.55, "n_replicates": 400})
    write_forecast(live_settings, revised, role="admin")

    loaded = load_forecasts(live_settings, config_id=CONFIG)
    assert len(loaded) == 1
    assert loaded[0].p_hat == pytest.approx(0.55)
    assert loaded[0].n_replicates == 400


def test_the_simulation_role_can_collapse_but_never_sees_an_outcome(
    live_settings: Settings, stored: Forecast
) -> None:
    """PHASE 3 runs under the role with no grant on `scenario_labels`.

    That is invariant 2 expressed as a phase boundary: the aggregation step
    cannot peek at the answer it is about to be scored against, and the reason
    is a permission rather than a convention.
    """
    import psycopg

    with psycopg.connect(live_settings.database_url("sim"), connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM forecasts WHERE config_id = %s", (CONFIG,))
            row = cur.fetchone()
            assert row is not None and row[0] == 1
        with conn.cursor() as cur, pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("SELECT outcome FROM scenario_labels LIMIT 1")
        conn.rollback()


def test_the_simulation_role_can_write_a_forecast(
    live_settings: Settings, scenario_id: str, stored: Forecast
) -> None:
    """The collapse writes as `sim`, so the grant has to allow the upsert."""
    write_forecast(live_settings, stored.model_copy(update={"p_hat": 0.6}), role="sim")
    assert load_forecasts(live_settings, config_id=CONFIG)[0].p_hat == pytest.approx(0.6)


def test_stats_report_what_is_stored(live_settings: Settings, stored: Forecast) -> None:
    stats = forecast_stats(live_settings, config_id=CONFIG)
    assert stats.forecasts == 1
    assert stats.configs == (CONFIG,)
    assert stats.multi_modal == 1
    assert stats.multi_modal_share == 1.0
    assert stats.mean_ci_width == pytest.approx(stored.ci_width)


def test_the_prompt_revision_audit_table_carries_the_r2_change(
    live_settings: Settings,
) -> None:
    """§1.3's third integrity mechanism, asserted against the deployed schema.

    A revision made before the first backtest has no before/after Brier, and
    records that as NULL rather than omitting the row.
    """
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("eval"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT subsystem, summary, brier_before, brier_after "
            "FROM prompt_revisions WHERE prompt_rev = 'r2'"
        )
        rows = cur.fetchall()
    assert rows, "the r2 compiler revision is not recorded"
    subsystem, summary, before, after = rows[0]
    assert subsystem == "compiler"
    assert "volatility" in summary
    assert before is None and after is None
