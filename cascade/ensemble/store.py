"""Persistence for collapsed forecasts (migration 010, spec §9.1, §3.3).

Reads and writes as ``cascade_sim``. That is the phase boundary of §2.2 made
enforceable: PHASE 3 turns runs into forecasts and must not be able to see an
outcome while doing it, so it runs under the role with no grant on
``scenario_labels``. PHASE 4 scores those forecasts and runs as ``eval``.

Scores come back ordered by replicate, never by insertion. §9.3's convergence
curve takes the first *n* replicates as its ensemble at each rung, and replicate
*k* is a fixed seeded world (§8.2) -- so the prefix is reproducible only if the
order is the replicate's, not the writer's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from cascade.config import Settings
from cascade.ensemble.schema import Forecast

__all__ = [
    "ForecastStats",
    "collapse_inputs",
    "forecast_stats",
    "load_forecasts",
    "scores_by_scenario",
    "write_forecast",
]

Role = Literal["admin", "sim", "eval"]


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


@dataclass(frozen=True)
class CollapseInput:
    """One (scenario, config)'s replicates, in replicate order."""

    scenario_id: str
    config_id: str
    scores: tuple[float, ...]
    mean_events: float
    mean_steps: float
    absorbed_runs: int


@dataclass(frozen=True)
class ForecastStats:
    """Measured figures over stored forecasts. Printed, never asserted against."""

    forecasts: int
    scenarios: int
    configs: tuple[str, ...]
    mean_p_hat: float
    mean_sigma: float
    multi_modal: int
    mean_ci_width: float
    mean_replicates: float

    @property
    def multi_modal_share(self) -> float:
        return self.multi_modal / self.forecasts if self.forecasts else 0.0


def collapse_inputs(
    settings: Settings, *, config_id: str, role: Role = "sim"
) -> tuple[CollapseInput, ...]:
    """Every scenario's replicate scores for one config, ready to collapse."""
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT scenario_id,
                   array_agg(outcome_score ORDER BY replicate) AS scores,
                   avg(decisions)  AS mean_events,
                   avg(steps_run)  AS mean_steps,
                   count(*) FILTER (WHERE termination = 'absorbed') AS absorbed
            FROM runs
            WHERE config_id = %s
            GROUP BY scenario_id
            ORDER BY scenario_id
            """,
            (config_id,),
        )
        rows = cur.fetchall()
    return tuple(
        CollapseInput(
            scenario_id=str(row[0]),
            config_id=config_id,
            scores=tuple(float(value) for value in row[1]),
            mean_events=float(row[2]),
            mean_steps=float(row[3]),
            absorbed_runs=int(row[4]),
        )
        for row in rows
    )


def scores_by_scenario(
    settings: Settings, *, config_id: str, role: Role = "sim"
) -> dict[str, tuple[float, ...]]:
    """Replicate-ordered scores keyed by scenario -- §9.3's convergence input."""
    return {
        item.scenario_id: item.scores
        for item in collapse_inputs(settings, config_id=config_id, role=role)
    }


def write_forecast(settings: Settings, forecast: Forecast, *, role: Role = "sim") -> None:
    """Upsert one collapsed forecast.

    An upsert rather than an insert because a forecast is derived: adding
    replicates and re-collapsing must replace the row, not accumulate a second
    one. That is not in tension with invariant 6 -- `events` stays append-only
    and this table is not the log.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO forecasts (
                scenario_id, config_id, p_hat, sigma, ci_lo, ci_hi, modality,
                dip_p, bimodality, n_replicates, mean_events, mean_steps, absorbed_runs
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scenario_id, config_id) DO UPDATE SET
                p_hat = EXCLUDED.p_hat,
                sigma = EXCLUDED.sigma,
                ci_lo = EXCLUDED.ci_lo,
                ci_hi = EXCLUDED.ci_hi,
                modality = EXCLUDED.modality,
                dip_p = EXCLUDED.dip_p,
                bimodality = EXCLUDED.bimodality,
                n_replicates = EXCLUDED.n_replicates,
                mean_events = EXCLUDED.mean_events,
                mean_steps = EXCLUDED.mean_steps,
                absorbed_runs = EXCLUDED.absorbed_runs,
                collapsed_at = now()
            """,
            (
                forecast.scenario_id,
                forecast.config_id,
                forecast.p_hat,
                forecast.sigma,
                forecast.ci_lo,
                forecast.ci_hi,
                forecast.modality,
                forecast.dip_p,
                forecast.bimodality,
                forecast.n_replicates,
                forecast.mean_events,
                forecast.mean_steps,
                forecast.absorbed_runs,
            ),
        )
        conn.commit()


def load_forecasts(
    settings: Settings, *, config_id: str | None = None, role: Role = "eval"
) -> tuple[Forecast, ...]:
    """Read stored forecasts back as boundary types."""
    clause, params = ("WHERE config_id = %s", (config_id,)) if config_id else ("", ())
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT scenario_id, config_id, p_hat, sigma, ci_lo, ci_hi, modality,
                   dip_p, bimodality, n_replicates, mean_events, mean_steps, absorbed_runs
            FROM forecasts {clause} ORDER BY scenario_id, config_id
            """,  # noqa: S608 -- `clause` is a literal chosen above, not input
            params,
        )
        rows = cur.fetchall()
    return tuple(
        Forecast(
            scenario_id=str(row[0]),
            config_id=str(row[1]),
            p_hat=float(row[2]),
            sigma=float(row[3]),
            ci_lo=float(row[4]),
            ci_hi=float(row[5]),
            modality="multi" if str(row[6]) == "multi" else "single",
            dip_p=None if row[7] is None else float(row[7]),
            bimodality=None if row[8] is None else float(row[8]),
            n_replicates=int(row[9]),
            mean_events=float(row[10]),
            mean_steps=float(row[11]),
            absorbed_runs=int(row[12]),
        )
        for row in rows
    )


def forecast_stats(
    settings: Settings, *, config_id: str | None = None, role: Role = "eval"
) -> ForecastStats:
    """Aggregate the stored forecasts for `cascade ensemble status`."""
    stored: Sequence[Forecast] = load_forecasts(settings, config_id=config_id, role=role)
    if not stored:
        return ForecastStats(
            forecasts=0,
            scenarios=0,
            configs=(),
            mean_p_hat=0.0,
            mean_sigma=0.0,
            multi_modal=0,
            mean_ci_width=0.0,
            mean_replicates=0.0,
        )
    count = len(stored)
    return ForecastStats(
        forecasts=count,
        scenarios=len({item.scenario_id for item in stored}),
        configs=tuple(sorted({item.config_id for item in stored})),
        mean_p_hat=sum(item.p_hat for item in stored) / count,
        mean_sigma=sum(item.sigma for item in stored) / count,
        multi_modal=sum(1 for item in stored if item.modality == "multi"),
        mean_ci_width=sum(item.ci_width for item in stored) / count,
        mean_replicates=sum(item.n_replicates for item in stored) / count,
    )
