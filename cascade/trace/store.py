"""Persistence for runs, steps and the event log (migration 009, spec §11.1).

Writes go in as ``cascade_sim``, the role the simulation actually runs as. That
role holds SELECT and INSERT and nothing else, so invariant 6 -- the event log
is append-only -- is enforced by the database rather than by this module being
careful. Re-inserting a replayed run is idempotent through ``ON CONFLICT DO
NOTHING``, which modifies no existing row.

A run is written **once, at completion**, in one transaction: the run row, its
step records and its events land together or not at all. A partially written
run is the failure mode that would make M6's "no duplicated and no lost runs"
criterion unverifiable, because a resumed worker could not tell a finished run
from an interrupted one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from cascade.config import Settings
from cascade.sim.kernel import RunResult
from cascade.trace.events import DecisionEvent

__all__ = [
    "RunStats",
    "completed_replicates",
    "load_events",
    "run_stats",
    "write_run",
]

Role = Literal["admin", "sim", "eval"]


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


@dataclass(frozen=True)
class RunStats:
    """Measured figures over stored runs. Printed by ``cascade simulate status``."""

    runs: int
    scenarios: int
    configs: tuple[str, ...]
    policies: tuple[str, ...]
    events: int
    mean_events_per_run: float
    mean_activation_rate: float
    mean_steps: float
    absorbed_runs: int
    cache_hits: int
    llm_calls: int
    distinct_event_hashes: int

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.llm_calls if self.llm_calls else 0.0


def write_run(
    settings: Settings,
    result: RunResult,
    *,
    started_at: datetime,
    graph_sha256: str,
    run_seed: int,
    numpy_version: str,
    cost_usd: Decimal = Decimal("0"),
    role: Role = "sim",
) -> None:
    """Persist one completed run atomically."""
    with _connect(settings, role) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, scenario_id, config_id, replicate, policy, agent_model,
                    prompt_rev, graph_sha256, run_seed, numpy_version, steps_run,
                    termination, outcome_score, activation_rate, decisions, llm_calls,
                    cache_hits, tokens_in, tokens_out, cost_usd, event_log_hash, started_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    result.spec.run_id,
                    result.spec.scenario_id,
                    result.spec.config_id,
                    result.spec.replicate,
                    result.spec.policy,
                    settings.models.agent,
                    settings.llm.prompt_rev,
                    graph_sha256,
                    run_seed,
                    numpy_version,
                    result.steps_run,
                    result.termination,
                    result.outcome_score,
                    result.activation_rate,
                    result.decisions,
                    result.llm_calls,
                    result.cache_hits,
                    result.tokens_in,
                    result.tokens_out,
                    cost_usd,
                    result.event_log_hash,
                    started_at,
                ),
            )
            cur.executemany(
                """
                INSERT INTO run_steps (
                    run_id, step, exogenous_delta, arrivals, contest_delta,
                    active, eligible, rng_counter, state_hash, absorbed
                ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, step) DO NOTHING
                """,
                [
                    (
                        record.run_id,
                        record.step,
                        json.dumps(record.exogenous_delta, sort_keys=True),
                        json.dumps(record.arrivals, sort_keys=True),
                        json.dumps(record.contest_delta, sort_keys=True),
                        list(record.active),
                        record.eligible,
                        record.rng_counter,
                        record.state_hash,
                        list(record.absorbed),
                    )
                    for record in sorted(result.step_records, key=lambda r: r.step)
                ],
            )
            cur.executemany(
                """
                INSERT INTO events (
                    run_id, step, seq, actor_id, obs_hash, action, caused_by,
                    factor_delta, cache_hit, tokens_in, tokens_out, latency_ms, coercion
                ) VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_id, step, seq) DO NOTHING
                """,
                [_event_row(event) for event in sorted(result.events, key=_event_key)],
            )
        conn.commit()


def _event_key(event: DecisionEvent) -> tuple[int, int]:
    return (event.step, event.seq)


def _event_row(event: DecisionEvent) -> tuple[Any, ...]:
    return (
        event.run_id,
        event.step,
        event.seq,
        event.actor_id,
        event.obs_hash,
        json.dumps(event.action, sort_keys=True),
        json.dumps([ref.as_json() for ref in event.caused_by], sort_keys=True),
        json.dumps(event.factor_delta, sort_keys=True),
        event.cache_hit,
        event.tokens_in,
        event.tokens_out,
        event.latency_ms,
        event.coercion,
    )


def completed_replicates(
    settings: Settings, *, scenario_id: str, config_id: str, role: Role = "sim"
) -> set[int]:
    """Replicate numbers already stored for one (scenario, config).

    The resume primitive (invariant 8): a run row exists only for a run that
    finished, so "what is left to do" is set difference and never a judgement
    call about a half-written run.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT replicate FROM runs WHERE scenario_id = %s AND config_id = %s",
            (scenario_id, config_id),
        )
        return {int(row[0]) for row in cur.fetchall()}


def run_stats(settings: Settings, *, config_id: str | None = None, role: Role = "eval") -> RunStats:
    """Aggregate the stored runs. Every figure measured, none configured."""
    clause, params = ("WHERE config_id = %s", (config_id,)) if config_id else ("", ())
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*), count(DISTINCT scenario_id),
                   coalesce(sum(decisions), 0),
                   coalesce(avg(decisions), 0), coalesce(avg(activation_rate), 0),
                   coalesce(avg(steps_run), 0),
                   count(*) FILTER (WHERE termination = 'absorbed'),
                   coalesce(sum(cache_hits), 0), coalesce(sum(llm_calls), 0),
                   count(DISTINCT event_log_hash)
            FROM runs {clause}
            """,  # noqa: S608 -- `clause` is a literal chosen above, not input
            params,
        )
        row = cur.fetchone()
        cur.execute(f"SELECT DISTINCT config_id FROM runs {clause}", params)  # noqa: S608
        configs = tuple(sorted(str(item[0]) for item in cur.fetchall()))
        cur.execute(f"SELECT DISTINCT policy FROM runs {clause}", params)  # noqa: S608
        policies = tuple(sorted(str(item[0]) for item in cur.fetchall()))
    if row is None:  # pragma: no cover - aggregate always returns a row
        raise RuntimeError("aggregate query returned no row")
    return RunStats(
        runs=int(row[0]),
        scenarios=int(row[1]),
        configs=configs,
        policies=policies,
        events=int(row[2]),
        mean_events_per_run=float(row[3]),
        mean_activation_rate=float(row[4]),
        mean_steps=float(row[5]),
        absorbed_runs=int(row[6]),
        cache_hits=int(row[7]),
        llm_calls=int(row[8]),
        distinct_event_hashes=int(row[9]),
    )


def load_events(
    settings: Settings, *, run_id: str, role: Role = "eval"
) -> tuple[dict[str, Any], ...]:
    """Read one run's events back in ``(step, seq)`` order.

    Ordered by the key rather than by insertion (spec §8.1): the log is a
    sequence, and a reader that depended on physical order would disagree with
    itself across a re-cluster or a partition-wise scan.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT step, seq, actor_id, obs_hash, action, caused_by, factor_delta,
                   cache_hit, tokens_in, tokens_out, latency_ms, coercion
            FROM events WHERE run_id = %s ORDER BY step, seq
            """,
            (run_id,),
        )
        columns = [
            "step",
            "seq",
            "actor_id",
            "obs_hash",
            "action",
            "caused_by",
            "factor_delta",
            "cache_hit",
            "tokens_in",
            "tokens_out",
            "latency_ms",
            "coercion",
        ]
        return tuple(dict(zip(columns, row, strict=True)) for row in cur.fetchall())
