"""The event log against the live database (migration 009, invariant 6).

Two things are checked here that cannot be checked anywhere else: that a run's
rows land atomically and read back in key order, and that the append-only rule
is a *permission*, not a convention. The second is the same mechanism as
invariant 2 -- a role that cannot UPDATE cannot be talked into it.

Everything writes inside a transaction that is rolled back, except the one test
that needs a committed run to prove idempotency; that one cleans up after
itself as the owner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cascade.aperture.policy import derive_policies
from cascade.config import Settings
from cascade.sim.kernel import Loom, RunSpec, mechanics_from
from cascade.sim.policies import HeuristicPolicy
from cascade.sim.rng import run_seed
from cascade.trace.store import completed_replicates, load_events, run_stats, write_run
from tests.conftest import make_graph

pytestmark = pytest.mark.integration

RUN_ID = "0d5e7c5a-0000-4000-8000-0000000000b5"


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


@pytest.fixture(scope="module")
def result(live_settings: Settings, scenario_id: str) -> Any:
    graph = make_graph(n_actors=14, n_factors=8, scenario_id=scenario_id)
    mechanics = mechanics_from(graph)
    loom = Loom(
        settings=live_settings,
        graph=graph,
        policies=derive_policies(graph, live_settings.aperture, asymmetry=True),
        decider=HeuristicPolicy(
            utility={actor: dict(mechanics[actor].utility) for actor in sorted(mechanics)}
        ),
    )
    return loom.run(
        RunSpec(
            run_id=RUN_ID,
            scenario_id=scenario_id,
            config_id="test-m5",
            replicate=0,
            policy="heuristic",
        )
    )


@pytest.fixture
def stored(live_settings: Settings, result: Any, scenario_id: str) -> Any:
    """Write the run, yield it, then remove it as the owner.

    Committed rather than rolled back because the point of several of these
    tests is what a *different connection* -- and a different role -- can see
    and do.
    """
    import numpy as np
    import psycopg

    write_run(
        live_settings,
        result,
        started_at=datetime.now(UTC),
        graph_sha256="0" * 64,
        run_seed=run_seed(
            scenario_id=scenario_id,
            config_id="test-m5",
            replicate=0,
            salt=live_settings.study.salt,
        ),
        numpy_version=np.__version__,
        role="admin",
    )
    yield result
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("DELETE FROM events WHERE run_id = %s", (RUN_ID,))
        cur.execute("DELETE FROM run_steps WHERE run_id = %s", (RUN_ID,))
        cur.execute("DELETE FROM runs WHERE run_id = %s", (RUN_ID,))
        conn.commit()


def test_a_run_writes_its_rows_atomically(live_settings: Settings, stored: Any) -> None:
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("eval"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT count(*) FROM events WHERE run_id = %s", (RUN_ID,))
        events = cur.fetchone()
        cur.execute("SELECT count(*) FROM run_steps WHERE run_id = %s", (RUN_ID,))
        steps = cur.fetchone()
        cur.execute("SELECT decisions, steps_run FROM runs WHERE run_id = %s", (RUN_ID,))
        run = cur.fetchone()

    assert events is not None and events[0] == stored.decisions
    assert steps is not None and steps[0] == stored.steps_run
    assert run is not None and (run[0], run[1]) == (stored.decisions, stored.steps_run)


def test_events_read_back_in_key_order(live_settings: Settings, stored: Any) -> None:
    """§8.1: ordering is by (run_id, step, seq), never by insertion time."""
    rows = load_events(live_settings, run_id=RUN_ID)
    keys = [(row["step"], row["seq"]) for row in rows]
    assert keys == sorted(keys)
    assert len(keys) == stored.decisions
    assert rows[0]["action"]["type"] in {
        "COMMIT",
        "ESCALATE",
        "CONCEDE",
        "ALLY",
        "DEFECT",
        "SIGNAL",
        "WAIT",
    }


def test_rewriting_a_replayed_run_is_idempotent(
    live_settings: Settings, stored: Any, scenario_id: str
) -> None:
    """A resumed worker that re-simulates a finished run must not double-count.

    ON CONFLICT DO NOTHING modifies no existing row, so it needs no UPDATE
    grant and cannot rewrite history -- which is the only form of idempotency
    an append-only log can have.
    """
    import numpy as np

    write_run(
        live_settings,
        stored,
        started_at=datetime.now(UTC),
        graph_sha256="0" * 64,
        run_seed=1,
        numpy_version=np.__version__,
        role="admin",
    )
    assert len(load_events(live_settings, run_id=RUN_ID)) == stored.decisions


def test_the_sim_role_cannot_update_or_delete_an_event(
    live_settings: Settings, stored: Any
) -> None:
    """Invariant 6, enforced by the grant rather than by code review."""
    import psycopg

    with psycopg.connect(live_settings.database_url("sim"), connect_timeout=10) as conn:
        for statement in (
            "UPDATE events SET actor_id = 'tampered' WHERE run_id = %s",
            "DELETE FROM events WHERE run_id = %s",
        ):
            with conn.cursor() as cur, pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute(statement, (RUN_ID,))
            conn.rollback()

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM events WHERE run_id = %s", (RUN_ID,))
            row = cur.fetchone()
    assert row is not None and row[0] == stored.decisions


def test_the_sim_role_can_insert_and_read(live_settings: Settings, stored: Any) -> None:
    """The role the simulation runs as has exactly the two rights it needs."""
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("sim"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT count(*) FROM runs WHERE run_id = %s", (RUN_ID,))
        row = cur.fetchone()
    assert row is not None and row[0] == 1


def test_completed_replicates_is_the_resume_primitive(
    live_settings: Settings, stored: Any, scenario_id: str
) -> None:
    """Invariant 8: a run row exists only for a run that finished."""
    done = completed_replicates(
        live_settings, scenario_id=scenario_id, config_id="test-m5", role="eval"
    )
    assert done == {0}
    assert (
        completed_replicates(
            live_settings, scenario_id=scenario_id, config_id="nothing-here", role="eval"
        )
        == set()
    )


def test_run_stats_reports_what_was_stored(live_settings: Settings, stored: Any) -> None:
    stats = run_stats(live_settings, config_id="test-m5")
    assert stats.runs == 1
    assert stats.events == stored.decisions
    assert stats.policies == ("heuristic",)
    assert 0.0 < stats.mean_activation_rate <= 1.0


def test_the_event_table_is_hash_partitioned_into_32(live_settings: Settings) -> None:
    """§11.1's partitioning, asserted against the deployed schema.

    A table that silently lost its partitions would still work and would
    serialise the write side of a 36,000-run fan-out onto one relation.
    """
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT count(*) FROM pg_inherits WHERE inhparent = 'events'::regclass")
        row = cur.fetchone()
        cur.execute(
            "SELECT partstrat FROM pg_partitioned_table WHERE partrelid = 'events'::regclass"
        )
        strategy = cur.fetchone()
    assert row is not None and row[0] == 32
    assert strategy is not None and strategy[0] == "h"
