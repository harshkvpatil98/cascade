"""Interrupting a fan-out and resuming it (M6 acceptance criterion 3).

"Kill the worker pool mid-phase and resume: no duplicated and no lost runs."
The full criterion is stated over 36,000 runs and needs M4's compiled graphs;
what is tested here is the mechanism it rests on, against the real database:
a run row exists only for a run that finished, so the work left is a set
difference and re-running a completed unit writes nothing new.

The interruption is modelled by executing part of the plan and then re-planning
-- which is exactly what a killed worker leaves behind, because a run that did
not reach its horizon never wrote a row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from cascade.aperture.policy import derive_policies
from cascade.config import Settings
from cascade.decompose.schema import graph_hash
from cascade.ensemble.runner import EnsembleRunner, run_id_for
from cascade.sim.kernel import Loom, mechanics_from
from cascade.sim.policies import HeuristicPolicy
from cascade.sim.rng import run_seed
from cascade.trace.store import completed_replicates, load_events, write_run
from tests.conftest import make_graph

pytestmark = pytest.mark.integration

CONFIG = "test-m6-resume"
REPLICATES = 4


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
def wiring(live_settings: Settings, scenario_id: str) -> Any:
    graph = make_graph(n_actors=14, n_factors=8, scenario_id=scenario_id)
    mechanics = mechanics_from(graph)
    started = datetime.now(UTC)

    def loom_for(_scenario_id: str) -> Loom:
        return Loom(
            settings=live_settings,
            graph=graph,
            policies=derive_policies(graph, live_settings.aperture, asymmetry=True),
            decider=HeuristicPolicy(
                utility={a: dict(mechanics[a].utility) for a in sorted(mechanics)}
            ),
        )

    def on_complete(result: Any) -> None:
        write_run(
            live_settings,
            result,
            started_at=started,
            graph_sha256=graph_hash(graph),
            run_seed=run_seed(
                scenario_id=result.spec.scenario_id,
                config_id=result.spec.config_id,
                replicate=result.spec.replicate,
                salt=live_settings.study.salt,
            ),
            numpy_version=np.__version__,
            role="admin",
        )

    def runner() -> EnsembleRunner:
        return EnsembleRunner(
            settings=live_settings,
            loom_for=loom_for,
            on_complete=on_complete,
            policy="heuristic",
            completed=lambda sid, cfg: completed_replicates(
                live_settings, scenario_id=sid, config_id=cfg, role="admin"
            ),
        )

    yield runner

    import psycopg

    run_ids = [run_id_for(scenario_id, CONFIG, index) for index in range(REPLICATES)]
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("DELETE FROM events WHERE run_id = ANY(%s::uuid[])", (run_ids,))
        cur.execute("DELETE FROM run_steps WHERE run_id = ANY(%s::uuid[])", (run_ids,))
        cur.execute("DELETE FROM runs WHERE run_id = ANY(%s::uuid[])", (run_ids,))
        conn.commit()


def test_an_interrupted_fanout_resumes_without_duplicating_or_losing_runs(
    live_settings: Settings, scenario_id: str, wiring: Any
) -> None:
    first = wiring()
    tasks, skipped = first.plan(scenario_ids=[scenario_id], config_id=CONFIG, replicates=REPLICATES)
    assert len(tasks) == REPLICATES and skipped == 0

    # The kill: only half the plan is executed. The runs that did not finish
    # wrote no row, which is what makes the resume a set difference.
    first.execute(tasks[:2], wave=2)
    assert completed_replicates(
        live_settings, scenario_id=scenario_id, config_id=CONFIG, role="admin"
    ) == {0, 1}

    resumed = wiring()
    remaining, already = resumed.plan(
        scenario_ids=[scenario_id], config_id=CONFIG, replicates=REPLICATES
    )
    assert already == 2
    assert [task.replicate for task in remaining] == [2, 3]

    report = resumed.execute(remaining, wave=2)
    assert report.completed == 2

    stored = completed_replicates(
        live_settings, scenario_id=scenario_id, config_id=CONFIG, role="admin"
    )
    assert stored == {0, 1, 2, 3}, "a run was lost across the interruption"


def test_re_running_a_completed_unit_writes_nothing_new(
    live_settings: Settings, scenario_id: str, wiring: Any
) -> None:
    """A worker that restarted mid-write must not double the event log.

    The run id is derived from (scenario, config, replicate), so a re-simulated
    run collides with itself and `ON CONFLICT DO NOTHING` keeps the first copy.
    Without that, the M6 event count would drift upward with every restart.
    """
    runner = wiring()
    tasks, _ = runner.plan(scenario_ids=[scenario_id], config_id=CONFIG, replicates=1)
    runner.execute(tasks, wave=1)

    run_id = run_id_for(scenario_id, CONFIG, 0)
    before = load_events(live_settings, run_id=run_id, role="admin")
    assert before

    # Force the same unit through again, as a restarted worker would.
    again = EnsembleRunner(
        settings=live_settings,
        loom_for=runner.loom_for,
        on_complete=runner.on_complete,
        policy="heuristic",
    )
    again.execute(tasks, wave=1)

    after = load_events(live_settings, run_id=run_id, role="admin")
    assert len(after) == len(before)
    assert [row["obs_hash"] for row in after] == [row["obs_hash"] for row in before]
