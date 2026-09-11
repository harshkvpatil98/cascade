"""The fan-out runner (spec §2.2 PHASE 2, M6 acceptance).

Two properties carry the milestone. The wavefront must produce *exactly* the
runs a one-at-a-time driver would -- batching is a cost optimisation and must
not be a behaviour change -- and the plan must be a set difference against what
is already stored, which is invariant 8 for a 36,000-run phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cascade.aperture.policy import derive_policies
from cascade.aperture.projection import Observation
from cascade.config import Settings
from cascade.ensemble.runner import EnsembleRunner, RunTask, run_id_for
from cascade.sim.actions import ActionSpace
from cascade.sim.agent import Decision
from cascade.sim.kernel import Loom, RunResult, mechanics_from
from cascade.sim.policies import HeuristicPolicy
from tests.conftest import make_graph


@pytest.fixture
def settings() -> Settings:
    return Settings()


def build_loom(settings: Settings, scenario_id: str, decider: Any | None = None) -> Loom:
    graph = make_graph(n_actors=14, n_factors=8, scenario_id=scenario_id)
    mechanics = mechanics_from(graph)
    return Loom(
        settings=settings,
        graph=graph,
        policies=derive_policies(graph, settings.aperture, asymmetry=True),
        decider=decider
        or HeuristicPolicy(
            utility={actor: dict(mechanics[actor].utility) for actor in sorted(mechanics)}
        ),
    )


@dataclass
class CountingDecider:
    """A stand-in that records the waves it was asked to prepare."""

    inner: Any
    waves: list[int] = field(default_factory=list)

    def prepare(self, turns: Any) -> int:
        self.waves.append(len(turns))
        return len(turns)

    def decide(self, **kwargs: Any) -> Decision:
        return self.inner.decide(**kwargs)


def runner_for(
    settings: Settings,
    *,
    results: list[RunResult],
    decider: Any | None = None,
    completed: Any = None,
) -> EnsembleRunner:
    looms: dict[str, Loom] = {}

    def loom_for(scenario_id: str) -> Loom:
        if scenario_id not in looms:
            looms[scenario_id] = build_loom(settings, scenario_id, decider)
        return looms[scenario_id]

    return EnsembleRunner(
        settings=settings,
        loom_for=loom_for,
        on_complete=results.append,
        policy="heuristic",
        completed=completed,
    )


# ---------------------------------------------------------------------------
# Planning and resumption
# ---------------------------------------------------------------------------


def test_the_plan_is_the_cross_product_minus_what_is_stored(settings: Settings) -> None:
    """Invariant 8: a run row exists only for a run that finished."""
    stored = {("s1", "base"): {0, 2}}
    runner = runner_for(
        settings,
        results=[],
        completed=lambda scenario_id, config_id: stored.get((scenario_id, config_id), set()),
    )
    tasks, skipped = runner.plan(scenario_ids=["s1", "s2"], config_id="base", replicates=3)

    assert skipped == 2
    assert [(t.scenario_id, t.replicate) for t in tasks] == [
        ("s1", 1),
        ("s2", 0),
        ("s2", 1),
        ("s2", 2),
    ]


def test_a_fully_stored_scenario_plans_no_work(settings: Settings) -> None:
    runner = runner_for(settings, results=[], completed=lambda scenario_id, config_id: {0, 1, 2})
    tasks, skipped = runner.plan(scenario_ids=["s1"], config_id="base", replicates=3)
    assert tasks == []
    assert skipped == 3


def test_run_ids_are_derived_so_a_restart_reuses_them(settings: Settings) -> None:
    """A random id would let an interrupted phase log every run twice."""
    assert run_id_for("s1", "base", 7) == run_id_for("s1", "base", 7)
    assert run_id_for("s1", "base", 7) != run_id_for("s1", "base", 8)
    assert run_id_for("s1", "C04", 7) != run_id_for("s1", "base", 7)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_the_wavefront_reproduces_the_single_run_driver(settings: Settings) -> None:
    """Batching is a cost optimisation; it must not change a single event.

    The same replicate run through the compiled graph and through the wavefront
    has to produce the same event-log hash, or M8's byte-identical criterion
    would be a property of which driver happened to run.
    """
    direct = build_loom(settings, "s1")
    task = RunTask(scenario_id="s1", config_id="base", replicate=3)
    expected = direct.run(task.spec(policy="heuristic"))

    results: list[RunResult] = []
    runner = runner_for(settings, results=results)
    runner.execute([task], wave=1)

    assert len(results) == 1
    assert results[0].event_log_hash == expected.event_log_hash
    assert results[0].outcome_score == expected.outcome_score


def test_wave_size_does_not_change_the_runs(settings: Settings) -> None:
    """The dial is cost and memory only."""
    tasks = [RunTask(scenario_id="s1", config_id="base", replicate=i) for i in range(6)]

    small: list[RunResult] = []
    runner_for(settings, results=small).execute(tasks, wave=1)
    large: list[RunResult] = []
    runner_for(settings, results=large).execute(tasks, wave=6)

    assert {r.spec.replicate: r.event_log_hash for r in small} == {
        r.spec.replicate: r.event_log_hash for r in large
    }


def test_a_wave_spans_scenarios_and_prepares_once_per_step(settings: Settings) -> None:
    """One decider serving every scenario is what makes a step one submission."""
    inner = build_loom(settings, "s1").decider
    decider = CountingDecider(inner=inner)
    tasks = [
        RunTask(scenario_id=name, config_id="base", replicate=index)
        for name in ("s1", "s2")
        for index in range(2)
    ]
    results: list[RunResult] = []
    runner_for(settings, results=results, decider=decider).execute(tasks, wave=4)

    assert len(results) == 4
    # One prepare per step, each carrying every live run's turns.
    assert len(decider.waves) == settings.kernel.steps
    assert decider.waves[0] > 4


def test_the_report_measures_what_ran(settings: Settings) -> None:
    tasks = [RunTask(scenario_id="s1", config_id="base", replicate=i) for i in range(3)]
    results: list[RunResult] = []
    report = runner_for(settings, results=results).execute(tasks, wave=3)

    assert report.tasks == 3
    assert report.completed == 3
    assert report.decisions == sum(r.decisions for r in results)
    assert report.waves == 1
    assert report.steps == settings.kernel.steps
    assert report.decisions_per_run == pytest.approx(report.decisions / 3)


def test_a_decider_without_prepare_still_runs(settings: Settings) -> None:
    """The stand-in has nothing to batch; the wavefront is still correct."""
    tasks = [RunTask(scenario_id="s1", config_id="base", replicate=0)]
    results: list[RunResult] = []
    report = runner_for(settings, results=results).execute(tasks, wave=1)
    assert report.batches == 0
    assert report.completed == 1


def test_hit_rate_is_measured_over_decisions(settings: Settings) -> None:
    """The M6 criterion is a rate over calls, not over runs."""
    tasks = [RunTask(scenario_id="s1", config_id="base", replicate=0)]
    results: list[RunResult] = []
    report = runner_for(settings, results=results).execute(tasks, wave=1)
    assert report.cache_hit_rate == 0.0  # the stand-in never calls a model
    assert report.llm_calls == results[0].llm_calls


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_a_partial_run_cannot_be_collected(settings: Settings) -> None:
    """A truncated run scored as complete would enter the ensemble as a forecast."""
    loom = build_loom(settings, "s1")
    handle = loom.start(
        RunTask(scenario_id="s1", config_id="base", replicate=0).spec(policy="heuristic")
    )
    loom.observe_step(handle)
    loom.complete_step(handle)
    with pytest.raises(RuntimeError, match="has not terminated"):
        loom.result(handle)


def test_a_step_cannot_be_observed_twice(settings: Settings) -> None:
    loom = build_loom(settings, "s1")
    handle = loom.start(
        RunTask(scenario_id="s1", config_id="base", replicate=0).spec(policy="heuristic")
    )
    loom.observe_step(handle)
    with pytest.raises(RuntimeError, match="already waiting"):
        loom.observe_step(handle)


def test_a_step_cannot_be_completed_before_it_is_observed(settings: Settings) -> None:
    loom = build_loom(settings, "s1")
    handle = loom.start(
        RunTask(scenario_id="s1", config_id="base", replicate=0).spec(policy="heuristic")
    )
    with pytest.raises(RuntimeError, match="no observed step"):
        loom.complete_step(handle)


def test_pending_turns_name_their_scenario(settings: Settings) -> None:
    """A wave spans scenarios, and actor ids repeat across graphs."""
    loom = build_loom(settings, "s1")
    handle = loom.start(
        RunTask(scenario_id="s1", config_id="base", replicate=0).spec(policy="heuristic")
    )
    turns = loom.observe_step(handle)
    assert turns
    assert all(turn.scenario_id == "s1" for turn in turns)
    assert all(isinstance(turn.observation, Observation) for turn in turns)
    assert all(isinstance(turn.space, ActionSpace) for turn in turns)
