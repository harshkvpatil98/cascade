"""The step loop end to end (spec §7.2, M5 acceptance criterion 3).

These run the real kernel -- real scheduler, real Aperture, real arbiter, real
seeded draw plan -- with the model replaced by a deterministic stand-in. That
substitution is the only one: everything the criterion is about (24 steps, the
stages in order, state that round-trips, a world that replays) is exercised as
built.

What they cannot do is measure the activation rate, which is a property of real
decompositions and real agents. That measurement is `cascade simulate
activation`, and it needs M4's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from cascade.aperture.policy import derive_policies
from cascade.aperture.projection import Observation
from cascade.config import Settings
from cascade.sim.actions import ActionSpace, Escalate, Signal, Wait
from cascade.sim.agent import Decision
from cascade.sim.kernel import Loom, RunSpec, mechanics_from, propagation_from
from cascade.sim.policies import HeuristicPolicy
from cascade.sim.rng import draws_per_step
from cascade.sim.state import WorldState
from tests.conftest import make_actor, make_factor, make_graph


@pytest.fixture
def settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class AlwaysWait:
    """Decides nothing. Isolates the world's own motion from the agents'."""

    def decide(
        self,
        *,
        scenario_id: str,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision:
        return Decision(action=Wait())


@dataclass(frozen=True)
class AlwaysIllegal:
    """Always reaches for a lever it does not have (ADR-0018)."""

    def decide(
        self,
        *,
        scenario_id: str,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision:
        return Decision(
            action=Escalate(target_factor="not_a_factor_here", magnitude=0.9),
            coercion=None,
        )


def build(
    settings: Settings, *, decider: object, asymmetry: bool = True, graph: object | None = None
) -> Loom:
    model = graph if graph is not None else make_graph(n_actors=14, n_factors=8)
    policies = derive_policies(model, settings.aperture, asymmetry=asymmetry)
    return Loom(settings=settings, graph=model, policies=policies, decider=decider)  # type: ignore[arg-type]


def heuristic_for(graph: object) -> HeuristicPolicy:
    mechanics = mechanics_from(graph)
    return HeuristicPolicy(
        utility={actor_id: dict(mechanics[actor_id].utility) for actor_id in sorted(mechanics)}
    )


def spec(replicate: int = 0, scenario_id: str = "scenario-1") -> RunSpec:
    return RunSpec(
        run_id=f"00000000-0000-0000-0000-{replicate:012d}",
        scenario_id=scenario_id,
        config_id="base",
        replicate=replicate,
        policy="heuristic",
    )


# ---------------------------------------------------------------------------
# Criterion 3: one scenario runs 24 steps end to end
# ---------------------------------------------------------------------------


def test_a_run_reaches_the_horizon_with_a_record_of_every_step(settings: Settings) -> None:
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())

    assert result.steps_run == settings.kernel.steps == 24
    assert result.termination == "horizon"
    assert [record.step for record in result.step_records] == list(range(24))
    assert result.world.step == 24
    assert 0.0 < result.outcome_score < 1.0


def test_the_langgraph_recursion_limit_does_not_truncate_a_run(settings: Settings) -> None:
    """Six nodes a step against a default limit of 25 would stop at step 4.

    A truncated run still returns a world and an outcome score, so nothing
    downstream would notice; only the step count says it happened.
    """
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    assert len(result.step_records) == settings.kernel.steps


def test_the_final_state_round_trips_exactly(settings: Settings) -> None:
    """Criterion 3, on real run output rather than a hand-built state."""
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    restored = WorldState.model_validate_json(result.world.model_dump_json())
    assert restored == result.world
    assert restored.canonical() == result.world.canonical()


def test_events_are_keyed_and_ordered_within_a_step(settings: Settings) -> None:
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())

    keys = [(event.step, event.seq) for event in result.events]
    assert len(set(keys)) == len(keys)
    for record in result.step_records:
        in_step = sorted(
            (event.seq, event.actor_id) for event in result.events if event.step == record.step
        )
        assert [actor for _, actor in in_step] == sorted(record.active)
        assert [seq for seq, _ in in_step] == list(range(len(record.active)))


def test_every_decision_carries_an_observation_hash(settings: Settings) -> None:
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    assert all(len(event.obs_hash) == 16 for event in result.events)
    assert len({event.obs_hash for event in result.events}) > 1


# ---------------------------------------------------------------------------
# Determinism (invariant 4, §8.1)
# ---------------------------------------------------------------------------


def test_the_same_seed_produces_a_byte_identical_event_log(settings: Settings) -> None:
    graph = make_graph(n_actors=14, n_factors=8)
    first = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    second = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())

    assert first.event_log_hash == second.event_log_hash
    assert [record.state_hash for record in first.step_records] == [
        record.state_hash for record in second.step_records
    ]


def test_a_different_replicate_produces_a_different_world(settings: Settings) -> None:
    """Otherwise 200 replicates would be one replicate counted 200 times."""
    graph = make_graph(n_actors=14, n_factors=8)
    loom = build(settings, decider=heuristic_for(graph), graph=graph)
    assert loom.run(spec(0)).event_log_hash != loom.run(spec(1)).event_log_hash


def test_the_stream_position_is_a_function_of_the_step_alone(settings: Settings) -> None:
    """Invariant 4, checked over a whole run rather than a single call."""
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    per_step = draws_per_step(n_actors=14, n_factors=8)
    for record in result.step_records:
        assert record.rng_counter == (record.step + 1) * per_step


def test_turning_asymmetry_off_changes_beliefs_and_not_the_world(
    settings: Settings,
) -> None:
    """§6.4: "Nothing else changes -- same graph, same agents, same seeds."

    Run with agents that do nothing, so the only thing the switch can touch is
    the projection. The factor trajectories must be identical, which they are
    only because the noise draws are made whether or not a channel uses them
    (ADR-0015). If the transparent policy skipped its draws, every exogenous
    shock after the first would shift.
    """
    graph = make_graph(n_actors=14, n_factors=8)
    opaque = build(settings, decider=AlwaysWait(), asymmetry=True, graph=graph).run(spec())
    clear = build(settings, decider=AlwaysWait(), asymmetry=False, graph=graph).run(spec())

    assert [record.exogenous_delta for record in opaque.step_records] == [
        record.exogenous_delta for record in clear.step_records
    ]
    assert opaque.world.factors == clear.world.factors
    assert opaque.outcome_score == clear.outcome_score
    # The observations differ, which is the entire point of the switch.
    assert {event.obs_hash for event in opaque.events} != {event.obs_hash for event in clear.events}


# ---------------------------------------------------------------------------
# Termination (§7.6)
# ---------------------------------------------------------------------------


def pinned_graph() -> object:
    """A graph whose outcome factor opens at a bound and reverts to it."""
    from cascade.decompose.schema import Edge, OutcomeRule, OutcomeTerm, UtilityTerm

    factors = (
        make_factor(0, state=1.0, volatility=0.0, inertia=1.0),
        make_factor(1, state=0.5, volatility=0.0, inertia=0.5),
        make_factor(2, state=0.5, volatility=0.0, inertia=0.5),
        make_factor(3, state=0.5, volatility=0.0, inertia=0.5),
    )
    actors = tuple(
        make_actor(
            index,
            utility_terms=(UtilityTerm(factor_id="factor_1", weight=0.5),),
            initial_beliefs={},
        )
        for index in range(8)
    )
    edges = tuple(
        [
            Edge(src=f"actor_{index}", dst="factor_1", sign=1, weight=0.3, lag=1)
            for index in range(8)
        ]
        + [
            Edge(src="factor_1", dst="factor_0", sign=1, weight=0.2, lag=1),
            Edge(src="factor_2", dst="factor_0", sign=1, weight=0.2, lag=1),
            Edge(src="factor_3", dst="factor_0", sign=1, weight=0.2, lag=1),
        ]
    )
    return make_graph(
        n_actors=8,
        n_factors=4,
        actors=actors,
        factors=factors,
        edges=edges,
        outcome_rule=OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_0", weight=0.8),
                OutcomeTerm(factor_id="factor_1", weight=-0.4),
            ),
            threshold=0.5,
            steepness=6.0,
        ),
    )


def test_a_pinned_outcome_factor_ends_the_run_early(settings: Settings) -> None:
    """§7.6: "a factor pinned at a bound for 3 consecutive steps"."""
    graph = pinned_graph()
    result = build(settings, decider=AlwaysWait(), graph=graph).run(spec())

    assert result.termination == "absorbed"
    assert result.steps_run == 3
    assert result.absorbed == ("factor_0",)


def test_the_activation_rate_is_measured_over_the_steps_that_ran(
    settings: Settings,
) -> None:
    """A run that stopped at step 3 has 3 steps of evidence, not 24."""
    graph = pinned_graph()
    result = build(settings, decider=AlwaysWait(), graph=graph).run(spec())
    expected = sum(record.activation_rate for record in result.step_records) / 3
    assert result.activation_rate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Provenance and admissibility
# ---------------------------------------------------------------------------


def test_an_inadmissible_action_is_recorded_as_a_coercion(settings: Settings) -> None:
    """ADR-0018: the run continues, and the reach is a column rather than a guess."""
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=AlwaysIllegal(), graph=graph).run(spec())

    assert result.events
    assert all(event.action["type"] == "WAIT" for event in result.events)
    assert all(event.coercion == "no_lever:not_a_factor_here" for event in result.events)


def test_decisions_acquire_antecedents_once_the_world_has_moved(
    settings: Settings,
) -> None:
    """§11.1: `caused_by` is what turns the log from a transcript into a graph."""
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())

    linked = [event for event in result.events if event.caused_by]
    assert linked, "no decision was ever attributed to an earlier one"
    for event in linked:
        for ref in event.caused_by:
            assert ref.step < event.step
            assert ref.run_id == event.run_id


def test_factor_delta_records_what_the_decision_actually_moved(
    settings: Settings,
) -> None:
    graph = make_graph(n_actors=14, n_factors=8)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    moved = [event for event in result.events if event.factor_delta]
    assert moved
    for event in moved:
        assert set(event.factor_delta) <= {factor.id for factor in graph.factors}
        assert all(
            abs(value) <= settings.kernel.max_step_delta for value in event.factor_delta.values()
        )


def test_propagation_carries_movement_between_factors(settings: Settings) -> None:
    """Without it the compiled edges are decoration and the factors are 8 walks."""
    graph = make_graph(n_actors=14, n_factors=8)
    assert propagation_from(graph)
    result = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())
    assert any(record.arrivals for record in result.step_records)


@dataclass(frozen=True)
class AlwaysSignal:
    """Always makes a claim to one fixed counterparty."""

    target: str

    def decide(
        self,
        *,
        scenario_id: str,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision:
        if self.target in space.counterparties and space.levers:
            return Decision(
                action=Signal(
                    target_actor=self.target,
                    claimed_factor=space.levers[0],
                    claimed_value=0.99,
                    truthful=False,
                )
            )
        return Decision(action=Wait())


def test_a_claim_waits_for_a_dormant_target_instead_of_being_dropped(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.4 alters "the target's observation" -- its next one, not nobody's.

    Dropping a claim addressed to an actor the scheduler left dormant would
    make a signal's effect depend on who happened to be salient, which is not
    something the sender can see or reason about.
    """
    graph = make_graph(n_actors=14, n_factors=8)
    observed: list[tuple[int, set[str], set[str]]] = []
    original = Loom.stage_observe

    def spy(self: Loom, ctx: Any) -> None:
        observed.append((ctx.world.step, set(ctx.pending_signals), set(ctx.active)))
        original(self, ctx)

    monkeypatch.setattr(Loom, "stage_observe", spy)
    build(settings, decider=AlwaysSignal(target="actor_9"), graph=graph).run(spec())

    carried = [step for step, pending, active in observed if pending and not (pending & active)]
    assert carried, "no claim ever had to wait for its target"
    # A claim that waited is still there on the following step.
    steps = {step: pending for step, pending, _ in observed}
    for step in carried:
        if step + 1 in steps:
            assert steps[step + 1], "a pending claim was dropped rather than delivered"


def test_both_drivers_produce_the_same_run(settings: Settings) -> None:
    """The compiled graph and the stepwise driver must agree, byte for byte.

    M6 batches a step's decisions across a wave, which needs a driver that can
    stop between OBSERVE and DECIDE. Two drivers over one `STAGE_SEQUENCE` is
    safe only while they stay identical, and "identical" here means the event
    log hashes to the same value -- not that both finish.
    """
    graph = make_graph(n_actors=14, n_factors=8)
    through_graph = build(settings, decider=heuristic_for(graph), graph=graph).run(spec())

    loom = build(settings, decider=heuristic_for(graph), graph=graph)
    handle = loom.start(spec())
    turns = 0
    while not handle.done:
        turns += len(loom.observe_step(handle))
        loom.complete_step(handle)
    stepwise = loom.result(handle)

    assert stepwise.event_log_hash == through_graph.event_log_hash
    assert stepwise.outcome_score == through_graph.outcome_score
    assert stepwise.steps_run == through_graph.steps_run
    assert turns == through_graph.decisions
