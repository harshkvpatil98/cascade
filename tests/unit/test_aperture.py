"""Aperture: derived visibility, projection and private memory (spec §6).

The claim these defend is §6.1's: asymmetry is a projection function, not a
persona. So the tests are about topology producing incompatible beliefs, and
about the ablation switch removing exactly that and nothing else.
"""

from __future__ import annotations

import pytest

from cascade.aperture.memory import AgentMemory
from cascade.aperture.policy import (
    actor_distances,
    counterparties,
    derive_policies,
    factor_hops,
    levers_by_actor,
)
from cascade.aperture.projection import (
    OBSERVATION_DECIMALS,
    Observation,
    ObservedAction,
    SignalClaim,
    observation_hash,
    project,
    quantize,
    visible_actions,
)
from cascade.config import Settings
from cascade.sim.actions import Commit, Escalate
from cascade.sim.state import WorldState, relation_key
from tests.conftest import make_actor, make_factor, make_graph


@pytest.fixture
def settings() -> Settings:
    return Settings()


def chain_graph() -> object:
    """A graph whose topology makes hop distance unambiguous.

    Every actor acts on ``factor_0`` except the last three, and the factors run
    in a chain ``factor_0 -> factor_1 -> factor_2 -> factor_3``. Nothing is
    public, so the only channel an actor has is the one its edges imply.
    """
    from cascade.decompose.schema import Edge, OutcomeRule, OutcomeTerm, UtilityTerm

    factors = tuple(make_factor(index, observable_by_default=False) for index in range(4))
    actors = tuple(
        make_actor(
            index,
            factor_id="factor_0",
            utility_terms=(UtilityTerm(factor_id="factor_0", weight=0.5),),
            initial_beliefs={},
        )
        for index in range(8)
    )
    edges = [
        Edge(src=f"actor_{index}", dst="factor_0", sign=1, weight=0.4, lag=0) for index in range(5)
    ]
    edges += [
        Edge(src=f"actor_{index}", dst="factor_3", sign=1, weight=0.4, lag=0)
        for index in range(5, 8)
    ]
    edges += [
        Edge(src="factor_0", dst="factor_1", sign=1, weight=0.5, lag=1),
        Edge(src="factor_1", dst="factor_2", sign=1, weight=0.5, lag=1),
        Edge(src="factor_2", dst="factor_3", sign=-1, weight=0.5, lag=1),
    ]
    return make_graph(
        n_actors=8,
        n_factors=4,
        actors=actors,
        factors=factors,
        edges=tuple(edges),
        outcome_rule=OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_3", weight=0.8),
                OutcomeTerm(factor_id="factor_0", weight=-0.4),
            ),
            threshold=0.5,
            steepness=6.0,
        ),
    )


# ---------------------------------------------------------------------------
# Derivation (§6.2)
# ---------------------------------------------------------------------------


def test_hop_distance_counts_intermediates_not_edges() -> None:
    """A direct outbound edge is zero hops -- "you see clearly what you act on"."""
    hops = factor_hops(chain_graph())
    assert hops["actor_0"]["factor_0"] == 0
    assert hops["actor_0"]["factor_1"] == 1
    assert hops["actor_0"]["factor_2"] == 2
    assert hops["actor_0"]["factor_3"] == 3


def test_the_derivation_applies_the_six_two_rules(settings: Settings) -> None:
    """§6.2's table, read off a topology built to exercise each row."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    policy = policies["actor_0"]

    direct = policy.channel("factor_0")
    assert direct is not None and direct.visible
    assert (direct.noise_sigma, direct.lag, direct.quantized) == (0.0, 0, False)

    one_hop = policy.channel("factor_1")
    assert one_hop is not None and one_hop.visible
    assert one_hop.noise_sigma == settings.aperture.hop1.noise_sigma
    assert one_hop.lag == settings.aperture.hop1.lag
    assert not one_hop.quantized

    two_hops = policy.channel("factor_2")
    assert two_hops is not None and two_hops.visible
    assert two_hops.noise_sigma == settings.aperture.hop2.noise_sigma
    assert two_hops.quantized

    three_hops = policy.channel("factor_3")
    assert three_hops is not None and not three_hops.visible


def test_a_public_factor_is_visible_to_everyone(settings: Settings) -> None:
    """§6.2: observable_by_default reaches every actor at sigma 0.03, lag 0."""
    from cascade.decompose.schema import Factor

    graph = chain_graph()
    public = tuple(
        Factor(**{**factor.model_dump(), "observable_by_default": factor.id == "factor_3"})
        for factor in graph.factors  # type: ignore[attr-defined]
    )
    graph = make_graph(
        n_actors=8,
        n_factors=4,
        actors=graph.actors,  # type: ignore[attr-defined]
        factors=public,
        edges=graph.edges,  # type: ignore[attr-defined]
        outcome_rule=graph.outcome_rule,  # type: ignore[attr-defined]
    )
    policies = derive_policies(graph, settings.aperture, asymmetry=True)
    channel = policies["actor_0"].channel("factor_3")
    assert channel is not None and channel.visible
    assert channel.noise_sigma == settings.aperture.public.noise_sigma
    assert channel.lag == 0


def test_the_better_of_two_channels_wins(settings: Settings) -> None:
    """An actor that acts on a published factor still sees it exactly.

    The alternative -- letting the public channel's noise degrade a direct
    observation -- would make an actor *less* informed about something it
    controls because the value is also published.
    """
    graph = make_graph(n_actors=8, n_factors=4)  # factor_0 is public in the fixture
    policies = derive_policies(graph, settings.aperture, asymmetry=True)
    channel = policies["actor_0"].channel("factor_0")
    assert channel is not None
    assert (channel.noise_sigma, channel.lag) == (0.0, 0)


def test_action_visibility_is_per_target(settings: Settings) -> None:
    """ADR-0016: §6.2's own rule assigns different visibility per target.

    Actors sharing a factor see each other in full; actors one further step
    apart see the action type only.
    """
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    distances = actor_distances(chain_graph())

    policy = policies["actor_0"]
    assert distances["actor_0"]["actor_1"] == 1
    assert policy.action_view("actor_1") == "full"
    assert policy.action_view("actor_5") == "none"
    assert policy.action_visibility == "none"


def test_the_ablation_switch_makes_every_policy_transparent(settings: Settings) -> None:
    """§6.4: all factors visible, zero noise, zero lag, all actions visible."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=False)
    for actor_id in sorted(policies):
        policy = policies[actor_id]
        assert policy.is_transparent()
        assert len(policy.visible_factors()) == 4
        assert policy.action_view("actor_7") == "full"


def test_levers_are_not_the_same_thing_as_visibility() -> None:
    """What an actor can touch and what it can see are different questions."""
    graph = chain_graph()
    levers = levers_by_actor(graph)
    assert levers["actor_0"] == ("factor_0",)
    assert levers["actor_7"] == ("factor_3",)


def test_counterparties_come_from_the_action_channels(settings: Settings) -> None:
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    parties = counterparties(policies)
    assert "actor_1" in parties["actor_0"]
    assert "actor_5" not in parties["actor_0"]


# ---------------------------------------------------------------------------
# Projection (§6.2, §7.2 stage 4)
# ---------------------------------------------------------------------------


def world_at(step: int) -> WorldState:
    history = {
        "factor_0": (0.20, 0.40, 0.60, 0.80),
        "factor_1": (0.10, 0.30, 0.50, 0.70),
        "factor_2": (0.90, 0.70, 0.50, 0.30),
        "factor_3": (0.50, 0.50, 0.50, 0.50),
    }
    return WorldState(
        step=step,
        factors={key: values[step] for key, values in sorted(history.items())},
        factor_history=history,
        resources={"actor_0": {"capital": 0.6}},
        relations={relation_key("actor_0", "actor_1"): 0.5},
    )


def test_lag_reads_the_value_from_that_many_steps_ago(settings: Settings) -> None:
    """A one-hop channel at step 3 sees step 2's value, exactly."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    observation = project(
        world_at(3),
        policies["actor_0"],
        noise={"factor_0": 0.0, "factor_1": 0.0, "factor_2": 0.0, "factor_3": 0.0},
    )
    assert observation.factors["factor_0"] == 0.80  # direct, no lag
    assert observation.factors["factor_1"] == 0.50  # lag 1
    assert "factor_3" in observation.unobserved


def test_noise_scales_the_draw_and_is_not_the_draw(settings: Settings) -> None:
    """Sigma is applied at use, so changing it never changes the stream."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    observation = project(
        world_at(3),
        policies["actor_0"],
        noise={"factor_0": 1.0, "factor_1": 1.0, "factor_2": 1.0, "factor_3": 1.0},
    )
    expected = round(0.50 + settings.aperture.hop1.noise_sigma, OBSERVATION_DECIMALS)
    assert observation.factors["factor_1"] == expected
    assert observation.factors["factor_0"] == 0.80  # sigma 0 ignores the draw


def test_a_quantized_channel_reports_a_band(settings: Settings) -> None:
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    observation = project(
        world_at(3),
        policies["actor_0"],
        noise=dict.fromkeys(["factor_0", "factor_1", "factor_2", "factor_3"], 0.0),
    )
    assert "factor_2" in observation.banded
    assert observation.factors["factor_2"] in {0.17, 0.5, 0.83}


def test_observed_values_are_rounded_to_two_decimals(settings: Settings) -> None:
    """The risk register's fix for a too-fine observation hash (§8.3, §12.2).

    The rounded value is what enters both the hash and the prompt, so they
    cannot disagree -- which is what makes the action cache and the LLM cache
    the same cache.
    """
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    observation = project(
        world_at(3),
        policies["actor_0"],
        noise={"factor_0": 0.0, "factor_1": 0.123456789, "factor_2": 0.0, "factor_3": 0.0},
    )
    for value in observation.factors.values():
        assert round(value, OBSERVATION_DECIMALS) == value


def test_a_signal_moves_a_blurry_reading_and_not_a_clear_one(settings: Settings) -> None:
    """§7.4: SIGNAL "alters the target's observation of that factor"."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    noise = dict.fromkeys(["factor_0", "factor_1", "factor_2", "factor_3"], 0.0)

    honest = project(world_at(3), policies["actor_0"], noise=noise)
    lied_to = project(
        world_at(3),
        policies["actor_0"],
        noise=noise,
        signals=(
            SignalClaim(from_actor="actor_1", factor_id="factor_1", claimed_value=1.0),
            SignalClaim(from_actor="actor_1", factor_id="factor_0", claimed_value=0.0),
        ),
    )
    assert lied_to.factors["factor_1"] > honest.factors["factor_1"]
    assert lied_to.factors["factor_0"] == honest.factors["factor_0"]


def test_two_actors_see_different_worlds(settings: Settings) -> None:
    """§6.1: the point of the subsystem, stated as a test.

    Same world, same step, same draws -- different observations, because the
    topology gives them different channels.
    """
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    noise = dict.fromkeys(["factor_0", "factor_1", "factor_2", "factor_3"], 0.8)
    near = project(world_at(3), policies["actor_0"], noise=noise)
    far = project(world_at(3), policies["actor_7"], noise=noise)

    assert near.factors != far.factors
    assert observation_hash(near) != observation_hash(far)
    assert set(near.unobserved) != set(far.unobserved)


def test_the_transparent_policy_gives_everyone_the_same_reading(settings: Settings) -> None:
    """§6.4 again, from the projection side: the same draws, no divergence."""
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=False)
    noise = dict.fromkeys(["factor_0", "factor_1", "factor_2", "factor_3"], 0.8)
    near = project(world_at(3), policies["actor_0"], noise=noise)
    far = project(world_at(3), policies["actor_7"], noise=noise)
    assert near.factors == far.factors
    assert near.unobserved == () == far.unobserved


def test_type_only_visibility_hides_the_target_and_the_magnitude(
    settings: Settings,
) -> None:
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=True)
    actions = {
        "actor_1": Commit(target_factor="factor_0", magnitude=0.7, resource_spend=0.4),
        "actor_6": Escalate(target_factor="factor_3", magnitude=0.9),
    }
    seen = {entry.actor_id: entry for entry in visible_actions(policies["actor_0"], actions)}
    assert seen["actor_1"].target == "factor_0"
    assert seen["actor_1"].magnitude == 0.7
    assert "actor_6" not in seen


def test_an_actor_never_sees_its_own_action_in_its_observation(
    settings: Settings,
) -> None:
    policies = derive_policies(chain_graph(), settings.aperture, asymmetry=False)
    actions = {"actor_0": Commit(target_factor="factor_0", magnitude=0.5, resource_spend=0.1)}
    assert visible_actions(policies["actor_0"], actions) == ()


def test_quantize_bands_at_the_thirds() -> None:
    assert quantize(0.1, banded=True) == 0.17
    assert quantize(0.5, banded=True) == 0.5
    assert quantize(0.9, banded=True) == 0.83
    assert quantize(0.123456, banded=False) == 0.12


def test_the_observation_hash_ignores_field_order() -> None:
    left = Observation(
        actor_id="a",
        step=1,
        factors={"f1": 0.2, "f0": 0.4},
        budget=0.5,
        pooled=0.0,
        actions=(
            ObservedAction(actor_id="b", action_type="WAIT", target=None, magnitude=None),
            ObservedAction(actor_id="a2", action_type="COMMIT", target="f0", magnitude=0.5),
        ),
    )
    right = Observation(
        actor_id="a",
        step=1,
        factors={"f0": 0.4, "f1": 0.2},
        budget=0.5,
        pooled=0.0,
        actions=(
            ObservedAction(actor_id="a2", action_type="COMMIT", target="f0", magnitude=0.5),
            ObservedAction(actor_id="b", action_type="WAIT", target=None, magnitude=None),
        ),
    )
    assert observation_hash(left) == observation_hash(right)


# ---------------------------------------------------------------------------
# Private memory (§6.3)
# ---------------------------------------------------------------------------


def observation_for(step: int, value: float) -> Observation:
    return Observation(
        actor_id="actor_0",
        step=step,
        factors={"factor_0": value},
        budget=0.5,
        pooled=0.0,
        trust={"actor_1": 0.4},
    )


def test_memory_is_capped_at_the_window() -> None:
    """§6.3: 12 observations, which is what keeps the prompt flat over 24 steps."""
    memory = AgentMemory(actor_id="actor_0")
    for step in range(20):
        memory = memory.remember(
            observation_for(step, 0.5),
            action=Commit(target_factor="factor_0", magnitude=0.3, resource_spend=0.1),
            window=12,
        )
    assert len(memory.entries) == 12
    assert memory.entries[0].step == 8
    assert memory.entries[-1].step == 19


def test_the_summary_refreshes_on_the_interval_and_not_between() -> None:
    memory = AgentMemory(actor_id="actor_0")
    memory = memory.maybe_refresh(observation_for(0, 0.40), interval=8)
    first = memory.summary_through_step
    memory = memory.maybe_refresh(observation_for(3, 0.55), interval=8)
    assert memory.summary_through_step == first
    memory = memory.maybe_refresh(observation_for(8, 0.70), interval=8)
    assert memory.summary_through_step == 8


def test_the_summary_is_computed_not_generated() -> None:
    """No model in the loop: a hallucinated memory is indistinguishable from one.

    It is also three calls a run against a budget of 10.5 (§12.1).
    """
    memory = AgentMemory(actor_id="actor_0").maybe_refresh(observation_for(0, 0.40), interval=8)
    memory = memory.remember(
        observation_for(1, 0.45),
        action=Commit(target_factor="factor_0", magnitude=0.3, resource_spend=0.1),
        window=12,
    )
    memory = memory.maybe_refresh(observation_for(8, 0.70), interval=8)
    # Drift is measured from the first observation this actor actually took,
    # not from step 0: it was not active at step 0 and has no reading from it.
    assert "factor_0 0.45->0.70" in memory.summary
    assert "COMMITx1" in memory.summary
    assert "actor_1+0.40" in memory.summary


def test_memory_is_per_actor_and_immutable() -> None:
    """§6.3: agents read their own namespace and never another's."""
    first = AgentMemory(actor_id="actor_0")
    second = first.remember(
        observation_for(0, 0.5),
        action=Commit(target_factor="factor_0", magnitude=0.3, resource_spend=0.1),
        window=12,
    )
    assert first.entries == ()
    assert len(second.entries) == 1
    assert second.actor_id == "actor_0"


def test_an_empty_memory_renders_as_such() -> None:
    assert AgentMemory(actor_id="actor_0").render() == "(no prior activity)"
