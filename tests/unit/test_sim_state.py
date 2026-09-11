"""World state: serialization, relation keys and the opening schedule (§7.1).

The round-trip test is M5 acceptance criterion 3, second half. "Round-trips
exactly" is taken literally: every field equal, and the content hash equal, so
a checkpoint resume cannot restore a world that merely looks like the one it
saved.
"""

from __future__ import annotations

import json
import math

import pytest

from cascade.sim.state import (
    RELATION_SEPARATOR,
    PendingEffect,
    WorldState,
    initial_state,
    relation_key,
    relation_pair,
    staggered_schedule,
    state_hash,
    trust_between,
)
from tests.conftest import make_graph


def awkward_state() -> WorldState:
    """A state whose floats are the ones that break naive serialisation.

    Values with long binary expansions, a negative trust, a pool, pending
    effects out of order, and a factor at a bound -- not round numbers, because
    round numbers round-trip through anything.
    """
    return WorldState(
        step=7,
        factors={"factor_0": 0.1 + 0.2, "factor_1": 1.0, "factor_2": 1 / 3},
        factor_history={
            "factor_0": (0.5, 0.4999999999999999, 0.30000000000000004),
            "factor_1": (0.5, 0.75, 1.0),
            "factor_2": (0.5, 0.5, 1 / 3),
        },
        resources={"actor_0": {"capital": 0.7000000000000001, "political": 0.2}},
        relations={relation_key("actor_1", "actor_0"): -0.30000000000000004},
        pools={"actor_0": {"actor_1": 0.15000000000000002}},
        pending=(
            PendingEffect(
                arrive_step=9,
                factor_id="factor_2",
                delta=-0.012345678901234,
                source_actor_id="actor_1",
                origin_step=7,
                origin_seq=3,
            ),
            PendingEffect(
                arrive_step=8,
                factor_id="factor_0",
                delta=0.0009765625,
                source_actor_id=None,
                origin_step=6,
                origin_seq=0,
            ),
        ),
        rng_counter=896,
        last_acted={"actor_0": 6, "actor_1": 2},
        last_observed_at={"actor_0": 7},
        pinned_streak={"factor_1": 2},
    )


def test_world_state_round_trips_exactly() -> None:
    """M5 criterion 3: serialization round-trips exactly, not approximately."""
    state = awkward_state()
    restored = WorldState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert state_hash(restored) == state_hash(state)
    for factor_id in sorted(state.factors):
        assert restored.factors[factor_id] == state.factors[factor_id]
        assert math.isclose(
            restored.factors[factor_id], state.factors[factor_id], rel_tol=0.0, abs_tol=0.0
        )


def test_the_round_trip_goes_through_real_json_text() -> None:
    """Guard the guard: a round trip through Python objects would prove nothing."""
    state = awkward_state()
    text = state.model_dump_json()
    assert isinstance(json.loads(text), dict)
    assert WorldState.model_validate(json.loads(text)) == state


def test_relation_keys_survive_json_but_tuple_keys_would_not() -> None:
    """ADR-0003: a `dict[tuple[str, str], float]` cannot round-trip through JSON."""
    key = relation_key("regional_bloc", "incumbent_party")
    assert key == f"incumbent_party{RELATION_SEPARATOR}regional_bloc"
    assert relation_pair(key) == ("incumbent_party", "regional_bloc")

    # What §7.1's literal type would do, for contrast: the key comes back as a
    # string spelling of a tuple, and a second write keys a *different* entry.
    naive = {("a", "b"): 0.5}
    revived = json.loads(json.dumps({str(k): v for k, v in naive.items()}))
    assert list(revived) == ["('a', 'b')"]
    assert ("a", "b") not in revived


def test_trust_is_mutual_and_absent_means_neutral() -> None:
    relations = {relation_key("a", "b"): 0.4}
    assert trust_between(relations, "a", "b") == 0.4
    assert trust_between(relations, "b", "a") == 0.4
    assert trust_between(relations, "a", "c") == 0.0


def test_an_actor_has_no_relation_with_itself() -> None:
    with pytest.raises(ValueError, match="no trust relation with itself"):
        relation_key("a", "a")


def test_a_separator_in_an_actor_id_is_refused() -> None:
    """The key would be ambiguous, and ambiguity here silently merges two pairs."""
    with pytest.raises(ValueError, match="relation separator"):
        relation_key("a|b", "c")


def test_factor_at_clamps_before_the_start_and_after_the_end() -> None:
    state = awkward_state()
    assert state.factor_at("factor_0", -3) == state.factor_history["factor_0"][0]
    assert state.factor_at("factor_0", 99) == state.factor_history["factor_0"][-1]
    assert state.factor_at("factor_0", 1) == state.factor_history["factor_0"][1]


def test_budget_sums_the_bundle_and_pool_sums_the_pledges() -> None:
    state = awkward_state()
    assert state.budget("actor_0") == pytest.approx(0.9000000000000001)
    assert state.pooled("actor_0") == pytest.approx(0.15000000000000002)
    assert state.budget("nobody") == 0.0


def test_initial_state_is_built_from_the_graph() -> None:
    graph = make_graph(n_actors=8, n_factors=4)
    state = initial_state(graph, forced_interval=5)

    assert state.step == 0
    assert sorted(state.factors) == [f.id for f in sorted(graph.factors, key=lambda f: f.id)]
    assert all(len(history) == 1 for history in state.factor_history.values())
    assert state.relations == {}
    assert state.rng_counter == 0


def test_the_opening_schedule_is_staggered_across_the_interval() -> None:
    """ADR-0017: a uniform opening makes all 14 actors fall due on one step."""
    schedule = staggered_schedule([f"actor_{i}" for i in range(14)], 5)
    due_at = sorted((5 - 1) - (-1 - value) for value in schedule.values())
    # Every actor is due within the first interval, spread across it.
    assert min(due_at) == 0
    assert max(due_at) == 4
    assert len(set(due_at)) == 5


def test_pending_effects_have_a_total_order() -> None:
    """ARRIVE must not depend on the order effects were appended."""
    state = awkward_state()
    keys = [effect.sort_key() for effect in state.pending]
    assert sorted(keys) != keys or len(keys) < 2
    assert len(set(keys)) == len(keys)
