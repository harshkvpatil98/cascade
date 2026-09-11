"""The CausalGraph contract (spec §5.1, ADR-0014).

§5.1's own framing is the reason these tests are strict: "if this schema is
loose, the whole system degrades into a chat room". Everything downstream reads
these fields mechanically, so a bound that is documented but not enforced is a
bound that does not exist.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cascade.decompose.schema import (
    MAX_ACTORS,
    MIN_ACTORS,
    Actor,
    CausalGraph,
    Edge,
    Factor,
    OutcomeRule,
    OutcomeTerm,
    UtilityTerm,
    graph_hash,
)
from tests.conftest import make_actor, make_factor, make_graph

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_a_baseline_graph_is_valid() -> None:
    graph = make_graph()
    assert len(graph.actors) == MIN_ACTORS
    assert graph.outcome_rule.factor_ids == ("factor_0", "factor_1")


@pytest.mark.parametrize("count", [0, 1, 7])
def test_too_few_actors_is_rejected(count: int) -> None:
    """Below 8 the information-asymmetry mechanism has nothing to work with."""
    with pytest.raises(ValidationError):
        make_graph(n_actors=count)


def test_too_many_actors_is_rejected() -> None:
    """Above 20 the per-run cost model breaks."""
    with pytest.raises(ValidationError):
        make_graph(n_actors=MAX_ACTORS + 1)


@pytest.mark.parametrize("count", [3, 13])
def test_factor_count_bounds_are_enforced(count: int) -> None:
    with pytest.raises(ValidationError):
        make_graph(n_factors=count)


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1])
def test_edge_weight_must_be_in_the_half_open_unit_interval(weight: float) -> None:
    """(0, 1] per §5.1: a zero-weight edge is an edge that is not there."""
    with pytest.raises(ValidationError):
        Edge(src="actor_0", dst="factor_0", sign=1, weight=weight, lag=0)


@pytest.mark.parametrize("lag", [-1, 7])
def test_edge_lag_is_bounded_to_six_steps(lag: int) -> None:
    with pytest.raises(ValidationError):
        Edge(src="actor_0", dst="factor_0", sign=1, weight=0.5, lag=lag)


def test_edge_sign_admits_only_plus_or_minus_one() -> None:
    with pytest.raises(ValidationError):
        Edge(src="actor_0", dst="factor_0", sign=0, weight=0.5, lag=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_factor_state_is_a_unit_interval(value: float) -> None:
    with pytest.raises(ValidationError):
        make_factor(0, state=value)


def test_ids_must_be_slugs() -> None:
    """Ids travel into JSON keys and prompts; free text there is a trap."""
    with pytest.raises(ValidationError):
        make_factor(0, id="Factor One")


def test_unknown_fields_are_refused() -> None:
    """`extra="forbid"` so a model inventing a field fails loudly."""
    with pytest.raises(ValidationError):
        make_factor(0, wobble=1.0)


# ---------------------------------------------------------------------------
# Zero weights -- a dependency in name only (ADR-0014)
# ---------------------------------------------------------------------------


def test_a_zero_utility_weight_is_refused() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        UtilityTerm(factor_id="factor_0", weight=0.0)


def test_a_zero_outcome_weight_is_refused() -> None:
    """It satisfies "depends on 2 factors" by arity while depending on one."""
    with pytest.raises(ValidationError, match="non-zero"):
        OutcomeTerm(factor_id="factor_0", weight=0.0)


def test_negative_utility_weights_are_allowed() -> None:
    """An actor that wants a factor low says so with a sign, not a mirror factor."""
    assert UtilityTerm(factor_id="factor_0", weight=-0.7).weight == pytest.approx(-0.7)


# ---------------------------------------------------------------------------
# Cross-reference integrity
# ---------------------------------------------------------------------------


def test_an_edge_to_an_unknown_factor_is_refused() -> None:
    graph = make_graph()
    with pytest.raises(ValidationError, match="not a factor"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=graph.factors,
            edges=(*graph.edges, Edge(src="actor_0", dst="ghost", sign=1, weight=0.2, lag=1)),
            outcome_rule=graph.outcome_rule,
        )


def test_an_edge_from_an_unknown_source_is_refused() -> None:
    graph = make_graph()
    with pytest.raises(ValidationError, match="neither an actor nor a factor"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=graph.factors,
            edges=(*graph.edges, Edge(src="ghost", dst="factor_0", sign=1, weight=0.2, lag=1)),
            outcome_rule=graph.outcome_rule,
        )


def test_an_edge_dst_may_not_be_an_actor() -> None:
    """§5.1: dst is always a factor. Actors influence the world, not each other."""
    graph = make_graph()
    with pytest.raises(ValidationError, match="not a factor"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=graph.factors,
            edges=(*graph.edges, Edge(src="actor_0", dst="actor_1", sign=1, weight=0.2, lag=1)),
            outcome_rule=graph.outcome_rule,
        )


def test_a_utility_term_on_an_unknown_factor_is_refused() -> None:
    graph = make_graph()
    stray = make_actor(99, factor_id="factor_0").model_copy(
        update={"utility_terms": (UtilityTerm(factor_id="ghost", weight=0.5),)}
    )
    with pytest.raises(ValidationError, match="unknown factor"):
        CausalGraph(
            scenario_id="s",
            actors=(*graph.actors, stray),
            factors=graph.factors,
            edges=graph.edges,
            outcome_rule=graph.outcome_rule,
        )


def test_an_outcome_rule_on_an_unknown_factor_is_refused() -> None:
    graph = make_graph()
    with pytest.raises(ValidationError, match="unknown factor"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=graph.factors,
            edges=graph.edges,
            outcome_rule=OutcomeRule(
                terms=(
                    OutcomeTerm(factor_id="factor_0", weight=0.5),
                    OutcomeTerm(factor_id="ghost", weight=0.5),
                ),
                threshold=0.5,
                steepness=4.0,
            ),
        )


def test_an_id_shared_between_an_actor_and_a_factor_is_refused() -> None:
    """An edge src would be ambiguous between the two."""
    graph = make_graph()
    # A ninth actor whose id collides with an existing factor. Added rather
    # than substituted so the existing edges stay valid and the id clash is
    # the only thing under test.
    clash = make_actor(9).model_copy(update={"id": "factor_0"})
    with pytest.raises(ValidationError, match="shared between"):
        CausalGraph(
            scenario_id="s",
            actors=(*graph.actors, clash),
            factors=graph.factors,
            edges=graph.edges,
            outcome_rule=graph.outcome_rule,
        )


def test_duplicate_edges_at_the_same_lag_are_refused() -> None:
    """Two influences at one lag are one influence with a doubled weight."""
    graph = make_graph()
    duplicate = Edge(src="actor_0", dst="factor_0", sign=1, weight=0.3, lag=1)
    with pytest.raises(ValidationError, match="duplicate edge"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=graph.factors,
            edges=(*graph.edges, duplicate),
            outcome_rule=graph.outcome_rule,
        )


def test_an_actor_may_not_repeat_a_utility_term() -> None:
    with pytest.raises(ValidationError, match="repeated utility terms"):
        Actor(
            id="a",
            name="A",
            objective="Something independent.",
            utility_terms=(
                UtilityTerm(factor_id="factor_0", weight=0.4),
                UtilityTerm(factor_id="factor_0", weight=-0.2),
            ),
            risk_posture="neutral",
        )


def test_duplicate_factor_ids_are_refused() -> None:
    graph = make_graph()
    with pytest.raises(ValidationError, match="not unique"):
        CausalGraph(
            scenario_id="s",
            actors=graph.actors,
            factors=(*graph.factors, Factor(**graph.factors[0].model_dump())),
            edges=graph.edges,
            outcome_rule=graph.outcome_rule,
        )


# ---------------------------------------------------------------------------
# OutcomeRule
# ---------------------------------------------------------------------------


def test_an_outcome_rule_needs_two_distinct_factors() -> None:
    """One factor reduces the whole simulation to a random walk (§5.3)."""
    with pytest.raises(ValidationError):
        OutcomeRule(
            terms=(OutcomeTerm(factor_id="factor_0", weight=0.5),),
            threshold=0.5,
            steepness=4.0,
        )


def test_two_terms_on_one_factor_do_not_count_as_two() -> None:
    """It would pass a naive arity check while depending on a single driver."""
    with pytest.raises(ValidationError, match="repeats a factor"):
        OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_0", weight=0.5),
                OutcomeTerm(factor_id="factor_0", weight=-0.3),
            ),
            threshold=0.5,
            steepness=4.0,
        )


def test_steepness_must_be_positive() -> None:
    """A non-positive steepness inverts or flattens every term at once."""
    with pytest.raises(ValidationError):
        OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_0", weight=0.5),
                OutcomeTerm(factor_id="factor_1", weight=0.5),
            ),
            threshold=0.5,
            steepness=0.0,
        )


def test_the_outcome_rule_is_strictly_interior() -> None:
    """Exactly 0 or 1 asserts certainty and degenerates the M7 Brier term."""
    rule = make_graph().outcome_rule
    extreme = {"factor_0": 1.0, "factor_1": 0.0, "factor_2": 1.0, "factor_3": 1.0}
    score = rule(extreme)
    assert 0.0 < score < 1.0


def test_the_outcome_rule_raises_on_a_missing_factor() -> None:
    """Defaulting an absent factor substitutes a guess for a measurement."""
    rule = make_graph().outcome_rule
    with pytest.raises(KeyError, match="absent from the terminal state"):
        rule({"factor_0": 0.5})


def test_the_outcome_rule_saturates_without_overflowing() -> None:
    """math.exp overflows near 709; the clamp must not raise on valid input."""
    rule = OutcomeRule(
        terms=(
            OutcomeTerm(factor_id="factor_0", weight=1.0),
            OutcomeTerm(factor_id="factor_1", weight=1.0),
        ),
        threshold=0.0,
        steepness=50.0,
    )
    assert rule({"factor_0": 1.0, "factor_1": 1.0}) == pytest.approx(1.0, abs=1e-9)
    assert rule({"factor_0": 0.0, "factor_1": 0.0}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Determinism (spec §5.3)
# ---------------------------------------------------------------------------


def test_the_same_graph_hashes_identically() -> None:
    assert graph_hash(make_graph()) == graph_hash(make_graph())


def test_emission_order_does_not_change_the_hash() -> None:
    """A model listing actors in another order produced the same graph.

    Without this, §5.3's determinism rule would fail for a reason that has
    nothing to do with the graph's content.
    """
    graph = make_graph()
    shuffled = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=tuple(reversed(graph.actors)),
        factors=tuple(reversed(graph.factors)),
        edges=tuple(reversed(graph.edges)),
        outcome_rule=OutcomeRule(
            terms=tuple(reversed(graph.outcome_rule.terms)),
            threshold=graph.outcome_rule.threshold,
            steepness=graph.outcome_rule.steepness,
        ),
    )
    assert graph_hash(shuffled) == graph_hash(graph)


def test_a_changed_weight_changes_the_hash() -> None:
    graph = make_graph()
    nudged = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=(
            Edge(src="actor_0", dst="factor_0", sign=1, weight=0.31, lag=1),
            *graph.edges[1:],
        ),
        outcome_rule=graph.outcome_rule,
    )
    assert graph_hash(nudged) != graph_hash(graph)


def test_a_different_scenario_id_changes_the_hash() -> None:
    """Graphs are keyed per scenario; two scenarios must not collide."""
    assert graph_hash(make_graph(scenario_id="a")) != graph_hash(make_graph(scenario_id="b"))


def test_the_hash_is_hex_and_stable_in_width() -> None:
    digest = graph_hash(make_graph())
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
