"""The §5.3 validator. Pure Python, so these tests need no model and no database.

The validator is what stands between a plausible-looking graph and 200
simulation runs built on it, so each rule is tested from both sides: it fires
on the defect it exists for, and it stays quiet on a graph that merely looks
unusual.
"""

from __future__ import annotations

import pytest

from cascade.decompose.schema import (
    CausalGraph,
    Edge,
    OutcomeRule,
    OutcomeTerm,
    UtilityTerm,
)
from cascade.decompose.validator import (
    FACTOR_SIMILARITY_MAX,
    MAX_INBOUND_WEIGHT,
    OBJECTIVE_SIMILARITY_MAX,
    cosine,
    validate,
)
from tests.conftest import fake_embed, make_actor, make_factor, make_graph

OUTCOME_TEXT = "Will the regulator block the merger before the deadline?"


def rules_fired(report: object) -> set[str]:
    return {violation.rule for violation in report.violations}  # type: ignore[attr-defined]


def check(graph: CausalGraph, *, embed: object | None = None) -> object:
    return validate(
        graph,
        outcome_text=OUTCOME_TEXT,
        embed=embed if embed is not None else fake_embed(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_a_sound_graph_passes_every_rule() -> None:
    report = check(make_graph())
    assert report.ok  # type: ignore[attr-defined]
    assert report.violations == ()  # type: ignore[attr-defined]
    assert "objective_independence" in report.checked_rules  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Skipped rules must not read as passed
# ---------------------------------------------------------------------------


def test_without_an_embedder_the_semantic_rules_are_skipped_not_passed() -> None:
    """A check that did not run must be distinguishable from one that succeeded.

    Otherwise the objective-independence rule quietly stops catching the
    failure §5.2 calls the most common one.
    """
    report = validate(make_graph(), outcome_text=OUTCOME_TEXT, embed=None)
    assert report.violations == ()
    assert report.passed_checks is True
    assert report.ok is False, "a skipped rule leaves the verdict unknown, not favourable"
    assert set(report.skipped_rules) == {"objective_independence", "factor_orthogonality"}


def test_structural_rules_still_run_without_an_embedder() -> None:
    graph = make_graph(n_actors=8)
    broken = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=(*graph.edges, Edge(src="factor_1", dst="factor_1", sign=1, weight=0.2, lag=0)),
        outcome_rule=graph.outcome_rule,
    )
    report = validate(broken, outcome_text=OUTCOME_TEXT, embed=None)
    assert "edge_sanity" in {v.rule for v in report.violations}


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


def test_an_actor_influencing_nothing_relevant_is_flagged() -> None:
    """Decorative actors inflate the agent count without informing the forecast."""
    graph = make_graph()
    # factor_3 is chained into factor_0 by the fixture; sever it so the actor
    # acting on it can no longer reach the outcome rule.
    kept = tuple(
        edge for edge in graph.edges if not (edge.src == "factor_3" and edge.dst == "factor_0")
    )
    severed = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=kept,
        outcome_rule=graph.outcome_rule,
    )
    report = check(severed)
    assert "reachability" in rules_fired(report)
    stranded = next(v for v in report.violations if v.rule == "reachability")  # type: ignore[attr-defined]
    assert "actor_3" in stranded.subjects


def test_an_actor_reaching_the_outcome_indirectly_is_accepted() -> None:
    """Reachability is a path property, not a direct-edge property."""
    report = check(make_graph())
    assert "reachability" not in rules_fired(report)


# ---------------------------------------------------------------------------
# Edge sanity
# ---------------------------------------------------------------------------


def test_a_zero_lag_factor_self_loop_is_flagged() -> None:
    """An instantaneous algebraic loop has no fixed point the arbiter can resolve."""
    graph = make_graph()
    looped = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=(*graph.edges, Edge(src="factor_2", dst="factor_2", sign=1, weight=0.2, lag=0)),
        outcome_rule=graph.outcome_rule,
    )
    report = check(looped)
    assert "edge_sanity" in rules_fired(report)


def test_a_lagged_factor_self_loop_is_allowed() -> None:
    """Momentum is a real dynamic; only the zero-lag form is unresolvable."""
    graph = make_graph()
    looped = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=(*graph.edges, Edge(src="factor_2", dst="factor_2", sign=1, weight=0.2, lag=2)),
        outcome_rule=graph.outcome_rule,
    )
    assert "edge_sanity" not in rules_fired(check(looped))


def test_excess_inbound_weight_is_flagged() -> None:
    """Above 3.0 the arbiter dynamics become explosive."""
    graph = make_graph(n_actors=20)
    heavy = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=tuple(
            Edge(src=actor.id, dst="factor_0", sign=1, weight=1.0, lag=1) for actor in graph.actors
        )
        + tuple(
            Edge(src=f"factor_{i}", dst="factor_0", sign=1, weight=0.2, lag=1) for i in range(1, 4)
        ),
        outcome_rule=graph.outcome_rule,
    )
    report = check(heavy)
    assert "edge_sanity" in rules_fired(report)
    violation = next(v for v in report.violations if v.rule == "edge_sanity")  # type: ignore[attr-defined]
    assert any("factor_0" in subject for subject in violation.subjects)


def test_inbound_weight_exactly_at_the_cap_is_allowed() -> None:
    """The rule is "must not exceed", so the boundary itself is fine."""
    graph = make_graph()
    edges = tuple(
        Edge(src=f"actor_{i}", dst="factor_0", sign=1, weight=MAX_INBOUND_WEIGHT / 8, lag=1)
        for i in range(8)
    ) + tuple(
        Edge(src=f"factor_{i}", dst="factor_1", sign=1, weight=0.2, lag=1) for i in range(2, 4)
    )
    at_cap = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=edges,
        outcome_rule=graph.outcome_rule,
    )
    assert "edge_sanity" not in rules_fired(check(at_cap))


# ---------------------------------------------------------------------------
# Objective independence -- the failure §5.2 names as most common
# ---------------------------------------------------------------------------


def test_an_objective_restating_the_outcome_is_flagged() -> None:
    """Every actor given the goal "make the outcome happen" is the classic failure."""
    graph = make_graph()
    parrot = make_actor(0, factor_id="factor_0").model_copy(update={"objective": OUTCOME_TEXT})
    restated = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=(parrot, *graph.actors[1:]),
        factors=graph.factors,
        edges=graph.edges,
        outcome_rule=graph.outcome_rule,
    )
    report = check(restated)
    assert "objective_independence" in rules_fired(report)
    violation = next(v for v in report.violations if v.rule == "objective_independence")  # type: ignore[attr-defined]
    assert any(subject.startswith("actor_0=") for subject in violation.subjects)


def test_independent_objectives_pass() -> None:
    assert "objective_independence" not in rules_fired(check(make_graph()))


def test_the_objective_threshold_is_the_documented_one() -> None:
    """The repair prompt quotes this number, so it must be the one enforced."""
    assert OBJECTIVE_SIMILARITY_MAX == 0.85


# ---------------------------------------------------------------------------
# Factor orthogonality
# ---------------------------------------------------------------------------


def test_near_duplicate_factor_names_are_flagged() -> None:
    """One driver split across three variables is counted three times."""
    graph = make_graph()
    twin = make_factor(1).model_copy(update={"name": graph.factors[0].name})
    duplicated = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=(graph.factors[0], twin, *graph.factors[2:]),
        edges=graph.edges,
        outcome_rule=graph.outcome_rule,
    )
    report = check(duplicated)
    assert "factor_orthogonality" in rules_fired(report)


def test_distinct_factor_names_pass() -> None:
    assert "factor_orthogonality" not in rules_fired(check(make_graph()))


def test_the_factor_threshold_is_the_documented_one() -> None:
    assert FACTOR_SIMILARITY_MAX == 0.80


# ---------------------------------------------------------------------------
# Outcome rule
# ---------------------------------------------------------------------------


def test_a_sound_outcome_rule_is_monotone_in_every_factor() -> None:
    assert "outcome_rule" not in rules_fired(check(make_graph()))


def test_monotonicity_is_checked_in_both_directions() -> None:
    """A negative weight must produce a falling score, not merely a changing one."""
    graph = make_graph()
    flipped = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=graph.actors,
        factors=graph.factors,
        edges=graph.edges,
        outcome_rule=OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_0", weight=-0.8),
                OutcomeTerm(factor_id="factor_1", weight=0.4),
            ),
            threshold=0.5,
            steepness=6.0,
        ),
    )
    assert "outcome_rule" not in rules_fired(check(flipped))
    rule = flipped.outcome_rule
    low = rule({"factor_0": 0.1, "factor_1": 0.5, "factor_2": 0.5, "factor_3": 0.5})
    high = rule({"factor_0": 0.9, "factor_1": 0.5, "factor_2": 0.5, "factor_3": 0.5})
    assert high < low, "a negative weight must lower the score as the factor rises"


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_ignores_magnitude() -> None:
    """Non-normalised inputs must not inflate a similarity past a threshold."""
    assert cosine([1.0, 0.0], [17.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_a_zero_vector_is_zero_not_an_error() -> None:
    """A degenerate embedding must not crash a 180-scenario compile."""
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Violation rendering -- read verbatim by the repair pass
# ---------------------------------------------------------------------------


def test_violations_render_with_their_subjects() -> None:
    graph = make_graph()
    parrot = make_actor(0, factor_id="factor_0").model_copy(update={"objective": OUTCOME_TEXT})
    restated = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=(parrot, *graph.actors[1:]),
        factors=graph.factors,
        edges=graph.edges,
        outcome_rule=graph.outcome_rule,
    )
    rendered = check(restated).render()  # type: ignore[attr-defined]
    assert "objective_independence" in rendered
    assert "actor_0" in rendered


def test_an_actor_count_violation_names_the_bound() -> None:
    """The repair pass needs the number it has to satisfy."""
    from cascade.decompose.validator import _check_actor_count

    class _Stub:
        actors = ()

    violations = _check_actor_count(_Stub())  # type: ignore[arg-type]
    assert violations and "minimum 8" in violations[0].message


def test_utility_terms_do_not_confer_reachability() -> None:
    """Wanting a factor is not influencing it.

    An actor with a utility term but no outbound edge is a spectator with an
    opinion, which is exactly what the reachability rule excludes.
    """
    graph = make_graph()
    watcher = make_actor(9, factor_id="factor_0")
    spectator = CausalGraph(
        scenario_id=graph.scenario_id,
        actors=(*graph.actors, watcher),
        factors=graph.factors,
        edges=graph.edges,
        outcome_rule=graph.outcome_rule,
    )
    report = check(spectator)
    assert "reachability" in rules_fired(report)
    violation = next(v for v in report.violations if v.rule == "reachability")  # type: ignore[attr-defined]
    assert "actor_9" in violation.subjects
    assert UtilityTerm(factor_id="factor_0", weight=0.5) in watcher.utility_terms
