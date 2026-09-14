"""The `causal_decomposition = off` arm (ADR-0025).

The panel is the mechanism behind a switch that had been configured,
documented and tested since M0 while nothing read it. These tests assert that
it is a real, valid, deterministic alternative structure -- and, more
importantly, that it is *different* from the compiled arm, because two
identical arms are what the defect looked like.
"""

from __future__ import annotations

import pytest

from cascade.decompose.schema import MAX_ACTORS, MIN_ACTORS, graph_hash
from cascade.decompose.validator import validate
from cascade.eval.panel import PANEL_ARCHETYPES, PANEL_FACTORS, panel_graph


class TestValidity:
    @pytest.mark.parametrize("actors", list(range(MIN_ACTORS, MAX_ACTORS + 1)))
    def test_every_panel_size_passes_the_real_structural_validator(self, actors: int) -> None:
        """Not a reimplementation of §5.3 -- the §5.3 validator itself.

        Per-actor edge weight is a function of panel size precisely because a
        fixed weight passes at eight panelists and exceeds the inbound-weight
        cap at fourteen. This is the test that found that.
        """
        report = validate(panel_graph("scenario-x", actors=actors), outcome_text="anything")
        assert [violation.rule for violation in report.violations] == []

    def test_a_panel_below_the_schema_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[8, 20\]"):
            panel_graph("s", actors=MIN_ACTORS - 1)

    def test_a_panel_above_the_schema_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[8, 20\]"):
            panel_graph("s", actors=MAX_ACTORS + 1)

    def test_there_are_enough_archetypes_for_the_largest_panel(self) -> None:
        """Two identical personas are one persona sampled twice, which is the
        self-consistency baseline wearing a different name."""
        assert len(PANEL_ARCHETYPES) >= MAX_ACTORS

    def test_the_outcome_rule_is_interior_at_the_midpoint(self) -> None:
        graph = panel_graph("s", actors=14)
        value = graph.outcome_rule({factor.id: 0.5 for factor in graph.factors})
        assert 0.0 < value < 1.0

    def test_the_outcome_rule_never_reaches_a_bound(self) -> None:
        """A terminal score of exactly 0 or 1 asserts certainty and makes the
        M7 log loss infinite for that scenario (ADR-0014)."""
        graph = panel_graph("s", actors=20)
        for state in (0.0, 1.0):
            value = graph.outcome_rule({factor.id: state for factor in graph.factors})
            assert 0.0 < value < 1.0


class TestDeterminism:
    def test_the_same_scenario_gives_the_same_graph(self) -> None:
        assert graph_hash(panel_graph("s", actors=14)) == graph_hash(panel_graph("s", actors=14))

    def test_different_scenarios_give_different_graphs(self) -> None:
        """Not scenario-specific *structure* -- the archetypes and factors are
        fixed -- but enough per-scenario variation that fourteen panelists are
        not fourteen copies of one."""
        assert graph_hash(panel_graph("a", actors=14)) != graph_hash(panel_graph("b", actors=14))

    def test_the_panel_is_balanced_so_no_prior_is_built_in(self) -> None:
        """An unbalanced archetype table would give the A=off arm a systematic
        lean, and the measured decomposition delta would contain it."""
        leans = [lean for *_rest, lean in PANEL_ARCHETYPES[:14]]
        assert sum(leans) == 0


class TestItIsActuallyDifferent:
    def test_the_panel_is_not_the_compiled_graph(self) -> None:
        """ADR-0025's regression: before the panel existed, the A=off cells ran
        the compiled graph and produced forecasts indistinguishable from the
        A=on cells."""
        graph = panel_graph("s", actors=14)
        assert {factor.id for factor in graph.factors} == {
            factor_id for factor_id, *_rest in PANEL_FACTORS
        }
        assert all(actor.id.startswith("panelist_") for actor in graph.actors)

    def test_every_factor_is_driven_by_at_least_one_panelist(self) -> None:
        graph = panel_graph("s", actors=14)
        actor_ids = {actor.id for actor in graph.actors}
        driven = {edge.dst for edge in graph.edges if edge.src in actor_ids}
        assert driven == {factor.id for factor in graph.factors}

    def test_no_panelist_owns_every_factor(self) -> None:
        graph = panel_graph("s", actors=14)
        for actor in graph.actors:
            owned = {edge.dst for edge in graph.edges if edge.src == actor.id}
            assert len(owned) < len(graph.factors)
