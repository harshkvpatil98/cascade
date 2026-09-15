"""The provenance walk's pure half (spec §11.2; M8).

`render` and `ChainLink.dominant` decide what an operator reads when they ask
why a run came out the way it did, so they are tested without a database. The
walk itself needs one and lives in the integration suite.
"""

from __future__ import annotations

from cascade.trace.provenance import (
    MAX_CHAIN_DEPTH,
    ChainLink,
    OutcomeExplanation,
    RootShock,
    render,
    terminal_factors,
)


def link(depth: int, step: int, **delta: float) -> ChainLink:
    return ChainLink(
        depth=depth,
        step=step,
        seq=0,
        actor_id=f"actor_{depth}",
        action_type="ESCALATE",
        factor_delta=dict(delta),
    )


def explanation(**overrides: object) -> OutcomeExplanation:
    base: dict[str, object] = {
        "run_id": "r1",
        "scenario_id": "s1",
        "config_id": "C01",
        "replicate": 0,
        "outcome_score": 0.83,
        "steps_run": 24,
        "termination": "horizon",
        "seed_factor": "coalition_stability",
        "seed_value": 0.11,
        "links": [link(0, 19, coalition_stability=0.11), link(1, 17, trust=-0.09)],
        "root": RootShock(step=4, factor_id="external_financing", delta=-0.14),
    }
    base.update(overrides)
    return OutcomeExplanation(**base)  # type: ignore[arg-type]


class TestDominantFactor:
    def test_the_largest_absolute_movement_wins(self) -> None:
        assert link(0, 1, a=0.02, b=-0.30).dominant() == ("b", -0.30)

    def test_a_tie_breaks_on_the_factor_id(self) -> None:
        """Invariant 7: the rendered line is a function of the row, not of the
        order a jsonb column happened to deserialise in."""
        assert link(0, 1, zeta=0.1, alpha=-0.1).dominant() == ("alpha", -0.1)

    def test_a_decision_that_moved_nothing_has_no_dominant_factor(self) -> None:
        assert link(0, 1).dominant() is None


class TestCompleteness:
    def test_a_chain_reaching_an_exogenous_shock_is_complete(self) -> None:
        """M8's criterion is "a complete chain to a root cause", so the command
        asserts this rather than printing whatever it found."""
        assert explanation().complete

    def test_a_chain_with_no_root_is_incomplete(self) -> None:
        assert not explanation(root=None).complete

    def test_a_truncated_chain_is_incomplete_even_with_a_root(self) -> None:
        assert not explanation(truncated=True).complete


class TestRender:
    def test_it_prints_the_spec_shape(self) -> None:
        lines = render(explanation())
        assert lines[0].startswith("outcome_score 0.83")
        assert "coalition_stability" in lines[0]
        assert lines[1].lstrip().startswith("d0 ")
        assert lines[-1].lstrip().startswith("root ")

    def test_every_link_gets_a_line(self) -> None:
        lines = render(explanation())
        assert len(lines) == 1 + 2 + 1  # header + two links + root

    def test_small_movements_are_not_rounded_to_zero(self) -> None:
        """A per-step movement is bounded by `kernel.max_step_delta` (0.12) and
        is routinely an order of magnitude under it. Two decimals printed
        `+0.00` for a real movement, which reads as "this decision did nothing"
        in the one display whose job is to show that it did.
        """
        lines = render(explanation(links=[link(0, 3, momentum=0.0007)]))
        assert "+0.0007" in lines[1]
        assert "+0.00 " not in lines[1]

    def test_a_missing_root_says_so_rather_than_printing_nothing(self) -> None:
        lines = render(explanation(root=None))
        assert "not reached" in lines[-1]

    def test_a_coerced_action_is_marked(self) -> None:
        """ADR-0018: an inadmissible action is coerced to WAIT and the reason
        recorded. A trace that hid that would explain a decision the actor did
        not get to make."""
        coerced = ChainLink(
            depth=0,
            step=5,
            seq=1,
            actor_id="a",
            action_type="WAIT",
            factor_delta={},
            coercion="lever_not_held",
        )
        assert "coerced: lever_not_held" in render(explanation(links=[coerced]))[1]

    def test_actor_names_are_aligned_to_the_longest(self) -> None:
        lines = render(
            explanation(
                links=[
                    ChainLink(0, 1, 0, "short", "WAIT", {}),
                    ChainLink(1, 0, 0, "a_very_long_actor_name", "WAIT", {}),
                ]
            )
        )
        assert lines[1].index("WAIT") == lines[2].index("WAIT")


class TestBounds:
    def test_the_depth_bound_exceeds_the_horizon(self) -> None:
        """A run is `kernel.steps` long and every antecedent is strictly
        earlier, so the walk cannot legitimately exceed the horizon."""
        from cascade.config import load_settings

        assert load_settings(None).kernel.steps < MAX_CHAIN_DEPTH

    def test_terminal_factors_are_sorted_and_deduplicated(self) -> None:
        assert terminal_factors(["b", "a", "b"]) == ("a", "b")
