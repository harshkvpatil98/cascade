"""The outcome rule is monotone in every factor it names (ADR-0014, spec §5.3).

ADR-0014 argues monotonicity holds by construction: the partial derivative is
``steepness * weight * p * (1 - p)``, and with ``steepness > 0`` and ``p``
strictly interior its sign is exactly the sign of the weight, everywhere.

An argument is not a test. §5.3 requires the property, M5 runs 24 steps of
dynamics on top of it, and M7's headline number is read off the result, so the
claim is quantified over rather than asserted.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.decompose.schema import OutcomeRule, OutcomeTerm

# Non-zero weights, as the schema requires: a zero weight is a dependency in
# name only and the rule would be flat in that factor rather than monotone.
weights = st.floats(min_value=0.01, max_value=1.0).flatmap(
    lambda magnitude: st.sampled_from([magnitude, -magnitude])
)
states = st.floats(min_value=0.0, max_value=1.0)


@st.composite
def rules(draw: st.DrawFn) -> tuple[OutcomeRule, list[str]]:
    count = draw(st.integers(min_value=2, max_value=6))
    factor_ids = [f"f{index}" for index in range(count)]
    terms = tuple(
        OutcomeTerm(factor_id=factor_id, weight=draw(weights)) for factor_id in factor_ids
    )
    return (
        OutcomeRule(
            terms=terms,
            threshold=draw(states),
            steepness=draw(st.floats(min_value=0.1, max_value=50.0)),
        ),
        factor_ids,
    )


@given(payload=rules(), base=states, low=states, high=states)
def test_the_score_moves_in_the_direction_of_each_weight(
    payload: tuple[OutcomeRule, list[str]], base: float, low: float, high: float
) -> None:
    """Raising a factor must move the score the way its weight says.

    Non-strict because the logistic saturates: at large ``steepness`` the score
    reaches the limit of float resolution and a further increase registers as
    exactly zero change. A saturated step is still monotone; a reversal is not.
    """
    rule, factor_ids = payload
    lower, upper = min(low, high), max(low, high)

    for term in rule.terms:
        state = dict.fromkeys(factor_ids, base)
        state[term.factor_id] = lower
        at_low = rule(state)
        state[term.factor_id] = upper
        at_high = rule(state)

        delta = at_high - at_low
        if term.weight > 0:
            assert delta >= 0.0, f"positive weight on {term.factor_id} lowered the score"
        else:
            assert delta <= 0.0, f"negative weight on {term.factor_id} raised the score"


@given(payload=rules(), base=states)
def test_the_score_is_always_a_probability(
    payload: tuple[OutcomeRule, list[str]], base: float
) -> None:
    rule, factor_ids = payload
    score = rule(dict.fromkeys(factor_ids, base))
    assert 0.0 <= score <= 1.0


@given(payload=rules(), base=states)
def test_the_score_is_strictly_interior_at_moderate_steepness(
    payload: tuple[OutcomeRule, list[str]], base: float
) -> None:
    """A terminal score of exactly 0 or 1 asserts certainty (ADR-0014).

    Saturation to the float boundary is possible at the extreme end of the
    steepness range, so this holds the rule to a moderate slope -- the regime
    the compiler is prompted to emit.
    """
    rule, factor_ids = payload
    gentle = OutcomeRule(
        terms=rule.terms, threshold=rule.threshold, steepness=min(rule.steepness, 10.0)
    )
    score = gentle(dict.fromkeys(factor_ids, base))
    assert 0.0 < score < 1.0


@given(payload=rules(), base=states)
def test_evaluation_is_deterministic(payload: tuple[OutcomeRule, list[str]], base: float) -> None:
    """M8 replays the event log and must reproduce the terminal score exactly."""
    rule, factor_ids = payload
    state = dict.fromkeys(factor_ids, base)
    assert rule(state) == rule(state)


@given(payload=rules(), base=states)
def test_extra_factors_in_the_state_are_ignored(
    payload: tuple[OutcomeRule, list[str]], base: float
) -> None:
    """The rule reads the factors it names; the world carries more than that."""
    rule, factor_ids = payload
    named = dict.fromkeys(factor_ids, base)
    extended = {**named, "unrelated_factor": 0.99, "another": 0.01}
    assert rule(extended) == pytest.approx(rule(named))


@given(payload=rules())
def test_a_missing_factor_raises_rather_than_defaulting(
    payload: tuple[OutcomeRule, list[str]],
) -> None:
    """Defaulting substitutes a guess for a measurement."""
    rule, factor_ids = payload
    partial = dict.fromkeys(factor_ids[1:], 0.5)
    with pytest.raises(KeyError):
        rule(partial)
