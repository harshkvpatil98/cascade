"""Properties of Assay's scoring rules (spec §10.1).

These are the invariants the metrics must satisfy for *every* input, not the
handful a unit test enumerates. They are what makes `metrics.py` worth keeping
pure: there is no settings object, no database handle and no global bin count,
so Hypothesis can quantify over the data directly.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from cascade.eval.metrics import (
    auc,
    brier,
    calibration,
    isotonic_fit,
    isotonic_predict,
    log_loss,
    murphy_decomposition,
    wilson_interval,
)

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
outcomes = st.integers(min_value=0, max_value=1)


@st.composite
def scored_sets(draw: st.DrawFn, min_size: int = 1, max_size: int = 60) -> tuple[list, list]:
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return (
        draw(st.lists(probabilities, min_size=size, max_size=size)),
        draw(st.lists(outcomes, min_size=size, max_size=size)),
    )


@given(scored_sets())
@settings(max_examples=150, deadline=None)
def test_brier_is_bounded_by_zero_and_one(data: tuple[list, list]) -> None:
    forecasts, ys = data
    assert 0.0 <= brier(forecasts, ys) <= 1.0


@given(scored_sets())
@settings(max_examples=150, deadline=None)
def test_brier_is_permutation_invariant(data: tuple[list, list]) -> None:
    """A metric that depended on scenario order would make the report depend on
    the query plan that loaded the forecasts."""
    forecasts, ys = data
    order = sorted(range(len(ys)), key=lambda index: (forecasts[index], ys[index]))
    shuffled_p = [forecasts[index] for index in order]
    shuffled_y = [ys[index] for index in order]
    assert brier(shuffled_p, shuffled_y) == brier(forecasts, ys)


@given(scored_sets())
@settings(max_examples=100, deadline=None)
def test_the_murphy_terms_reconstruct_the_brier_with_the_stated_residual(
    data: tuple[list, list],
) -> None:
    """The identity the report prints, checked for every input rather than the
    one where it happens to be exact."""
    forecasts, ys = data
    terms = murphy_decomposition(forecasts, ys)
    assert (
        abs(
            terms.brier
            - (terms.reliability - terms.resolution + terms.uncertainty + terms.residual)
        )
        < 1e-9
    )


@given(scored_sets())
@settings(max_examples=100, deadline=None)
def test_the_murphy_terms_are_non_negative(data: tuple[list, list]) -> None:
    forecasts, ys = data
    terms = murphy_decomposition(forecasts, ys)
    assert terms.reliability >= -1e-12
    assert terms.resolution >= -1e-12
    assert 0.0 <= terms.uncertainty <= 0.25


@given(scored_sets())
@settings(max_examples=100, deadline=None)
def test_calibration_bins_partition_the_set(data: tuple[list, list]) -> None:
    """Every forecast lands in exactly one bin, including p = 1.0.

    A forecast that fell through the bins would be missing from ECE while
    still counting in the Brier, and the two would disagree for a reason no
    reader could see.
    """
    forecasts, ys = data
    report = calibration(forecasts, ys)
    assert sum(row.count for row in report.bins) == len(forecasts)
    assert report.n == len(forecasts)


@given(scored_sets())
@settings(max_examples=100, deadline=None)
def test_ece_never_exceeds_mce(data: tuple[list, list]) -> None:
    forecasts, ys = data
    report = calibration(forecasts, ys)
    assert report.ece <= report.mce + 1e-12


@given(scored_sets(min_size=2))
@settings(max_examples=100, deadline=None)
def test_auc_is_bounded_and_complementary(data: tuple[list, list]) -> None:
    """Flipping every label flips the AUC around 0.5."""
    forecasts, ys = data
    value = auc(forecasts, ys)
    if value is None:
        return
    assert 0.0 <= value <= 1.0
    flipped = auc(forecasts, [1 - y for y in ys])
    assert flipped is not None
    assert abs((value + flipped) - 1.0) < 1e-9


@given(scored_sets())
@settings(max_examples=100, deadline=None)
def test_log_loss_is_finite_and_bounded_by_the_clip(data: tuple[list, list]) -> None:
    """The clip is the reason this holds; unclipped, one confident miss is
    infinite and owns the metric."""
    import math

    forecasts, ys = data
    value = log_loss(forecasts, ys)
    assert math.isfinite(value)
    assert 0.0 <= value <= -math.log(0.01) + 1e-9


@given(scored_sets(min_size=2))
@settings(max_examples=100, deadline=None)
def test_isotonic_is_non_decreasing_and_bounded(data: tuple[list, list]) -> None:
    forecasts, ys = data
    thresholds, values = isotonic_fit(forecasts, ys)
    assert list(values) == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)
    mapped = isotonic_predict(thresholds, values, forecasts)
    assert all(0.0 <= value <= 1.0 for value in mapped)


@given(scored_sets(min_size=2))
@settings(max_examples=80, deadline=None)
def test_isotonic_never_worsens_the_brier_on_its_own_fitting_set(
    data: tuple[list, list],
) -> None:
    """Which is exactly why §10.5 requires a held-out half and forbids
    reporting the recalibrated figure as the headline."""
    forecasts, ys = data
    thresholds, values = isotonic_fit(forecasts, ys)
    mapped = isotonic_predict(thresholds, values, forecasts)
    assert brier(list(mapped), ys) <= brier(forecasts, ys) + 1e-9


@given(st.integers(min_value=0, max_value=200), st.integers(min_value=1, max_value=200))
@settings(max_examples=200, deadline=None)
def test_the_wilson_interval_contains_the_observed_proportion(successes: int, trials: int) -> None:
    if successes > trials:
        successes, trials = trials, successes or 1
    lo, hi = wilson_interval(successes, trials)
    observed = successes / trials
    assert 0.0 <= lo <= hi <= 1.0
    assert lo - 1e-9 <= observed <= hi + 1e-9
