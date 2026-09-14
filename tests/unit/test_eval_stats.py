"""Assay's inference (spec §10.4).

The bootstrap, the multiple-comparison control and the correlation tests are
what turn a 0.035 Brier difference on 180 scenarios into a claim. Each is
checked against a case whose answer is known before the code runs.
"""

from __future__ import annotations

import pytest

from cascade.eval.schema import BootstrapInterval, Comparison
from cascade.eval.stats import (
    adjust_family,
    bootstrap_seed,
    holm_bonferroni,
    paired_bootstrap,
    pearson,
    permutation_p,
    spearman,
)


def _comparison(name: str, p_value: float) -> Comparison:
    return Comparison(
        name=name,
        config_a="A",
        config_b="B",
        brier_a=0.2,
        brier_b=0.2,
        n_paired=10,
        interval=BootstrapInterval(point=0.0, lo=-0.1, hi=0.1, b=100, p_value=p_value),
    )


class TestPairedBootstrap:
    def test_identical_configurations_give_a_zero_interval(self) -> None:
        """Two copies of one forecast column differ by exactly zero in every
        resample, so the interval is a point and the p-value cannot reject."""
        forecasts = [0.2, 0.9, 0.5, 0.31]
        outcomes = [0, 1, 1, 0]
        interval = paired_bootstrap(forecasts, list(forecasts), outcomes, seed=7, b_resamples=500)
        assert interval.point == 0.0
        assert (interval.lo, interval.hi) == (0.0, 0.0)
        assert interval.p_value == 1.0

    def test_the_point_estimate_is_the_brier_difference(self) -> None:
        from cascade.eval.metrics import brier

        a = [0.9, 0.1, 0.4]
        b = [0.5, 0.5, 0.5]
        outcomes = [1, 0, 1]
        interval = paired_bootstrap(a, b, outcomes, seed=3, b_resamples=200)
        assert interval.point == pytest.approx(brier(a, outcomes) - brier(b, outcomes))

    def test_a_large_consistent_difference_excludes_zero(self) -> None:
        a = [0.99] * 20
        b = [0.01] * 20
        outcomes = [0] * 20
        interval = paired_bootstrap(a, b, outcomes, seed=11, b_resamples=2000)
        assert interval.excludes_zero
        assert interval.lo > 0.0

    def test_it_is_reproducible_from_the_seed(self) -> None:
        args = ([0.3, 0.7, 0.55], [0.5, 0.5, 0.5], [1, 0, 1])
        first = paired_bootstrap(*args, seed=42, b_resamples=1000)
        second = paired_bootstrap(*args, seed=42, b_resamples=1000)
        assert (first.lo, first.hi, first.p_value) == (second.lo, second.hi, second.p_value)

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        args = ([0.3, 0.7, 0.55, 0.2, 0.8], [0.5] * 5, [1, 0, 1, 0, 1])
        first = paired_bootstrap(*args, seed=1, b_resamples=1000)
        second = paired_bootstrap(*args, seed=2, b_resamples=1000)
        assert (first.lo, first.hi) != (second.lo, second.hi)

    def test_the_p_value_is_floored_at_the_bootstrap_resolution(self) -> None:
        """A bootstrap cannot resolve finer than 1/B, and printing p = 0.0
        claims that it can."""
        interval = paired_bootstrap([1.0] * 30, [0.0] * 30, [0] * 30, seed=5, b_resamples=500)
        assert interval.p_value == pytest.approx(1 / 500)

    def test_a_single_scenario_cannot_be_resampled_into_a_claim(self) -> None:
        interval = paired_bootstrap([0.9], [0.1], [1], seed=1, b_resamples=100)
        assert interval.lo == interval.hi == interval.point
        assert interval.p_value == 1.0

    def test_unpaired_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="paired lengths disagree"):
            paired_bootstrap([0.1, 0.2], [0.3], [1, 0], seed=1)

    def test_an_empty_set_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            paired_bootstrap([], [], [], seed=1)


class TestHolmBonferroni:
    def test_the_smallest_p_is_multiplied_by_the_family_size(self) -> None:
        adjusted = holm_bonferroni([0.01, 0.04, 0.03])
        assert adjusted[0] == pytest.approx(0.03)

    def test_adjustment_is_monotone_in_the_raw_order(self) -> None:
        """Without the running maximum a test can come out with a smaller
        adjusted p than a test with a smaller raw p, which is not a step-down
        procedure and is not what "adjusted" means."""
        raw = [0.001, 0.4, 0.41, 0.42]
        adjusted = holm_bonferroni(raw)
        by_raw = [adjusted[index] for index in sorted(range(len(raw)), key=lambda i: raw[i])]
        assert by_raw == sorted(by_raw)

    def test_adjusted_values_never_exceed_one(self) -> None:
        assert all(value <= 1.0 for value in holm_bonferroni([0.9, 0.95, 0.99]))

    def test_order_is_preserved(self) -> None:
        adjusted = holm_bonferroni([0.5, 0.01])
        assert adjusted[1] < adjusted[0]

    def test_a_single_test_is_unadjusted(self) -> None:
        assert holm_bonferroni([0.02]) == (0.02,)

    def test_an_empty_family_is_empty(self) -> None:
        assert holm_bonferroni([]) == ()

    def test_it_is_never_weaker_than_bonferroni(self) -> None:
        raw = [0.002, 0.02, 0.2]
        adjusted = holm_bonferroni(raw)
        bonferroni = [min(1.0, value * len(raw)) for value in raw]
        assert all(a <= b + 1e-12 for a, b in zip(adjusted, bonferroni, strict=True))

    def test_a_p_value_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_bonferroni([1.4])


class TestAdjustFamily:
    def test_every_comparison_gets_an_adjusted_p(self) -> None:
        family = adjust_family([_comparison("a", 0.01), _comparison("b", 0.5)])
        assert [item.p_adjusted for item in family] == [pytest.approx(0.02), pytest.approx(0.5)]

    def test_the_originals_are_not_mutated(self) -> None:
        original = _comparison("a", 0.01)
        adjust_family([original])
        assert original.p_adjusted is None


class TestCorrelations:
    def test_perfect_positive_correlation(self) -> None:
        assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)

    def test_a_constant_column_is_undefined_not_zero(self) -> None:
        """Reporting 0.0 would read as "tested and found nothing"."""
        assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_spearman_sees_a_monotone_relation_pearson_understates(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 2.0, 4.0, 8.0, 64.0]
        assert spearman(xs, ys) == pytest.approx(1.0)
        assert pearson(xs, ys) < 1.0

    def test_spearman_handles_ties_with_mid_ranks(self) -> None:
        assert spearman([1.0, 1.0, 2.0], [3.0, 3.0, 5.0]) == pytest.approx(1.0)

    def test_a_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            pearson([1.0], [1.0, 2.0])


class TestPermutationP:
    def test_a_perfect_relation_is_significant(self) -> None:
        xs = [float(index) for index in range(12)]
        ys = [float(index) for index in range(12)]
        assert permutation_p(xs, ys, seed=1, permutations=400) < 0.05

    def test_an_unrelated_pair_is_not(self) -> None:
        xs = [0.1, 0.9, 0.4, 0.6, 0.5, 0.2, 0.8, 0.3]
        ys = [0.5, 0.5, 0.5, 0.51, 0.49, 0.5, 0.5, 0.5]
        value = permutation_p(xs, ys, seed=2, permutations=400)
        assert value is not None and value > 0.05

    def test_it_is_floored_at_the_test_resolution(self) -> None:
        xs = [float(index) for index in range(20)]
        value = permutation_p(xs, list(xs), seed=3, permutations=200)
        assert value is not None and value >= 1 / 200

    def test_an_undefined_statistic_reports_none(self) -> None:
        assert permutation_p([1.0, 1.0], [1.0, 2.0], seed=1, permutations=50) is None

    def test_it_is_reproducible_from_the_seed(self) -> None:
        xs = [0.2, 0.4, 0.1, 0.9, 0.5, 0.7]
        ys = [0.3, 0.5, 0.2, 0.8, 0.4, 0.6]
        assert permutation_p(xs, ys, seed=9, permutations=300) == permutation_p(
            xs, ys, seed=9, permutations=300
        )


class TestBootstrapSeed:
    def test_the_seed_is_a_function_of_the_salt_and_the_pair(self) -> None:
        assert bootstrap_seed("salt", "C05", "C01") == bootstrap_seed("salt", "C05", "C01")

    def test_a_different_salt_gives_a_different_seed(self) -> None:
        assert bootstrap_seed("salt-a", "C05", "C01") != bootstrap_seed("salt-b", "C05", "C01")

    def test_the_order_of_the_pair_matters(self) -> None:
        """Brier(A) - Brier(B) and Brier(B) - Brier(A) are different
        comparisons and must not share a resampling stream."""
        assert bootstrap_seed("salt", "C05", "C01") != bootstrap_seed("salt", "C01", "C05")
