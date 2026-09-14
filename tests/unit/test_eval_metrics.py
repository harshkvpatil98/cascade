"""Assay's scoring rules, against closed forms (spec §10.1, §10.5).

Every function here is the study's headline arithmetic, so it is tested
against values computable by hand rather than against a golden file. A golden
file records what the code did; a closed form records what the answer is.
"""

from __future__ import annotations

import math

import pytest

from cascade.eval.metrics import (
    auc,
    brier,
    brier_skill_score,
    calibration,
    isotonic_fit,
    isotonic_predict,
    log_loss,
    murphy_decomposition,
    per_domain,
    wilson_interval,
)


class TestBrier:
    def test_a_perfect_forecast_scores_zero(self) -> None:
        assert brier([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_a_maximally_wrong_forecast_scores_one(self) -> None:
        assert brier([0.0, 1.0], [1, 0]) == 1.0

    def test_a_constant_half_scores_a_quarter(self) -> None:
        assert brier([0.5] * 4, [1, 0, 1, 0]) == 0.25

    def test_the_base_rate_forecast_scores_p_times_one_minus_p(self) -> None:
        """The identity M1's climatology relies on and M7's UNC term reuses."""
        outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        base = sum(outcomes) / len(outcomes)
        assert brier([base] * len(outcomes), outcomes) == pytest.approx(base * (1 - base))

    def test_it_agrees_with_the_m1_implementation(self) -> None:
        """Two independent implementations of the headline metric must agree.

        M1 computes the climatology Brier in pure Python from the definition
        and seals it with the split. M7 computes every Brier in numpy. They are
        the same number and the study quotes both.
        """
        from cascade.ledger.climatology import brier_score

        forecasts = (0.12, 0.83, 0.5, 0.61, 0.04)
        outcomes = (0, 1, 1, 0, 0)
        assert brier(list(forecasts), list(outcomes)) == pytest.approx(
            brier_score(forecasts, outcomes), abs=1e-15
        )

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            brier([0.5], [1, 0])

    def test_an_empty_set_raises(self) -> None:
        with pytest.raises(ValueError, match="empty set"):
            brier([], [])

    def test_a_forecast_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier([1.5], [1])

    def test_a_non_binary_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier([0.5], [2])


class TestSkillScore:
    def test_matching_the_reference_scores_zero_skill(self) -> None:
        assert brier_skill_score(0.25, 0.25) == 0.0

    def test_the_sign_of_the_claim(self) -> None:
        """-30.5% in the spec's framing is +0.305 skill; the sign must not flip."""
        assert brier_skill_score(0.141, 0.203) == pytest.approx(0.3054187, abs=1e-6)

    def test_a_perfect_reference_is_undefined_not_infinite(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            brier_skill_score(0.1, 0.0)


class TestLogLoss:
    def test_it_is_the_clipped_definition(self) -> None:
        value = log_loss([0.9, 0.2], [1, 0])
        expected = -(math.log(0.9) + math.log(0.8)) / 2
        assert value == pytest.approx(expected)

    def test_the_clip_bounds_a_confident_miss(self) -> None:
        """Unclipped this is infinite, and one scenario would own the metric."""
        assert log_loss([0.0], [1]) == pytest.approx(-math.log(0.01))

    def test_the_clip_is_stated_and_changeable(self) -> None:
        assert log_loss([0.0], [1], clip=0.001) == pytest.approx(-math.log(0.001))

    def test_a_nonsensical_clip_raises(self) -> None:
        with pytest.raises(ValueError, match="clip"):
            log_loss([0.5], [1], clip=0.7)


class TestAUC:
    def test_perfect_ranking_scores_one(self) -> None:
        assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_inverted_ranking_scores_zero(self) -> None:
        assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0

    def test_a_constant_forecast_scores_one_half(self) -> None:
        """Mid-ranks are why. Climatology is a constant forecast, and an AUC
        that depended on sort order would make the floor look like a signal."""
        assert auc([0.5] * 6, [1, 0, 1, 0, 1, 0]) == 0.5

    def test_one_class_absent_is_not_measurable(self) -> None:
        assert auc([0.2, 0.8], [1, 1]) is None
        assert auc([0.2, 0.8], [0, 0]) is None

    def test_it_is_invariant_under_a_monotone_transform(self) -> None:
        forecasts = [0.1, 0.35, 0.6, 0.77]
        outcomes = [0, 1, 0, 1]
        squashed = [value**3 for value in forecasts]
        assert auc(forecasts, outcomes) == auc(squashed, outcomes)


class TestWilson:
    def test_it_matches_the_closed_form_evaluated_independently(self) -> None:
        """9 successes in 20 trials, computed here from the formula itself.

        Written out rather than quoted so the expected value is derived from
        Wilson's definition and not from a remembered table -- a remembered
        table is how a rounded constant becomes an assertion.
        """
        p, n, z = 9 / 20, 20, 1.959963984540054
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

        lo, hi = wilson_interval(9, 20)
        assert lo == pytest.approx(centre - half, abs=1e-12)
        assert hi == pytest.approx(centre + half, abs=1e-12)
        # And the interval is where a reader would expect it to be.
        assert (lo, hi) == pytest.approx((0.25820, 0.65791), abs=5e-5)

    def test_it_stays_inside_the_unit_interval_at_the_extremes(self) -> None:
        """Where the normal approximation degenerates and §10.5's bins live."""
        lo, hi = wilson_interval(0, 5)
        assert lo == 0.0
        assert 0.0 < hi < 1.0
        lo, hi = wilson_interval(5, 5)
        assert hi == 1.0
        assert 0.0 < lo < 1.0

    def test_more_trials_narrow_the_interval(self) -> None:
        narrow = wilson_interval(50, 100)
        wide = wilson_interval(5, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_impossible_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="impossible"):
            wilson_interval(6, 5)
        with pytest.raises(ValueError, match="undefined"):
            wilson_interval(0, 0)


class TestCalibration:
    def test_every_bin_appears_including_the_empty_ones(self) -> None:
        """§10.5's point is that the interesting bins may hold nine scenarios.

        Dropping empty bins hides the same thing a missing count hides.
        """
        report = calibration([0.45, 0.55], [0, 1])
        assert len(report.bins) == 10
        assert [row.count for row in report.bins] == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

    def test_a_forecast_of_exactly_one_lands_in_the_top_bin(self) -> None:
        report = calibration([1.0], [1])
        assert report.bins[9].count == 1

    def test_a_perfectly_calibrated_set_has_zero_ece(self) -> None:
        forecasts = [0.25] * 4 + [0.75] * 4
        outcomes = [1, 0, 0, 0, 1, 1, 1, 0]
        report = calibration(forecasts, outcomes)
        assert report.ece == pytest.approx(0.0)
        assert report.mce == pytest.approx(0.0)

    def test_ece_is_recomputable_from_the_printed_table(self) -> None:
        """The property that makes the table worth printing."""
        forecasts = [0.05, 0.15, 0.45, 0.65, 0.85, 0.95, 0.35, 0.55]
        outcomes = [0, 0, 1, 0, 1, 1, 0, 1]
        report = calibration(forecasts, outcomes)
        recomputed = sum(
            (row.count / report.n) * abs(row.mean_pred - row.obs_freq)
            for row in report.bins
            if row.count and row.mean_pred is not None and row.obs_freq is not None
        )
        assert report.ece == pytest.approx(recomputed)

    def test_mce_is_the_largest_gap(self) -> None:
        report = calibration([0.05, 0.95], [1, 1])
        assert report.mce == pytest.approx(0.95)

    def test_zero_bins_raise(self) -> None:
        with pytest.raises(ValueError, match="bins must be positive"):
            calibration([0.5], [1], bins=0)


class TestMurphy:
    def test_the_identity_is_exact_when_forecasts_sit_on_bin_means(self) -> None:
        """Grouping by equal value makes BS = REL - RES + UNC exactly."""
        forecasts = [0.25] * 4 + [0.75] * 4
        outcomes = [1, 0, 0, 0, 1, 1, 1, 0]
        terms = murphy_decomposition(forecasts, outcomes)
        assert terms.residual == pytest.approx(0.0, abs=1e-12)
        assert terms.brier == pytest.approx(
            terms.reliability - terms.resolution + terms.uncertainty
        )

    def test_the_binning_residual_is_reported_not_absorbed(self) -> None:
        """A spread of forecasts inside one bin leaves a cross term.

        Reporting it is the difference between a decomposition a reader can
        check and three numbers printed near a fourth.
        """
        forecasts = [0.41, 0.49, 0.42, 0.48, 0.44]
        outcomes = [1, 0, 1, 0, 1]
        terms = murphy_decomposition(forecasts, outcomes)
        assert terms.residual != 0.0
        assert terms.brier == pytest.approx(
            terms.reliability - terms.resolution + terms.uncertainty + terms.residual
        )

    def test_uncertainty_is_the_climatology_brier(self) -> None:
        outcomes = [1, 1, 0, 0, 0]
        terms = murphy_decomposition([0.5] * 5, outcomes)
        base = sum(outcomes) / len(outcomes)
        assert terms.uncertainty == pytest.approx(base * (1 - base))

    def test_a_constant_forecast_has_no_resolution(self) -> None:
        terms = murphy_decomposition([0.4] * 6, [1, 0, 1, 0, 1, 0])
        assert terms.resolution == pytest.approx(0.0)


class TestPerDomain:
    def test_it_reports_counts_alongside_every_brier(self) -> None:
        rows = per_domain([0.9, 0.1, 0.5], [1, 0, 1], ["elections", "elections", "macro_policy"])
        assert [(row.domain, row.n) for row in rows] == [("elections", 2), ("macro_policy", 1)]
        assert rows[0].brier == pytest.approx(0.01)

    def test_domains_come_back_sorted(self) -> None:
        rows = per_domain([0.5] * 3, [1, 0, 1], ["z", "a", "m"])
        assert [row.domain for row in rows] == ["a", "m", "z"]

    def test_a_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="domain/forecast"):
            per_domain([0.5, 0.5], [1, 0], ["only_one"])


class TestIsotonic:
    def test_an_already_monotone_fit_is_the_identity_on_its_knots(self) -> None:
        thresholds, values = isotonic_fit([0.1, 0.4, 0.9], [0, 1, 1])
        assert list(values) == sorted(values)
        assert isotonic_predict(thresholds, values, [0.1]) == (0.0,)

    def test_violations_are_pooled(self) -> None:
        """The whole algorithm: a decreasing pair becomes one block at its mean."""
        _, values = isotonic_fit([0.2, 0.4], [1, 0])
        assert values == (0.5,)

    def test_the_fit_is_non_decreasing(self) -> None:
        _, values = isotonic_fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])
        assert list(values) == sorted(values)

    def test_prediction_clamps_outside_the_training_support(self) -> None:
        """Extrapolating would invent calibration where nothing was observed."""
        thresholds, values = isotonic_fit([0.3, 0.7], [0, 1])
        assert isotonic_predict(thresholds, values, [0.0])[0] == values[0]
        assert isotonic_predict(thresholds, values, [1.0])[0] == values[-1]

    def test_recalibration_cannot_worsen_the_brier_on_its_own_fitting_set(self) -> None:
        """Isotonic regression minimises squared error under monotonicity.

        On the set it was fitted to it is therefore never worse than the raw
        forecasts -- which is exactly why §10.5 requires a held-out half and
        forbids reporting the recalibrated figure as the headline.
        """
        forecasts = [0.1, 0.3, 0.35, 0.6, 0.62, 0.9]
        outcomes = [0, 1, 0, 1, 0, 1]
        thresholds, values = isotonic_fit(forecasts, outcomes)
        mapped = isotonic_predict(thresholds, values, forecasts)
        assert brier(list(mapped), outcomes) <= brier(forecasts, outcomes) + 1e-12

    def test_equal_forecasts_are_pooled_before_the_sweep(self) -> None:
        """The regression a property test found.

        Unpooled, `p = [0.0, 0.5, 0.5]` with `y = [0, 0, 1]` produced two
        blocks both ending at x = 0.5 with values 0.0 and 1.0 -- a step
        function that is two-valued at 0.5. `searchsorted` then took the 0.0
        branch for both, mapped every point to 0.0, and *doubled* the Brier on
        the fit's own training set, which is the one thing isotonic regression
        cannot do.
        """
        forecasts = [0.0, 0.5, 0.5]
        outcomes = [0, 0, 1]
        thresholds, values = isotonic_fit(forecasts, outcomes)
        assert list(thresholds) == sorted(set(thresholds))
        mapped = isotonic_predict(thresholds, values, forecasts)
        assert mapped[1] == mapped[2]
        assert brier(list(mapped), outcomes) <= brier(forecasts, outcomes) + 1e-12

    def test_thresholds_are_strictly_increasing(self) -> None:
        """Which is what makes prediction at a knot well defined."""
        thresholds, _ = isotonic_fit([0.2, 0.2, 0.4, 0.4, 0.9], [1, 0, 0, 1, 1])
        assert list(thresholds) == sorted(set(thresholds))

    def test_a_mismatched_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold/value"):
            isotonic_predict([0.1], [0.0, 1.0], [0.5])

    def test_applying_an_empty_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="no blocks"):
            isotonic_predict([], [], [0.5])
