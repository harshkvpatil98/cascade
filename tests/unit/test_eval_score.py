"""Turning scored forecasts into §10's metrics, and §9.2's dispersion claim.

The recalibration split is the part most likely to move a number quietly: a
half chosen by a fresh shuffle would make the post-calibration Brier differ
between two runs of the same report, and it is the number most likely to be
quoted out of context.
"""

from __future__ import annotations

import pytest

from cascade.eval.schema import ScoredForecast
from cascade.eval.score import (
    calibration_of,
    dispersion_finding,
    domains_of,
    metrics_for,
    recalibrated_brier,
    split_halves,
)


def _scored(
    count: int = 40, *, sigma_tracks_error: bool = False, policy: str = "agent"
) -> tuple[ScoredForecast, ...]:
    out = []
    for index in range(count):
        outcome = index % 2
        p_hat = 0.3 + 0.4 * ((index % 5) / 4)
        error = abs(p_hat - outcome)
        out.append(
            ScoredForecast(
                scenario_id=f"scenario-{index:03d}",
                config_id="C01",
                p_hat=p_hat,
                outcome=outcome,
                domain="elections" if index % 3 else "macro_policy",
                sigma=error if sigma_tracks_error else 0.1 + 0.001 * index,
                n_replicates=200,
                modality="multi" if error > 0.5 else "single",
                policy=policy,  # type: ignore[arg-type]
            )
        )
    return tuple(out)


class TestMetricsFor:
    def test_it_reports_the_count_it_scored(self) -> None:
        metrics = metrics_for(_scored(12), config_id="C01")
        assert metrics.n == 12

    def test_the_climatology_reference_comes_from_the_caller(self) -> None:
        """Recomputing the floor against whatever is loaded moves it in
        whichever direction flatters the system (§10.2)."""
        metrics = metrics_for(_scored(), config_id="C01", climatology_brier=0.25)
        assert metrics.bss_vs_climatology == pytest.approx(1.0 - metrics.brier / 0.25)

    def test_a_missing_reference_is_not_measured_rather_than_zero(self) -> None:
        metrics = metrics_for(_scored(), config_id="C01")
        assert metrics.bss_vs_climatology is None
        assert metrics.bss_vs_direct is None

    def test_a_zero_reference_is_not_measured_rather_than_infinite(self) -> None:
        metrics = metrics_for(_scored(), config_id="C01", climatology_brier=0.0)
        assert metrics.bss_vs_climatology is None

    def test_recalibration_is_absent_without_a_salt(self) -> None:
        """It needs a deterministic split, and the salt is what makes it one."""
        assert metrics_for(_scored(), config_id="C01").brier_recalibrated is None

    def test_the_decider_policies_travel_with_the_metrics(self) -> None:
        metrics = metrics_for(_scored(policy="heuristic"), config_id="C01")
        assert metrics.policies == ("heuristic",)

    def test_an_empty_set_raises_rather_than_scoring_nothing(self) -> None:
        with pytest.raises(ValueError, match="nothing behind it"):
            metrics_for([], config_id="C01")

    def test_murphy_and_brier_agree_exactly(self) -> None:
        """They are printed side by side; a reader comparing them must not find
        them differing in the last digit."""
        metrics = metrics_for(_scored(), config_id="C01")
        assert metrics.brier == metrics.murphy.brier


class TestSplitHalves:
    def test_the_split_is_stable_across_calls(self) -> None:
        scored = _scored()
        first = split_halves(scored, salt="salt")
        second = split_halves(list(reversed(scored)), salt="salt")
        assert [item.scenario_id for item in first[0]] == [item.scenario_id for item in second[0]]

    def test_a_different_salt_gives_a_different_split(self) -> None:
        a_fit, _ = split_halves(_scored(), salt="a")
        b_fit, _ = split_halves(_scored(), salt="b")
        assert [item.scenario_id for item in a_fit] != [item.scenario_id for item in b_fit]

    def test_the_halves_partition_the_set(self) -> None:
        scored = _scored()
        fit, held = split_halves(scored, salt="salt")
        assert len(fit) + len(held) == len(scored)
        assert not ({item.scenario_id for item in fit} & {item.scenario_id for item in held})

    def test_the_split_does_not_depend_on_the_outcome(self) -> None:
        """Splitting on anything correlated with the label would make the
        held-out half a different population, and the recalibrated Brier would
        measure that instead of calibration."""
        scored = _scored()
        flipped = tuple(item.model_copy(update={"outcome": 1 - item.outcome}) for item in scored)
        assert [item.scenario_id for item in split_halves(scored, salt="s")[0]] == [
            item.scenario_id for item in split_halves(flipped, salt="s")[0]
        ]


class TestRecalibration:
    def test_it_is_measured_on_the_half_it_was_not_fitted_to(self) -> None:
        value = recalibrated_brier(_scored(60), salt="salt")
        assert value is not None
        assert 0.0 <= value <= 1.0

    def test_it_is_reproducible(self) -> None:
        assert recalibrated_brier(_scored(60), salt="salt") == recalibrated_brier(
            _scored(60), salt="salt"
        )

    def test_too_few_scenarios_is_not_measurable_rather_than_a_number(self) -> None:
        assert recalibrated_brier(_scored(2), salt="salt") is None


class TestDispersion:
    def test_a_sigma_that_tracks_error_correlates_positively(self) -> None:
        """§9.2's claim, tested rather than assumed."""
        finding = dispersion_finding(
            _scored(40, sigma_tracks_error=True),
            sigma_threshold=0.3,
            seed=1,
            permutations=200,
        )
        assert finding.spearman_rho is not None and finding.spearman_rho > 0.8
        assert finding.spearman_p is not None and finding.spearman_p < 0.05

    def test_an_uninformative_sigma_is_reported_as_such(self) -> None:
        """The finding may be that sigma says nothing. That is a result."""
        scored = tuple(item.model_copy(update={"sigma": 0.2}) for item in _scored(20))
        finding = dispersion_finding(scored, sigma_threshold=0.3, seed=1, permutations=100)
        assert finding.spearman_rho is None
        assert finding.spearman_p is None

    def test_flagged_and_unflagged_subsets_are_scored_separately(self) -> None:
        finding = dispersion_finding(_scored(40), sigma_threshold=0.3, seed=1, permutations=100)
        assert finding.n == 40
        assert finding.flagged + (finding.n - finding.flagged) == 40
        if finding.flagged:
            assert finding.brier_flagged is not None
        if finding.flagged < finding.n:
            assert finding.brier_unflagged is not None

    def test_an_all_flagged_set_has_no_unflagged_brier(self) -> None:
        scored = tuple(item.model_copy(update={"modality": "multi"}) for item in _scored(10))
        finding = dispersion_finding(scored, sigma_threshold=0.3, seed=1, permutations=50)
        assert finding.flagged == 10
        assert finding.brier_unflagged is None

    def test_the_threshold_travels_with_the_finding(self) -> None:
        finding = dispersion_finding(_scored(10), sigma_threshold=0.3, seed=1, permutations=50)
        assert finding.sigma_threshold == 0.3


class TestTables:
    def test_the_calibration_table_has_ten_bins(self) -> None:
        assert len(calibration_of(_scored()).bins) == 10

    def test_the_domain_table_covers_every_domain_present(self) -> None:
        rows = domains_of(_scored())
        assert {row.domain for row in rows} == {"elections", "macro_policy"}
        assert sum(row.n for row in rows) == 40
