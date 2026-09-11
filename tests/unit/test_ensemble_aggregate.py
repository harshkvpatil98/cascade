"""Collapsing replicates into a forecast (spec §9.1, §9.3)."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.ensemble.aggregate import (
    bootstrap_percentile,
    collapse,
    convergence_curve,
    standard_error,
)

SALT = "cascade-2026-study-01"


def collapsed(scores: list[float], **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "scenario_id": "s1",
        "config_id": "base",
        "salt": SALT,
        "sigma_threshold": 0.30,
        "bootstrap_b": 500,
        "dip_samples": 200,
    }
    kwargs.update(overrides)
    return collapse(scores, **kwargs)  # type: ignore[arg-type]


def test_the_forecast_is_the_mean_not_the_median() -> None:
    """§9.1 is explicit: the mean is what the Brier score is defined against."""
    scores = [0.0] * 60 + [1.0] * 40
    forecast = collapsed(scores)
    assert forecast.p_hat == pytest.approx(0.4)  # type: ignore[attr-defined]
    assert forecast.p_hat != np.median(scores)  # type: ignore[attr-defined]


def test_sigma_is_the_sample_standard_deviation() -> None:
    rng = np.random.Generator(np.random.PCG64(3))
    scores = list(rng.normal(0.5, 0.1, 200).clip(0, 1))
    forecast = collapsed(scores)
    assert forecast.sigma == pytest.approx(float(np.std(scores, ddof=1)))  # type: ignore[attr-defined]


def test_a_dispersed_ensemble_is_flagged_multi_modal() -> None:
    """Either arm of §9.1's disjunction flags it; this sample trips both."""
    scores = [0.02] * 100 + [0.98] * 100
    forecast = collapsed(scores)
    assert forecast.modality == "multi"  # type: ignore[attr-defined]
    assert forecast.sigma > 0.30  # type: ignore[attr-defined]
    assert forecast.dip_p < 0.05  # type: ignore[attr-defined]


def test_a_tight_ensemble_is_single_modal() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    forecast = collapsed(list(rng.normal(0.6, 0.05, 200).clip(0, 1)))
    assert forecast.modality == "single"  # type: ignore[attr-defined]


def test_the_dip_alone_can_flag_a_low_sigma_ensemble() -> None:
    """Two tight clusters close together: sigma stays under 0.30, the dip does not.

    This is the case the second statistic exists for -- a threshold on sigma
    alone would call it unimodal.
    """
    rng = np.random.Generator(np.random.PCG64(13))
    scores = list(np.concatenate([rng.normal(0.40, 0.01, 100), rng.normal(0.60, 0.01, 100)]))
    forecast = collapsed(scores)
    assert forecast.sigma < 0.30  # type: ignore[attr-defined]
    assert forecast.dip_p < 0.05  # type: ignore[attr-defined]
    assert forecast.modality == "multi"  # type: ignore[attr-defined]


def test_the_bimodality_coefficient_is_recorded_but_does_not_decide() -> None:
    """§9.2 asks for it as corroboration; a third silent vote would change the flag."""
    rng = np.random.Generator(np.random.PCG64(17))
    scores = list(rng.random(200))  # uniform: BC ~0.56, above the 0.555 line
    forecast = collapsed(scores)
    assert forecast.bimodality > 0.5  # type: ignore[attr-defined]
    assert forecast.modality == "single"  # type: ignore[attr-defined]


def test_the_bootstrap_is_seeded_by_scenario_and_config() -> None:
    """A CI that moved between processes would make the report irreproducible."""
    rng = np.random.Generator(np.random.PCG64(19))
    scores = list(rng.normal(0.5, 0.15, 200).clip(0, 1))
    first = collapsed(scores)
    second = collapsed(scores)
    other = collapsed(scores, config_id="C04")
    assert (first.ci_lo, first.ci_hi) == (second.ci_lo, second.ci_hi)  # type: ignore[attr-defined]
    assert (first.ci_lo, first.ci_hi) != (other.ci_lo, other.ci_hi)  # type: ignore[attr-defined]


def test_the_interval_brackets_the_mean_and_stays_in_the_unit_interval() -> None:
    """Percentile rather than normal: a bounded score must not get a CI outside [0, 1]."""
    scores = [0.0] * 195 + [1.0] * 5
    forecast = collapsed(scores)
    assert 0.0 <= forecast.ci_lo <= forecast.p_hat <= forecast.ci_hi <= 1.0  # type: ignore[attr-defined]


def test_the_interval_narrows_as_replicates_accumulate() -> None:
    rng = np.random.Generator(np.random.PCG64(23))
    scores = list(rng.normal(0.5, 0.2, 800).clip(0, 1))
    narrow = collapsed(scores)
    wide = collapsed(scores[:25])
    assert narrow.ci_width < wide.ci_width  # type: ignore[attr-defined]


def test_collapsing_nothing_is_an_error_not_a_forecast() -> None:
    with pytest.raises(ValueError, match="no replicates"):
        collapsed([])


def test_standard_error_is_sigma_over_root_n() -> None:
    """§9.3 chooses 200 against this quantity, so it is computed and not assumed."""
    rng = np.random.Generator(np.random.PCG64(29))
    scores = list(rng.normal(0.5, 0.35, 200))
    assert standard_error(scores) == pytest.approx(float(np.std(scores, ddof=1)) / np.sqrt(200))
    assert standard_error([0.5]) == 0.0


def test_the_convergence_curve_uses_replicate_prefixes() -> None:
    """Replicate k is a fixed seeded world, so the first n are a reproducible ensemble."""
    rng = np.random.Generator(np.random.PCG64(31))
    scores = {name: list(rng.normal(0.5, 0.2, 400).clip(0, 1)) for name in ("a", "b")}
    curve = convergence_curve(scores)
    assert [point.n for point in curve] == [25, 50, 100, 200, 400]
    assert all(point.scenarios == 2 for point in curve)
    assert convergence_curve(scores) == curve


def test_the_convergence_curve_skips_rungs_a_scenario_cannot_reach() -> None:
    curve = convergence_curve({"a": [0.5] * 60, "b": [0.4] * 400})
    by_n = {point.n: point for point in curve}
    assert by_n[50].scenarios == 2
    assert by_n[100].scenarios == 1


def test_the_bootstrap_of_a_single_score_is_that_score() -> None:
    assert bootstrap_percentile([0.7], seed=1, b=10) == (0.7, 0.7)
