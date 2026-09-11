"""The dip test and the bimodality coefficient (spec §9.2).

§9.2 makes a claim these statistics have to earn: that sigma is *informative* --
that the system can tell you in advance which of its own forecasts to distrust.
A dip test that did not separate a bimodal ensemble from a unimodal one would
make that claim unfalsifiable, so the cases below are the ones with known
answers rather than the ones that are convenient to generate.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.ensemble.dip import (
    BIMODALITY_THRESHOLD,
    bimodality_coefficient,
    dip_p_value,
    dip_statistic,
    dip_test,
)

SALT = "cascade-2026-study-01"


def test_a_two_point_split_at_the_bounds_is_the_maximum_dip() -> None:
    """0.25 is the largest dip attainable, and this is the sample that attains it."""
    assert dip_statistic([0.0] * 100 + [1.0] * 100) == pytest.approx(0.25, abs=1e-9)


def test_a_uniform_grid_dips_by_one_over_two_n() -> None:
    """The closed-form case: an exactly uniform sample of size n dips 1/(2n).

    It is the sharpest available check that the statistic is calibrated and not
    merely monotone in "spread".
    """
    for n in (50, 100, 200):
        assert dip_statistic(np.linspace(0.0, 1.0, n)) == pytest.approx(1.0 / (2 * n), abs=1e-9)


def test_a_degenerate_or_tiny_sample_dips_zero() -> None:
    """Fewer than four points cannot evidence multimodality, so it claims none."""
    assert dip_statistic([0.5] * 40) == 0.0
    assert dip_statistic([0.1, 0.5, 0.9]) == 0.0
    assert dip_statistic([]) == 0.0


def test_a_bimodal_sample_dips_far_more_than_a_unimodal_one() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    unimodal = rng.normal(0.5, 0.12, 200)
    bimodal = np.concatenate([rng.normal(0.2, 0.04, 100), rng.normal(0.8, 0.04, 100)])
    assert dip_statistic(bimodal) > 5 * dip_statistic(unimodal)


def test_the_p_value_separates_the_two_cases() -> None:
    """The decision §9.1 makes is p < 0.05; this is that decision, both ways."""
    rng = np.random.Generator(np.random.PCG64(11))
    unimodal = list(rng.normal(0.5, 0.12, 200))
    bimodal = list(np.concatenate([rng.normal(0.2, 0.04, 100), rng.normal(0.8, 0.04, 100)]))
    assert dip_test(unimodal, salt=SALT, samples=300).p > 0.05
    assert dip_test(bimodal, salt=SALT, samples=300).p < 0.05


def test_the_p_value_is_never_exactly_zero() -> None:
    """300 draws do not contain the certainty a p of 0 would claim."""
    result = dip_test([0.0] * 100 + [1.0] * 100, salt=SALT, samples=300)
    assert 0.0 < result.p <= 1.0


def test_the_null_is_seeded_and_reproducible() -> None:
    """A p-value that moved between processes would make the report irreproducible."""
    assert dip_p_value(0.03, 200, salt=SALT, samples=300) == dip_p_value(
        0.03, 200, salt=SALT, samples=300
    )
    assert dip_p_value(0.03, 200, salt=SALT, samples=300) != dip_p_value(
        0.03, 200, salt="other-salt", samples=300
    )


def test_bimodality_coefficient_matches_its_reference_values() -> None:
    """BC is 5/9 for a uniform and 1/3 for a normal -- which is why 0.555 is the line."""
    rng = np.random.Generator(np.random.PCG64(5))
    assert bimodality_coefficient(rng.random(20_000)) == pytest.approx(5 / 9, abs=0.02)
    assert bimodality_coefficient(rng.normal(0.0, 1.0, 20_000)) == pytest.approx(1 / 3, abs=0.02)
    assert BIMODALITY_THRESHOLD == 0.555


def test_bimodality_coefficient_flags_a_two_cluster_sample() -> None:
    rng = np.random.Generator(np.random.PCG64(9))
    bimodal = np.concatenate([rng.normal(0.2, 0.04, 500), rng.normal(0.8, 0.04, 500)])
    assert bimodality_coefficient(bimodal) > BIMODALITY_THRESHOLD


def test_bimodality_coefficient_is_zero_for_a_constant() -> None:
    """No spread is no shape; dividing by a zero variance would flag a constant."""
    assert bimodality_coefficient([0.4] * 50) == 0.0
