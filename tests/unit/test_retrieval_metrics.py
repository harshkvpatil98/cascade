"""Pure measurement functions for the M3 bench (spec §4.2).

The acceptance criteria are stated to three significant figures, so an
unstated percentile or recall definition is not a reproducible criterion.
These tests pin both definitions against hand-computable cases.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.retrieval.metrics import (
    histogram,
    percentile,
    recall_at_k,
    summarise_latency,
)

# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_matches_numpy_linear_interpolation() -> None:
    """The definition the criteria are read against, asserted against numpy.

    numpy is a pinned dependency, so this is a cross-check rather than an
    implementation: if the two ever disagree, the reported p95 has moved.
    """
    import numpy as np

    samples = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0]
    for q in (0.0, 25.0, 50.0, 90.0, 95.0, 99.0, 100.0):
        assert percentile(samples, q) == pytest.approx(float(np.percentile(samples, q)))


def test_percentile_interpolates_between_ranks() -> None:
    """p50 of an even-length sample is the mean of the middle two, not either."""
    assert percentile([0.0, 10.0], 50.0) == pytest.approx(5.0)


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    assert percentile([7.5], 95.0) == pytest.approx(7.5)


def test_percentile_of_an_empty_sample_raises() -> None:
    """Returning 0.0 would read as a passing latency on a bench that ran nothing."""
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 95.0)


@pytest.mark.parametrize("q", [-1.0, 100.1])
def test_percentile_rejects_out_of_range_q(q: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1.0, 2.0], q)


@given(st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=200))
def test_percentiles_are_monotonic_in_q(samples: list[float]) -> None:
    """p50 <= p95 <= p99 always, whatever the sample."""
    assert percentile(samples, 50.0) <= percentile(samples, 95.0) <= percentile(samples, 99.0)


@given(st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=200))
def test_percentiles_lie_within_the_sample_range(samples: list[float]) -> None:
    assert min(samples) <= percentile(samples, 95.0) <= max(samples)


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


def test_histogram_counts_sum_to_the_sample_size() -> None:
    """Nothing may be silently dropped -- the top bucket is unbounded."""
    samples = [0.5, 1.5, 14.9, 15.0, 99.0, 1000.0]
    assert sum(count for _, _, count in histogram(samples)) == len(samples)


def test_histogram_buckets_are_half_open_upward() -> None:
    """A value on an edge lands in exactly one bucket."""
    buckets = {(low, high): count for low, high, count in histogram([15.0])}
    assert buckets[(12.5, 15.0)] == 0
    assert buckets[(15.0, 20.0)] == 1


def test_histogram_needs_at_least_two_edges() -> None:
    with pytest.raises(ValueError, match="two edges"):
        histogram([1.0], edges=[0.0])


def test_summarise_latency_of_nothing_reports_n_zero() -> None:
    """A caller must be able to tell "fast" from "did not run"."""
    summary = summarise_latency([])
    assert summary.n == 0
    assert summary.p95 == 0.0


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


def test_recall_is_the_intersection_over_the_ground_truth() -> None:
    approximate = ["a", "b", "x", "y"]
    exact = ["a", "b", "c", "d"]
    assert recall_at_k([(approximate, exact)], k=4).mean == pytest.approx(0.5)


def test_recall_ignores_rank_order_within_k() -> None:
    """recall@20 is a set measure; requiring order would be a different metric."""
    assert recall_at_k([(["c", "b", "a"], ["a", "b", "c"])], k=3).mean == pytest.approx(1.0)


def test_recall_truncates_both_sides_to_k() -> None:
    """An approximate result must not be credited for hits outside the window."""
    approximate = ["z", "a", "b"]
    exact = ["a", "b", "c"]
    # At k=1 only "z" vs "a" is compared, so nothing is found.
    assert recall_at_k([(approximate, exact)], k=1).mean == pytest.approx(0.0)


def test_queries_with_no_ground_truth_are_counted_not_scored() -> None:
    """Scoring them 1.0 would let an empty corpus report perfect recall."""
    summary = recall_at_k([(["a"], []), (["a"], ["a"])], k=20)
    assert summary.empty_ground_truth == 1
    assert summary.scored == 1
    assert summary.mean == pytest.approx(1.0)


def test_recall_reports_not_measured_when_nothing_could_be_scored() -> None:
    """0.0 must not be readable as a measured result."""
    summary = recall_at_k([(["a"], []), (["b"], [])], k=20)
    assert summary.scored == 0
    assert summary.measured is False


def test_recall_tracks_the_worst_query_and_the_perfect_count() -> None:
    summary = recall_at_k(
        [(["a", "b"], ["a", "b"]), (["a", "z"], ["a", "b"])],
        k=2,
    )
    assert summary.perfect == 1
    assert summary.minimum == pytest.approx(0.5)
    assert summary.mean == pytest.approx(0.75)


def test_recall_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k([], k=0)


@given(
    st.lists(st.sampled_from("abcdefghij"), min_size=1, max_size=10, unique=True),
    st.lists(st.sampled_from("abcdefghij"), min_size=1, max_size=10, unique=True),
)
def test_recall_is_always_a_probability(approximate: list[str], exact: list[str]) -> None:
    assert 0.0 <= recall_at_k([(approximate, exact)], k=10).mean <= 1.0


# ---------------------------------------------------------------------------
# The bench verdict
# ---------------------------------------------------------------------------


def _result(**overrides):  # type: ignore[no-untyped-def]
    from cascade.retrieval.bench import BenchResult
    from cascade.retrieval.metrics import RecallSummary, summarise_latency

    defaults = dict(
        queries=10_000,
        k=20,
        latency=summarise_latency([10.0] * 100),
        recall=RecallSummary(
            k=20, scored=500, empty_ground_truth=0, mean=0.94, minimum=0.5, perfect=300
        ),
        empty_results=0,
        distinct_cutoffs=180,
        earliest_cutoff="2018-01-12T00:00:00+00:00",
        latest_cutoff="2026-07-19T00:00:00+00:00",
        probes=40,
        elapsed_s=150.0,
    )
    defaults.update(overrides)
    return BenchResult(**defaults)  # type: ignore[arg-type]


def test_a_result_meeting_both_criteria_passes(settings) -> None:  # type: ignore[no-untyped-def]
    passed, failures = _result().meets(settings)
    assert passed and failures == ()


def test_a_p95_over_budget_fails_with_the_measured_value(settings) -> None:  # type: ignore[no-untyped-def]
    """Never a tolerance. 15.4 ms against a 15 ms budget failed at 15.4 ms."""
    passed, failures = _result(latency=summarise_latency([15.4] * 100)).meets(settings)
    assert not passed
    assert any("15.40 ms" in failure for failure in failures)


def test_a_tiny_recall_sample_reports_the_sample_not_a_verdict(settings) -> None:  # type: ignore[no-untyped-def]
    """`--queries 100` samples 5 queries; one miss moves the mean by 0.04.

    Reporting "recall 0.9000 is not above 0.92" from that would be a verdict
    the measurement cannot support, and would read as a real regression.
    """
    from cascade.retrieval.metrics import RecallSummary

    passed, failures = _result(
        recall=RecallSummary(
            k=20, scored=5, empty_ground_truth=0, mean=0.90, minimum=0.5, perfect=2
        )
    ).meets(settings)
    assert not passed, "a sample too small to decide must still not pass"
    assert any("too few to decide" in failure for failure in failures)


def test_an_unmeasurable_recall_is_not_reported_as_zero(settings) -> None:  # type: ignore[no-untyped-def]
    from cascade.retrieval.metrics import RecallSummary

    passed, failures = _result(
        recall=RecallSummary(
            k=20, scored=0, empty_ground_truth=500, mean=0.0, minimum=0.0, perfect=0
        )
    ).meets(settings)
    assert not passed
    assert any("could not be measured" in failure for failure in failures)
