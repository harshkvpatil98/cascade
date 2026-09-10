"""IVFFlat index sizing (ADR-0012, spec §4.2).

The sizing rule is pure so it can be tested without a database. The cases that
matter are the degenerate ones: an empty partition, a one-row partition, and a
partition large enough to hit the configured cap.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.retrieval.index import (
    PGVECTOR_MAX_LISTS,
    index_name_for,
    plan_all,
    plan_partition,
    target_lists,
)
from cascade.retrieval.schema import PartitionIndex

MAX = 2000
TOL = 0.25


def partition(name: str, rows: int, lists: int | None = None) -> PartitionIndex:
    return PartitionIndex(
        partition=name,
        rows=rows,
        index_name=index_name_for(name) if lists is not None else None,
        lists=lists,
    )


# ---------------------------------------------------------------------------
# target_lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (0, 0),  # empty: no index at all
        (1, 1),
        (100, 10),
        (2_400, 49),  # round, not truncate: sqrt is 48.99
        (130_073, 361),  # the real 2016q4 partition
        (1_300_000, 1140),  # the study target for one partition
    ],
)
def test_target_lists_is_the_rounded_square_root(rows: int, expected: int) -> None:
    assert target_lists(rows, max_lists=MAX) == expected


def test_target_lists_rounds_rather_than_truncating() -> None:
    """Truncation biases every partition low, widening lists and costing recall."""
    assert target_lists(2_400, max_lists=MAX) == 49
    assert int(math.sqrt(2_400)) == 48


def test_target_lists_is_clamped_to_the_configured_maximum() -> None:
    assert target_lists(10**9, max_lists=MAX) == MAX


def test_target_lists_never_exceeds_pgvectors_own_ceiling() -> None:
    """`lists` above 32768 is rejected by pgvector at CREATE INDEX time."""
    assert target_lists(10**12, max_lists=10**9) == PGVECTOR_MAX_LISTS


@pytest.mark.parametrize(("rows", "max_lists"), [(-1, MAX), (10, 0)])
def test_target_lists_rejects_nonsense_inputs(rows: int, max_lists: int) -> None:
    with pytest.raises(ValueError):
        target_lists(rows, max_lists=max_lists)


@given(st.integers(min_value=1, max_value=5_000_000))
def test_target_lists_is_always_at_least_one_for_a_non_empty_partition(rows: int) -> None:
    """A non-empty partition must never be planned with lists = 0."""
    assert target_lists(rows, max_lists=MAX) >= 1


# ---------------------------------------------------------------------------
# plan_partition
# ---------------------------------------------------------------------------


def test_an_empty_partition_is_skipped_not_indexed() -> None:
    """An index built on no rows sends every later insert into one list."""
    plan = plan_partition(partition("chunks_2027q4", 0), max_lists=MAX, tolerance=TOL)
    assert plan.action == "skip-empty"
    assert plan.target_lists == 0


def test_an_unindexed_non_empty_partition_is_created() -> None:
    plan = plan_partition(partition("chunks_2016q4", 130_073), max_lists=MAX, tolerance=TOL)
    assert plan.action == "create"
    assert plan.target_lists == 361


def test_a_correctly_sized_index_is_kept() -> None:
    plan = plan_partition(
        partition("chunks_2016q4", 130_073, lists=361), max_lists=MAX, tolerance=TOL
    )
    assert plan.action == "keep"


def test_an_index_within_tolerance_is_kept_rather_than_thrashed() -> None:
    """Rebuilding a 130k-row partition on every insert would never finish."""
    plan = plan_partition(
        partition("chunks_2016q4", 130_073, lists=330), max_lists=MAX, tolerance=TOL
    )
    assert plan.action == "keep"


def test_an_index_outside_tolerance_is_rebuilt() -> None:
    """Regression: a probe sweep left the corpus at sqrt(n)/8 and it was caught."""
    plan = plan_partition(
        partition("chunks_2016q4", 130_073, lists=45), max_lists=MAX, tolerance=TOL
    )
    assert plan.action == "rebuild"
    assert plan.target_lists == 361
    assert "drift" in plan.reason


def test_drift_is_measured_against_the_target_not_the_current_value() -> None:
    """An index at half the target must rebuild; at 0.8x it must not."""
    rows = 10_000  # target 100
    assert (
        plan_partition(partition("p", rows, lists=50), max_lists=MAX, tolerance=TOL).action
        == "rebuild"
    )
    assert (
        plan_partition(partition("p", rows, lists=80), max_lists=MAX, tolerance=TOL).action
        == "keep"
    )


# ---------------------------------------------------------------------------
# plan_all
# ---------------------------------------------------------------------------


def test_plans_are_ordered_largest_partition_first() -> None:
    """An interrupted pass should already have fixed what dominates latency."""
    plans = plan_all(
        [partition("small", 10), partition("huge", 100_000), partition("mid", 5_000)],
        max_lists=MAX,
        tolerance=TOL,
    )
    assert [plan.partition for plan in plans] == ["huge", "mid", "small"]


def test_ties_break_on_name_so_the_order_is_deterministic() -> None:
    """Invariant 7: iteration order is never left to chance."""
    plans = plan_all(
        [partition("b", 100), partition("a", 100), partition("c", 100)],
        max_lists=MAX,
        tolerance=TOL,
    )
    assert [plan.partition for plan in plans] == ["a", "b", "c"]


def test_planning_is_idempotent_after_a_pass() -> None:
    """A second run must be a no-op, or the command is not safe to re-run."""
    measured = [partition("chunks_2016q4", 130_073), partition("chunks_2027q4", 0)]
    first = plan_all(measured, max_lists=MAX, tolerance=TOL)

    applied = [
        (
            partition(plan.partition, plan.rows, lists=plan.target_lists)
            if plan.action in {"create", "rebuild"}
            else partition(plan.partition, plan.rows)
        )
        for plan in first
    ]
    second = plan_all(applied, max_lists=MAX, tolerance=TOL)
    assert {plan.action for plan in second} <= {"keep", "skip-empty"}


def test_index_names_are_derived_from_the_partition() -> None:
    """`verify` must be able to say "unindexed" without a registry to consult."""
    assert index_name_for("chunks_2016q4") == "chunks_2016q4_embedding_ivfflat_idx"
    assert len(index_name_for("chunks_2016q4")) < 64  # Postgres identifier limit
