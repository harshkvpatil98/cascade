"""HNSW index planning (ADR-0012, ADR-0026, spec §4.2).

The planning rule is pure so it can be tested without a database. What matters
here is different from what mattered under IVFFlat: there is no row-dependent
build parameter any more, so the interesting cases are the degenerate ones (an
empty partition, an unindexed one) and the one remaining rebuild trigger (the
configured parameters changed). A partition that has merely *grown* must plan
as `keep` -- that is the drift class ADR-0026 retires, and a test is what keeps
it retired.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.retrieval.index import (
    PGVECTOR_MAX_EF_CONSTRUCTION,
    PGVECTOR_MAX_M,
    index_name_for,
    plan_all,
    plan_partition,
)
from cascade.retrieval.schema import PartitionIndex

M = 16
EFC = 64


def partition(
    name: str, rows: int, m: int | None = None, ef_construction: int | None = None
) -> PartitionIndex:
    indexed = m is not None
    return PartitionIndex(
        partition=name,
        rows=rows,
        index_name=index_name_for(name) if indexed else None,
        m=m,
        ef_construction=ef_construction if indexed else None,
    )


# ---------------------------------------------------------------------------
# plan_partition
# ---------------------------------------------------------------------------


class TestPlanning:
    def test_an_empty_partition_is_skipped(self) -> None:
        """Building on nothing produces an index describing an empty graph."""
        plan = plan_partition(partition("chunks_2027q4", 0), m=M, ef_construction=EFC)
        assert plan.action == "skip-empty"

    def test_an_unindexed_partition_is_created(self) -> None:
        plan = plan_partition(partition("chunks_2016q4", 130_073), m=M, ef_construction=EFC)
        assert plan.action == "create"
        assert plan.target_m == M
        assert plan.target_ef_construction == EFC

    def test_a_correctly_built_index_is_kept(self) -> None:
        plan = plan_partition(
            partition("chunks_2016q4", 130_073, m=M, ef_construction=EFC),
            m=M,
            ef_construction=EFC,
        )
        assert plan.action == "keep"

    @pytest.mark.parametrize("rows", [130_073, 600_000, 5_000_000])
    def test_growth_alone_never_triggers_a_rebuild(self, rows: int) -> None:
        """ADR-0026's central claim, asserted rather than asserted-in-prose.

        Under IVFFlat `lists = sqrt(rows)`, so a partition that grew carried a
        degraded index until a rebuild pass ran -- and that drift interrupted
        M4, M5, M6 and M7 in turn. HNSW has no row-dependent parameter.
        """
        plan = plan_partition(
            partition("chunks_2017q4", rows, m=M, ef_construction=EFC),
            m=M,
            ef_construction=EFC,
        )
        assert plan.action == "keep"
        assert "no row-dependent parameter" in plan.reason

    @pytest.mark.parametrize(
        ("built_m", "built_efc"),
        [(8, EFC), (M, 128), (32, 200)],
    )
    def test_changed_build_parameters_trigger_a_rebuild(self, built_m: int, built_efc: int) -> None:
        """The only remaining rebuild trigger, and it is an operator action."""
        plan = plan_partition(
            partition("chunks_2016q4", 130_073, m=built_m, ef_construction=built_efc),
            m=M,
            ef_construction=EFC,
        )
        assert plan.action == "rebuild"
        assert str(built_m) in plan.reason or str(built_efc) in plan.reason

    def test_an_empty_partition_is_skipped_even_with_odd_parameters(self) -> None:
        plan = plan_partition(
            partition("chunks_2027q4", 0, m=8, ef_construction=32), m=M, ef_construction=EFC
        )
        assert plan.action == "skip-empty"


class TestParameterValidation:
    """Rejected here rather than by Postgres, so a misconfiguration fails
    before the pass starts dropping indexes rather than midway through."""

    @pytest.mark.parametrize("m", [1, 0, -4, PGVECTOR_MAX_M + 1])
    def test_m_outside_pgvectors_range_is_refused(self, m: int) -> None:
        with pytest.raises(ValueError, match="hnsw m must lie"):
            plan_partition(partition("p", 10), m=m, ef_construction=1000)

    @pytest.mark.parametrize("efc", [3, 0, PGVECTOR_MAX_EF_CONSTRUCTION + 1])
    def test_ef_construction_outside_pgvectors_range_is_refused(self, efc: int) -> None:
        with pytest.raises(ValueError, match="ef_construction must lie"):
            plan_partition(partition("p", 10), m=M, ef_construction=efc)

    def test_ef_construction_below_twice_m_is_refused(self) -> None:
        """pgvector enforces this; a smaller value builds a degraded graph."""
        with pytest.raises(ValueError, match="at least 2 \\* m"):
            plan_partition(partition("p", 10), m=32, ef_construction=40)

    def test_the_pinned_defaults_are_valid(self) -> None:
        from cascade.config import load_settings

        retrieval = load_settings(None).retrieval
        plan = plan_partition(
            partition("p", 10), m=retrieval.hnsw_m, ef_construction=retrieval.hnsw_ef_construction
        )
        assert plan.action == "create"


# ---------------------------------------------------------------------------
# plan_all
# ---------------------------------------------------------------------------


class TestPlanAll:
    def test_plans_are_ordered_largest_partition_first(self) -> None:
        """An interrupted pass should already have indexed what dominates latency."""
        plans = plan_all(
            [partition("small", 10), partition("huge", 100_000), partition("mid", 5_000)],
            m=M,
            ef_construction=EFC,
        )
        assert [plan.partition for plan in plans] == ["huge", "mid", "small"]

    def test_ties_break_on_name_so_the_order_is_deterministic(self) -> None:
        """Invariant 7: iteration order is never left to chance."""
        plans = plan_all(
            [partition("b", 100), partition("a", 100), partition("c", 100)],
            m=M,
            ef_construction=EFC,
        )
        assert [plan.partition for plan in plans] == ["a", "b", "c"]

    def test_planning_is_idempotent_after_a_pass(self) -> None:
        """A second run must be a no-op, or the command is not safe to re-run."""
        measured = [partition("chunks_2016q4", 130_073), partition("chunks_2027q4", 0)]
        first = plan_all(measured, m=M, ef_construction=EFC)

        applied = [
            (
                partition(
                    plan.partition,
                    plan.rows,
                    m=plan.target_m,
                    ef_construction=plan.target_ef_construction,
                )
                if plan.action in {"create", "rebuild"}
                else partition(plan.partition, plan.rows)
            )
            for plan in first
        ]
        second = plan_all(applied, m=M, ef_construction=EFC)
        assert {plan.action for plan in second} <= {"keep", "skip-empty"}

    @given(st.integers(min_value=1, max_value=10_000_000))
    def test_a_built_partition_at_any_size_stays_kept(self, rows: int) -> None:
        """The property the whole change rests on, quantified over sizes."""
        plan = plan_partition(
            partition("p", rows, m=M, ef_construction=EFC), m=M, ef_construction=EFC
        )
        assert plan.action == "keep"

    def test_every_plan_is_accounted_for(self) -> None:
        measured = [
            partition("a", 0),
            partition("b", 10),
            partition("c", 10, m=M, ef_construction=EFC),
        ]
        plans = plan_all(measured, m=M, ef_construction=EFC)
        assert len(plans) == 3
        assert {p.action for p in plans} == {"skip-empty", "create", "keep"}


def test_index_names_carry_the_access_method() -> None:
    """So an IVFFlat index left by a pre-ADR-0026 database is not mistaken for
    this one -- it does not exist under the name the command looks for, and the
    partition plans as `create`."""
    assert index_name_for("chunks_2016q4") == "chunks_2016q4_embedding_hnsw_idx"
    assert "ivfflat" not in index_name_for("chunks_2016q4")
    assert len(index_name_for("chunks_2016q4")) < 64  # Postgres identifier limit
