"""The §12.4 cost-ledger gate (M8).

Two independent records of the same spend -- the meter's, summed from `runs`,
and Langfuse's -- must agree within 2%. The arithmetic is pure and is tested
here; reading either side needs a service and lives in the integration suite.

The case that matters most is the one that looks like a pass: zero against
zero agrees perfectly, and a gate that accepts it is a gate that cannot fail.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cascade.trace.ledger import RECONCILE_TOLERANCE, LedgerReconciliation, PhaseSpend


def spend(source: str, usd: str, calls: int = 100) -> PhaseSpend:
    return PhaseSpend(
        source=source,
        total_usd=Decimal(usd),
        calls=calls,
        input_tokens=calls * 450,
        output_tokens=calls * 45,
    )


def pair(local: str, remote: str | None, *, calls: int = 100) -> LedgerReconciliation:
    return LedgerReconciliation(
        local=spend("runs ledger", local, calls),
        remote=None if remote is None else spend("langfuse", remote, calls),
    )


class TestAgreement:
    def test_identical_totals_reconcile(self) -> None:
        assert pair("1.234567", "1.234567").reconciled

    def test_a_difference_inside_the_tolerance_reconciles(self) -> None:
        assert pair("100.00", "99.00").reconciled  # 1%

    def test_a_difference_above_the_tolerance_does_not(self) -> None:
        result = pair("100.00", "95.00")  # 5%
        assert not result.reconciled
        assert result.relative_difference == pytest.approx(Decimal("0.05"))

    def test_the_boundary_is_inclusive(self) -> None:
        """'above 2%' fails; exactly 2% does not."""
        result = pair("100.00", "98.00")
        assert result.relative_difference == pytest.approx(Decimal("0.02"))
        assert result.reconciled

    def test_the_tolerance_is_the_spec_value(self) -> None:
        assert str(RECONCILE_TOLERANCE) == "0.02"

    def test_the_direction_of_the_difference_does_not_change_the_verdict(self) -> None:
        """Divided by the larger side, so the figure does not depend on which
        record is called the reference."""
        assert (
            pair("100.00", "95.00").relative_difference
            == pair("95.00", "100.00").relative_difference
        )


class TestVacuousComparison:
    def test_zero_against_zero_is_not_a_pass(self) -> None:
        """The failure this exists to prevent: a study run entirely in replay,
        or under a stand-in decider, books nothing on either side. Zero agrees
        with zero to 0.0000% and would pass a gate guarding the cost claim."""
        result = pair("0", "0", calls=0)
        assert result.vacuous
        assert not result.reconciled
        assert result.within_tolerance  # the arithmetic agrees; the gate still refuses

    def test_zero_spend_with_real_calls_is_not_vacuous(self) -> None:
        """A fully cached phase legitimately costs nothing and did make calls."""
        result = pair("0", "0", calls=5_000)
        assert not result.vacuous
        assert result.reconciled

    def test_a_non_zero_total_is_never_vacuous(self) -> None:
        assert not pair("0.01", "0.01", calls=0).vacuous


class TestUnreachableLangfuse:
    def test_a_missing_remote_total_is_not_reconciled(self) -> None:
        """Observability must never fail a run (`tracing.py`), so an outage is
        reported as unreconciled rather than as a discrepancy -- but "we could
        not check" is not "we checked"."""
        result = pair("12.34", None)
        assert result.remote is None
        assert not result.reconciled
        assert result.relative_difference is None
        assert result.difference_usd is None

    def test_it_is_distinguishable_from_a_disagreement(self) -> None:
        unreachable = pair("12.34", None)
        disagreeing = pair("12.34", "6.00")
        assert unreachable.relative_difference is None
        assert disagreeing.relative_difference is not None


class TestArithmetic:
    def test_the_difference_is_signed(self) -> None:
        assert pair("10.00", "8.00").difference_usd == Decimal("2.00")
        assert pair("8.00", "10.00").difference_usd == Decimal("-2.00")

    def test_a_zero_remote_against_a_real_local_reads_as_total_disagreement(self) -> None:
        """Not a division by zero: divided by the larger side, this is 100%."""
        result = pair("10.00", "0", calls=100)
        assert result.relative_difference == Decimal(1)
        assert not result.reconciled

    def test_totals_stay_exact(self) -> None:
        """The meter prices in exact Decimal; the reconciliation must not
        round the comparison through a float."""
        result = pair("0.000001", "0.000001")
        assert result.difference_usd == Decimal("0")
        assert isinstance(result.local.total_usd, Decimal)
