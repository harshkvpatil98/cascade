"""Reconciling the local cost ledger against Langfuse (spec §12.4; M8).

§12.4 makes this a gate, not a dashboard: "Token counts and cost are written to
the runs table on completion of each run and reconciled against the Langfuse
total at the end of each phase. A discrepancy above 2% is a bug in the meter
and blocks the report."

Two independent records of the same spend exist by construction. The **meter**
prices every call from the pinned per-token table (`llm/meter.py`) in exact
`Decimal`, and the per-run total is written to `runs.cost_usd`. The **tracer**
emits one Langfuse generation per call carrying the same cost. They are
produced by different code from the same event, which is what makes agreement
evidence rather than tautology -- and why the reconciliation is worth running.

The comparison is deliberately asymmetric about failure. A Langfuse that is
unreachable is **not** a reconciliation failure: `tracing.py` is allowed to
degrade to a no-op because observability must never fail a run, so a missing
remote total is reported as *unreconciled*, which is a different verdict from
*disagreeing*. Only a measured difference above the threshold fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from cascade.config import Settings

__all__ = [
    "RECONCILE_TOLERANCE",
    "LedgerReconciliation",
    "PhaseSpend",
    "local_spend",
    "reconcile",
    "remote_spend",
]

Role = Literal["admin", "sim", "eval"]

# §12.4's threshold, stated once. "A discrepancy above 2% is a bug in the
# meter", so this is a criterion rather than a tuning knob.
RECONCILE_TOLERANCE = Decimal("0.02")


@dataclass(frozen=True, slots=True)
class PhaseSpend:
    """What one source says a phase cost."""

    source: str
    total_usd: Decimal
    calls: int
    input_tokens: int
    output_tokens: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class LedgerReconciliation:
    """The §12.4 gate, measured."""

    local: PhaseSpend
    remote: PhaseSpend | None
    tolerance: Decimal = RECONCILE_TOLERANCE

    @property
    def vacuous(self) -> bool:
        """True when there is no spend on either side to reconcile.

        Zero against zero agrees to 0.0000%, and reporting that as reconciled
        is a guard that cannot fail: a study run entirely in replay, or under a
        stand-in decider, books nothing on either side and would pass §12.4's
        gate without either record having been tested against the other.
        """
        if self.remote is None:
            return False
        return (
            self.local.total_usd == 0
            and self.remote.total_usd == 0
            and self.local.calls == 0
            and self.remote.calls == 0
        )

    @property
    def reconciled(self) -> bool:
        """True only when both totals exist, are non-trivial, and agree.

        A missing remote total reads as *not reconciled* rather than as
        reconciled-by-default: §12.4 blocks the report on a discrepancy, and
        "we could not check" is not "we checked". A vacuous comparison reads
        the same way, for the same reason.
        """
        return self.remote is not None and not self.vacuous and self.within_tolerance

    @property
    def difference_usd(self) -> Decimal | None:
        if self.remote is None:
            return None
        return self.local.total_usd - self.remote.total_usd

    @property
    def relative_difference(self) -> Decimal | None:
        """``|local - remote| / max(local, remote)``, or ``None``.

        Divided by the larger of the two rather than by one of them, so the
        figure does not depend on which side is called the reference -- and so
        a remote total of zero against a non-zero local one reads as 100%
        rather than as a division by zero.
        """
        if self.remote is None:
            return None
        base = max(self.local.total_usd, self.remote.total_usd)
        if base == 0:
            return Decimal(0)
        difference = self.local.total_usd - self.remote.total_usd
        return abs(difference) / base

    @property
    def within_tolerance(self) -> bool:
        relative = self.relative_difference
        return relative is not None and relative <= self.tolerance


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


def local_spend(
    settings: Settings, *, config_id: str | None = None, role: Role = "eval"
) -> PhaseSpend:
    """What the meter booked, summed from the run ledger.

    ``runs`` is the ledger §12.4 names: one row per completed run carrying the
    cost the meter priced for it. Summing it rather than re-pricing keeps this
    a *reconciliation* -- re-pricing here would compare the price table with
    itself and agree by construction.
    """
    clause = "WHERE config_id = %(config_id)s" if config_id else ""
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COALESCE(sum(cost_usd), 0)::text,
                   COALESCE(sum(llm_calls), 0),
                   COALESCE(sum(tokens_in), 0),
                   COALESCE(sum(tokens_out), 0),
                   count(*)
            FROM runs {clause}
            """,  # noqa: S608 -- `clause` is a literal chosen above, not input
            {"config_id": config_id},
        )
        row = cur.fetchone()
    total, calls, tokens_in, tokens_out, runs = row if row else ("0", 0, 0, 0, 0)
    return PhaseSpend(
        source="runs ledger",
        total_usd=Decimal(str(total)),
        calls=int(calls),
        input_tokens=int(tokens_in),
        output_tokens=int(tokens_out),
        detail=f"{int(runs):,} run(s)" + (f", config {config_id}" if config_id else ""),
    )


def remote_spend(settings: Settings, *, timeout_s: float = 30.0) -> PhaseSpend | None:
    """What Langfuse recorded, or ``None`` when it cannot be asked.

    ``None`` is returned for an unreachable or unconfigured Langfuse rather
    than a zero total. A zero would compare as a 100% discrepancy and fail the
    gate for an observability outage, which inverts the rule that observability
    must never fail a run (`tracing.py`).

    Reads the daily metrics endpoint, which Langfuse v2 (ADR-0006) serves as an
    aggregate rather than requiring a walk over every observation -- 4.2M
    generations is not something to paginate for a total.
    """
    if not settings.langfuse.enabled:
        return None
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None

    import httpx

    auth = (
        settings.langfuse_public_key.get_secret_value(),
        settings.langfuse_secret_key.get_secret_value(),
    )
    url = f"{settings.langfuse.host.rstrip('/')}/api/public/metrics/daily"
    total = Decimal(0)
    calls = tokens_in = tokens_out = 0
    page = 1
    try:
        with httpx.Client(timeout=timeout_s) as client:
            while True:
                response = client.get(url, auth=auth, params={"page": page, "limit": 100})
                if response.status_code != 200:
                    return None
                payload = response.json()
                rows = payload.get("data") or []
                for day in rows:
                    total += Decimal(str(day.get("totalCost") or 0))
                    for usage in day.get("usage") or []:
                        calls += int(usage.get("countObservations") or 0)
                        tokens_in += int(usage.get("inputUsage") or 0)
                        tokens_out += int(usage.get("outputUsage") or 0)
                meta = payload.get("meta") or {}
                if page >= int(meta.get("totalPages") or 1) or not rows:
                    break
                page += 1
    except Exception:  # noqa: BLE001 -- an unreachable Langfuse is not a discrepancy
        return None

    return PhaseSpend(
        source="langfuse",
        total_usd=total,
        calls=calls,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        detail=f"{page} page(s) of daily metrics",
    )


def reconcile(
    settings: Settings,
    *,
    config_id: str | None = None,
    tolerance: Decimal = RECONCILE_TOLERANCE,
) -> LedgerReconciliation:
    """Compare the two records of the same spend (§12.4)."""
    return LedgerReconciliation(
        local=local_spend(settings, config_id=config_id),
        remote=remote_spend(settings),
        tolerance=tolerance,
    )
