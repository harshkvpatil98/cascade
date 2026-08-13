"""Token and USD accounting with a hard per-phase ceiling (spec §12.4).

Every figure here is :class:`~decimal.Decimal`. Binary floats would make the
6-decimal-place acceptance criterion fail for reasons unrelated to the meter,
and the M8 ledger reconciliation (±2% against Langfuse) has no tolerance to
spare for accumulated representation error over ~378,000 calls.

A breach is never a warning. The meter writes a resumable checkpoint and
raises :class:`BudgetExceeded`, which the CLI boundary maps to exit code 2.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from cascade.config import CacheTTL, PricingEntry, PricingMultipliers, Settings
from cascade.llm.types import BudgetExceeded, Usage

__all__ = [
    "USD",
    "CostMeter",
    "PhaseEstimate",
    "compute_cost",
    "estimate_phase",
    "quantize_usd",
]

_PER_MTOK = Decimal(1_000_000)
USD = Decimal("0.000001")


def quantize_usd(value: Decimal) -> Decimal:
    """Round to the reporting precision, six decimal places.

    Used only at the boundary. Internal accumulation stays exact so that
    rounding is applied once, not once per call.
    """
    return value.quantize(USD)


def compute_cost(
    *,
    usage: Usage,
    price: PricingEntry,
    multipliers: PricingMultipliers,
    batch: bool,
    cache_ttl: CacheTTL,
) -> Decimal:
    """Return the exact USD cost of one call.

    Preserves the invariant that every billed token category is priced with
    its own multiplier. Cached reads are ~10% of list and cache writes carry a
    TTL-dependent premium; collapsing them into plain input tokens is the
    single easiest way to produce a cost model that looks right and is not.

    The result is exact, not rounded -- see :func:`quantize_usd`.
    """
    write_multiplier = (
        multipliers.cache_write_1h if cache_ttl == "1h" else multipliers.cache_write_5m
    )
    subtotal = (
        Decimal(usage.input_tokens) * price.input_per_mtok
        + Decimal(usage.output_tokens) * price.output_per_mtok
        + Decimal(usage.cache_read_input_tokens) * price.input_per_mtok * multipliers.cache_read
        + Decimal(usage.cache_creation_input_tokens) * price.input_per_mtok * write_multiplier
    ) / _PER_MTOK
    if batch:
        subtotal *= multipliers.batch
    return subtotal


@dataclass(frozen=True)
class PhaseEstimate:
    """Extrapolation from a sample of units to a whole phase (spec §12.4)."""

    phase: str
    sample_units: int
    sample_usd: Decimal
    total_units: int
    projected_usd: Decimal
    ceiling_usd: Decimal

    @property
    def within_ceiling(self) -> bool:
        return self.projected_usd <= self.ceiling_usd

    @property
    def per_unit_usd(self) -> Decimal:
        return self.sample_usd / Decimal(self.sample_units) if self.sample_units else Decimal(0)


def estimate_phase(
    *,
    phase: str,
    sample_units: int,
    sample_usd: Decimal,
    total_units: int,
    ceiling_usd: Decimal,
) -> PhaseEstimate:
    """Project a phase's spend from a measured sample.

    Preserves the guardrail that no full phase launches unmeasured: a 36,000
    run phase whose per-unit cost is 3x the model would otherwise be
    discovered by the invoice.
    """
    if sample_units <= 0:
        raise ValueError("cannot extrapolate from a zero-unit sample")
    projected = sample_usd / Decimal(sample_units) * Decimal(total_units)
    return PhaseEstimate(
        phase=phase,
        sample_units=sample_units,
        sample_usd=sample_usd,
        total_units=total_units,
        projected_usd=projected,
        ceiling_usd=ceiling_usd,
    )


@dataclass
class CostMeter:
    """Accumulates spend for one phase and enforces its ceiling.

    Construction takes the phase name so the ceiling comes from config rather
    than from the call site; passing ``ceiling_usd`` explicitly is for tests
    and the ``dev budget-probe`` acceptance hook.
    """

    settings: Settings
    phase: str
    ceiling_usd: Decimal | None = None
    total_usd: Decimal = field(default=Decimal(0), init=False)
    calls: int = field(default=0, init=False)
    cached_calls: int = field(default=0, init=False)
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0), init=False)
    cached_usage: Usage = field(
        default_factory=lambda: Usage(input_tokens=0, output_tokens=0), init=False
    )
    _resume_state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.ceiling_usd is None:
            self.ceiling_usd = self.settings.phase_ceiling(self.phase)

    # -- resumability -------------------------------------------------------

    def set_resume_state(self, state: dict[str, Any]) -> None:
        """Record what a restart would need to skip completed units.

        Preserves invariant 8 (every phase is resumable). The meter holds this
        rather than the runner because the meter is what aborts, and a
        checkpoint written after the abort decision would race it.
        """
        self._resume_state = dict(state)

    def checkpoint_path(self) -> Path:
        return self.settings.checkpoint_path() / f"{self.phase}.checkpoint.json"

    def write_checkpoint(self) -> Path:
        """Persist the resume state atomically and return where it went."""
        path = self.checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {
                "phase": self.phase,
                "spent_usd": str(quantize_usd(self.total_usd)),
                "ceiling_usd": str(quantize_usd(self._ceiling())),
                "calls": self.calls,
                "cached_calls": self.cached_calls,
                "resume": self._resume_state,
            },
            indent=2,
            sort_keys=True,
        )
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{self.phase}-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            # Cleanup and re-raise. A checkpoint that half-exists is worse than
            # none: resume would read it and skip work that never ran.
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    # -- accounting ---------------------------------------------------------

    def _ceiling(self) -> Decimal:
        assert self.ceiling_usd is not None  # noqa: S101 -- set in __post_init__
        return self.ceiling_usd

    def price_of(self, model: str, usage: Usage, *, batch: bool) -> Decimal:
        """Price a call without booking it."""
        return compute_cost(
            usage=usage,
            price=self.settings.price_for(model),
            multipliers=self.settings.pricing_multipliers,
            batch=batch,
            cache_ttl=self.settings.prompt_cache.ttl,
        )

    def record(self, *, model: str, usage: Usage, batch: bool) -> Decimal:
        """Book one billed call, then enforce the ceiling.

        The ceiling is checked *after* the spend is added, so the reported
        figure is the true amount spent rather than the amount that would have
        been spent had the call been allowed. Returns the call's cost.
        """
        cost = self.price_of(model, usage, batch=batch)
        self.total_usd += cost
        self.calls += 1
        self.usage = self.usage + usage
        self._enforce()
        return cost

    def record_cache_hit(self, *, model: str, usage: Usage) -> None:
        """Book a cache hit as a zero-cost call.

        Tracked rather than ignored so that hit rate and spend can be read off
        the same ledger (spec §12.2); the hit rate is the biggest single lever
        on total cost and an untracked lever cannot be tuned.
        """
        del model  # priced at zero by construction; kept for call-site symmetry
        self.cached_calls += 1
        self.cached_usage = self.cached_usage + usage

    def _enforce(self) -> None:
        if not self.settings.budget.abort_on_breach:
            return
        ceiling = self._ceiling()
        if self.total_usd <= ceiling:
            return
        path = self.write_checkpoint()
        raise BudgetExceeded(
            phase=self.phase,
            spent=quantize_usd(self.total_usd),
            ceiling=quantize_usd(ceiling),
            checkpoint=str(path),
        )

    # -- reporting ----------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Fraction of calls served from the recording cache."""
        total = self.calls + self.cached_calls
        return self.cached_calls / total if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe summary for the run ledger and the phase report."""
        return {
            "phase": self.phase,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "hit_rate": self.hit_rate,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            "total_usd": str(quantize_usd(self.total_usd)),
            "ceiling_usd": str(quantize_usd(self._ceiling())),
        }
