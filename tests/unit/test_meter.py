"""Cost meter: exact USD to six decimal places, and a ceiling that aborts.

Two M0 acceptance criteria live here:

* computed USD matches a known-token fixture to 6 dp;
* a phase with a $0.01 ceiling aborts, checkpoints and exits non-zero (the
  exit code itself is asserted end-to-end in ``test_cli.py``).

The fixtures are not invented. The first one reproduces the spec's own §12.1
derivation from raw token counts, so if the price table drifts from the spec
the test says so.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from cascade.config import Settings
from cascade.llm.meter import (
    CostMeter,
    compute_cost,
    estimate_phase,
    quantize_usd,
)
from cascade.llm.types import BudgetExceeded, Usage

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Known-token fixtures. (usage, model, batch, expected USD at 6 dp)
# ---------------------------------------------------------------------------

FIXTURES: list[tuple[str, Usage, str, bool, str]] = [
    (
        # Spec §12.1, one uncached agent call: 1,900-token prompt-cached prefix
        # read at 10%, 260 dynamic input tokens, 45 output tokens, Batch API.
        "spec-12.1-agent-call",
        Usage(input_tokens=260, output_tokens=45, cache_read_input_tokens=1900),
        HAIKU,
        True,
        "0.000338",  # exact 0.0003375, half-even to 6 dp
    ),
    (
        "haiku-interactive-round-numbers",
        Usage(input_tokens=1000, output_tokens=500),
        HAIKU,
        False,
        "0.003500",
    ),
    (
        "sonnet-interactive-round-numbers",
        Usage(input_tokens=1000, output_tokens=500),
        SONNET,
        False,
        "0.010500",
    ),
    (
        # A cache *write* at the 1h TTL carries a 2.0x premium (config), so a
        # 4,096-token prefix write costs 2x its list input price.
        "haiku-cache-write-1h",
        Usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=4096),
        HAIKU,
        False,
        "0.008192",
    ),
    (
        "zero-usage-is-free",
        Usage(input_tokens=0, output_tokens=0),
        HAIKU,
        False,
        "0.000000",
    ),
]


@pytest.mark.parametrize(("name", "usage", "model", "batch", "expected"), FIXTURES)
def test_cost_matches_fixture_to_six_decimal_places(
    settings: Settings, name: str, usage: Usage, model: str, batch: bool, expected: str
) -> None:
    """M0 acceptance: computed USD equals the fixture exactly at 6 dp."""
    cost = compute_cost(
        usage=usage,
        price=settings.price_for(model),
        multipliers=settings.pricing_multipliers,
        batch=batch,
        cache_ttl=settings.prompt_cache.ttl,
    )
    assert quantize_usd(cost) == Decimal(expected), name


def test_cache_write_premium_depends_on_ttl(settings: Settings) -> None:
    """A 5m write is 1.25x list; a 1h write is 2.0x. Confusing them misprices M6."""
    usage = Usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=4096)
    kwargs = {
        "usage": usage,
        "price": settings.price_for(HAIKU),
        "multipliers": settings.pricing_multipliers,
        "batch": False,
    }
    assert quantize_usd(compute_cost(cache_ttl="5m", **kwargs)) == Decimal("0.005120")
    assert quantize_usd(compute_cost(cache_ttl="1h", **kwargs)) == Decimal("0.008192")


def test_batch_api_halves_the_bill(settings: Settings) -> None:
    """The 50% Batch discount applies to every token category, not just input."""
    usage = Usage(input_tokens=1000, output_tokens=500, cache_read_input_tokens=2000)
    kwargs = {
        "usage": usage,
        "price": settings.price_for(HAIKU),
        "multipliers": settings.pricing_multipliers,
        "cache_ttl": settings.prompt_cache.ttl,
    }
    interactive = compute_cost(batch=False, **kwargs)
    batched = compute_cost(batch=True, **kwargs)
    assert batched * 2 == interactive


def test_reproduces_the_spec_marginal_cost_per_run(settings: Settings) -> None:
    """Independently re-derive spec §12.1's $0.0035/run from raw token counts.

    This is the check that catches a price-table edit: the study's headline
    cost claim is downstream of these five numbers and nothing else.
    """
    per_call = compute_cost(
        usage=Usage(input_tokens=260, output_tokens=45, cache_read_input_tokens=1900),
        price=settings.price_for(HAIKU),
        multipliers=settings.pricing_multipliers,
        batch=True,
        cache_ttl=settings.prompt_cache.ttl,
    )
    uncached_calls_per_run = Decimal("10.5")  # 116.6 decision events x 9% miss rate
    per_run = per_call * uncached_calls_per_run
    assert quantize_usd(per_run) == Decimal("0.003544")
    # Spec quotes $0.0035; agreement to 4 dp is the claim being checked.
    assert per_run.quantize(Decimal("0.0001")) == Decimal("0.0035")


def test_arithmetic_is_exact_not_floating_point(settings: Settings) -> None:
    """Accumulating a repeating value must not drift.

    ``0.1 * 3 != 0.3`` in binary floating point. Over ~378,000 calls that
    error is what would make the M8 ledger reconciliation fail against a
    correct meter.
    """
    meter = CostMeter(settings, "bench", ceiling_usd=Decimal("1000"))
    usage = Usage(input_tokens=100, output_tokens=0)  # exactly $0.0001 at haiku list
    for _ in range(10_000):
        meter.record(model=HAIKU, usage=usage, batch=False)
    assert meter.total_usd == Decimal("1.0000")
    assert meter.calls == 10_000


def test_unpriced_model_raises_rather_than_costing_zero(settings: Settings) -> None:
    """A model missing from the price table must not silently bill $0."""
    with pytest.raises(KeyError, match="no pricing entry"):
        settings.price_for("claude-not-in-the-table")


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------


def test_ceiling_breach_aborts_and_checkpoints(settings: Settings) -> None:
    """M0 acceptance: $0.01 ceiling aborts and writes a resumable checkpoint."""
    meter = CostMeter(settings, "simulate", ceiling_usd=Decimal("0.01"))
    usage = Usage(input_tokens=100_000, output_tokens=10_000)  # $0.15/call at haiku list

    meter.set_resume_state({"completed_units": 41, "next_unit": 42})
    with pytest.raises(BudgetExceeded) as caught:
        meter.record(model=HAIKU, usage=usage, batch=False)

    breach = caught.value
    assert breach.phase == "simulate"
    assert breach.spent > breach.ceiling
    assert breach.ceiling == Decimal("0.010000")

    checkpoint = Path(breach.checkpoint)
    assert checkpoint.name == "simulate.checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["phase"] == "simulate"
    assert payload["resume"] == {"completed_units": 41, "next_unit": 42}
    assert Decimal(payload["spent_usd"]) > Decimal(payload["ceiling_usd"])


def test_spend_is_booked_before_the_ceiling_is_checked(settings: Settings) -> None:
    """The reported figure is what was spent, not what was authorised.

    The call has already been made and billed by the time the meter sees it;
    reporting the pre-call total would understate the real liability.
    """
    meter = CostMeter(settings, "simulate", ceiling_usd=Decimal("0.01"))
    with pytest.raises(BudgetExceeded):
        meter.record(model=HAIKU, usage=Usage(input_tokens=100_000, output_tokens=0), batch=False)
    assert meter.total_usd == Decimal("0.1")


def test_exact_ceiling_is_not_a_breach(settings: Settings) -> None:
    """Spending exactly the ceiling is allowed; one token past it is not."""
    meter = CostMeter(settings, "bench", ceiling_usd=Decimal("0.0001"))
    meter.record(model=HAIKU, usage=Usage(input_tokens=100, output_tokens=0), batch=False)
    assert meter.total_usd == Decimal("0.0001")
    with pytest.raises(BudgetExceeded):
        meter.record(model=HAIKU, usage=Usage(input_tokens=1, output_tokens=0), batch=False)


def test_abort_on_breach_false_does_not_raise(settings: Settings) -> None:
    """The flag exists for the estimator, which must survive its own overrun."""
    relaxed = settings.model_copy(
        update={"budget": settings.budget.model_copy(update={"abort_on_breach": False})}
    )
    meter = CostMeter(relaxed, "simulate", ceiling_usd=Decimal("0.01"))
    meter.record(model=HAIKU, usage=Usage(input_tokens=100_000, output_tokens=0), batch=False)
    assert meter.total_usd > Decimal("0.01")


def test_ceiling_defaults_to_the_configured_phase_budget(settings: Settings) -> None:
    """A meter built without an explicit ceiling uses config, never infinity."""
    meter = CostMeter(settings, "simulate")
    assert meter.ceiling_usd == settings.phase_ceiling("simulate")
    assert meter.ceiling_usd == Decimal("240.0")


def test_unknown_phase_has_no_implicit_budget(settings: Settings) -> None:
    """An unbudgeted phase must fail loudly rather than run uncapped."""
    with pytest.raises(KeyError, match="no budget ceiling"):
        CostMeter(settings, "phase-that-does-not-exist")


# ---------------------------------------------------------------------------
# Cache hits and estimation
# ---------------------------------------------------------------------------


def test_cache_hits_are_free_but_counted(settings: Settings) -> None:
    """Hit rate is the biggest cost lever (§12.2); an untracked lever is useless."""
    meter = CostMeter(settings, "simulate")
    usage = Usage(input_tokens=260, output_tokens=45)
    meter.record(model=HAIKU, usage=usage, batch=True)
    for _ in range(9):
        meter.record_cache_hit(model=HAIKU, usage=usage)

    assert meter.calls == 1
    assert meter.cached_calls == 9
    assert meter.hit_rate == pytest.approx(0.9)
    assert meter.cached_usage.output_tokens == 405
    assert meter.total_usd == meter.price_of(HAIKU, usage, batch=True)


def test_snapshot_is_json_serialisable(settings: Settings) -> None:
    """The snapshot goes into the run ledger, so it must survive json.dumps."""
    meter = CostMeter(settings, "simulate")
    meter.record(model=HAIKU, usage=Usage(input_tokens=10, output_tokens=5), batch=True)
    round_tripped = json.loads(json.dumps(meter.snapshot()))
    assert round_tripped["phase"] == "simulate"
    assert Decimal(round_tripped["total_usd"]) > 0


def test_estimate_extrapolates_a_phase() -> None:
    """`--estimate` projects from a measured sample (spec §12.4)."""
    estimate = estimate_phase(
        phase="simulate",
        sample_units=20,
        sample_usd=Decimal("0.0709"),
        total_units=36_000,
        ceiling_usd=Decimal("240.0"),
    )
    assert estimate.per_unit_usd == Decimal("0.003545")
    assert estimate.projected_usd == Decimal("127.62")
    assert estimate.within_ceiling


def test_estimate_flags_a_projected_overrun() -> None:
    """A projection over the ceiling is reported, not silently truncated."""
    estimate = estimate_phase(
        phase="simulate",
        sample_units=20,
        sample_usd=Decimal("1.00"),
        total_units=36_000,
        ceiling_usd=Decimal("240.0"),
    )
    assert estimate.projected_usd == Decimal("1800.00")
    assert not estimate.within_ceiling


def test_estimate_refuses_a_zero_unit_sample() -> None:
    """Extrapolating from nothing would report $0 for any phase."""
    with pytest.raises(ValueError, match="zero-unit sample"):
        estimate_phase(
            phase="simulate",
            sample_units=0,
            sample_usd=Decimal("0"),
            total_units=36_000,
            ceiling_usd=Decimal("240.0"),
        )
