"""Selection: the Q2 precedence, base-rate control and domain stratification.

The rule under test is that the *count* is the only thing that gives. A
selection that reaches 180 by admitting a rule violation, or by dropping
scenarios until the base rate looks right, is a selection effect on the
headline metric -- and an undisclosable one, because nothing downstream can
detect it afterwards.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cascade.ledger.schema import Scenario, ScenarioLabel, ScenarioRecord
from cascade.ledger.select import deduplicate_event_groups, select

SALT = "test-salt"
BASE = datetime(2024, 1, 1, tzinfo=UTC)

# The real domain tags. `Domain` is a closed Literal, so a synthetic name is a
# validation error -- which is the schema doing its job.
DOMAINS = (
    "elections",
    "geopolitics",
    "conflict",
    "corporate",
    "regulation",
    "labor",
    "macro_policy",
    "technology",
    "health",
    "sports",
    "other",
)


def record(
    scenario_id: str,
    *,
    outcome: int,
    domain: str = "elections",
    event_group: str | None = None,
) -> ScenarioRecord:
    resolve = BASE + timedelta(days=60)
    return ScenarioRecord(
        scenario=Scenario(
            scenario_id=scenario_id,
            question=f"Will {scenario_id}?",
            resolution_criterion="criterion",
            cutoff_ts=BASE,
            resolve_ts=resolve,
            domain=domain,  # type: ignore[arg-type]
            source="curated",
            source_ref=scenario_id,
            party_rule="curated",
            party_names=("A", "B", "C"),
            event_group=event_group,
        ),
        label=ScenarioLabel(
            scenario_id=scenario_id, outcome=outcome, resolved_at=resolve  # type: ignore[arg-type]
        ),
    )


def pool(
    n_yes: int, n_no: int, *, domains: tuple[str, ...] = ("elections", "conflict", "corporate")
) -> tuple[ScenarioRecord, ...]:
    out = [record(f"y{i:04d}", outcome=1, domain=domains[i % len(domains)]) for i in range(n_yes)]
    out += [record(f"n{i:04d}", outcome=0, domain=domains[i % len(domains)]) for i in range(n_no)]
    return tuple(out)


def run(records: tuple[ScenarioRecord, ...], *, target: int = 180, cap: float = 0.25):  # type: ignore[no-untyped-def]
    return select(
        records,
        rejections=(),
        n_candidates=len(records),
        target_n=target,
        yes_rate_min=0.40,
        yes_rate_max=0.60,
        max_domain_share=cap,
        salt=SALT,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_rich_pool_hits_the_target_with_a_balanced_base_rate() -> None:
    result = run(pool(400, 400, domains=DOMAINS[:8]))
    report = result.report
    assert report.n_selected == 180
    assert report.met_target
    assert 0.40 <= report.yes_rate <= 0.60
    assert report.max_domain_share <= 0.25
    assert report.shortfall_reason == ""


def test_selection_is_deterministic() -> None:
    """Same pool, same salt, same set -- the manifest depends on it."""
    records = pool(400, 400, domains=DOMAINS[:8])
    first = run(records)
    second = run(tuple(reversed(records)))
    assert [r.scenario.scenario_id for r in first.records] == [
        r.scenario.scenario_id for r in second.records
    ]


def test_a_different_salt_selects_a_different_set() -> None:
    """Ordering is keyed, so the salt is a real degree of freedom."""
    records = pool(400, 400, domains=DOMAINS[:8])
    other = select(
        records,
        rejections=(),
        n_candidates=len(records),
        target_n=180,
        yes_rate_min=0.40,
        yes_rate_max=0.60,
        max_domain_share=0.25,
        salt="a-different-salt",
    )
    assert [r.scenario.scenario_id for r in other.records] != [
        r.scenario.scenario_id for r in run(records).records
    ]


# ---------------------------------------------------------------------------
# Q2 precedence: the count gives, never the rules
# ---------------------------------------------------------------------------


def test_a_thin_pool_yields_fewer_scenarios_rather_than_breaking_the_base_rate() -> None:
    """Only 20 YES available: N shrinks so the rate stays inside the band."""
    result = run(pool(20, 400, domains=DOMAINS[:8]))
    report = result.report
    assert not report.met_target
    assert report.n_selected < 180
    assert 0.40 <= report.yes_rate <= 0.60
    # 20 YES at a 0.40 floor caps the set at 50.
    assert report.n_selected == 50
    assert "the count gives, never the rules" in report.shortfall_reason


def test_the_domain_cap_is_never_exceeded_even_at_the_cost_of_the_target() -> None:
    """A single-domain pool cannot fill 180 without breaching 25%."""
    result = run(pool(400, 400, domains=("elections",)))
    report = result.report
    assert report.max_domain_share <= 0.25 or report.n_selected == 0
    assert not report.met_target


def test_selection_never_exceeds_the_target() -> None:
    result = run(pool(1000, 1000, domains=DOMAINS))
    assert result.report.n_selected == 180


def test_an_empty_pool_reports_rather_than_raising() -> None:
    result = run(())
    assert result.records == ()
    assert result.report.n_selected == 0
    assert "no constraint-satisfying set" in result.report.shortfall_reason


@pytest.mark.parametrize("n_yes", [0, 1, 5, 90, 400])
def test_the_base_rate_band_always_holds_whatever_the_pool(n_yes: int) -> None:
    """The band is invariant across pool shapes; only N responds."""
    result = run(pool(n_yes, 400, domains=DOMAINS[:8]))
    if result.report.n_selected:
        assert 0.40 <= result.report.yes_rate <= 0.60


# ---------------------------------------------------------------------------
# Independence: one scenario per real-world event
# ---------------------------------------------------------------------------


def test_only_one_scenario_survives_per_event_group() -> None:
    """Sixty markets under one event are sixty views of one event.

    Admitting several would inflate the effective sample size while the M7
    paired bootstrap treats them as independent draws over scenarios.
    """
    records = tuple(
        record(f"c{i:03d}", outcome=i % 2, event_group="election-2024") for i in range(60)
    )
    kept, dropped = deduplicate_event_groups(records, salt=SALT)
    assert len(kept) == 1
    assert dropped == 59


def test_ungrouped_scenarios_are_never_deduplicated() -> None:
    """Curated and standalone questions are independent by construction."""
    records = tuple(record(f"s{i:03d}", outcome=i % 2) for i in range(25))
    kept, dropped = deduplicate_event_groups(records, salt=SALT)
    assert len(kept) == 25
    assert dropped == 0


def test_deduplication_keeps_a_deterministic_representative() -> None:
    records = tuple(record(f"c{i:03d}", outcome=1, event_group="g") for i in range(30))
    first, _ = deduplicate_event_groups(records, salt=SALT)
    second, _ = deduplicate_event_groups(tuple(reversed(records)), salt=SALT)
    assert first[0].scenario.scenario_id == second[0].scenario.scenario_id


def test_selection_reports_the_duplicates_it_dropped() -> None:
    records = (
        *[record(f"g{i:03d}", outcome=i % 2, event_group="shared") for i in range(40)],
        *pool(100, 100, domains=DOMAINS[:8]),
    )
    result = run(records)
    assert result.report.dropped_duplicate_groups == 39
    groups = [r.scenario.event_group for r in result.records if r.scenario.event_group]
    assert len(groups) == len(set(groups))
