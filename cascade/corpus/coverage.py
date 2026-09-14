"""Where the corpus is, measured against where the scenarios are.

M2's acceptance criterion counts chunks. M3's measures latency and recall.
Neither asks the question that decides whether the evidence is *usable*: for a
scenario with a 2026-05 cutoff, is the nearest admissible document from 2026-04
or from 2017-11? Chronofence answers both queries correctly and quickly. Only
one of them is evidence.

This module is the missing measurement and the ordering that acts on it. Both
halves are pure: demand is a function of the scenario cutoffs alone, and the
ordering is a function of the demand and the unit keys. Nothing here reads an
outcome -- ``scenarios.cutoff_ts`` is in the registry table the simulation role
can read, never in ``scenario_labels`` (invariant 2).

**Why ingest order is a correctness concern, not a scheduling preference.**
Units were walked in key order, which for every date-keyed source is
chronological. A run that is stopped -- and a multi-day ingest is always
stopped -- therefore leaves a corpus that is complete at the beginning of the
window and empty at the end. Measured before this change: 99.6% of 1.76M
chunks fell before 2018-04 while 169 of 180 scenario cutoffs fell after
2024-01. The corpus was over target and under-covering the entire study.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "DEFAULT_LOOKBACK_MONTHS",
    "UnitPlan",
    "demand_profile",
    "month_index",
    "order_units",
    "unit_depth",
    "unit_month",
]

# How far back from a cutoff a document still counts as evidence *for* that
# scenario. 18 months is deliberately generous: the point of the measure is to
# separate "stale by a quarter" from "stale by seven years", not to adjudicate
# what a forecaster would have read.
DEFAULT_LOOKBACK_MONTHS = 18

# Unit-key shapes across the five sources, most specific first. Each captures
# a year and a month (a quarter is mapped to its first month), and optionally
# a trailing depth index -- CC-NEWS's WARC file ordinal, GDELT's query index.
_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<year>\d{4})[/-](?P<month>\d{2})(?:[#:](?P<depth>\d+))?$"),
    re.compile(r"^(?P<year>\d{4})[Qq](?P<quarter>[1-4])$"),
)


def month_index(moment: datetime) -> int:
    """Months since year 0 -- the integer that makes month arithmetic exact.

    Preserves the invariant that comparisons between a unit key and a cutoff
    never involve a timezone, a day-of-month or a leap year, because every one
    of those is a way for an off-by-one to become a silent leak.
    """
    return moment.year * 12 + (moment.month - 1)


def unit_month(unit_key: str) -> int | None:
    """The month a unit covers, or ``None`` if the key is not date-shaped.

    Wikipedia units are keyed by scenario and are already anchored at that
    scenario's cutoff (ADR-0010), so they carry no month and are never
    reordered by demand.
    """
    for pattern in _KEY_PATTERNS:
        match = pattern.match(unit_key)
        if match is None:
            continue
        groups = match.groupdict()
        year = int(groups["year"])
        if groups.get("quarter") is not None:
            return year * 12 + (int(groups["quarter"]) - 1) * 3
        return year * 12 + (int(groups["month"]) - 1)
    return None


def unit_depth(unit_key: str) -> int:
    """The unit's position within its month, or 0 when it is the whole month.

    Depth is what makes a breadth-first pass expressible: every source's
    depth-0 unit is ingested for every month before any month's depth-1 unit
    is touched, so an ingest interrupted at any point has covered the span
    rather than the prefix.
    """
    for pattern in _KEY_PATTERNS:
        match = pattern.match(unit_key)
        if match is None:
            continue
        depth = match.groupdict().get("depth")
        return int(depth) if depth is not None else 0
    return 0


def demand_profile(
    cutoffs: Iterable[datetime],
    *,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
) -> dict[int, float]:
    """Evidence demand per month, from the scenario cutoffs alone.

    A month earns weight from every scenario whose cutoff falls within
    ``lookback_months`` after it, decaying linearly with the gap: the month
    immediately before a cutoff is worth ``1.0`` to that scenario and the
    oldest month in the window is worth nearly nothing. Months at or after a
    cutoff earn nothing from it -- they are inadmissible for that scenario by
    construction, and weighting them would be asking the ingest to fetch
    documents Chronofence will refuse to return.

    Returns a mapping keyed by :func:`month_index`; absent months have zero
    demand.
    """
    if lookback_months <= 0:
        raise ValueError(f"lookback_months must be positive, got {lookback_months}")
    profile: dict[int, float] = {}
    for cutoff in cutoffs:
        target = month_index(cutoff)
        for gap in range(lookback_months):
            # gap 0 is the month the cutoff falls in: partially admissible, and
            # the most valuable month there is.
            month = target - gap
            weight = (lookback_months - gap) / lookback_months
            profile[month] = profile.get(month, 0.0) + weight
    return profile


@dataclass(frozen=True, slots=True)
class UnitPlan:
    """One pending unit and why it sits where it does in the queue."""

    unit_key: str
    depth: int
    demand: float
    month: int | None


def order_units(
    unit_keys: Sequence[str],
    *,
    demand: Mapping[int, float],
) -> list[UnitPlan]:
    """Order pending units breadth-first over months, by scenario demand.

    The sort key is ``(depth, -demand, unit_key)``:

    * **depth first** so the queue sweeps the whole span before deepening any
      month. Depth-major ordering is what makes an interrupted ingest leave a
      corpus shaped like the study rather than like its first quarter.
    * **demand next** so within a sweep the months the scenarios actually need
      are reached first.
    * **key last** so the order is total and reproducible; two units with the
      same depth and demand never swap between runs (invariant 7).

    Units with no month -- Wikipedia's per-scenario snapshots -- sort after
    every dated unit at the same depth, because their demand is unmeasurable
    here and they are already cutoff-anchored.
    """
    plans = [
        UnitPlan(
            unit_key=key,
            depth=unit_depth(key),
            demand=demand.get(month, 0.0) if (month := unit_month(key)) is not None else -1.0,
            month=unit_month(key),
        )
        for key in unit_keys
    ]
    return sorted(plans, key=lambda plan: (plan.depth, -plan.demand, plan.unit_key))
