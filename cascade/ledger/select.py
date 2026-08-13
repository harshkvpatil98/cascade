"""Deterministic selection of the study set (spec §3.1, CLAUDE.md Q2).

Pure: no I/O, no clock, no global RNG.

**Q2 is resolved here.** §3.1 requires, simultaneously, exactly 180 scenarios,
a resolved-YES rate in [0.40, 0.60] *by construction*, no domain above 25%, and
>= 3 distinguishable parties. The spec does not say what gives when the
available pool cannot satisfy all four. The precedence this module enforces,
weakest constraint sacrificed first:

1. **Eligibility is absolute** -- binary and unambiguous, >= 3 parties, a
   horizon over 21 days. These define what a scenario *is*. Relaxing one does
   not shrink the study, it changes what the study is about.
2. **Base rate in [0.40, 0.60]** is next. It is the only constraint protecting
   the headline number from a degenerate always-predict-the-base-rate model.
   Outside it, Brier is not interpretable and neither is any comparison drawn
   from it.
3. **Domain cap at 25%** is next. A breach is visible in the per-domain table
   the report prints anyway, so it degrades a result that can still be read
   honestly.
4. **Exactly 180 is sacrificed first.** Reducing N costs statistical power,
   which is disclosable and bounded. Admitting scenarios that violate 1-3, or
   dropping scenarios until the numbers work, is a selection effect on the
   headline metric -- and an undisclosable one, because nothing downstream
   could detect it.

So: the largest N <= 180 for which every other constraint holds, and a report
that states N. **Never a set that hits 180 by bending a rule.**
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from hashlib import blake2b

from cascade.ledger.rules import Rejected
from cascade.ledger.schema import ScenarioRecord

__all__ = ["SelectionReport", "SelectionResult", "deduplicate_event_groups", "select"]


@dataclass(frozen=True)
class SelectionReport:
    """What the selection did, in enough detail to audit it afterwards."""

    n_candidates: int
    n_eligible: int
    n_selected: int
    n_yes: int
    target_n: int
    domain_counts: tuple[tuple[str, int], ...]
    party_rule_counts: tuple[tuple[str, int], ...]
    source_counts: tuple[tuple[str, int], ...]
    rejection_counts: tuple[tuple[str, int], ...]
    dropped_duplicate_groups: int
    shortfall_reason: str = ""

    @property
    def yes_rate(self) -> float:
        return self.n_yes / self.n_selected if self.n_selected else 0.0

    @property
    def max_domain_share(self) -> float:
        if not self.n_selected:
            return 0.0
        return max(count for _, count in self.domain_counts) / self.n_selected

    @property
    def met_target(self) -> bool:
        return self.n_selected == self.target_n


@dataclass(frozen=True)
class SelectionResult:
    records: tuple[ScenarioRecord, ...] = ()
    report: SelectionReport = field(
        default_factory=lambda: SelectionReport(0, 0, 0, 0, 0, (), (), (), (), 0, "empty")
    )


def _order_key(scenario_id: str, salt: str) -> bytes:
    """Seeded, outcome-independent ordering key.

    Ordering by anything the sources supply -- volume, recency, popularity --
    would correlate selection with question difficulty and with the outcome.
    A keyed hash of the id is reproducible across machines and processes and
    is uncorrelated with both.
    """
    return blake2b(scenario_id.encode("utf-8"), key=salt.encode("utf-8"), digest_size=16).digest()


def deduplicate_event_groups(
    records: tuple[ScenarioRecord, ...], *, salt: str
) -> tuple[tuple[ScenarioRecord, ...], int]:
    """Keep at most one scenario per real-world event.

    Preserves the independence the 180-scenario claim rests on. A market group
    like "who wins the 2024 presidential election" carries one binary market
    per candidate: they are the *same* event, their outcomes are mutually
    determined, and admitting several would inflate the effective sample size
    while the paired bootstrap over scenarios silently treats them as
    independent draws.

    Ungrouped scenarios are always kept -- ``event_group`` is ``None`` for
    curated and standalone questions, which are independent by construction.
    """
    ordered = sorted(records, key=lambda record: _order_key(record.scenario.scenario_id, salt))
    seen: set[str] = set()
    kept: list[ScenarioRecord] = []
    dropped = 0
    for record in ordered:
        group = record.scenario.event_group
        if group is None:
            kept.append(record)
            continue
        if group in seen:
            dropped += 1
            continue
        seen.add(group)
        kept.append(record)
    return tuple(kept), dropped


def _feasible_yes_count(n: int, y_max: int, n_max: int, lo: float, hi: float) -> int | None:
    """Return the YES count closest to balance that satisfies the base rate.

    ``None`` when no split of size ``n`` can land inside [lo, hi] given the
    pools available.
    """
    lowest = max(math.ceil(lo * n), n - n_max, 0)
    highest = min(math.floor(hi * n), y_max, n)
    if lowest > highest:
        return None
    balanced = n // 2
    return min(max(balanced, lowest), highest)


def _greedy_fill(
    ordered: tuple[ScenarioRecord, ...], *, n_yes: int, n_no: int, domain_cap: int
) -> tuple[ScenarioRecord, ...] | None:
    """Take records in order until both outcome buckets are full.

    Greedy rather than exhaustive: a search over domain-cap-feasible subsets is
    exponential, and a deterministic greedy pass that occasionally reports a
    smaller N than some clever packing could achieve is the conservative error
    to make. It never admits a scenario that violates a constraint.
    """
    need = {0: n_no, 1: n_yes}
    per_domain: Counter[str] = Counter()
    chosen: list[ScenarioRecord] = []
    for record in ordered:
        outcome = record.label.outcome
        if need[outcome] == 0:
            continue
        domain = record.scenario.domain
        if per_domain[domain] >= domain_cap:
            continue
        need[outcome] -= 1
        per_domain[domain] += 1
        chosen.append(record)
        if need[0] == 0 and need[1] == 0:
            return tuple(chosen)
    return None


def select(
    records: tuple[ScenarioRecord, ...],
    *,
    rejections: tuple[Rejected, ...],
    n_candidates: int,
    target_n: int,
    yes_rate_min: float,
    yes_rate_max: float,
    max_domain_share: float,
    salt: str,
) -> SelectionResult:
    """Choose the largest constraint-satisfying set of at most ``target_n``.

    Preserves the Q2 precedence documented in this module's docstring: the
    scenario count is the only thing that gives.
    """
    deduped, dropped = deduplicate_event_groups(records, salt=salt)
    ordered = tuple(sorted(deduped, key=lambda r: _order_key(r.scenario.scenario_id, salt)))

    available_yes = sum(1 for record in ordered if record.label.outcome == 1)
    available_no = len(ordered) - available_yes

    def build_report(chosen: tuple[ScenarioRecord, ...], shortfall: str) -> SelectionReport:
        return SelectionReport(
            n_candidates=n_candidates,
            n_eligible=len(ordered),
            n_selected=len(chosen),
            n_yes=sum(1 for record in chosen if record.label.outcome == 1),
            target_n=target_n,
            domain_counts=tuple(sorted(Counter(r.scenario.domain for r in chosen).items())),
            party_rule_counts=tuple(sorted(Counter(r.scenario.party_rule for r in chosen).items())),
            source_counts=tuple(sorted(Counter(r.scenario.source for r in chosen).items())),
            rejection_counts=tuple(sorted(Counter(r.reason for r in rejections).items())),
            dropped_duplicate_groups=dropped,
            shortfall_reason=shortfall,
        )

    for size in range(min(target_n, len(ordered)), 0, -1):
        yes_count = _feasible_yes_count(
            size, available_yes, available_no, yes_rate_min, yes_rate_max
        )
        if yes_count is None:
            continue
        domain_cap = max(1, math.floor(max_domain_share * size))
        chosen = _greedy_fill(
            ordered, n_yes=yes_count, n_no=size - yes_count, domain_cap=domain_cap
        )
        if chosen is None:
            continue
        shortfall = (
            ""
            if size == target_n
            else (
                f"pool yields {size} of {target_n} under the inclusion rules "
                f"({len(ordered)} eligible, {available_yes} YES / {available_no} NO, "
                f"domain cap {domain_cap}). Q2 precedence: the count gives, "
                "never the rules."
            )
        )
        return SelectionResult(records=chosen, report=build_report(chosen, shortfall))

    return SelectionResult(
        report=build_report(
            (),
            f"no constraint-satisfying set exists: {len(ordered)} eligible "
            f"({available_yes} YES / {available_no} NO)",
        )
    )
