"""Inclusion rules as executable predicates (spec §3.1).

Pure: no I/O, no clock, no RNG. The rules are enforced in code rather than
prose because a prose rule is one a tired operator can decide not to apply to
"just this one" borderline question, and a backtest is only as honest as its
question set.

Every rejection carries a reason. The counts by reason are printed by
``cascade ledger build`` and stored with the manifest, so the composition of
the set -- and what it excluded -- is auditable rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from cascade.ledger.schema import (
    MIN_HORIZON,
    Domain,
    PartyRule,
    RawQuestion,
    RejectionReason,
    Scenario,
    ScenarioLabel,
    ScenarioRecord,
)
from cascade.ledger.taxonomy import (
    classify_domain,
    extract_known_actors,
    extract_parties,
    is_single_quantity,
)

__all__ = [
    "MIN_PARTIES",
    "Rejected",
    "ScreenOutcome",
    "derive_cutoff",
    "screen",
]

# Spec §3.1: "At least three distinguishable parties with non-identical
# objectives." Two-party questions are the ones a single model already handles;
# the whole claim of this system is about what happens above that.
MIN_PARTIES = 3


@dataclass(frozen=True)
class Rejected:
    """Why one candidate did not enter the set."""

    source_ref: str
    reason: RejectionReason
    detail: str = ""


ScreenOutcome = ScenarioRecord | Rejected


def derive_cutoff(*, open_ts: datetime, resolve_ts: datetime, fraction: float) -> datetime | None:
    """Place the cutoff a fixed fraction into the question's public life.

    Preserves the two halves of spec §3.1's cutoff requirement as well as they
    can be preserved without price history: *enough public information existed
    to reason* (a fraction of the question's life has already elapsed, so the
    evidence corpus has something to say) and *the horizon is meaningful*
    (what remains is checked against the 21-day rule by the caller).

    A proportional cutoff rather than a fixed offset, because a fixed offset
    would sit near the open of a long-running question and near the close of a
    short one -- systematically different amounts of evidence for reasons that
    have nothing to do with the question.

    Returns ``None`` when the interval is degenerate, so the caller rejects
    rather than fabricating a timestamp.

    **Stated limitation.** "Genuinely unsettled" is not verified here. Doing so
    needs the price series at the cutoff, which neither source returns without
    a per-question request; the M3 parametric probe measures the related risk
    directly and reports it.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"cutoff fraction must be strictly between 0 and 1; got {fraction}")
    if resolve_ts <= open_ts:
        return None
    span = resolve_ts - open_ts
    return open_ts + timedelta(seconds=span.total_seconds() * fraction)


def _party_verdict(raw: RawQuestion) -> tuple[PartyRule, tuple[str, ...]] | None:
    """Establish >= 3 distinguishable parties, or return ``None``.

    Ordered strongest evidence first. Structural evidence beats textual
    evidence: three mutually exclusive outcomes in one event *are* three
    parties with non-identical objectives, whereas three capitalised names in
    a sentence merely suggest it.
    """
    text = f"{raw.question} {raw.resolution_criterion}"
    if len(raw.curated_parties) >= MIN_PARTIES:
        return "curated", tuple(sorted(raw.curated_parties))
    if raw.event_sibling_count >= MIN_PARTIES:
        # Structure already established the parties; the extracted names are
        # recorded for audit but are not what admitted the question.
        return "event_siblings", extract_parties(text)
    # Recognised institutional actors only -- not proper nouns. See
    # taxonomy.extract_known_actors for why the distinction is load-bearing.
    actors = extract_known_actors(text)
    if len(actors) >= MIN_PARTIES:
        return "named_parties", actors
    return None


def screen(
    raw: RawQuestion,
    *,
    scenario_id: str,
    cutoff_fraction: float,
    min_volume: float,
) -> ScreenOutcome:
    """Apply every inclusion rule to one candidate.

    Returns the assembled record, or the first rule it failed. Rules are
    ordered cheapest-and-most-decisive first so the rejection histogram is
    dominated by the reason that actually disqualifies most of the pool.
    """
    # Market sources only. Metaculus is not a market and the curated set is
    # authored, so neither has a volume to screen on.
    if raw.source in ("polymarket", "manifold") and raw.volume < min_volume:
        return Rejected(raw.source_ref, "low_volume", f"{raw.volume:.0f}")

    text = f"{raw.question} {raw.resolution_criterion}"

    if is_single_quantity(raw.question):
        return Rejected(raw.source_ref, "single_quantity", raw.question[:100])

    verdict = _party_verdict(raw)
    if verdict is None:
        return Rejected(raw.source_ref, "insufficient_parties", raw.question[:100])
    party_rule, party_names = verdict

    cutoff = raw.explicit_cutoff or derive_cutoff(
        open_ts=raw.open_ts, resolve_ts=raw.resolved_at, fraction=cutoff_fraction
    )
    if cutoff is None:
        return Rejected(raw.source_ref, "cutoff_not_derivable", "resolve_ts <= open_ts")
    if cutoff <= raw.open_ts:
        return Rejected(raw.source_ref, "cutoff_not_derivable", "cutoff precedes the open")

    # Strictly greater, matching the Appendix A CHECK constraint exactly. A
    # scenario that passed here and failed there would abort the load halfway.
    if raw.resolved_at - cutoff <= MIN_HORIZON:
        return Rejected(
            raw.source_ref,
            "horizon_too_short",
            f"{(raw.resolved_at - cutoff).days}d <= {MIN_HORIZON.days}d",
        )

    domain: Domain = raw.curated_domain or classify_domain(text)

    scenario = Scenario(
        scenario_id=scenario_id,
        question=raw.question.strip(),
        resolution_criterion=raw.resolution_criterion.strip(),
        cutoff_ts=cutoff,
        resolve_ts=raw.resolved_at,
        domain=domain,
        source=raw.source,
        source_ref=raw.source_ref,
        party_rule=party_rule,
        party_names=party_names,
        event_group=raw.event_group,
    )
    label = ScenarioLabel(scenario_id=scenario_id, outcome=raw.outcome, resolved_at=raw.resolved_at)
    return ScenarioRecord(scenario=scenario, label=label)
