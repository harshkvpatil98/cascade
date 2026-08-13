"""Boundary types for the scenario registry (spec §3.1, Appendix A).

Two shapes matter here and they are deliberately separate.

``RawQuestion`` is what a source produces: normalised across Metaculus,
Polymarket, Manifold and the curated set, but not yet screened. It still
carries the outcome, because a loader has to read it to normalise it.

``Scenario`` and ``ScenarioLabel`` are what the registry stores, and they are
split exactly as the database splits them (invariant 2). Nothing that reaches
the simulation carries an outcome field at all -- the type system says so
before the Postgres grant does.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "MIN_HORIZON",
    "Domain",
    "PartyRule",
    "RawQuestion",
    "RejectionReason",
    "Scenario",
    "ScenarioLabel",
    "ScenarioRecord",
    "SourceName",
]

# Spec Appendix A: `CHECK (resolve_ts > cutoff_ts + interval '21 days')`.
# §3.1 prose says "at least 21 days"; the DDL is the executable form and is
# strictly greater, so the Python rule matches the DDL exactly rather than
# admitting a scenario the database would then reject.
MIN_HORIZON = timedelta(days=21)

SourceName = Literal["metaculus", "polymarket", "manifold", "curated"]

Domain = Literal[
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
]

PartyRule = Literal[
    "curated",  # authored with an explicit party list
    "event_siblings",  # >= 3 mutually exclusive outcomes in one real-world event
    "named_parties",  # >= 3 distinct recognised actors in the question text
]

RejectionReason = Literal[
    "not_binary",
    "unresolved",
    "ambiguous_resolution",
    "single_quantity",
    "insufficient_parties",
    "horizon_too_short",
    "cutoff_not_derivable",
    "low_volume",
    "duplicate_event_group",
    "missing_timestamp",
    "naive_timestamp",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_aware(value: datetime, field: str) -> datetime:
    """Reject a timezone-naive timestamp.

    Preserves invariant 1's premise: a naive timestamp has no defined instant,
    so any ``as_of`` comparison against it is a guess. The corpus applies the
    same rule to ``published_at`` at M2 -- a document with an uncertain date is
    dropped, never inferred.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware; got a naive datetime")
    return value


class RawQuestion(_Frozen):
    """One candidate question, normalised across sources but not yet screened.

    Carries the outcome because a loader must read it to normalise it. This
    type never reaches the simulation: :func:`split` produces the two records
    that do.
    """

    source: SourceName
    source_ref: str
    question: str = Field(min_length=1)
    resolution_criterion: str
    open_ts: datetime
    close_ts: datetime
    resolved_at: datetime
    outcome: Literal[0, 1]
    # Source-native units: USD notional for Polymarket, mana for Manifold,
    # zero for non-market sources. Not comparable across sources and never
    # summed -- it is a per-source "non-trivial activity" screen only.
    volume: float = Field(default=0.0, ge=0.0)
    event_group: str | None = None
    # Number of mutually exclusive sibling outcomes in the same real-world
    # event. Three or more establishes >= 3 distinguishable parties with
    # non-identical objectives structurally rather than by reading the text.
    event_sibling_count: int = Field(default=1, ge=1)
    curated_parties: tuple[str, ...] = ()
    curated_domain: Domain | None = None
    # Curated entries name their own cutoff. Choosing the moment at which
    # "enough public information existed to reason but the outcome was
    # genuinely unsettled" (spec §3.1) is a judgement about a specific
    # episode -- it is the substance of curation, and a proportional rule
    # derived for market data cannot make it.
    explicit_cutoff: datetime | None = None
    provenance: str = ""

    @field_validator("open_ts", "close_ts", "resolved_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "timestamp")

    @field_validator("explicit_cutoff")
    @classmethod
    def _optional_timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_aware(value, "explicit_cutoff")


class Scenario(_Frozen):
    """A sealed registry entry. Contains no outcome, by construction.

    The simulation reads this type. There is no field it could read an answer
    from, which is the type-level half of invariant 2; the Postgres grant is
    the half that survives someone adding one.
    """

    scenario_id: str = Field(min_length=1)
    question: str
    resolution_criterion: str
    cutoff_ts: datetime
    resolve_ts: datetime
    domain: Domain
    source: SourceName
    source_ref: str
    party_rule: PartyRule
    party_names: tuple[str, ...]
    event_group: str | None

    @field_validator("cutoff_ts", "resolve_ts")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "timestamp")

    def horizon(self) -> timedelta:
        return self.resolve_ts - self.cutoff_ts


class ScenarioLabel(_Frozen):
    """The outcome. Stored in its own table, granted only to ``cascade_eval``."""

    scenario_id: str
    outcome: Literal[0, 1]
    resolved_at: datetime

    @field_validator("resolved_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "resolved_at")


class ScenarioRecord(_Frozen):
    """A scenario and its label, held together only during registry assembly.

    Assembly needs both -- the base-rate control is a property of the outcome
    distribution. Everything downstream takes the halves separately, so this
    type exists in one module's working set and is never persisted whole.
    """

    scenario: Scenario
    label: ScenarioLabel
