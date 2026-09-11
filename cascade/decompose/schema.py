"""The ``CausalGraph`` contract (spec §5.1).

The compiler's output is a typed artifact, not prose. Everything downstream --
agent construction at M5, the Aperture visibility policy at §6.2, arbiter
dynamics at §7.5 -- is derived from this mechanically. Spec §5.1 puts it
plainly: if this schema is loose, the whole system degrades into a chat room.

So the models here are strict in ways a draft-generating model will find
inconvenient, on purpose. Bounds are enforced at parse time (`extra="forbid"`,
`frozen=True`), cross-references are checked in
:meth:`CausalGraph.model_post_init`, and a graph that survives construction is
structurally sound before the validator in :mod:`cascade.decompose.validator`
looks at whether it is *sensible*.

``UtilityTerm`` and ``OutcomeRule`` are referenced by §5.1 but never defined
there; ADR-0014 records the design and why the outcome rule is monotone by
construction rather than by inspection.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cascade.canonical import canonical_json

__all__ = [
    "MAX_ACTORS",
    "MAX_FACTORS",
    "MIN_ACTORS",
    "MIN_FACTORS",
    "Actor",
    "CausalGraph",
    "Edge",
    "Factor",
    "OutcomeRule",
    "OutcomeTerm",
    "RiskPosture",
    "UtilityTerm",
    "graph_hash",
]

# Spec §5.1 / §5.3. Enforced on `CausalGraph` so a graph outside these bounds
# cannot be constructed at all; the validator reports the same bounds as a
# repairable violation, because the compiler needs a message rather than an
# exception when a draft comes back with six actors.
MIN_ACTORS = 8
MAX_ACTORS = 20
MIN_FACTORS = 4
MAX_FACTORS = 12

RiskPosture = Literal["averse", "neutral", "seeking"]

# See `OutcomeRule.__call__`: beyond this the float64 logistic saturates onto
# the bound it is documented never to reach.
_MAX_LOGIT = 36.0

# A slug: stable across recompilations, safe in a JSON key and in a prompt.
Slug = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
# Signed and non-zero: see `_reject_zero_weight`.
SignedWeight = Annotated[float, Field(ge=-1.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_zero_weight(value: float, field: str) -> float:
    """A zero weight claims a dependency that does not exist.

    It satisfies "depends on N factors" by arity while depending on N-1 in
    substance, and it makes the outcome rule flat in a factor it names --
    which is exactly the degenerate case §5.3's monotonicity rule exists to
    exclude (ADR-0014).
    """
    if value == 0.0:
        raise ValueError(f"{field} must be non-zero; a zero weight is a dependency in name only")
    return value


class UtilityTerm(_Frozen):
    """One weighted reference from an actor's utility to a factor.

    The weight is signed: an actor that wants a factor *low* carries a negative
    weight on that factor rather than a positive weight on an inverted twin.
    M5's agents read this mechanically, so the direction has to live in a
    number and not in prose (ADR-0014).
    """

    factor_id: Slug
    weight: SignedWeight

    @field_validator("weight")
    @classmethod
    def _non_zero(cls, value: float) -> float:
        return _reject_zero_weight(value, "UtilityTerm.weight")


class Actor(_Frozen):
    """A party with objectives, resources and constraints (spec §5.1)."""

    id: Slug
    name: Annotated[str, Field(min_length=1, max_length=200)]
    objective: Annotated[str, Field(min_length=1, max_length=400)]
    utility_terms: Annotated[tuple[UtilityTerm, ...], Field(min_length=1)]
    resources: dict[str, Unit] = Field(default_factory=dict)
    constraints: tuple[Annotated[str, Field(min_length=1, max_length=300)], ...] = ()
    risk_posture: RiskPosture
    initial_beliefs: dict[str, Unit] = Field(default_factory=dict)

    @field_validator("utility_terms")
    @classmethod
    def _one_term_per_factor(cls, value: tuple[UtilityTerm, ...]) -> tuple[UtilityTerm, ...]:
        """Two terms on one factor are an ambiguous preference, not a stronger one."""
        seen = [term.factor_id for term in value]
        if len(set(seen)) != len(seen):
            duplicated = sorted({item for item in seen if seen.count(item) > 1})
            raise ValueError(f"actor has repeated utility terms on factors: {duplicated}")
        return value


class Factor(_Frozen):
    """A world variable with its own exogenous dynamics (spec §5.1)."""

    id: Slug
    name: Annotated[str, Field(min_length=1, max_length=200)]
    state: Unit
    volatility: Annotated[float, Field(ge=0.0, le=1.0)]
    inertia: Unit
    observable_by_default: bool


class Edge(_Frozen):
    """A signed, lagged influence from an actor or factor onto a factor."""

    src: Slug
    dst: Slug
    sign: Literal[-1, 1]
    # (0, 1] per §5.1: an edge with zero weight is an edge that is not there.
    weight: Annotated[float, Field(gt=0.0, le=1.0)]
    lag: Annotated[int, Field(ge=0, le=6)]


class OutcomeTerm(_Frozen):
    """One monotone contribution of a factor to the outcome score."""

    factor_id: Slug
    weight: SignedWeight

    @field_validator("weight")
    @classmethod
    def _non_zero(cls, value: float) -> float:
        return _reject_zero_weight(value, "OutcomeTerm.weight")


class OutcomeRule(_Frozen):
    """Maps terminal factor state to a probability in (0, 1). See ADR-0014.

    ``p = sigmoid(steepness * Σ wᵢ (sᵢ - threshold))``. The partial derivative
    in each referenced factor is ``steepness · wᵢ · p · (1 - p)``, and since
    ``steepness > 0`` and ``p`` is strictly interior, its sign is exactly the
    sign of ``wᵢ`` everywhere. A rule that parses is strictly monotone in every
    factor it names -- §5.3's rule holds by construction, not by inspection.

    The range is open: a terminal score of exactly 0 or 1 asserts certainty and
    degenerates the M7 Brier decomposition for that scenario.
    """

    terms: Annotated[tuple[OutcomeTerm, ...], Field(min_length=2)]
    threshold: Unit
    steepness: Annotated[float, Field(gt=0.0, le=50.0)]

    @field_validator("terms")
    @classmethod
    def _distinct_factors(cls, value: tuple[OutcomeTerm, ...]) -> tuple[OutcomeTerm, ...]:
        """Two terms on one factor is one dependency wearing two hats.

        It would pass a naive "len(terms) >= 2" reading of §5.3 while the rule
        reduces to a random walk on a single driver -- the exact failure the
        rule exists to prevent.
        """
        ids = [term.factor_id for term in value]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"outcome rule repeats a factor: {sorted({i for i in ids if ids.count(i) > 1})}; "
                "it must depend on at least two distinct factors (spec §5.3)"
            )
        return value

    @property
    def factor_ids(self) -> tuple[str, ...]:
        return tuple(term.factor_id for term in self.terms)

    def __call__(self, factors: Mapping[str, float]) -> float:
        """Evaluate at terminal state. Spec §7.2 calls this at the end of a run.

        A missing factor raises rather than defaulting to 0.0 or 0.5: the rule
        names the factors it needs, and scoring a run against a state that does
        not contain one of them is a silent substitution of a guess for a
        measurement.
        """
        total = 0.0
        for term in self.terms:
            if term.factor_id not in factors:
                raise KeyError(
                    f"outcome rule references factor {term.factor_id!r}, absent from the "
                    f"terminal state (present: {sorted(factors)})"
                )
            total += term.weight * (float(factors[term.factor_id]) - self.threshold)
        # The logistic is mathematically open on (0, 1) for every finite
        # exponent, and float64 is not: at |x| >= 37, `1 + exp(-x)` rounds to
        # 1.0 and the result lands exactly on a bound. That matters here
        # because §7.6 drives runs toward states where factors pin at their
        # bounds, which is precisely where a rule with several strong weights
        # saturates -- and a terminal score of exactly 0 or 1 asserts certainty
        # and makes the M7 log-loss infinite for that scenario.
        #
        # 36 is the largest magnitude at which the result is still
        # representably distinct from 1.0 (measured: 1 - 2.22e-16 at 36,
        # exactly 1.0 at 37). Clamping there keeps the promise ADR-0014 makes
        # about the range, and it only touches inputs whose value was already
        # saturated to the bound.
        exponent = max(-_MAX_LOGIT, min(_MAX_LOGIT, -self.steepness * total))
        return 1.0 / (1.0 + math.exp(exponent))


class CausalGraph(_Frozen):
    """The compiled decomposition of one scenario (spec §5.1).

    Cross-reference integrity is enforced here rather than left to the
    validator: an edge pointing at a factor that does not exist is not a
    *defect the model should repair*, it is a graph that cannot be simulated,
    and letting it reach the repair loop would spend two LLM calls discovering
    what parsing already knew.
    """

    scenario_id: Annotated[str, Field(min_length=1)]
    actors: Annotated[tuple[Actor, ...], Field(min_length=MIN_ACTORS, max_length=MAX_ACTORS)]
    factors: Annotated[tuple[Factor, ...], Field(min_length=MIN_FACTORS, max_length=MAX_FACTORS)]
    edges: Annotated[tuple[Edge, ...], Field(min_length=1)]
    outcome_rule: OutcomeRule

    @property
    def actor_ids(self) -> frozenset[str]:
        return frozenset(actor.id for actor in self.actors)

    @property
    def factor_ids(self) -> frozenset[str]:
        return frozenset(factor.id for factor in self.factors)

    @model_validator(mode="after")
    def _references_resolve(self) -> CausalGraph:
        actor_ids = self.actor_ids
        factor_ids = self.factor_ids

        if len(actor_ids) != len(self.actors):
            raise ValueError("actor ids are not unique")
        if len(factor_ids) != len(self.factors):
            raise ValueError("factor ids are not unique")
        overlap = actor_ids & factor_ids
        if overlap:
            raise ValueError(
                f"ids shared between actors and factors: {sorted(overlap)}; "
                "an edge src would be ambiguous"
            )

        for actor in self.actors:
            for term in actor.utility_terms:
                if term.factor_id not in factor_ids:
                    raise ValueError(
                        f"actor {actor.id!r} has a utility term on unknown factor "
                        f"{term.factor_id!r}"
                    )
            for belief in sorted(actor.initial_beliefs):
                if belief not in factor_ids:
                    raise ValueError(
                        f"actor {actor.id!r} holds a belief about unknown factor {belief!r}"
                    )

        for edge in self.edges:
            if edge.src not in actor_ids and edge.src not in factor_ids:
                raise ValueError(f"edge src {edge.src!r} is neither an actor nor a factor")
            if edge.dst not in factor_ids:
                raise ValueError(f"edge dst {edge.dst!r} is not a factor (spec §5.1)")

        for outcome_term in self.outcome_rule.terms:
            if outcome_term.factor_id not in factor_ids:
                raise ValueError(
                    f"outcome rule references unknown factor {outcome_term.factor_id!r}"
                )

        seen: set[tuple[str, str, int]] = set()
        for edge in self.edges:
            key = (edge.src, edge.dst, edge.lag)
            if key in seen:
                raise ValueError(
                    f"duplicate edge {edge.src!r} -> {edge.dst!r} at lag {edge.lag}; "
                    "two influences at the same lag are one influence"
                )
            seen.add(key)

        return self

    def canonical(self) -> dict[str, Any]:
        """The graph as sorted, primitive data -- the input to :func:`graph_hash`.

        Every collection is sorted (invariant 7). The model's emission order is
        not part of the graph's identity, so two drafts that differ only in the
        order they listed actors must hash identically; otherwise the §5.3
        determinism rule would fail for a reason that has nothing to do with
        the content.
        """
        return {
            "scenario_id": self.scenario_id,
            "actors": [
                {
                    "id": actor.id,
                    "name": actor.name,
                    "objective": actor.objective,
                    "utility_terms": [
                        {"factor_id": term.factor_id, "weight": term.weight}
                        for term in sorted(actor.utility_terms, key=lambda t: t.factor_id)
                    ],
                    "resources": dict(sorted(actor.resources.items())),
                    "constraints": sorted(actor.constraints),
                    "risk_posture": actor.risk_posture,
                    "initial_beliefs": dict(sorted(actor.initial_beliefs.items())),
                }
                for actor in sorted(self.actors, key=lambda a: a.id)
            ],
            "factors": [
                {
                    "id": factor.id,
                    "name": factor.name,
                    "state": factor.state,
                    "volatility": factor.volatility,
                    "inertia": factor.inertia,
                    "observable_by_default": factor.observable_by_default,
                }
                for factor in sorted(self.factors, key=lambda f: f.id)
            ],
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "sign": edge.sign,
                    "weight": edge.weight,
                    "lag": edge.lag,
                }
                for edge in sorted(self.edges, key=lambda e: (e.src, e.dst, e.lag))
            ],
            "outcome_rule": {
                "terms": [
                    {"factor_id": outcome.factor_id, "weight": outcome.weight}
                    for outcome in sorted(self.outcome_rule.terms, key=lambda t: t.factor_id)
                ],
                "threshold": self.outcome_rule.threshold,
                "steepness": self.outcome_rule.steepness,
            },
        }


def graph_hash(graph: CausalGraph) -> str:
    """Content hash of a compiled graph (spec §5.3, determinism rule).

    Uses the one canonical serialisation the whole system agrees on, so a
    graph hashes identically across machines, processes and PYTHONHASHSEED
    values. A scenario compiles once for the entire study and this is the
    identity that makes "unchanged input" checkable.
    """
    payload = canonical_json(graph.canonical()).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=32).hexdigest()
