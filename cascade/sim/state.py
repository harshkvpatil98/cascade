"""World state for one simulation run (spec §7.1).

State is small, typed and fully serializable, because two things depend on it
being exactly that: a checkpoint written every step across 36,000 runs has to
be cheap, and the M5 acceptance criterion is that it round-trips *exactly* --
not approximately, not up to float formatting.

Three fields are here that §7.1's listing does not name, and each is required
by a rule stated elsewhere in the spec (ADR-0017):

``pools``
    §7.4 gives ALLY the effect "pools resources for contests next step" and
    DEFECT "releases pooled resources back". Neither is expressible without a
    ledger of who pledged what to whom, and folding pledges into ``resources``
    would double-count them in the conservation check.
``last_acted`` / ``last_observed_at``
    §7.3 activates an actor when an observed factor "moved by more than 0.06
    since that actor's last observation" and forces one action "every 5
    steps". Both are per-actor history that the scheduler cannot recompute
    from ``factors`` alone.
``pinned_streak``
    §7.6 terminates early on "a factor pinned at a bound for 3 consecutive
    steps", which is a streak, not a predicate on the current state.

Every field is treated as immutable: the step loop builds a new ``WorldState``
per stage rather than mutating one in place. Pydantic's ``frozen=True`` stops
field rebinding but not mutation of a dict a field points at, so the rule is
enforced by the helpers in this module always constructing fresh containers,
and by :func:`state_hash` being cheap enough to assert against in tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.canonical import canonical_json

__all__ = [
    "RELATION_SEPARATOR",
    "PendingEffect",
    "WorldState",
    "initial_state",
    "relation_key",
    "relation_pair",
    "staggered_schedule",
    "state_hash",
    "trust_between",
]

# ADR-0003: `dict[tuple[str, str], float]` as §7.1 writes it cannot round-trip
# through JSON -- json.dumps renders a tuple key as the string "('a', 'b')" and
# json.loads gives back that string, so a checkpoint resume would silently grow
# a second, differently-keyed relation matrix. The canonical key is the two
# actor ids joined by a separator no slug can contain (the `Slug` pattern in
# `cascade.decompose.schema` admits only [a-z0-9_-]).
RELATION_SEPARATOR = "|"

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Trust = Annotated[float, Field(ge=-1.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def relation_key(actor_a: str, actor_b: str) -> str:
    """Return the canonical key for the trust between two actors.

    Trust is mutual (§7.4: ALLY "raises mutual trust", DEFECT "collapses
    trust"), so the pair is unordered and the key sorts its members. Storing
    both directions would let them drift apart, and nothing in the spec says
    which one the contest should then read.
    """
    if actor_a == actor_b:
        raise ValueError(f"an actor has no trust relation with itself: {actor_a!r}")
    for part in (actor_a, actor_b):
        if RELATION_SEPARATOR in part:
            raise ValueError(
                f"actor id {part!r} contains the relation separator {RELATION_SEPARATOR!r}; "
                "the key would be ambiguous"
            )
    low, high = sorted((actor_a, actor_b))
    return f"{low}{RELATION_SEPARATOR}{high}"


def relation_pair(key: str) -> tuple[str, str]:
    """Split a canonical relation key back into its two actor ids."""
    left, separator, right = key.partition(RELATION_SEPARATOR)
    if not separator:
        raise ValueError(f"not a relation key: {key!r}")
    return left, right


def trust_between(relations: Mapping[str, float], actor_a: str, actor_b: str) -> float:
    """Trust between two actors, defaulting to 0.0 (neither allied nor hostile).

    Absent is neutral rather than an error: a 14-actor graph has 91 pairs and
    most never interact, so materialising every pair at step 0 would triple the
    size of a checkpoint to record zeros.
    """
    return float(relations.get(relation_key(actor_a, actor_b), 0.0))


class PendingEffect(_Frozen):
    """A lagged delta waiting for its arrival step (spec §7.2, stage ARRIVE).

    Carries its origin so the ARRIVE stage can attribute the movement it
    applies. Without that, a factor that moves three steps after the decision
    that moved it has no traceable cause and §11.2's chain stops short.
    """

    arrive_step: int = Field(ge=0)
    factor_id: str
    delta: float
    source_actor_id: str | None = None
    """The actor whose action scheduled this, or None for factor propagation."""
    origin_step: int = Field(ge=0)
    origin_seq: int = Field(ge=0)

    def sort_key(self) -> tuple[int, str, int, int, str]:
        """Total order over pending effects, so ARRIVE never depends on insertion order."""
        return (
            self.arrive_step,
            self.factor_id,
            self.origin_step,
            self.origin_seq,
            self.source_actor_id or "",
        )


class WorldState(_Frozen):
    """Everything an agent could ever influence. Nothing else (spec §7.1)."""

    step: int = Field(ge=0)
    factors: dict[str, Unit]
    factor_history: dict[str, tuple[Unit, ...]]
    """``factor_history[f][t]`` is the value of ``f`` at the *start* of step t.

    Indexed by step rather than by recency so a lag-2 observation at step 9
    reads index 7 directly. Recency indexing would silently shift the whole
    observation model by one step at the start of a run, when the history is
    shorter than the lag.
    """
    resources: dict[str, dict[str, Unit]]
    relations: dict[str, Trust]
    pools: dict[str, dict[str, Unit]] = Field(default_factory=dict)
    """``pools[ally][backer]`` -- resource pledged by ``backer`` to ``ally``."""
    pending: tuple[PendingEffect, ...] = ()
    rng_counter: int = Field(default=0, ge=0)
    last_acted: dict[str, int] = Field(default_factory=dict)
    """actor -> the step at which it last acted. Absent means "never acted"."""
    last_observed_at: dict[str, int] = Field(default_factory=dict)
    """actor -> the step at which it last observed. Absent means "never".

    The step, not the values it saw. §7.3's trigger is "an observed factor
    moved by more than 0.06 since that actor's last observation", and the
    scheduler evaluates that against the factor's own trajectory between the
    two steps rather than against the noisy reading the actor took. Comparing
    against the noisy reading would make a two-hop channel (sigma 0.15) fire
    the 0.06 trigger on noise alone, so the activation rate -- and with it the
    per-run cost -- would be a function of the visibility policy. §6.4 needs
    the asymmetry switch to change what agents *believe*, not how often they
    are woken up by their own sensor noise.
    """
    pinned_streak: dict[str, int] = Field(default_factory=dict)
    """factor -> consecutive steps at 0.0 or 1.0 (spec §7.6)."""

    def factor_at(self, factor_id: str, step: int) -> float:
        """Value of ``factor_id`` as of ``step``, clamped to the recorded history.

        "As of step t" means the value after that step's exogenous walk and
        arriving effects -- what an actor with zero lag observes at step t.
        Actions taken during step t land in step t+1's entry, which is what
        makes an action observable to others one step after it is taken.

        A lag that reaches before step 0 returns the opening state rather than
        raising: an actor observing with lag 2 at step 1 has no older state to
        see, and inventing one would be worse than showing it the start.
        """
        history = self.factor_history.get(factor_id)
        if not history:
            raise KeyError(f"no history for factor {factor_id!r}")
        index = max(0, min(step, len(history) - 1))
        return float(history[index])

    def budget(self, actor_id: str) -> float:
        """Total resource left to an actor, summed over its resource kinds.

        The contest in §7.5 takes a single scalar ``r`` per push, while §5.1
        gives actors a named resource bundle. Summing is the only reading that
        keeps conservation checkable: the arbiter debits the bundle
        proportionally (see :mod:`cascade.sim.arbiter`) so the parts and the
        total cannot disagree.
        """
        kinds = self.resources.get(actor_id, {})
        return float(sum(kinds[name] for name in sorted(kinds)))

    def pooled(self, actor_id: str) -> float:
        """Resource pledged to ``actor_id`` by allies, available for contests."""
        pledges = self.pools.get(actor_id, {})
        return float(sum(pledges[name] for name in sorted(pledges)))

    def canonical(self) -> dict[str, Any]:
        """Sorted, primitive projection of the state -- the input to :func:`state_hash`."""
        return {
            "step": self.step,
            "factors": {key: self.factors[key] for key in sorted(self.factors)},
            "factor_history": {
                key: list(self.factor_history[key]) for key in sorted(self.factor_history)
            },
            "resources": {
                actor: {kind: self.resources[actor][kind] for kind in sorted(self.resources[actor])}
                for actor in sorted(self.resources)
            },
            "relations": {key: self.relations[key] for key in sorted(self.relations)},
            "pools": {
                actor: {backer: self.pools[actor][backer] for backer in sorted(self.pools[actor])}
                for actor in sorted(self.pools)
            },
            "pending": [
                {
                    "arrive_step": effect.arrive_step,
                    "factor_id": effect.factor_id,
                    "delta": effect.delta,
                    "source_actor_id": effect.source_actor_id,
                    "origin_step": effect.origin_step,
                    "origin_seq": effect.origin_seq,
                }
                for effect in sorted(self.pending, key=PendingEffect.sort_key)
            ],
            "rng_counter": self.rng_counter,
            "last_acted": {key: self.last_acted[key] for key in sorted(self.last_acted)},
            "last_observed_at": {
                actor: self.last_observed_at[actor] for actor in sorted(self.last_observed_at)
            },
            "pinned_streak": {key: self.pinned_streak[key] for key in sorted(self.pinned_streak)},
        }


def state_hash(state: WorldState) -> str:
    """Content hash of a world state.

    Used by the determinism suite to localise a replay divergence to the step
    that produced it. Hashing the canonical projection rather than the pydantic
    dump means the hash does not change when a field is reordered in the model,
    which would otherwise read as a divergence.
    """
    return hashlib.blake2b(
        canonical_json(state.canonical()).encode("utf-8"), digest_size=16
    ).hexdigest()


def initial_state(graph: Any, *, forced_interval: int) -> WorldState:
    """Build step-0 state from a compiled graph (spec §7.1, §5.1).

    Factor values, volatility and inertia come from the graph; resources come
    from each actor's declared bundle. Relations start empty -- ``trust_between``
    reads an absent pair as 0.0 -- so a 14-actor run does not open with 91 rows
    of zero.

    ``last_acted`` opens **staggered** rather than uniform (ADR-0017). §7.3
    requires every actor to act "at least once every 5 steps"; initialising the
    whole cast as equally overdue satisfies that too, but it makes all 14
    actors fall due on the same step, where the cap of 8 then decides who acts
    by salience and pushes the rest into a synchronised pulse every 5 steps.
    Staggering by sorted position spreads the floor across the interval, which
    is the behaviour "no party goes fully silent" is asking for, and it is a
    deterministic function of the graph so it replays identically.

    Typed as ``Any`` to keep :mod:`cascade.sim.state` free of a dependency on
    the decomposition package: the kernel is the place those two meet, and a
    cycle between them would make the pure core import the compiler.
    """
    factors = {factor.id: float(factor.state) for factor in sorted(graph.factors, key=_by_id)}
    actor_ids = sorted(actor.id for actor in graph.actors)
    return WorldState(
        step=0,
        factors=dict(factors),
        factor_history={key: (factors[key],) for key in sorted(factors)},
        resources={
            actor.id: {kind: float(actor.resources[kind]) for kind in sorted(actor.resources)}
            for actor in sorted(graph.actors, key=_by_id)
        },
        relations={},
        pools={},
        pending=(),
        rng_counter=0,
        last_acted=staggered_schedule(actor_ids, forced_interval),
        last_observed_at={},
        pinned_streak={},
    )


def staggered_schedule(actor_ids: Sequence[str], forced_interval: int) -> dict[str, int]:
    """Opening ``last_acted`` values that spread the forced floor over the interval.

    Actor *i* in sorted order is first due at step ``interval - 1 - (i % interval)``,
    so an interval of 5 has roughly a fifth of the cast due on each of the
    first five steps and every actor has acted by step 4 -- the floor §7.3
    states, reached without a cohort of 14 arriving at the cap together.
    """
    if forced_interval <= 0:
        return {actor_id: -1 for actor_id in sorted(actor_ids)}
    ordered = sorted(actor_ids)
    return {actor_id: -(1 + (index % forced_interval)) for index, actor_id in enumerate(ordered)}


def _by_id(item: Any) -> str:
    return str(item.id)
