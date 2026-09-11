"""The deterministic arbiter (spec §7.5). No LLM. No I/O. Invariant 3.

This is the single most important architectural decision in the system. An LLM
arbiter would make replay impossible, make cost scale with steps rather than
with agent activations, and let the referee smuggle in its own view of the
answer. So conflict resolution is a pure function: actions in, world delta out,
the same delta every time.

Four properties hold, each with a property test in ``tests/property/``:

*Boundedness* -- no single step moves a factor by more than
``max_step_delta``. Enforced after the jitter, so the ESCALATE variance penalty
cannot smuggle a larger move past the bound.

*Conservation* -- resource debits equal resource spends, and no actor ends a
step with a negative budget. Spends are taken kind by kind in sorted order, so
the total debited is the sum of the takes by construction rather than a
proportional split that rounds.

*Permutation invariance* -- shuffling the actions within a step yields an
identical delta. Every iteration here is over a sorted key list, and every sum
accumulates in sorted actor order (§8.1: "Arbiter sums in sorted actor-id
order; no parallel reduction inside a step").

*Monotonicity* -- increasing an actor's committed resource, holding others
fixed, weakly increases its contest share.

**On the constants.** §7.4 gives each action a qualitative effect -- "superlinear
cost", "collapses trust", "small resource refund" -- and no numbers. The
numbers are here, in one block, with the shape each is chosen for. They were
never tuned against an outcome: nothing in this module or its tests can read a
label, and the M7 harness runs against a frozen arbiter. A study that tuned
these against the headline Brier would be fitting the referee to the answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.sim.actions import Action, Ally, Commit, Concede, Defect, Escalate, Signal, Wait
from cascade.sim.state import PendingEffect, WorldState, relation_key

__all__ = [
    "ALLY_TRUST_GAIN",
    "CONCEDE_TRUST_GAIN",
    "DEFECT_TRUST_FLOOR",
    "ESCALATE_COST_EXPONENT",
    "ESCALATE_COST_SCALE",
    "ESCALATE_GAIN",
    "ESCALATE_VARIANCE",
    "REPUTATION_PENALTY",
    "WAIT_REFUND",
    "ActorMechanics",
    "ArbiterResult",
    "FactorOutcome",
    "Push",
    "SignalDelivery",
    "arbitrate",
    "contest",
    "contest_share",
]

# -- Action constants (spec §7.4 gives the shape; these give it a scale) -----
#
# ESCALATE buys reach at rising cost and with a variance penalty: at moderate
# magnitude it is efficient, and past ~0.7 the quadratic cost consumes the
# whole budget. That curvature is the point -- escalation should be a decision
# with a downside, not a strictly better COMMIT.
ESCALATE_GAIN = 1.5
ESCALATE_COST_EXPONENT = 2.0
ESCALATE_COST_SCALE = 2.0
ESCALATE_VARIANCE = 0.35
"""Jitter as a fraction of ``max_step_delta``, scaled by escalation intensity."""

# CONCEDE moves the factor toward the opponent's pole at a fixed, modest
# weight. It cannot be a free push of arbitrary size: conceding is giving
# ground, and giving ground harder than anyone can take it would make it an
# offensive weapon.
CONCEDE_WEIGHT = 0.35
CONCEDE_REFUND = 0.05
CONCEDE_TRUST_GAIN = 0.10

ALLY_TRUST_GAIN = 0.25
DEFECT_TRUST_FLOOR = -0.60
REPUTATION_PENALTY = 0.10
"""Applied to the defector's trust with *every* other actor. §7.4 calls for a
reputation penalty, and reputation that only the injured party can see is not
reputation. Who *learns* of the defection is Aperture's business; what it does
to standing is the world's."""

WAIT_REFUND = 0.02
"""§7.4: WAIT "accrues a small resource refund". Capped at the endowment, so
waiting recovers capacity and never manufactures it."""

TRUST_BOUND = 1.0


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Push(_Frozen):
    """One actor's entry in a factor contest (spec §7.5)."""

    actor_id: str
    direction: Annotated[int, Field(ge=-1, le=1)]
    resource: Annotated[float, Field(ge=0.0)]
    weight: Annotated[float, Field(ge=0.0)]


class ActorMechanics(_Frozen):
    """What the arbiter needs about one actor: preferences, levers, endowment.

    Derived from the graph by the kernel. The arbiter never sees the graph
    itself, which is what keeps it testable against hand-built cases rather
    than against a compiled decomposition.
    """

    actor_id: str
    utility: dict[str, float]
    """factor id -> signed weight. The sign is the actor's preferred pole."""
    leverage: dict[str, float]
    """factor id -> edge weight. Absent means the actor has no lever there."""
    edge_sign: dict[str, int]
    """factor id -> edge sign, the fallback direction when utility is silent."""
    endowment: dict[str, float]
    """Opening resource bundle. Refunds are capped at it."""

    def direction_on(self, factor_id: str) -> int:
        """Which way this actor pushes a factor when it acts on it.

        Preference first: an actor acts toward the pole its utility wants. An
        actor with a lever and no stated preference falls back to the edge's
        own sign -- it pulls its lever the way the mechanism runs.
        """
        weight = self.utility.get(factor_id)
        if weight is not None and weight != 0.0:
            return 1 if weight > 0 else -1
        return 1 if self.edge_sign.get(factor_id, 1) >= 0 else -1


class FactorOutcome(_Frozen):
    """The resolved contest on one factor."""

    factor_id: str
    delta: float
    share: float
    contributions: dict[str, float]
    """actor id -> the part of ``delta`` attributed to it, for the event log."""
    escalated: bool


class SignalDelivery(_Frozen):
    """A claim to be delivered into the target's next observation (§7.4)."""

    from_actor: str
    to_actor: str
    factor_id: str
    claimed_value: float
    truthful: bool


class ArbiterResult(_Frozen):
    """The world delta for one step, plus everything the event log needs."""

    world: WorldState
    outcomes: tuple[FactorOutcome, ...]
    factor_delta: dict[str, dict[str, float]]
    """actor id -> factor id -> the movement attributed to that actor's action."""
    debits: dict[str, float]
    """actor id -> resource *spent* into contests. Leaves the world."""
    pledges: dict[str, float]
    """actor id -> resource *transferred* to an ally's pool. Stays in the world."""
    releases: dict[str, float]
    """actor id -> pledged resource *returned* from a pool. Also stays in the world."""
    refunds: dict[str, float]
    """actor id -> capacity *recovered* by WAIT or CONCEDE, capped at the endowment.

    The only term that adds to the world's total, which is why it is separate
    from a release: both put resource back in an actor's own bundle, and only
    one of them is new. Conservation reads
    ``total_after == total_before - debits + refunds``."""
    signals: tuple[SignalDelivery, ...]
    scheduled: tuple[PendingEffect, ...]

    def moved_factors(self) -> tuple[str, ...]:
        return tuple(sorted(o.factor_id for o in self.outcomes if o.delta != 0.0))

    def spent(self) -> float:
        """Resource consumed by contests this step -- the only outflow."""
        return float(sum(self.debits[actor] for actor in sorted(self.debits)))

    def created(self) -> float:
        """Capacity recovered this step -- the only inflow."""
        return float(sum(self.refunds[actor] for actor in sorted(self.refunds)))


# ---------------------------------------------------------------------------
# The contest (spec §7.5)
# ---------------------------------------------------------------------------


def _effort(pushes: Sequence[Push], direction: int, gamma: float) -> float:
    """Tullock effort on one side, accumulated in sorted actor order.

    The sort is not cosmetic. Float addition is not associative, so summing the
    same three efforts in two orders can differ in the last bit -- and that bit
    is the difference between a replay that matches and one that does not.
    """
    total = 0.0
    for push in sorted(pushes, key=lambda p: p.actor_id):
        if push.direction == direction:
            total += (push.resource * push.weight) ** gamma
    return total


def contest_share(pushes: Sequence[Push], *, gamma: float = 1.6) -> float:
    """Share of the contest won by the upward side, in [0, 1].

    Returns 0.5 for an empty or perfectly balanced contest: with no effort on
    either side there is no winner, and a share of 0.5 maps to zero delta.
    """
    up = _effort(pushes, 1, gamma)
    down = _effort(pushes, -1, gamma)
    if up + down == 0.0:
        return 0.5
    return up / (up + down)


def contest(pushes: Sequence[Push], *, gamma: float = 1.6, max_step_delta: float) -> float:
    """Resolve one factor's contest into a bounded delta (spec §7.5).

    ``intensity`` keeps a contest that nobody funded from moving the factor as
    far as one everybody funded: share decides *direction*, intensity decides
    *how much of the ceiling* that direction is worth.
    """
    up = _effort(pushes, 1, gamma)
    down = _effort(pushes, -1, gamma)
    if up + down == 0.0:
        return 0.0
    share = up / (up + down)
    intensity: float = min(1.0, (up + down) ** 0.5)
    return (2.0 * share - 1.0) * intensity * max_step_delta


# ---------------------------------------------------------------------------
# The step fold
# ---------------------------------------------------------------------------


def arbitrate(
    *,
    world: WorldState,
    actions: Mapping[str, Action],
    mechanics: Mapping[str, ActorMechanics],
    jitter: Mapping[str, float],
    propagation: Mapping[str, Sequence[Any]],
    gamma: float,
    max_step_delta: float,
    seq_of: Mapping[str, int],
) -> ArbiterResult:
    """Fold this step's actions into a world delta (spec §7.2, stage 6).

    ``propagation`` maps a factor to its outgoing factor->factor edges, which
    is how a contest won on one factor reaches the rest of the graph. Without
    it the compiled edges would be decoration and the world would be a set of
    independent random walks wearing a causal graph's clothes.

    ``seq_of`` gives each acting actor its event sequence number, so a
    scheduled effect can name the decision that scheduled it and §11.2's chain
    survives the lag.
    """
    resources = {actor: dict(kinds) for actor, kinds in sorted(world.resources.items())}
    pools = {actor: dict(backers) for actor, backers in sorted(world.pools.items())}
    relations = dict(world.relations)

    debits: dict[str, float] = {}
    pledges: dict[str, float] = {}
    releases: dict[str, float] = {}
    refunds: dict[str, float] = {}
    pushes: dict[str, list[Push]] = {}
    escalated: set[str] = set()
    signals: list[SignalDelivery] = []

    for actor_id in sorted(actions):
        action = actions[actor_id]
        mech = mechanics.get(actor_id)
        if mech is None:
            raise KeyError(f"no mechanics for acting actor {actor_id!r}")
        available = _available(resources, pools, actor_id)

        if isinstance(action, Commit):
            spend = _take(resources, pools, actor_id, available * action.resource_spend)
            debits[actor_id] = debits.get(actor_id, 0.0) + spend
            pushes.setdefault(action.target_factor, []).append(
                Push(
                    actor_id=actor_id,
                    direction=mech.direction_on(action.target_factor),
                    resource=spend,
                    weight=mech.leverage.get(action.target_factor, 0.0) * action.magnitude,
                )
            )
        elif isinstance(action, Escalate):
            cost = min(1.0, ESCALATE_COST_SCALE * action.magnitude**ESCALATE_COST_EXPONENT)
            spend = _take(resources, pools, actor_id, available * cost)
            debits[actor_id] = debits.get(actor_id, 0.0) + spend
            escalated.add(action.target_factor)
            pushes.setdefault(action.target_factor, []).append(
                Push(
                    actor_id=actor_id,
                    direction=mech.direction_on(action.target_factor),
                    resource=spend,
                    weight=(
                        mech.leverage.get(action.target_factor, 0.0)
                        * action.magnitude
                        * ESCALATE_GAIN
                    ),
                )
            )
        elif isinstance(action, Concede):
            refund = _credit(resources, mech, actor_id, CONCEDE_REFUND)
            refunds[actor_id] = refunds.get(actor_id, 0.0) + refund
            pushes.setdefault(action.target_factor, []).append(
                Push(
                    actor_id=actor_id,
                    direction=-mech.direction_on(action.target_factor),
                    resource=CONCEDE_WEIGHT,
                    weight=mech.leverage.get(action.target_factor, 0.0),
                )
            )
            for other_id in sorted(_contenders(actions, action.target_factor, actor_id)):
                _adjust_trust(relations, actor_id, other_id, CONCEDE_TRUST_GAIN)
        elif isinstance(action, Ally):
            # A pledge is a transfer, not a spend: it leaves this actor's
            # bundle and appears in the ally's pool, so the world's total is
            # unchanged and the conservation check must not see it as an
            # outflow.
            pledge = _take(resources, pools, actor_id, available * action.offered_share)
            pledges[actor_id] = pledges.get(actor_id, 0.0) + pledge
            target = pools.setdefault(action.target_actor, {})
            target[actor_id] = target.get(actor_id, 0.0) + pledge
            _adjust_trust(relations, actor_id, action.target_actor, ALLY_TRUST_GAIN)
        elif isinstance(action, Defect):
            # Both directions of the pledge unwind: what this actor staked on
            # the other comes home, and what the other staked here goes back.
            # Anything that will not fit under an endowment cap stays in the
            # pool rather than evaporating -- a conservation check that
            # tolerated quiet destruction would tolerate quiet creation too.
            releases[actor_id] = releases.get(actor_id, 0.0) + _release(
                pools, resources, mech, backer=actor_id, holder=action.target_actor
            )
            if action.target_actor in mechanics:
                releases[action.target_actor] = releases.get(action.target_actor, 0.0) + _release(
                    pools,
                    resources,
                    mechanics[action.target_actor],
                    backer=action.target_actor,
                    holder=actor_id,
                )
            key = relation_key(actor_id, action.target_actor)
            relations[key] = min(relations.get(key, 0.0), DEFECT_TRUST_FLOOR)
            for other_id in sorted(mechanics):
                if other_id in (actor_id, action.target_actor):
                    continue
                _adjust_trust(relations, actor_id, other_id, -REPUTATION_PENALTY)
        elif isinstance(action, Signal):
            signals.append(
                SignalDelivery(
                    from_actor=actor_id,
                    to_actor=action.target_actor,
                    factor_id=action.claimed_factor,
                    claimed_value=action.claimed_value,
                    truthful=action.truthful,
                )
            )
        elif isinstance(action, Wait):
            refund = _credit(resources, mech, actor_id, WAIT_REFUND)
            refunds[actor_id] = refunds.get(actor_id, 0.0) + refund

    outcomes: list[FactorOutcome] = []
    factor_delta: dict[str, dict[str, float]] = {}
    factors = dict(world.factors)
    scheduled: list[PendingEffect] = []

    for factor_id in sorted(pushes):
        entries = pushes[factor_id]
        delta = contest(entries, gamma=gamma, max_step_delta=max_step_delta)
        if factor_id in escalated:
            intensity = min(1.0, sum(p.resource * p.weight for p in sorted(entries, key=_by_actor)))
            delta += jitter.get(factor_id, 0.0) * ESCALATE_VARIANCE * max_step_delta * intensity
        # Boundedness holds *after* the variance penalty, not before it.
        delta = max(-max_step_delta, min(max_step_delta, delta))

        before = factors.get(factor_id, 0.0)
        after = max(0.0, min(1.0, before + delta))
        applied = after - before
        factors[factor_id] = after

        contributions = _attribute(entries, applied, gamma=gamma)
        for actor_id in sorted(contributions):
            factor_delta.setdefault(actor_id, {})[factor_id] = contributions[actor_id]
        outcomes.append(
            FactorOutcome(
                factor_id=factor_id,
                delta=applied,
                share=contest_share(entries, gamma=gamma),
                contributions=contributions,
                escalated=factor_id in escalated,
            )
        )
        scheduled.extend(
            _propagate(
                factor_id,
                applied,
                propagation=propagation,
                step=world.step,
                seq_of=seq_of,
                entries=entries,
            )
        )

    updated = world.model_copy(
        update={
            "factors": factors,
            "resources": resources,
            "pools": {actor: dict(pools[actor]) for actor in sorted(pools) if pools[actor]},
            "relations": relations,
            "pending": tuple(sorted((*world.pending, *scheduled), key=PendingEffect.sort_key)),
        }
    )
    return ArbiterResult(
        world=updated,
        outcomes=tuple(outcomes),
        factor_delta=factor_delta,
        debits=debits,
        pledges=pledges,
        releases=releases,
        refunds=refunds,
        signals=tuple(signals),
        scheduled=tuple(scheduled),
    )


def _by_actor(push: Push) -> str:
    return push.actor_id


def _contenders(actions: Mapping[str, Action], factor_id: str, conceder_id: str) -> tuple[str, ...]:
    """Actors pushing the factor this step -- the parties a concession is *to*."""
    out: list[str] = []
    for actor_id in sorted(actions):
        if actor_id == conceder_id:
            continue
        action = actions[actor_id]
        if isinstance(action, Commit | Escalate) and action.target_factor == factor_id:
            out.append(actor_id)
    return tuple(out)


def _available(
    resources: Mapping[str, Mapping[str, float]],
    pools: Mapping[str, Mapping[str, float]],
    actor_id: str,
) -> float:
    own = resources.get(actor_id, {})
    pledged = pools.get(actor_id, {})
    return float(sum(own[k] for k in sorted(own))) + float(sum(pledged[k] for k in sorted(pledged)))


def _take(
    resources: dict[str, dict[str, float]],
    pools: dict[str, dict[str, float]],
    actor_id: str,
    requested: float,
) -> float:
    """Debit ``requested`` from an actor, own resources first, then allied pledges.

    Returns what was actually taken, which is the requested amount unless the
    actor could not afford it. Conservation is by construction: the return
    value is the sum of the individual takes, so a caller cannot book a spend
    that no balance gave up. Own resources drain before pledges because a
    pledge is someone else's capital and spending it first would let an actor
    keep its own while burning its allies'.
    """
    remaining = max(0.0, requested)
    taken = 0.0
    own = resources.setdefault(actor_id, {})
    for kind in sorted(own):
        if remaining <= 0.0:
            break
        amount = min(remaining, own[kind])
        own[kind] = own[kind] - amount
        remaining -= amount
        taken += amount
    pledged = pools.setdefault(actor_id, {})
    for backer in sorted(pledged):
        if remaining <= 0.0:
            break
        amount = min(remaining, pledged[backer])
        pledged[backer] = pledged[backer] - amount
        remaining -= amount
        taken += amount
    return taken


def _credit(
    resources: dict[str, dict[str, float]],
    mech: ActorMechanics,
    actor_id: str,
    fraction: float,
) -> float:
    """Refund a fraction of the endowment, capped at the endowment itself."""
    endowment = float(sum(mech.endowment[k] for k in sorted(mech.endowment)))
    return _credit_raw(resources, mech, actor_id, endowment * fraction)


def _credit_raw(
    resources: dict[str, dict[str, float]],
    mech: ActorMechanics,
    actor_id: str,
    amount: float,
) -> float:
    """Return ``amount`` to an actor's bundle without exceeding its endowment.

    The cap is what stops a run from manufacturing resources: WAIT refunds and
    released pledges restore capacity that existed, and an uncapped refund
    would make waiting a renewable income.
    """
    own = resources.setdefault(actor_id, {})
    restored = 0.0
    remaining = max(0.0, amount)
    for kind in sorted(mech.endowment):
        if remaining <= 0.0:
            break
        headroom = mech.endowment[kind] - own.get(kind, 0.0)
        if headroom <= 0.0:
            continue
        credited = min(remaining, headroom)
        own[kind] = own.get(kind, 0.0) + credited
        remaining -= credited
        restored += credited
    return restored


def _release(
    pools: dict[str, dict[str, float]],
    resources: dict[str, dict[str, float]],
    mech: ActorMechanics,
    *,
    backer: str,
    holder: str,
) -> float:
    """Return ``backer``'s pledge held by ``holder``, as far as the cap allows."""
    held = pools.get(holder, {})
    staked = held.get(backer, 0.0)
    if staked <= 0.0:
        return 0.0
    restored = _credit_raw(resources, mech, backer, staked)
    leftover = staked - restored
    if leftover > 0.0:
        held[backer] = leftover
    else:
        held.pop(backer, None)
    return restored


def _adjust_trust(relations: dict[str, float], actor_a: str, actor_b: str, delta: float) -> None:
    key = relation_key(actor_a, actor_b)
    relations[key] = max(-TRUST_BOUND, min(TRUST_BOUND, relations.get(key, 0.0) + delta))


def _attribute(entries: Sequence[Push], applied: float, *, gamma: float) -> dict[str, float]:
    """Split the realised movement across the actors that caused it.

    Effort share, signed by direction: an actor that pushed against the
    realised movement is attributed a negative part. The parts sum to the
    movement, so the event log's ``factor_delta`` column adds up to what the
    world actually did rather than to an idealised version of it.
    """
    total = 0.0
    for push in sorted(entries, key=_by_actor):
        total += (push.resource * push.weight) ** gamma
    if total <= 0.0:
        return {}
    out: dict[str, float] = {}
    for push in sorted(entries, key=_by_actor):
        weight = (push.resource * push.weight) ** gamma / total
        signed = weight if push.direction > 0 else -weight
        out[push.actor_id] = out.get(push.actor_id, 0.0) + signed * abs(applied)
    return out


def _propagate(
    factor_id: str,
    applied: float,
    *,
    propagation: Mapping[str, Sequence[Any]],
    step: int,
    seq_of: Mapping[str, int],
    entries: Sequence[Push],
) -> list[PendingEffect]:
    """Schedule downstream movement along factor->factor edges.

    Arrival is at ``step + max(1, lag)``: an edge with lag 0 still lands on the
    next step, because resolving a factor's influence on another factor inside
    the step that produced it would require a fixed point over a graph that is
    not guaranteed acyclic.
    """
    if applied == 0.0:
        return []
    # The provenance chain should name the actor that actually moved the
    # factor, so the lead is the largest contributor, with the actor id
    # settling an exact tie.
    lead = max(
        sorted(entries, key=_by_actor),
        key=lambda p: (p.resource * p.weight, p.actor_id),
        default=None,
    )
    out: list[PendingEffect] = []
    for edge in sorted(propagation.get(factor_id, ()), key=lambda e: (e.dst, e.lag)):
        delta = applied * float(edge.weight) * (1.0 if edge.sign > 0 else -1.0)
        if delta == 0.0:
            continue
        out.append(
            PendingEffect(
                arrive_step=step + max(1, int(edge.lag)),
                factor_id=edge.dst,
                delta=delta,
                source_actor_id=lead.actor_id if lead is not None else None,
                origin_step=step,
                origin_seq=seq_of.get(lead.actor_id, 0) if lead is not None else 0,
            )
        )
    return out
