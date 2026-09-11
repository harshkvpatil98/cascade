"""The activation scheduler (spec §7.3). Pure: no clock, no RNG, no I/O.

Every agent acting on every step is wasteful and unrealistic -- most parties in
a real situation are dormant most of the time -- and it is also the difference
between a $126 study and a $360 one. Agents activate on salience, under three
rules:

*Triggered* -- an observed factor moved by more than ``salience_threshold``
since that actor's last observation, or an actor it can see acted on a factor
in its own utility terms.

*Scheduled* -- every actor acts at least once every ``forced_interval`` steps,
so no party goes fully silent.

*Capped* -- at most ``max_active_per_step`` actors act, ties broken by
utility-weighted salience, deterministically.

The 34.7% mean activation rate the cost model rests on is an **emergent
property of these rules, not a knob**. Nothing in this module takes a target
rate, and the M5 acceptance criterion measures what the rules produce over a
50-run sample. If that lands outside the spec's band, the number to change is
the report, not the threshold.

Two readings are fixed here because the spec leaves them open, and both are
chosen to keep the §6.4 ablation clean:

* Movement is measured on the factor's own trajectory between the two steps,
  not on the noisy readings the actor took. Otherwise a two-hop channel
  (sigma 0.15) trips a 0.06 threshold on sensor noise alone and the activation
  rate -- hence the run's cost -- becomes a function of the visibility policy.
* An actor that has never observed is not "triggered": it has no baseline to
  have moved from. It still acts under the scheduled floor, which the
  staggered opening in :func:`cascade.sim.state.staggered_schedule` spreads
  across the first interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.sim.state import WorldState

__all__ = ["Activation", "ActorProfile", "activate", "profiles_from"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActorProfile(_Frozen):
    """What the scheduler needs to know about one actor, and nothing more.

    Narrow on purpose: the scheduler is one of the four pure modules the
    engineering standards name, and a function that takes the whole graph is a
    function whose tests need the whole graph.
    """

    actor_id: str
    utility: dict[str, float]
    """factor id -> signed weight. Magnitude is what weights salience."""
    observed: tuple[str, ...]
    """Factors this actor can see at all. Invisible factors cannot wake it."""
    watches: dict[str, str]
    """actor id -> "full" | "type_only". Only a full view carries a target."""


class Activation(_Frozen):
    """Who acts this step, and why -- the record the M5 rate is measured from."""

    step: int = Field(ge=0)
    active: tuple[str, ...]
    forced: tuple[str, ...]
    triggered: tuple[str, ...]
    crowded_out: tuple[str, ...]
    """Actors that qualified but lost to the cap. A forced actor appearing here
    means the floor was missed on this step; it becomes more overdue and takes
    priority on the next one."""
    salience: dict[str, float]

    @property
    def rate(self) -> float:
        """Fraction of the cast that acted. Undefined for an empty cast."""
        total = len(self.salience)
        return len(self.active) / total if total else 0.0


def profiles_from(graph: Any, policies: Mapping[str, Any]) -> dict[str, ActorProfile]:
    """Adapt a compiled graph and its visibility policies into scheduler inputs."""
    out: dict[str, ActorProfile] = {}
    for actor in sorted(graph.actors, key=lambda a: a.id):
        policy = policies[actor.id]
        out[actor.id] = ActorProfile(
            actor_id=actor.id,
            utility={term.factor_id: float(term.weight) for term in actor.utility_terms},
            observed=policy.visible_factors(),
            watches={
                entry.actor_id: entry.visibility
                for entry in sorted(policy.sees_actions_of, key=lambda e: e.actor_id)
            },
        )
    return out


def activate(
    *,
    profiles: Mapping[str, ActorProfile],
    world: WorldState,
    last_actions: Mapping[str, Any],
    salience_threshold: float,
    forced_interval: int,
    max_active: int,
) -> Activation:
    """Select the acting subset for ``world.step`` (spec §7.3).

    ``last_actions`` maps actor id -> the action it took on the previous step.
    Deterministic in every branch: the priority order is a total order (overdue
    count, then salience, then actor id), so the cap never has to break a tie
    by iteration order.
    """
    salience: dict[str, float] = {}
    triggered: list[str] = []
    forced: list[str] = []

    for actor_id in sorted(profiles):
        profile = profiles[actor_id]
        moved = _movement_salience(profile, world, threshold=salience_threshold)
        provoked = _action_salience(profile, last_actions)
        salience[actor_id] = round(moved.score + provoked, 6)
        if moved.tripped or provoked > 0.0:
            triggered.append(actor_id)
        if _overdue(world, actor_id, forced_interval) >= 0:
            forced.append(actor_id)

    due = set(forced)
    candidates = sorted(set(triggered) | due)
    # Total order, so the cap never falls back on iteration order: the
    # scheduled floor outranks salience, the most overdue actor outranks the
    # rest of the floor, and the actor id settles an exact tie.
    ordered = sorted(
        candidates,
        key=lambda actor_id: (
            0 if actor_id in due else 1,
            -_overdue(world, actor_id, forced_interval),
            -salience[actor_id],
            actor_id,
        ),
    )
    cap = max(0, max_active)
    chosen = ordered[:cap]
    crowded = ordered[cap:]

    return Activation(
        step=world.step,
        active=tuple(sorted(chosen)),
        forced=tuple(sorted(forced)),
        triggered=tuple(sorted(triggered)),
        crowded_out=tuple(sorted(crowded)),
        salience=salience,
    )


class _Movement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    tripped: bool


def _movement_salience(profile: ActorProfile, world: WorldState, *, threshold: float) -> _Movement:
    """Utility-weighted movement of this actor's observed factors since it looked."""
    since = world.last_observed_at.get(profile.actor_id)
    if since is None:
        return _Movement(score=0.0, tripped=False)
    score = 0.0
    tripped = False
    for factor_id in sorted(profile.observed):
        try:
            then = world.factor_at(factor_id, since)
            now = world.factor_at(factor_id, world.step)
        except KeyError:
            continue
        delta = abs(now - then)
        if delta > threshold:
            tripped = True
        score += abs(profile.utility.get(factor_id, 0.0)) * delta
    return _Movement(score=score, tripped=tripped)


def _action_salience(profile: ActorProfile, last_actions: Mapping[str, Any]) -> float:
    """Weight of visible actions aimed at factors this actor cares about.

    Only a ``full`` view counts. Under ``type_only`` the actor knows a party
    did *something*; it cannot know the something was aimed at its own
    interests, and letting it react anyway would hand it information the
    visibility policy denied.
    """
    score = 0.0
    for other_id in sorted(last_actions):
        if other_id == profile.actor_id:
            continue
        if profile.watches.get(other_id) != "full":
            continue
        target = getattr(last_actions[other_id], "target_factor", None)
        if not isinstance(target, str):
            continue
        weight = abs(profile.utility.get(target, 0.0))
        if weight <= 0.0:
            continue
        magnitude = getattr(last_actions[other_id], "magnitude", 1.0)
        score += weight * float(magnitude)
    return score


def _overdue(world: WorldState, actor_id: str, forced_interval: int) -> int:
    """Steps past the scheduled floor. Negative means not yet due."""
    last = world.last_acted.get(actor_id, -1)
    return (world.step - last) - forced_interval
