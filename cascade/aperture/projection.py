"""Projecting the world through one actor's policy (spec §6.2, §7.2 stage 4).

This is the function that makes information asymmetry real rather than
cosmetic. Two actors handed the same world produce different observations
because their channels differ in noise, lag and granularity -- and they produce
the *same* different observations on every replay, because the noise comes from
the run's seeded draw plan rather than from a generator this module holds.

Three details are load-bearing:

**Observed values are rounded to two decimals.** Not for display: the rounded
value is what goes into the observation hash *and* into the prompt, so two
steps that differ only in the fourth decimal of a factor are one cache entry.
The spec's risk register names a too-fine observation hash as the reason an
action cache misses its 88% target, and rounding at the projection is what
makes the hash and the prompt agree by construction instead of by review.

**A signal alters the target's observation and is not otherwise announced**
(§7.4: "Alters the target's observation of that factor; truth is not
enforced"). The target is not handed the claim to weigh -- a claim you can
inspect separately is an announcement, not a signal. How far it moves the
observation depends on how blurry the channel already is: an actor observing a
factor it directly controls (sigma 0) cannot be told otherwise, which is why
misdirection has to be aimed at what someone half-sees.

**Lag reads history, not a decayed value.** ``lag = 2`` means the actor sees
the state as of two steps ago, exactly, which is why ``WorldState`` keeps
``factor_history`` indexed by step.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.aperture.policy import VisibilityPolicy
from cascade.canonical import canonical_json
from cascade.sim.state import WorldState, trust_between

__all__ = [
    "OBSERVATION_DECIMALS",
    "QUANTIZED_BANDS",
    "SIGNAL_ANCHOR_SIGMA",
    "Observation",
    "ObservedAction",
    "SignalClaim",
    "observation_hash",
    "project",
    "quantize",
]

# Spec risk register: "Quantize observed floats to 2 decimals before hashing;
# that alone typically moves hit rate 15-20 points."
OBSERVATION_DECIMALS = 2

# §6.2's "quantized: observe only {low, mid, high}". The representatives are
# the band midpoints, so a quantized reading is unbiased with respect to the
# value it replaces.
QUANTIZED_BANDS: tuple[float, float, float] = (0.17, 0.5, 0.83)

# The sigma at which an actor gives a claim equal weight with its own reading.
# Pinned to the one-hop channel (0.08): an actor one hop from a factor is
# exactly the audience a signal is for. A direct observer (sigma 0) is immune,
# and a two-hop observer (0.15) is more susceptible than immune but not blindly
# so.
SIGNAL_ANCHOR_SIGMA = 0.08


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SignalClaim(_Frozen):
    """One claim made to this actor last step by another (spec §7.4)."""

    from_actor: str
    factor_id: str
    claimed_value: Annotated[float, Field(ge=0.0, le=1.0)]


class ObservedAction(_Frozen):
    """Another actor's action as this actor saw it.

    ``target`` and ``magnitude`` are None under ``type_only`` visibility: the
    actor knows a party escalated and not on what or how hard. That is the
    distinction §6.2 draws, and collapsing it would hand every agent the same
    action log with a different label on it.
    """

    actor_id: str
    action_type: str
    target: str | None
    magnitude: float | None


class Observation(_Frozen):
    """What one actor sees at one step. The unit the action cache is keyed on."""

    actor_id: str
    step: int = Field(ge=0)
    factors: dict[str, float]
    """Observed values, rounded. Only factors this actor can see appear."""
    banded: tuple[str, ...] = ()
    """Factors observed only as low/mid/high, so the prompt can say so."""
    unobserved: tuple[str, ...] = ()
    budget: float = Field(ge=0.0)
    pooled: float = Field(ge=0.0)
    trust: dict[str, float] = Field(default_factory=dict)
    actions: tuple[ObservedAction, ...] = ()

    def canonical(self) -> dict[str, Any]:
        """Sorted, primitive projection -- the input to :func:`observation_hash`."""
        return {
            "actor_id": self.actor_id,
            "step": self.step,
            "factors": {key: self.factors[key] for key in sorted(self.factors)},
            "banded": sorted(self.banded),
            "unobserved": sorted(self.unobserved),
            "budget": self.budget,
            "pooled": self.pooled,
            "trust": {key: self.trust[key] for key in sorted(self.trust)},
            "actions": [
                {
                    "actor_id": action.actor_id,
                    "action_type": action.action_type,
                    "target": action.target,
                    "magnitude": action.magnitude,
                }
                for action in sorted(
                    self.actions,
                    key=lambda a: (a.actor_id, a.action_type, a.target or "", a.magnitude or 0.0),
                )
            ],
        }


def observation_hash(observation: Observation) -> bytes:
    """The ``obs_hash`` written to the event log (spec §11.1).

    Raw digest bytes rather than hex: the column is ``bytea``, and 4.2M rows
    is a place where 16 bytes against 32 characters is worth having.
    """
    return hashlib.blake2b(
        canonical_json(observation.canonical()).encode("utf-8"), digest_size=16
    ).digest()


def quantize(value: float, *, banded: bool) -> float:
    """Round an observed value, banding it to low/mid/high when the channel is coarse."""
    if banded:
        if value < 1.0 / 3.0:
            return QUANTIZED_BANDS[0]
        if value < 2.0 / 3.0:
            return QUANTIZED_BANDS[1]
        return QUANTIZED_BANDS[2]
    return round(value, OBSERVATION_DECIMALS)


def project(
    world: WorldState,
    policy: VisibilityPolicy,
    *,
    noise: Mapping[str, float],
    signals: Sequence[SignalClaim] = (),
    actions: Sequence[ObservedAction] = (),
) -> Observation:
    """Project ``world`` through ``policy`` into one actor's observation.

    ``noise`` is the actor's row of the step's draw plan -- one unit normal per
    factor, drawn whether or not the channel uses it (see
    :mod:`cascade.sim.rng`). Taking the draws as an argument rather than a
    generator is what keeps this function pure and what lets the §6.4 ablation
    run the identical stream through a transparent policy.
    """
    observed: dict[str, float] = {}
    banded: list[str] = []
    unobserved: list[str] = []

    claims: dict[str, list[SignalClaim]] = {}
    for claim in sorted(signals, key=lambda c: (c.factor_id, c.from_actor)):
        claims.setdefault(claim.factor_id, []).append(claim)

    for channel in sorted(policy.channels, key=lambda c: c.factor_id):
        if not channel.visible:
            unobserved.append(channel.factor_id)
            continue
        truth = world.factor_at(channel.factor_id, world.step - channel.lag)
        draw = float(noise.get(channel.factor_id, 0.0))
        value = truth + channel.noise_sigma * draw
        for claim in claims.get(channel.factor_id, ()):
            value = _apply_claim(
                value,
                claim=claim,
                sigma=channel.noise_sigma,
                trust=trust_between(world.relations, policy.actor_id, claim.from_actor),
            )
        value = min(1.0, max(0.0, value))
        observed[channel.factor_id] = quantize(value, banded=channel.quantized)
        if channel.quantized:
            banded.append(channel.factor_id)

    trust = {
        entry.actor_id: round(
            trust_between(world.relations, policy.actor_id, entry.actor_id),
            OBSERVATION_DECIMALS,
        )
        for entry in sorted(policy.sees_actions_of, key=lambda e: e.actor_id)
    }

    return Observation(
        actor_id=policy.actor_id,
        step=world.step,
        factors=observed,
        banded=tuple(banded),
        unobserved=tuple(unobserved),
        budget=round(world.budget(policy.actor_id), OBSERVATION_DECIMALS),
        pooled=round(world.pooled(policy.actor_id), OBSERVATION_DECIMALS),
        trust=trust,
        actions=tuple(actions),
    )


def _apply_claim(value: float, *, claim: SignalClaim, sigma: float, trust: float) -> float:
    """Blend a claimed value into an observation (spec §7.4, SIGNAL).

    Susceptibility rises with the channel's own noise and with trust in the
    speaker. Both terms matter: an actor that sees a factor perfectly cannot be
    told otherwise however much it trusts the source, and an actor that sees
    nothing clearly still discounts a party it distrusts.
    """
    if sigma <= 0.0:
        return value
    blur = sigma / (sigma + SIGNAL_ANCHOR_SIGMA)
    credence = 0.5 * (1.0 + max(-1.0, min(1.0, trust)))
    weight = blur * credence
    return (1.0 - weight) * value + weight * claim.claimed_value


def visible_actions(
    policy: VisibilityPolicy,
    actions: Mapping[str, Any],
) -> tuple[ObservedAction, ...]:
    """Filter last step's actions through this actor's action channels (§6.2).

    ``actions`` maps actor id -> the action it took. An actor never sees its
    own action here -- it knows what it did, and including it would spend
    prompt tokens telling it.
    """
    out: list[ObservedAction] = []
    for actor_id in sorted(actions):
        if actor_id == policy.actor_id:
            continue
        view = policy.action_view(actor_id)
        if view == "none":
            continue
        action = actions[actor_id]
        if view == "type_only":
            out.append(
                ObservedAction(
                    actor_id=actor_id, action_type=action.type, target=None, magnitude=None
                )
            )
            continue
        out.append(
            ObservedAction(
                actor_id=actor_id,
                action_type=action.type,
                target=_target(action),
                magnitude=_magnitude(action),
            )
        )
    return tuple(out)


def _target(action: Any) -> str | None:
    for attribute in ("target_factor", "target_actor"):
        value = getattr(action, attribute, None)
        if isinstance(value, str):
            return value
    return None


def _magnitude(action: Any) -> float | None:
    for attribute in ("magnitude", "offered_share"):
        value = getattr(action, attribute, None)
        if isinstance(value, float):
            return round(value, OBSERVATION_DECIMALS)
    return None
