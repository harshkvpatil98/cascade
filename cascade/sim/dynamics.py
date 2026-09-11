"""Stages 1, 2 and 6b of the step loop: the world moving on its own (§7.2, §7.6).

Pure, like the arbiter and for the same reason. These three functions are the
parts of a step that no agent decides:

EXOGENOUS
    Each factor takes a seeded mean-reverting step toward the value the
    compiler gave it, with its own volatility. Mean reversion is what stops a
    24-step random walk from wandering to a bound and staying there, which
    would make the terminal outcome a function of the walk rather than of the
    agents.
ARRIVE
    Pending effects whose lag has expired are applied. Arrivals are clamped to
    the same ``max_step_delta`` as a contest: a factor that three lagged
    effects converge on must not move further in one step than one contested
    by every actor in the graph.
Observability and absorption
    After both, the step's observable value is written into
    ``factor_history``. That is the value an actor with zero lag sees, and the
    value a lagged actor will see later -- there is exactly one array, so an
    agent's memory of "what it was two steps ago" cannot disagree with anyone
    else's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.sim.state import PendingEffect, WorldState

__all__ = [
    "ABSORBING_STEPS",
    "BOUND_EPSILON",
    "FactorDynamics",
    "absorbed_factors",
    "arrive",
    "dynamics_from",
    "exogenous",
    "record_observable",
    "update_streaks",
]

# §7.6: "a factor pinned at a bound for 3 consecutive steps".
ABSORBING_STEPS = 3
# A factor clipped to a bound lands exactly on 0.0 or 1.0, but a factor that
# reverts to within a rounding error of one is pinned in every sense that
# matters to an observer reading two decimals.
BOUND_EPSILON = 1e-9


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorDynamics(_Frozen):
    """One factor's exogenous law of motion (spec §5.1, §7.2)."""

    factor_id: str
    anchor: Annotated[float, Field(ge=0.0, le=1.0)]
    """The value it reverts toward -- the compiler's initial state for it."""
    inertia: Annotated[float, Field(ge=0.0, le=1.0)]
    volatility: Annotated[float, Field(ge=0.0, le=1.0)]


def dynamics_from(graph: Any) -> dict[str, FactorDynamics]:
    """Read each factor's law of motion off the compiled graph."""
    return {
        factor.id: FactorDynamics(
            factor_id=factor.id,
            anchor=float(factor.state),
            inertia=float(factor.inertia),
            volatility=float(factor.volatility),
        )
        for factor in sorted(graph.factors, key=lambda f: f.id)
    }


def exogenous(
    world: WorldState,
    dynamics: Mapping[str, FactorDynamics],
    noise: Mapping[str, float],
) -> WorldState:
    """Advance every factor by its seeded mean-reverting step (spec §7.2, stage 1).

    ``noise`` carries unit normals from the step's draw plan; volatility scales
    them here rather than at the draw, so changing a factor's volatility is a
    change to the world model and never to the RNG stream.
    """
    factors = dict(world.factors)
    for factor_id in sorted(factors):
        law = dynamics.get(factor_id)
        if law is None:
            continue
        value = factors[factor_id]
        drift = law.inertia * (law.anchor - value)
        shock = law.volatility * float(noise.get(factor_id, 0.0))
        factors[factor_id] = max(0.0, min(1.0, value + drift + shock))
    return world.model_copy(update={"factors": factors})


def arrive(
    world: WorldState, *, max_step_delta: float
) -> tuple[WorldState, tuple[PendingEffect, ...]]:
    """Apply every pending effect whose lag has expired (spec §7.2, stage 2).

    Returns the new state and the effects that landed, so the caller can
    attribute the movement -- an arriving effect names the decision that
    scheduled it, which is what lets §11.2's chain cross a lag boundary.
    """
    due: list[PendingEffect] = []
    rest: list[PendingEffect] = []
    for effect in sorted(world.pending, key=PendingEffect.sort_key):
        (due if effect.arrive_step <= world.step else rest).append(effect)

    if not due:
        return world, ()

    totals: dict[str, float] = {}
    for effect in due:
        totals[effect.factor_id] = totals.get(effect.factor_id, 0.0) + effect.delta

    factors = dict(world.factors)
    for factor_id in sorted(totals):
        if factor_id not in factors:
            continue
        delta = max(-max_step_delta, min(max_step_delta, totals[factor_id]))
        factors[factor_id] = max(0.0, min(1.0, factors[factor_id] + delta))

    return (
        world.model_copy(update={"factors": factors, "pending": tuple(rest)}),
        tuple(due),
    )


def record_observable(world: WorldState) -> WorldState:
    """Write the step's observable value into the history (spec §7.1).

    Called once per step, after the exogenous walk and the arrivals and before
    anyone observes. Index ``step`` is overwritten rather than appended so a
    resumed step cannot leave two entries for one step number.
    """
    history = dict(world.factor_history)
    for factor_id in sorted(world.factors):
        previous = history.get(factor_id, ())
        head = previous[: world.step]
        if len(head) < world.step:
            # A gap would silently shift every lagged read that spans it.
            filler = head[-1] if head else world.factors[factor_id]
            head = head + (filler,) * (world.step - len(head))
        history[factor_id] = (*head, world.factors[factor_id])
    return world.model_copy(update={"factor_history": history})


def update_streaks(world: WorldState) -> WorldState:
    """Count consecutive steps each factor has spent pinned at a bound (§7.6)."""
    streaks = dict(world.pinned_streak)
    for factor_id in sorted(world.factors):
        value = world.factors[factor_id]
        pinned = value <= BOUND_EPSILON or value >= 1.0 - BOUND_EPSILON
        streaks[factor_id] = streaks.get(factor_id, 0) + 1 if pinned else 0
    return world.model_copy(update={"pinned_streak": streaks})


def absorbed_factors(
    world: WorldState, outcome_factor_ids: Sequence[str], *, limit: int = ABSORBING_STEPS
) -> tuple[str, ...]:
    """Outcome-rule factors pinned at a bound for ``limit`` consecutive steps.

    Restricted to the factors the outcome rule reads, because §7.6 ties
    termination to "an absorbing condition in the outcome rule". A peripheral
    factor stuck at 1.0 is a fact about the world, not a reason to stop
    simulating one whose drivers are still moving.
    """
    return tuple(
        factor_id
        for factor_id in sorted(set(outcome_factor_ids))
        if world.pinned_streak.get(factor_id, 0) >= limit
    )
