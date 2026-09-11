"""Agent prompts and the action tool schema (spec §7.4, §12.1, §12.2).

Kept apart from the kernel for the same reason the compiler's prompts are kept
apart from the compiler: prompt text is the part of this subsystem that will be
revised on evidence, and §1.3 requires every revision to be logged in
``prompt_revisions`` with the before/after Brier. A revision should be a diff of
this file.

The prompt is built in three layers, and the split is a cost decision as much
as a clarity one (§12.2):

``RULES`` (static, study-wide)
    The world model, the action semantics and the arbiter's actual mechanics.
    Identical for every one of the ~4.2M decisions in the study.
``persona`` (static per scenario and actor)
    Who this actor is, what it wants, what it can move, who it can address --
    and its evidence. Evidence is retrieved **once per (scenario, actor)**, not
    once per step (ADR-0019): ``as_of`` is the scenario cutoff for the whole
    run, so a per-step query would return near-identical chunks 24 times, cost
    4.2M vector searches, and blow §12.1's 260-token dynamic budget by an order
    of magnitude.
``turn`` (dynamic)
    The observation and the memory digest. This is the only part that varies
    within a run, which is exactly what makes the trajectory-prefix action
    cache work: two replicates whose observations round to the same two
    decimals produce the same bytes and the same cache key.

Everything the model is told about the mechanics is true of the arbiter, and
the numbers are imported from it rather than restated. A prompt that describes
a different game than the one being played produces actors that look
deliberative and are not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.aperture.projection import Observation
from cascade.sim.actions import (
    RATIONALE_MAX_CHARS,
    Ally,
    Commit,
    Concede,
    Defect,
    Escalate,
    Signal,
    Wait,
)
from cascade.sim.arbiter import (
    ALLY_TRUST_GAIN,
    CONCEDE_WEIGHT,
    DEFECT_TRUST_FLOOR,
    ESCALATE_GAIN,
    WAIT_REFUND,
)

__all__ = ["ACTION_TOOL", "ACTION_TOOL_NAME", "RULES", "persona_block", "turn_message"]

ACTION_TOOL_NAME = "emit_action"


class _ActionEnvelope(BaseModel):
    """The tool input: exactly one action from the closed union.

    A single ``action`` field with a discriminated union underneath, rather
    than seven optional fields, so a model cannot emit two actions and leave
    the arbiter to choose.
    """

    model_config = ConfigDict(extra="forbid")

    action: Commit | Signal | Ally | Defect | Escalate | Concede | Wait = Field(
        discriminator="type"
    )


ACTION_TOOL: dict[str, Any] = {
    "name": ACTION_TOOL_NAME,
    "description": (
        "Take exactly one action this step. This is the only way to act; "
        "prose is not read by the simulation."
    ),
    # Generated from the Pydantic models, never hand-written beside them: the
    # schema the model is held to and the schema the parser enforces cannot
    # drift if there is only one of them.
    "input_schema": _ActionEnvelope.model_json_schema(),
}


RULES = f"""\
You are one party in a multi-party situation that is being simulated forward, \
one step at a time, to a horizon. You are not a forecaster and you are not an \
assistant. You act in your own interest, given what you can see.

# The world

The situation is described by a small set of *factors*, each a number in \
[0, 1]. A factor is a real quantity -- a coalition's stability, a regulator's \
posture, a price level -- normalised so that 0 and 1 are its extremes. Between \
steps, every factor drifts on its own: it reverts toward its starting level and \
takes a random shock scaled by its volatility. Your actions push against that \
drift.

You have a *utility*: a list of factors with signed weights. A positive weight \
means you want that factor high; a negative weight means you want it low. The \
magnitude is how much you care. Your objective text says the same thing in \
words; when they seem to differ, the weights are what you are.

You have *levers*: the factors you can actually move. You cannot act on a \
factor that is not one of your levers, however much you care about it. Acting \
on something you cannot move is the most common wasted turn.

You have *resources*: a single budget, drawn down by acting and slowly \
recovered by waiting. Resources spent are gone. An ally's pledge can be spent \
by the ally, not by you.

# What you can see

You do not see the world. You see a projection of it through your own \
channels, and other parties see different projections. Specifically:

- Factors you act on directly, you see exactly.
- Factors one step away from your levers, you see with noise and a one-step \
delay.
- Factors two steps away, you see as a band -- low, mid or high -- with more \
noise and a two-step delay.
- Factors further away than that, you do not see at all. They are listed as \
unobserved, and they are still moving.

Other parties' actions reach you the same way. Some you see in full; some you \
see only as "that party did something of this kind"; some you never see.

Two consequences worth holding onto. First, a number you are shown may be \
wrong, and a party that wants you to believe something can send you a claim \
that shifts what you see -- most easily about a factor you see only vaguely. \
Second, the absence of news is not the absence of movement.

# How actions are resolved

A deterministic referee resolves every step. It is not a negotiator and it \
does not read your reasoning; it reads your action.

When several parties push the same factor, the referee runs a contest. Each \
push is worth (resources committed x your leverage on that factor), raised to \
a power, and the two sides are compared. The side with more effort moves the \
factor its way. Two things follow: a contest nobody funds moves nothing, and \
doubling your spend against an unopposed factor does far less than spending \
the same amount where you are actually contested. No single step moves a \
factor very far -- there is a hard ceiling per step -- so nothing is decided \
in one turn, and a factor cannot be rushed to its bound.

The actions, and exactly what the referee does with each:

- COMMIT (factor, magnitude, resource_spend): push a lever toward your \
preferred pole. `resource_spend` is the fraction of your budget you are \
putting behind it; `magnitude` is how hard you are pushing, which scales your \
leverage. This is the ordinary action.
- ESCALATE (factor, magnitude): the same push at {ESCALATE_GAIN}x leverage, at \
a cost that rises with the square of the magnitude, plus added variance -- the \
result is noisier in both directions. Efficient at moderate magnitude, \
ruinous near 1.0.
- CONCEDE (factor): give ground. Pushes the factor toward the *other* side at \
weight {CONCEDE_WEIGHT}, refunds a little of your budget, and raises trust \
with whoever was contesting it. Sometimes the cheapest way to buy an ally.
- ALLY (actor, offered_share): pledge that share of your budget to another \
party -- they can spend it, you cannot -- and raise mutual trust by \
{ALLY_TRUST_GAIN}. Trust changes how much weight your claims carry with them.
- DEFECT (actor): break with a party. Trust between you collapses to \
{DEFECT_TRUST_FLOOR}, every pledge between you returns to its owner, and your \
standing with *every other party* falls. Defection is visible in your \
reputation whether or not anyone saw the act.
- SIGNAL (actor, factor, claimed_value, truthful): tell a party what a factor \
is. It shifts what they see, in proportion to how blurry their view of it \
already is and how much they trust you. It is free. `truthful` records whether \
the claim matches your own reading; it is never shown to anyone.
- WAIT: do nothing and recover {WAIT_REFUND:.0%} of your endowment.

# How to answer

Call `{ACTION_TOOL_NAME}` with exactly one action. Do not explain yourself \
outside the tool call; nothing outside it is read. The `rationale` field takes \
at most {RATIONALE_MAX_CHARS} characters and exists for the record -- it is \
stored and never parsed, so it changes nothing about what happens.

Act on what you see, not on what a well-informed observer would see. If your \
view is poor, that is part of your situation.\
"""


class ActorBrief(BaseModel):
    """The static, per-(scenario, actor) half of the prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: str
    name: str
    objective: str
    risk_posture: str
    constraints: tuple[str, ...]
    utility: tuple[tuple[str, float], ...]
    levers: tuple[tuple[str, float], ...]
    """factor id -> leverage weight."""
    counterparties: tuple[str, ...]
    horizon: int
    question_context: str
    evidence: tuple[tuple[str, str, str], ...]
    """(published_iso, source, excerpt) -- pre-cutoff, retrieved once per run."""


def persona_block(brief: ActorBrief, *, evidence_chars: int) -> str:
    """Render the cacheable per-actor block.

    Evidence excerpts are truncated to a fixed width so the block's length is
    bounded and, more importantly, *stable*: the cached prefix is only worth
    what it is when the same bytes recur across every call for this actor.
    """
    utility = ", ".join(f"{factor} {weight:+.2f}" for factor, weight in brief.utility)
    levers = ", ".join(f"{factor} (leverage {weight:.2f})" for factor, weight in brief.levers)
    constraints = "\n".join(f"- {item}" for item in brief.constraints) or "- none stated"
    evidence = (
        "\n\n".join(
            f"[{published}] {source}\n{excerpt[:evidence_chars]}"
            for published, source, excerpt in brief.evidence
        )
        or "(no admissible evidence was found before the cutoff)"
    )
    counterparties = ", ".join(brief.counterparties) or "(none reachable)"
    return f"""\
# You

{brief.name} (`{brief.actor_id}`). Risk posture: {brief.risk_posture}.

Objective: {brief.objective}

Utility weights: {utility}
Your levers: {levers or "(none -- you can move nothing directly)"}
Parties you can address: {counterparties}

Constraints you operate under:
{constraints}

# The situation

{brief.question_context}

The simulation runs {brief.horizon} steps.

# What was known before the cutoff

Everything below was published before the situation's cutoff date. It is the \
only outside information you have, it does not update as the simulation runs, \
and nothing after the cutoff exists for you.

{evidence}\
"""


def turn_message(
    observation: Observation,
    *,
    memory: str,
    horizon: int,
) -> str:
    """Render the dynamic half: this step's observation and the memory digest.

    Deliberately terse. §12.1 budgets 260 tokens for everything that varies
    within a run, and every token here is billed on every uncached call and
    changes the cache key on every call.
    """
    lines = [f"Step {observation.step} of {horizon}."]
    if observation.factors:
        readings = ", ".join(
            f"{factor}={observation.factors[factor]:.2f}"
            + (" (band)" if factor in observation.banded else "")
            for factor in sorted(observation.factors)
        )
        lines.append(f"You observe: {readings}")
    if observation.unobserved:
        lines.append(f"Unobserved: {', '.join(sorted(observation.unobserved))}")
    lines.append(
        f"Budget {observation.budget:.2f}"
        + (f" (+{observation.pooled:.2f} pledged to you)" if observation.pooled else "")
    )
    trust = ", ".join(
        f"{actor} {observation.trust[actor]:+.2f}"
        for actor in sorted(observation.trust)
        if observation.trust[actor] != 0.0
    )
    if trust:
        lines.append(f"Trust: {trust}")
    if observation.actions:
        seen = "; ".join(_render_action(action) for action in observation.actions)
        lines.append(f"Last step you saw: {seen}")
    lines.append(f"Your record:\n{memory}")
    lines.append("Take one action.")
    return "\n".join(lines)


def _render_action(action: Any) -> str:
    if action.target is None:
        return f"{action.actor_id} {action.action_type}"
    magnitude = f" {action.magnitude:.2f}" if action.magnitude is not None else ""
    return f"{action.actor_id} {action.action_type} {action.target}{magnitude}"


def brief_from(
    actor: Any,
    *,
    levers: Mapping[str, float],
    counterparties: Sequence[str],
    horizon: int,
    question_context: str,
    evidence: Sequence[tuple[str, str, str]],
) -> ActorBrief:
    """Assemble one actor's brief from the compiled graph and its evidence."""
    return ActorBrief(
        actor_id=actor.id,
        name=actor.name,
        objective=actor.objective,
        risk_posture=actor.risk_posture,
        constraints=tuple(actor.constraints),
        utility=tuple(
            (term.factor_id, float(term.weight))
            for term in sorted(actor.utility_terms, key=lambda t: t.factor_id)
        ),
        levers=tuple((factor, float(levers[factor])) for factor in sorted(levers)),
        counterparties=tuple(sorted(counterparties)),
        horizon=horizon,
        question_context=question_context,
        evidence=tuple(evidence),
    )
