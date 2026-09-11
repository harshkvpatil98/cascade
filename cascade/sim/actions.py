"""The closed action union agents emit (spec §7.4).

Agents do not emit free text into the world. They emit one member of a
discriminated union, which is what makes the arbiter deterministic and the
output tokens cheap. Everything here is pure: parsing, bounds and
admissibility are decided without a model, a clock or a database.

``rationale`` is capped and **never parsed**. It exists so a provenance chain
reads as something other than a column of enum values; §7.4 caps it at 30
tokens because output tokens are the only part of a decision's cost that scales
with what the model feels like saying.

Admissibility (ADR-0018) is enforced here rather than trusted to the prompt. An
actor may only push a factor it has an outbound edge to, and may only address
an actor it can see. A model that asks for anything else is not corrected and
not re-prompted -- that would cost a second call at 36,000 runs and make the
cache hit rate depend on model mood -- it is coerced to WAIT and the coercion
is recorded on the event, so "how often did agents reach for a lever they do
not have" is a query rather than a guess.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "ACTION_TYPES",
    "RATIONALE_MAX_CHARS",
    "Action",
    "ActionSpace",
    "Ally",
    "Commit",
    "Concede",
    "Defect",
    "Escalate",
    "Signal",
    "Wait",
    "admit",
    "parse_action",
    "refusal",
]

# §7.4 caps the rationale at 30 tokens. Characters are what a schema can
# enforce; at Claude's ~3.6 chars/token on English prose 120 characters is
# about 33 tokens, and the field is truncated rather than rejected because a
# long rationale is a cosmetic overrun and re-asking for one would cost a call.
RATIONALE_MAX_CHARS = 120

Slug = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Magnitude = Annotated[float, Field(gt=0.0, le=1.0)]
Rationale = Annotated[str, Field(default="", max_length=RATIONALE_MAX_CHARS)]


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rationale: Rationale = ""


class Commit(_Action):
    """Push a factor toward this actor's preferred pole, scaled by contest share."""

    type: Literal["COMMIT"] = "COMMIT"
    target_factor: Slug
    magnitude: Magnitude
    resource_spend: Unit


class Signal(_Action):
    """Tell another actor a value for a factor. Truth is not enforced (§7.4).

    ``truthful`` is private: it is recorded in the event log and never shown to
    the target, which is the whole point -- a signal the recipient could verify
    is not a signal, it is an announcement.
    """

    type: Literal["SIGNAL"] = "SIGNAL"
    target_actor: Slug
    claimed_factor: Slug
    claimed_value: Unit
    truthful: bool


class Ally(_Action):
    """Offer a share of this actor's resources; raises mutual trust and pools."""

    type: Literal["ALLY"] = "ALLY"
    target_actor: Slug
    offered_share: Magnitude


class Defect(_Action):
    """Break with another actor: trust collapses, pledges return, reputation pays."""

    type: Literal["DEFECT"] = "DEFECT"
    target_actor: Slug


class Escalate(_Action):
    """A large push with superlinear cost and a variance penalty (§7.4)."""

    type: Literal["ESCALATE"] = "ESCALATE"
    target_factor: Slug
    magnitude: Magnitude


class Concede(_Action):
    """Move a factor toward the opponent's pole; refunds resources, raises trust."""

    type: Literal["CONCEDE"] = "CONCEDE"
    target_factor: Slug


class Wait(_Action):
    """No delta; accrues a small resource refund. The default under low salience."""

    type: Literal["WAIT"] = "WAIT"


Action = Annotated[
    Commit | Signal | Ally | Defect | Escalate | Concede | Wait,
    Field(discriminator="type"),
]

ACTION_TYPES: tuple[str, ...] = (
    "ALLY",
    "COMMIT",
    "CONCEDE",
    "DEFECT",
    "ESCALATE",
    "SIGNAL",
    "WAIT",
)

_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


class ActionSpace(BaseModel):
    """What one actor may do this step -- derived, not authored.

    ``levers`` are the factors the actor has an outbound edge to (§5.1: an edge
    from an actor to a factor *is* the claim that it can act on it).
    ``counterparties`` are the actors it can see well enough to address, which
    comes from the Aperture policy, so an actor cannot ally with a party it has
    no channel to.

    Both are sorted tuples: they are rendered into the prompt, and an unsorted
    render would change the prompt string -- and therefore the cache key -- for
    a state that is otherwise identical (invariant 7, and the 88% hit rate the
    cost model needs).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: str
    levers: tuple[str, ...]
    counterparties: tuple[str, ...]

    def permits(self, action: Action) -> str | None:
        """Return None if the action is admissible, else a short reason code.

        Reason codes are stable strings, not prose: they are written to the
        event log and aggregated across 36,000 runs.
        """
        if isinstance(action, Commit | Escalate | Concede):
            if action.target_factor not in self.levers:
                return "no_lever"
            return None
        if isinstance(action, Signal):
            # A claim about a factor the speaker cannot act on is still a claim
            # it is free to make -- lying about something you do not control is
            # the ordinary case. What it cannot do is address an actor it has
            # no channel to.
            if action.target_actor not in self.counterparties:
                return "no_channel"
            if action.target_actor == self.actor_id:
                return "self_target"
            return None
        if isinstance(action, Ally | Defect):
            if action.target_actor not in self.counterparties:
                return "no_channel"
            if action.target_actor == self.actor_id:
                return "self_target"
            return None
        return None


def parse_action(payload: Any) -> Action:
    """Validate a model's tool input into the union.

    Raises :class:`pydantic.ValidationError` on anything that is not a member.
    The caller decides what a malformed action means -- at M5 it becomes a WAIT
    with a recorded coercion, because one unparseable decision must not abort a
    run that has 23 more steps of useful behaviour in it.
    """
    return _ADAPTER.validate_python(payload)


def admit(payload: Any, space: ActionSpace) -> tuple[Action, str | None]:
    """Parse and admit one action, coercing an inadmissible one to WAIT.

    Returns ``(executed_action, coercion_reason)``. The reason is None when the
    model's action was executed as emitted; otherwise it names why it was not,
    and the executed action is a WAIT carrying the original rationale so the
    provenance chain still says what the actor thought it was doing.
    """
    try:
        action = parse_action(payload)
    except ValidationError as exc:
        return Wait(rationale=_rationale_of(payload)), f"unparseable:{exc.error_count()}"
    reason = refusal(action, space)
    if reason is None:
        return action, None
    return Wait(rationale=action.rationale), reason


def refusal(action: Action, space: ActionSpace) -> str | None:
    """Why ``space`` refuses ``action``, as a stable code, or None if it does not.

    Shared by the agent adapter and the kernel on purpose. The adapter admits
    what a *model* emitted; the kernel admits what *any* decider returned,
    including a stand-in policy that never went near the schema. Admissibility
    has to hold at the boundary the arbiter is behind, not at the boundary the
    model happens to be in front of.
    """
    reason = space.permits(action)
    if reason is None:
        return None
    return f"{reason}:{_target_of(action)}"


def _rationale_of(payload: Any) -> str:
    """Salvage the rationale from an unparseable payload, if there is one."""
    if isinstance(payload, dict):
        value = payload.get("rationale")
        if isinstance(value, str):
            return value[:RATIONALE_MAX_CHARS]
    return ""


def _target_of(action: Action) -> str:
    if isinstance(action, Commit | Escalate | Concede):
        return action.target_factor
    if isinstance(action, Signal | Ally | Defect):
        return action.target_actor
    return ""
