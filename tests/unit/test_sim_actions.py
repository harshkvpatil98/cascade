"""The closed action union and its admissibility rules (§7.4, ADR-0018)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cascade.sim.actions import (
    ACTION_TYPES,
    RATIONALE_MAX_CHARS,
    ActionSpace,
    Ally,
    Commit,
    Escalate,
    Signal,
    Wait,
    admit,
    parse_action,
)

SPACE = ActionSpace(
    actor_id="incumbent_party",
    levers=("coalition_stability", "external_financing"),
    counterparties=("regional_bloc", "external_guarantor"),
)


def test_every_action_type_parses_from_the_wire_shape() -> None:
    """The union is closed: seven members, and the discriminator selects one."""
    payloads = [
        {
            "type": "COMMIT",
            "target_factor": "coalition_stability",
            "magnitude": 0.4,
            "resource_spend": 0.2,
        },
        {"type": "ESCALATE", "target_factor": "coalition_stability", "magnitude": 0.8},
        {"type": "CONCEDE", "target_factor": "coalition_stability"},
        {"type": "ALLY", "target_actor": "regional_bloc", "offered_share": 0.3},
        {"type": "DEFECT", "target_actor": "regional_bloc"},
        {
            "type": "SIGNAL",
            "target_actor": "regional_bloc",
            "claimed_factor": "coalition_stability",
            "claimed_value": 0.9,
            "truthful": False,
        },
        {"type": "WAIT"},
    ]
    parsed = [parse_action(payload) for payload in payloads]
    assert sorted(action.type for action in parsed) == sorted(ACTION_TYPES)


def test_an_unknown_action_type_does_not_parse() -> None:
    """A closed union is the reason the arbiter can be exhaustive."""
    with pytest.raises(ValidationError):
        parse_action({"type": "BRIBE", "target_actor": "regional_bloc"})


def test_bounds_are_enforced_at_parse_time() -> None:
    with pytest.raises(ValidationError):
        parse_action(
            {"type": "COMMIT", "target_factor": "f", "magnitude": 0.0, "resource_spend": 0.5}
        )
    with pytest.raises(ValidationError):
        parse_action(
            {"type": "COMMIT", "target_factor": "f", "magnitude": 1.5, "resource_spend": 0.5}
        )
    with pytest.raises(ValidationError):
        parse_action({"type": "ALLY", "target_actor": "x", "offered_share": 1.2})


def test_an_over_long_rationale_is_rejected_by_the_schema() -> None:
    """§7.4 caps it; the cap is what keeps output cost near $0.0002 a decision."""
    with pytest.raises(ValidationError):
        parse_action({"type": "WAIT", "rationale": "x" * (RATIONALE_MAX_CHARS + 1)})


def test_an_extra_field_is_rejected() -> None:
    """A field nobody reads is a field the model thinks it is using."""
    with pytest.raises(ValidationError):
        parse_action({"type": "WAIT", "urgency": "high"})


def test_an_admissible_action_passes_through_unchanged() -> None:
    action, coercion = admit(
        {
            "type": "COMMIT",
            "target_factor": "coalition_stability",
            "magnitude": 0.4,
            "resource_spend": 0.2,
        },
        SPACE,
    )
    assert coercion is None
    assert isinstance(action, Commit)


def test_a_factor_the_actor_cannot_move_is_coerced_to_wait() -> None:
    """ADR-0018: recorded, not repaired. A second call would cost 36,000 of them."""
    action, coercion = admit(
        {"type": "ESCALATE", "target_factor": "global_rates", "magnitude": 0.9}, SPACE
    )
    assert isinstance(action, Wait)
    assert coercion == "no_lever:global_rates"


def test_an_actor_with_no_channel_cannot_be_addressed() -> None:
    action, coercion = admit(
        {"type": "ALLY", "target_actor": "unseen_party", "offered_share": 0.5}, SPACE
    )
    assert isinstance(action, Wait)
    assert coercion == "no_channel:unseen_party"


def test_an_unparseable_payload_becomes_a_wait_and_keeps_its_rationale() -> None:
    """One bad answer costs one turn, not the remaining 23 steps of the run."""
    action, coercion = admit({"type": "COMMIT", "rationale": "press the advantage"}, SPACE)
    assert isinstance(action, Wait)
    assert action.rationale == "press the advantage"
    assert coercion is not None and coercion.startswith("unparseable:")


def test_a_coerced_action_keeps_the_rationale_it_came_with() -> None:
    action, coercion = admit(
        {
            "type": "ESCALATE",
            "target_factor": "global_rates",
            "magnitude": 0.9,
            "rationale": "escalate on rates",
        },
        SPACE,
    )
    assert action.rationale == "escalate on rates"
    assert coercion == "no_lever:global_rates"


def test_a_signal_may_name_a_factor_the_speaker_cannot_move() -> None:
    """Lying about what you do not control is the ordinary case, not an error."""
    action, coercion = admit(
        {
            "type": "SIGNAL",
            "target_actor": "regional_bloc",
            "claimed_factor": "global_rates",
            "claimed_value": 0.1,
            "truthful": False,
        },
        SPACE,
    )
    assert coercion is None
    assert isinstance(action, Signal)


def test_an_actor_cannot_address_itself() -> None:
    space = SPACE.model_copy(update={"counterparties": ("incumbent_party",)})
    _, coercion = admit(
        {"type": "ALLY", "target_actor": "incumbent_party", "offered_share": 0.5}, space
    )
    assert coercion == "self_target:incumbent_party"


def test_actions_are_frozen() -> None:
    """An action is a decision, not a mutable buffer the arbiter can edit."""
    action = Escalate(target_factor="coalition_stability", magnitude=0.5)
    with pytest.raises(ValidationError):
        action.magnitude = 0.9  # type: ignore[misc]


def test_ally_and_commit_are_distinguished_by_the_discriminator_only() -> None:
    """Two actions with the same shape must not be interchangeable."""
    ally = Ally(target_actor="regional_bloc", offered_share=0.3)
    assert ally.type == "ALLY"
    assert not isinstance(ally, Commit)
