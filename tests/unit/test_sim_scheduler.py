"""The activation scheduler (spec §7.3). Pure, so this needs nothing live.

The 34.7% rate is not asserted here and cannot be: it is an emergent property
of these rules over real graphs, measured by `cascade simulate activation`.
What is asserted is that each of the three rules does what §7.3 says, that the
cap is deterministic, and that the scheduled floor is not quietly optional.
"""

from __future__ import annotations

from cascade.sim.actions import Commit, Wait
from cascade.sim.scheduler import ActorProfile, activate
from cascade.sim.state import WorldState

FACTORS = ("factor_0", "factor_1")


def profile(
    actor_id: str, *, observed: tuple[str, ...] = FACTORS, **overrides: object
) -> ActorProfile:
    base: dict[str, object] = {
        "actor_id": actor_id,
        "utility": {"factor_0": 0.5, "factor_1": -0.5},
        "observed": observed,
        "watches": {},
    }
    base.update(overrides)
    return ActorProfile(**base)  # type: ignore[arg-type]


def world(
    *,
    step: int,
    history: dict[str, tuple[float, ...]],
    last_acted: dict[str, int],
    last_observed_at: dict[str, int],
) -> WorldState:
    return WorldState(
        step=step,
        factors={
            key: values[min(step, len(values) - 1)] for key, values in sorted(history.items())
        },
        factor_history=history,
        resources={},
        relations={},
        last_acted=last_acted,
        last_observed_at=last_observed_at,
    )


def test_movement_past_the_threshold_triggers_an_actor() -> None:
    """§7.3: "any observed factor moved by more than 0.06 since ... last observation"."""
    state = world(
        step=2,
        history={"factor_0": (0.50, 0.55, 0.62), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2},
        last_observed_at={"actor_0": 0},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0")},
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ("actor_0",)
    assert decision.active == ("actor_0",)


def test_movement_below_the_threshold_does_not() -> None:
    state = world(
        step=2,
        history={"factor_0": (0.50, 0.52, 0.55), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2},
        last_observed_at={"actor_0": 0},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0")},
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ()
    assert decision.active == ()


def test_an_invisible_factor_cannot_wake_an_actor() -> None:
    """Only *observed* factors trigger -- otherwise the policy does nothing here."""
    state = world(
        step=2,
        history={"factor_0": (0.50, 0.70, 0.90), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2},
        last_observed_at={"actor_0": 0},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0", observed=("factor_1",))},
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ()


def test_an_actor_that_never_observed_is_not_triggered_by_movement() -> None:
    """It has no baseline to have moved from; the floor is what wakes it."""
    state = world(
        step=3,
        history={"factor_0": (0.1, 0.4, 0.7, 0.9), "factor_1": (0.5, 0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2},
        last_observed_at={},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0")},
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ()


def test_a_visible_action_on_a_factor_i_care_about_triggers_me() -> None:
    state = world(
        step=2,
        history={"factor_0": (0.5, 0.5, 0.5), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2, "actor_1": 1},
        last_observed_at={"actor_0": 1},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0", watches={"actor_1": "full"})},
        world=state,
        last_actions={
            "actor_1": Commit(target_factor="factor_0", magnitude=0.6, resource_spend=0.3)
        },
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ("actor_0",)


def test_a_type_only_view_does_not_trigger_on_the_target() -> None:
    """The actor cannot know what the action was aimed at, so it cannot react to it."""
    state = world(
        step=2,
        history={"factor_0": (0.5, 0.5, 0.5), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2, "actor_1": 1},
        last_observed_at={"actor_0": 1},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0", watches={"actor_1": "type_only"})},
        world=state,
        last_actions={
            "actor_1": Commit(target_factor="factor_0", magnitude=0.6, resource_spend=0.3)
        },
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ()


def test_the_scheduled_floor_wakes_a_silent_actor() -> None:
    """§7.3: "every actor acts at least once every 5 steps regardless"."""
    state = world(
        step=6,
        history={"factor_0": (0.5,) * 7, "factor_1": (0.5,) * 7},
        last_acted={"actor_0": 1},
        last_observed_at={"actor_0": 1},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0")},
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.forced == ("actor_0",)
    assert decision.active == ("actor_0",)


def test_the_cap_holds_and_prefers_the_floor_then_salience() -> None:
    """§7.3: at most 8 act, ties broken by utility-weighted salience."""
    history = {"factor_0": (0.10, 0.30, 0.70), "factor_1": (0.5, 0.5, 0.5)}
    profiles = {
        f"actor_{index}": profile(
            f"actor_{index}", utility={"factor_0": (index + 1) / 12.0, "factor_1": -0.1}
        )
        for index in range(12)
    }
    state = world(
        step=2,
        history=history,
        last_acted={actor_id: 2 for actor_id in profiles},
        last_observed_at={actor_id: 0 for actor_id in profiles},
    )
    decision = activate(
        profiles=profiles,
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert len(decision.active) == 8
    assert len(decision.crowded_out) == 4
    # The four dropped are the four with the smallest utility weight on the
    # factor that moved -- selection is by salience, not by name.
    assert set(decision.crowded_out) == {f"actor_{index}" for index in range(4)}


def test_an_overdue_actor_outranks_a_merely_salient_one() -> None:
    """A missed floor becomes more overdue and takes priority on the next step."""
    profiles = {f"actor_{index}": profile(f"actor_{index}") for index in range(3)}
    state = world(
        step=9,
        history={"factor_0": (0.1,) * 9 + (0.9,), "factor_1": (0.5,) * 10},
        last_acted={"actor_0": 8, "actor_1": 8, "actor_2": 1},
        last_observed_at={actor_id: 8 for actor_id in profiles},
    )
    decision = activate(
        profiles=profiles,
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=1,
    )
    assert decision.active == ("actor_2",)
    assert "actor_2" in decision.forced


def test_activation_is_identical_under_a_reordered_profile_map() -> None:
    """Invariant 7: iteration order must not decide who acts."""
    history = {"factor_0": (0.10, 0.30, 0.70), "factor_1": (0.5, 0.5, 0.5)}
    names = [f"actor_{index}" for index in range(10)]
    profiles = {name: profile(name) for name in names}
    reversed_profiles = {name: profiles[name] for name in reversed(names)}
    state = world(
        step=2,
        history=history,
        last_acted={name: 2 for name in names},
        last_observed_at={name: 0 for name in names},
    )
    kwargs = {
        "world": state,
        "last_actions": {},
        "salience_threshold": 0.06,
        "forced_interval": 5,
        "max_active": 6,
    }
    assert activate(profiles=profiles, **kwargs) == activate(  # type: ignore[arg-type]
        profiles=reversed_profiles, **kwargs  # type: ignore[arg-type]
    )


def test_the_rate_is_over_the_eligible_cast() -> None:
    profiles = {f"actor_{index}": profile(f"actor_{index}") for index in range(4)}
    state = world(
        step=1,
        history={"factor_0": (0.5, 0.5), "factor_1": (0.5, 0.5)},
        last_acted={"actor_0": -5, "actor_1": 0, "actor_2": 0, "actor_3": 0},
        last_observed_at={},
    )
    decision = activate(
        profiles=profiles,
        world=state,
        last_actions={},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.rate == len(decision.active) / 4


def test_a_waiting_actor_does_not_trigger_anyone() -> None:
    state = world(
        step=2,
        history={"factor_0": (0.5, 0.5, 0.5), "factor_1": (0.5, 0.5, 0.5)},
        last_acted={"actor_0": 2, "actor_1": 1},
        last_observed_at={"actor_0": 1},
    )
    decision = activate(
        profiles={"actor_0": profile("actor_0", watches={"actor_1": "full"})},
        world=state,
        last_actions={"actor_1": Wait()},
        salience_threshold=0.06,
        forced_interval=5,
        max_active=8,
    )
    assert decision.triggered == ()
