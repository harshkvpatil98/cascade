"""The event log's boundary types and the replay hash (spec §11.1, §8.4)."""

from __future__ import annotations

from cascade.trace.events import (
    CAUSED_BY_LIMIT,
    CausalLedger,
    DecisionEvent,
    EventRef,
    StepRecord,
    event_log_hash,
)

RUN = "00000000-0000-0000-0000-000000000001"


def event(step: int, seq: int, *, latency: int | None = 12, **overrides: object) -> DecisionEvent:
    base: dict[str, object] = {
        "run_id": RUN,
        "step": step,
        "seq": seq,
        "actor_id": f"actor_{seq}",
        "obs_hash": bytes([step, seq] * 8),
        "action": {"type": "WAIT", "rationale": ""},
        "factor_delta": {"factor_0": 0.01 * seq},
        "cache_hit": False,
        "tokens_in": 4200,
        "tokens_out": 45,
        "latency_ms": latency,
    }
    base.update(overrides)
    return DecisionEvent(**base)  # type: ignore[arg-type]


def step_record(step: int) -> StepRecord:
    return StepRecord(
        run_id=RUN,
        step=step,
        exogenous_delta={"factor_0": 0.01},
        arrivals={},
        contest_delta={"factor_0": 0.02},
        active=("actor_0", "actor_1"),
        eligible=14,
        rng_counter=(step + 1) * 128,
        state_hash=f"hash-{step}",
    )


# ---------------------------------------------------------------------------
# The replay hash
# ---------------------------------------------------------------------------


def test_the_hash_ignores_insertion_order() -> None:
    """§8.1: ordering is by (run_id, step, seq), never by insertion time."""
    forward = [event(0, 0), event(0, 1), event(1, 0)]
    shuffled = [forward[2], forward[0], forward[1]]
    assert event_log_hash(forward) == event_log_hash(shuffled)


def test_the_hash_ignores_latency() -> None:
    """Wall-clock differs between a recording and its replay by construction.

    Including it would make M8's byte-identical criterion unachievable for a
    reason that has nothing to do with determinism.
    """
    assert event_log_hash([event(0, 0, latency=11)]) == event_log_hash([event(0, 0, latency=9999)])


def test_the_hash_covers_the_action_and_the_delta() -> None:
    assert event_log_hash([event(0, 0)]) != event_log_hash(
        [event(0, 0, action={"type": "COMMIT", "rationale": ""})]
    )
    assert event_log_hash([event(0, 0)]) != event_log_hash(
        [event(0, 0, factor_delta={"factor_0": 0.99})]
    )


def test_the_hash_covers_the_world_as_well_as_the_decisions() -> None:
    """Two runs that decided the same things from different worlds are not one run."""
    events = [event(0, 0)]
    left = event_log_hash(events, [step_record(0)])
    right = event_log_hash(events, [step_record(0).model_copy(update={"state_hash": "different"})])
    assert left != right
    assert left != event_log_hash(events)


# ---------------------------------------------------------------------------
# The causal ledger
# ---------------------------------------------------------------------------


def test_antecedents_are_windowed_on_when_the_movement_was_felt() -> None:
    """A lagged effect is caused at one step and felt at another.

    An effect scheduled at step 2 that arrives at step 8 is what an actor
    reacting at step 9 is responding to; windowing on the *causing* step would
    drop it and the chain would stop at the lag boundary.
    """
    ledger = CausalLedger()
    origin = EventRef(run_id=RUN, step=2, seq=0)
    ledger.record("factor_0", magnitude=0.05, ref=origin, at_step=8)

    assert ledger.antecedents(["factor_0"], since=7, before=9) == (origin,)
    assert ledger.antecedents(["factor_0"], since=0, before=3) == ()


def test_antecedents_exclude_movement_the_actor_already_saw() -> None:
    ledger = CausalLedger()
    ledger.record("factor_0", magnitude=0.05, ref=EventRef(run_id=RUN, step=1, seq=0), at_step=1)
    recent = EventRef(run_id=RUN, step=5, seq=2)
    ledger.record("factor_0", magnitude=0.02, ref=recent, at_step=5)

    assert ledger.antecedents(["factor_0"], since=4, before=6) == (recent,)


def test_antecedents_are_capped_and_keep_the_largest_movers() -> None:
    """Provenance is a chain, not a census; 4.2M rows make the cap a size decision."""
    ledger = CausalLedger()
    for seq in range(CAUSED_BY_LIMIT + 4):
        ledger.record(
            "factor_0",
            magnitude=0.001 * (seq + 1),
            ref=EventRef(run_id=RUN, step=1, seq=seq),
            at_step=1,
        )
    chosen = ledger.antecedents(["factor_0"], since=0, before=2)
    assert len(chosen) == CAUSED_BY_LIMIT
    # The four smallest movers are what got dropped.
    assert {ref.seq for ref in chosen} == {4, 5, 6, 7, 8, 9}
    assert [ref.seq for ref in chosen] == sorted(ref.seq for ref in chosen)


def test_a_zero_movement_is_not_an_antecedent() -> None:
    """An action that moved nothing did not cause what happened next."""
    ledger = CausalLedger()
    ledger.record("factor_0", magnitude=0.0, ref=EventRef(run_id=RUN, step=1, seq=0), at_step=1)
    assert ledger.antecedents(["factor_0"], since=0, before=2) == ()


def test_pruning_drops_only_what_nobody_can_be_responding_to() -> None:
    ledger = CausalLedger()
    old = EventRef(run_id=RUN, step=1, seq=0)
    new = EventRef(run_id=RUN, step=9, seq=0)
    ledger.record("factor_0", magnitude=0.05, ref=old, at_step=1)
    ledger.record("factor_0", magnitude=0.05, ref=new, at_step=9)

    ledger.prune(before_step=5)
    assert ledger.antecedents(["factor_0"], since=0, before=12) == (new,)
