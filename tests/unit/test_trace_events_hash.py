"""The replay hash's contract (spec §8.4; M8 criterion 1).

`event_log_hash` is what the determinism criterion is stated in, so what it
covers and what it deliberately ignores are both properties worth pinning. A
hash that included wall-clock latency would make the criterion unachievable;
one that ignored the world state would make two different runs look identical.
"""

from __future__ import annotations

from cascade.trace.events import DecisionEvent, EventRef, StepRecord, event_log_hash


def event(step: int, seq: int, **overrides: object) -> DecisionEvent:
    base: dict[str, object] = {
        "run_id": "r1",
        "step": step,
        "seq": seq,
        "actor_id": f"actor_{seq}",
        "obs_hash": b"\x01\x02",
        "action": {"type": "WAIT"},
        "factor_delta": {"a": 0.1},
        "cache_hit": False,
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 42,
    }
    base.update(overrides)
    return DecisionEvent(**base)  # type: ignore[arg-type]


def step_record(step: int, **overrides: object) -> StepRecord:
    base: dict[str, object] = {
        "run_id": "r1",
        "step": step,
        "exogenous_delta": {"a": 0.01},
        "arrivals": {},
        "contest_delta": {"a": 0.02},
        "active": ("actor_0",),
        "eligible": 14,
        "rng_counter": step * 100,
        "state_hash": f"hash-{step}",
    }
    base.update(overrides)
    return StepRecord(**base)  # type: ignore[arg-type]


class TestStability:
    def test_the_hash_is_independent_of_insertion_order(self) -> None:
        """§8.1: ordering is by key, never by insertion time. A log read back
        from a partition-wise scan must hash to what the writer computed."""
        events = [event(1, 0), event(0, 1), event(0, 0)]
        assert event_log_hash(events) == event_log_hash(sorted(events, key=lambda e: e.step))

    def test_wall_clock_latency_is_excluded(self) -> None:
        """It differs between a run and its replay by construction. Including
        it would make the byte-identical criterion unachievable for a reason
        with nothing to do with determinism."""
        assert event_log_hash([event(0, 0, latency_ms=1)]) == event_log_hash(
            [event(0, 0, latency_ms=9_999)]
        )

    def test_the_hash_is_stable_across_calls(self) -> None:
        events = [event(0, 0), event(0, 1)]
        assert event_log_hash(events) == event_log_hash(events)


class TestSensitivity:
    def test_a_changed_action_changes_the_hash(self) -> None:
        assert event_log_hash([event(0, 0)]) != event_log_hash(
            [event(0, 0, action={"type": "ESCALATE"})]
        )

    def test_a_changed_factor_delta_changes_the_hash(self) -> None:
        assert event_log_hash([event(0, 0)]) != event_log_hash(
            [event(0, 0, factor_delta={"a": 0.2})]
        )

    def test_a_changed_cause_changes_the_hash(self) -> None:
        """`caused_by` is the provenance chain. A replay that produced the same
        decisions from different antecedents is not the same run."""
        caused = (EventRef(run_id="r1", step=0, seq=0),)
        assert event_log_hash([event(1, 0)]) != event_log_hash([event(1, 0, caused_by=caused)])

    def test_a_coercion_changes_the_hash(self) -> None:
        """ADR-0018 coerces an inadmissible action to WAIT and records why. Two
        runs where one actor was coerced and the other was not are different
        runs even if the executed actions match."""
        assert event_log_hash([event(0, 0)]) != event_log_hash(
            [event(0, 0, coercion="lever_not_held")]
        )

    def test_the_world_is_folded_in_as_well_as_the_decisions(self) -> None:
        """Two runs that produced identical decisions from different worlds are
        not the same run."""
        events = [event(0, 0)]
        assert event_log_hash(events, [step_record(0)]) != event_log_hash(
            events, [step_record(0, state_hash="different")]
        )

    def test_an_exogenous_shock_changes_the_hash(self) -> None:
        events = [event(0, 0)]
        assert event_log_hash(events, [step_record(0)]) != event_log_hash(
            events, [step_record(0, exogenous_delta={"a": 0.99})]
        )

    def test_the_rng_counter_is_covered(self) -> None:
        """ADR-0015 asserts `rng_counter == step x draws_per_step` at the end
        of every step. Folding it into the hash means a replay that drew a
        different number of values diverges here rather than silently later."""
        events = [event(0, 0)]
        assert event_log_hash(events, [step_record(0)]) != event_log_hash(
            events, [step_record(0, rng_counter=7)]
        )


class TestShape:
    def test_it_is_a_256_bit_hex_digest(self) -> None:
        digest = event_log_hash([event(0, 0)])
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_an_empty_log_still_hashes(self) -> None:
        """A run that reached its horizon with no active actor is a real run."""
        assert len(event_log_hash([])) == 64
