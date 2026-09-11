"""The append-only event log's boundary types (spec §11.1, invariant 6).

One row per agent decision -- about 4.2M for the main study. The shapes here
are the schema's shapes, so a column that changes meaning changes a Pydantic
model first and fails a test, rather than changing a dict key and failing a
report three milestones later.

``caused_by`` is what turns the log from a transcript into a causal graph. It
is populated from the :class:`CausalLedger`, which records which decision moved
which factor; when an actor acts, the events that moved the factors *it can
see*, since *it last looked*, are the observations it is responding to. That is
a claim about attribution, so it is derived from the world's own bookkeeping
rather than asserted by the agent.

The log is append-only. There is no update helper in this module and no DELETE
in the migration's grants -- ``cascade_sim`` holds INSERT and SELECT and
nothing else (migration 009), because invariant 6 is the kind of rule that
survives exactly as long as it is impossible to break.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.canonical import canonical_json

__all__ = [
    "CAUSED_BY_LIMIT",
    "CausalLedger",
    "DecisionEvent",
    "EventRef",
    "StepRecord",
    "event_log_hash",
]

# Provenance is a chain, not a census: six antecedents is enough to follow a
# movement back and keeps the jsonb column small across 4.2M rows. The ledger
# keeps the largest movers, so what is dropped is the least of what happened.
CAUSED_BY_LIMIT = 6


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventRef(_Frozen):
    """A pointer into the event log. The primary key, spelled as a value."""

    run_id: str
    step: int = Field(ge=0)
    seq: int = Field(ge=0)

    def as_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "step": self.step, "seq": self.seq}


class DecisionEvent(_Frozen):
    """One agent decision, exactly as §11.1's table records it."""

    run_id: str
    step: int = Field(ge=0)
    seq: int = Field(ge=0)
    actor_id: str
    obs_hash: bytes
    action: dict[str, Any]
    caused_by: tuple[EventRef, ...] = ()
    factor_delta: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: int | None = None
    coercion: str | None = None
    """Why the emitted action was not executed as given (ADR-0018), or None."""

    @property
    def ref(self) -> EventRef:
        return EventRef(run_id=self.run_id, step=self.step, seq=self.seq)

    def canonical(self) -> dict[str, Any]:
        """Sorted, primitive projection -- the unit the M8 replay hash is over.

        ``latency_ms`` is deliberately **excluded**: it is wall-clock, it
        differs between a recorded run and its replay by construction, and
        including it would make the byte-identical criterion unachievable for
        a reason that has nothing to do with determinism.
        """
        return {
            "step": self.step,
            "seq": self.seq,
            "actor_id": self.actor_id,
            "obs_hash": self.obs_hash.hex(),
            "action": self.action,
            "caused_by": [ref.as_json() for ref in self.caused_by],
            "factor_delta": {key: self.factor_delta[key] for key in sorted(self.factor_delta)},
            "cache_hit": self.cache_hit,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "coercion": self.coercion,
        }


class StepRecord(_Frozen):
    """What one world step did, apart from the decisions (spec §7.2).

    Exogenous shocks live here rather than in ``events``: §11.1's table is one
    row per *agent decision*, and 864,000 synthetic non-decision rows would put
    the M6 event count 20% over a criterion stated as 4.2M ± 5%. §11.2's trace
    still reaches a root exogenous shock, because the chain terminates in this
    table by ``(run_id, step)``.
    """

    run_id: str
    step: int = Field(ge=0)
    exogenous_delta: dict[str, float]
    arrivals: dict[str, float]
    contest_delta: dict[str, float]
    active: tuple[str, ...]
    eligible: int = Field(ge=0)
    rng_counter: int = Field(ge=0)
    state_hash: str
    absorbed: tuple[str, ...] = ()

    @property
    def activation_rate(self) -> float:
        return len(self.active) / self.eligible if self.eligible else 0.0

    def canonical(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "exogenous_delta": {
                key: self.exogenous_delta[key] for key in sorted(self.exogenous_delta)
            },
            "arrivals": {key: self.arrivals[key] for key in sorted(self.arrivals)},
            "contest_delta": {key: self.contest_delta[key] for key in sorted(self.contest_delta)},
            "active": sorted(self.active),
            "rng_counter": self.rng_counter,
            "state_hash": self.state_hash,
            "absorbed": sorted(self.absorbed),
        }


class CausalLedger:
    """Which decision last moved which factor, and by how much.

    Mutable by design and owned by one run: it is the kernel's working memory
    for attribution, not state anyone replays. Everything it produces is
    deterministic -- entries are selected by magnitude and returned in
    (step, seq) order, with the reference settling an exact tie.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[tuple[int, int, int, float, EventRef]]] = {}

    def record(self, factor_id: str, *, magnitude: float, ref: EventRef, at_step: int) -> None:
        """Note that ``ref`` moved ``factor_id``, and when the movement landed.

        ``at_step`` is not always ``ref.step``: a lagged effect scheduled at
        step 2 and arriving at step 8 is *caused* at 2 and *felt* at 8. An
        actor reacting at step 9 is responding to the arrival, so the window is
        matched against ``at_step`` while the reference still points at the
        decision -- which is what lets §11.2's chain cross a lag boundary
        rather than stopping at it.
        """
        if magnitude == 0.0:
            return
        self._entries.setdefault(factor_id, []).append(
            (at_step, ref.step, ref.seq, abs(magnitude), ref)
        )

    def antecedents(
        self, factors: Sequence[str], *, since: int, before: int, limit: int = CAUSED_BY_LIMIT
    ) -> tuple[EventRef, ...]:
        """The decisions that moved ``factors`` in ``[since, before)``.

        ``since`` is the step the acting actor last observed: movement it has
        already seen and acted on is not what it is responding to now.
        """
        candidates: list[tuple[float, int, int, EventRef]] = []
        for factor_id in sorted(set(factors)):
            for at_step, step, seq, magnitude, ref in self._entries.get(factor_id, ()):
                if since <= at_step < before:
                    candidates.append((magnitude, step, seq, ref))
        chosen = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:limit]
        return tuple(ref for _, _, _, ref in sorted(chosen, key=lambda item: (item[1], item[2])))

    def prune(self, *, before_step: int) -> None:
        """Drop entries older than a step nobody can still be responding to.

        Bounds memory on a long run without changing any answer: an actor's
        antecedent window opens at its last observation, and the scheduled
        floor guarantees that is never more than ``forced_interval`` steps ago.
        """
        for factor_id in sorted(self._entries):
            self._entries[factor_id] = [
                entry for entry in self._entries[factor_id] if entry[0] >= before_step
            ]


def event_log_hash(events: Sequence[DecisionEvent], steps: Sequence[StepRecord] = ()) -> str:
    """The replay hash (spec §8.4, M8 criterion 1).

    Ordered by ``(step, seq)`` -- §8.1's rule that ordering is by key and never
    by insertion time -- and computed over the canonical projection, so it is
    stable across processes, workers and JSON libraries. Step records are
    folded in as well: two runs that produced identical decisions from
    different worlds are not the same run.
    """
    payload = {
        "events": [
            event.canonical() for event in sorted(events, key=lambda e: (e.run_id, e.step, e.seq))
        ],
        "steps": [step.canonical() for step in sorted(steps, key=lambda s: (s.run_id, s.step))],
    }
    return hashlib.blake2b(canonical_json(payload).encode("utf-8"), digest_size=32).hexdigest()


def action_payload(action: Any) -> dict[str, Any]:
    """Project an action onto the jsonb column, sorted and primitive."""
    dumped: Mapping[str, Any] = action.model_dump(mode="json")
    return {key: dumped[key] for key in sorted(dumped)}
