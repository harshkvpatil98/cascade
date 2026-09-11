"""Private per-agent memory (spec §6.3).

Each agent has its own namespace, keyed ``(run_id, actor_id)``, holding only
observations it actually received and actions it actually took. Agents read
their own namespace and Chronofence -- never another agent's. That is enforced
structurally here: a memory carries one ``actor_id`` and there is no method
that merges two.

Capacity is the spec's: the most recent 12 observations plus a rolling summary
refreshed every 8 steps. Both numbers come from config, and both exist for the
same reason -- to bound prompt growth so the §12.1 cost model stays flat across
the 24-step horizon instead of growing linearly with it.

**The summary is computed, not generated.** An LLM summariser would add roughly
three calls per run against a budget of 10.5, would have to be cached and
replayed like every other call, and would put a model in the loop of what an
agent remembers -- where a hallucination becomes indistinguishable from an
observation. The summary here is a deterministic digest: factor drift since the
last refresh, action counts, and the trust that moved.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.aperture.projection import OBSERVATION_DECIMALS, Observation

__all__ = ["AgentMemory", "MemoryEntry"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryEntry(_Frozen):
    """One step this actor was active: what it saw and what it did."""

    step: int = Field(ge=0)
    factors: dict[str, float]
    action_type: str
    action_target: str | None = None
    action_magnitude: float | None = None

    def render(self) -> str:
        """One compact line. Read by the model; never parsed back."""
        target = f" {self.action_target}" if self.action_target else ""
        magnitude = f" {self.action_magnitude:.2f}" if self.action_magnitude is not None else ""
        return f"s{self.step}: {self.action_type}{target}{magnitude}"


class AgentMemory(_Frozen):
    """One actor's private namespace within one run.

    Immutable: :meth:`remember` returns a new memory. That is what lets the
    kernel checkpoint an agent's memory alongside the world state without
    worrying that a later step mutated an object the checkpoint still points
    at.
    """

    actor_id: str
    entries: tuple[MemoryEntry, ...] = ()
    summary: str = ""
    summary_through_step: int = -1
    anchor: dict[str, float] = Field(default_factory=dict)
    """Factor values when the current summary window opened."""
    action_counts: dict[str, int] = Field(default_factory=dict)
    """Cumulative over the whole run, so the summary can report the run, not the window."""

    def remember(
        self,
        observation: Observation,
        *,
        action: Any,
        window: int,
    ) -> AgentMemory:
        """Record one active step, evicting anything past the window."""
        entry = MemoryEntry(
            step=observation.step,
            factors=dict(observation.factors),
            action_type=str(action.type),
            action_target=_target(action),
            action_magnitude=_magnitude(action),
        )
        entries = (*self.entries, entry)[-window:] if window > 0 else ()
        counts = dict(self.action_counts)
        counts[entry.action_type] = counts.get(entry.action_type, 0) + 1
        anchor = self.anchor or dict(observation.factors)
        return self.model_copy(
            update={"entries": entries, "action_counts": counts, "anchor": anchor}
        )

    def maybe_refresh(self, observation: Observation, *, interval: int) -> AgentMemory:
        """Refresh the rolling summary if ``interval`` steps have passed.

        Refreshed on the observation the actor is about to act on, so the
        summary an agent reads is never staler than the cap promises.
        """
        if interval <= 0:
            return self
        if observation.step < self.summary_through_step + interval:
            return self
        summary = self._compose(observation)
        return self.model_copy(
            update={
                "summary": summary,
                "summary_through_step": observation.step,
                "anchor": dict(observation.factors),
            }
        )

    def _compose(self, observation: Observation) -> str:
        """Build the digest. Deterministic, sorted, and bounded in length."""
        drift: list[str] = []
        for factor_id in sorted(observation.factors):
            start = self.anchor.get(factor_id)
            if start is None:
                continue
            delta = observation.factors[factor_id] - start
            if abs(delta) < 10.0**-OBSERVATION_DECIMALS:
                continue
            drift.append(f"{factor_id} {start:.2f}->{observation.factors[factor_id]:.2f}")
        acted = ", ".join(
            f"{kind}x{self.action_counts[kind]}" for kind in sorted(self.action_counts)
        )
        parts = [f"through s{observation.step}"]
        if drift:
            parts.append("moved: " + "; ".join(drift))
        if acted:
            parts.append("you: " + acted)
        allies = [
            f"{actor}{observation.trust[actor]:+.2f}"
            for actor in sorted(observation.trust)
            if abs(observation.trust[actor]) >= 0.1
        ]
        if allies:
            parts.append("trust: " + " ".join(allies))
        return " | ".join(parts)

    def render(self) -> str:
        """The memory block for the prompt: summary first, then recent steps."""
        lines: list[str] = []
        if self.summary:
            lines.append(self.summary)
        lines.extend(entry.render() for entry in self.entries)
        return "\n".join(lines) if lines else "(no prior activity)"


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
