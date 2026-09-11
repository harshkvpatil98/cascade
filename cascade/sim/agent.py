"""The DECIDE stage: one agent turn, through the one LLM call site (§7.2, §8.3).

Everything model-facing about a turn is here, and nothing else is. The kernel
hands this module an observation and gets back an action; whether that action
came from the network, from the record/replay cache or from a stand-in policy
is invisible to the step loop and visible in the event log.

The action cache and the LLM cache are **the same cache**, which is the point
of rounding observed values at the projection. A turn's request is a pure
function of (rules, persona, evidence, observation, memory), so two replicates
whose trajectories have not diverged produce identical request bytes and the
second is served from disk. §12.1's 91% hit rate is therefore a property of the
prompt's construction rather than a separate subsystem to build and keep
consistent.

Prompt caching is applied per (scenario, actor) and only when the prefix
actually clears the provider's floor. ADR-0001: below 4,096 tokens Anthropic
silently declines to cache while still charging the write premium, so marking a
short prefix is strictly worse than not marking it. The decision is measured,
recorded on the run, and never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cascade.aperture.projection import Observation
from cascade.config import Settings
from cascade.llm.client import estimate_tokens
from cascade.llm.types import LLMRequest
from cascade.sim.actions import Action, ActionSpace, Wait, admit
from cascade.sim.prompts import ACTION_TOOL, ACTION_TOOL_NAME, RULES, ActorBrief, persona_block
from cascade.sim.prompts import turn_message as render_turn

__all__ = ["Decision", "DecisionPolicy", "LLMAgents", "PreparedActor"]


class Decision(BaseModel):
    """One agent turn's outcome, with the accounting the event log needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    coercion: str | None = None
    cache_hit: bool = False
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: int | None = None


class DecisionPolicy(Protocol):
    """What the kernel needs from whatever is deciding.

    A Protocol rather than a base class so the LLM-backed agents and the
    deterministic stand-in in :mod:`cascade.sim.policies` are interchangeable
    at the kernel boundary without either knowing about the other.
    """

    def decide(
        self,
        *,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision: ...


@dataclass(frozen=True)
class PreparedActor:
    """One actor's cacheable prefix, built once per (scenario, actor)."""

    brief: ActorBrief
    persona: str
    prefix_tokens: int
    cacheable: bool


def prepare_actor(brief: ActorBrief, settings: Settings) -> PreparedActor:
    """Render an actor's persona block and decide whether to mark it cacheable.

    The measurement is over the whole static prefix -- tool schema, rules and
    persona -- because that is what the provider caches, not the last block
    alone.
    """
    persona = persona_block(brief, evidence_chars=settings.kernel.evidence_chars)
    tokens = estimate_tokens(RULES) + estimate_tokens(persona) + estimate_tokens(str(ACTION_TOOL))
    cacheable = settings.prompt_cache.enabled and tokens >= settings.prompt_cache.min_prefix_tokens
    return PreparedActor(brief=brief, persona=persona, prefix_tokens=tokens, cacheable=cacheable)


@dataclass
class LLMAgents:
    """Decides every actor's turn through :class:`~cascade.llm.client.LLMClient`.

    One instance per run; the prepared prefixes are shared across the 200
    replicates of a scenario, which is where the cache economics live.
    """

    settings: Settings
    client: Any
    prepared: dict[str, PreparedActor]
    calls: int = field(default=0)
    cache_hits: int = field(default=0)

    def decide(
        self,
        *,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision:
        """Ask the model for one action and admit it (spec §7.2, stage 5)."""
        actor = self.prepared.get(actor_id)
        if actor is None:
            raise KeyError(
                f"no prepared prompt for actor {actor_id!r}; the kernel prepares every "
                "actor before step 0, so this means the graph and the run disagree"
            )
        persona: dict[str, Any] = {"type": "text", "text": actor.persona}
        if actor.cacheable:
            # ADR-0007: stripped from the cache key, so this is a pure cost
            # change and does not invalidate a single recorded call.
            persona["cache_control"] = {
                "type": "ephemeral",
                "ttl": self.settings.prompt_cache.ttl,
            }
        request = LLMRequest(
            model=self.settings.models.agent,
            system=[{"type": "text", "text": RULES}, persona],
            messages=[
                {
                    "role": "user",
                    "content": render_turn(
                        observation,
                        memory=memory,
                        horizon=self.settings.kernel.steps,
                    ),
                }
            ],
            tools=[ACTION_TOOL],
            tool_choice={"type": "tool", "name": ACTION_TOOL_NAME},
            temperature=self.settings.models.temperature,
            max_tokens=self.settings.models.agent_max_tokens,
            prompt_rev=self.settings.llm.prompt_rev,
        )
        result = self.client.complete(request, trace_name="loom.decide")
        self.calls += 1
        if result.served_from_cache:
            self.cache_hits += 1

        payload = _tool_input(result)
        if payload is None:
            # Not an error the run should die of: one unusable answer costs one
            # actor one turn, and the run has 23 more steps of behaviour in it.
            return Decision(
                action=Wait(),
                coercion="no_tool_call",
                cache_hit=result.served_from_cache,
                tokens_in=result.usage.input_tokens + result.usage.cache_read_input_tokens,
                tokens_out=result.usage.output_tokens,
                latency_ms=int(result.latency_ms),
            )
        action, coercion = admit(payload, space)
        return Decision(
            action=action,
            coercion=coercion,
            cache_hit=result.served_from_cache,
            tokens_in=result.usage.input_tokens + result.usage.cache_read_input_tokens,
            tokens_out=result.usage.output_tokens,
            latency_ms=int(result.latency_ms),
        )

    @property
    def hit_rate(self) -> float:
        """Measured action-cache hit rate. The M6 criterion is read off this."""
        return self.cache_hits / self.calls if self.calls else 0.0


def _tool_input(result: Any) -> Any:
    """Pull the action payload out of a tool call, or None if there is none."""
    for block in result.tool_calls:
        if block.get("name") != ACTION_TOOL_NAME:
            continue
        payload = block.get("input")
        if isinstance(payload, dict):
            inner = payload.get("action")
            return inner if isinstance(inner, dict) else payload
    return None
