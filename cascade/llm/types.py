"""Boundary types for the LLM subsystem, and the failure modes that get their
own exit codes.

These are Pydantic models rather than dicts because they cross a subsystem
boundary (spec §13): the cache persists them, the meter prices them and the
tracer reports them, and a silently renamed key would be discovered at M8 as a
replay divergence rather than here as a validation error.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BudgetExceeded",
    "CacheMiss",
    "CachedCall",
    "LLMError",
    "LLMRequest",
    "LLMResult",
    "PromptTooShortToCache",
    "Usage",
]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base for every failure the LLM subsystem raises deliberately."""


class CacheMiss(LLMError):
    """Replay mode was asked for a call that was never recorded.

    Preserves the invariant that replay never reaches the network: the only
    honest response to a miss is to stop. Falling back to a live call would
    make the run non-reproducible while still appearing to succeed, which is
    the failure this whole mechanism exists to prevent. Exit code 4.
    """


class BudgetExceeded(LLMError):
    """A phase spent past its configured ceiling.

    Carries the accounting so the operator does not have to reconstruct it
    from logs. Exit code 2.
    """

    def __init__(self, phase: str, spent: Decimal, ceiling: Decimal, checkpoint: str) -> None:
        self.phase = phase
        self.spent = spent
        self.ceiling = ceiling
        self.checkpoint = checkpoint
        super().__init__(
            f"phase {phase!r} spent ${spent:.6f} against a ${ceiling:.6f} ceiling; "
            f"resumable checkpoint written to {checkpoint}"
        )


class PromptTooShortToCache(LLMError):
    """A prefix marked for caching sits below the provider's cache floor.

    Anthropic does not error on a short cacheable prefix -- it silently
    declines to cache it and bills every token at list price. The §12.1 cost
    model assumes a 10% cached-read rate on 1,900 prefix tokens, so a silent
    non-cache is a ~4.8x input-cost underestimate that shows up only in the
    final ledger. See ADR-0001. Exit code 3.
    """

    def __init__(self, measured_tokens: int, minimum_tokens: int) -> None:
        self.measured_tokens = measured_tokens
        self.minimum_tokens = minimum_tokens
        super().__init__(
            f"cacheable prefix is {measured_tokens} tokens but the model's cache floor is "
            f"{minimum_tokens}; the provider will silently not cache it and every call will "
            "be billed at list price. Pad the prefix or set "
            "prompt_cache.enforce_min_prefix=false and accept the cost. See ADR-0001."
        )


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Usage(_Frozen):
    """Token counts for one call, in the provider's own vocabulary.

    Kept in the provider's four-field shape rather than collapsed to
    ``input``/``output`` because cached reads and cache writes are billed at
    different multipliers; collapsing them loses the information the meter
    needs and the loss is invisible until the ledger fails to reconcile.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across calls, field by field."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    @property
    def billable_input_tokens(self) -> int:
        """Uncached input tokens only -- what the base input price applies to."""
        return self.input_tokens


class LLMRequest(_Frozen):
    """Everything that determines a response, and nothing that does not.

    The field set here *is* the cache key domain (spec §8.3). Adding a field
    invalidates every recorded call, so anything that does not change the
    model's output -- request ids, timestamps, retry counts, ``cache_control``
    markers (ADR-0007) -- must stay out.
    """

    model: str
    system: str | list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float
    max_tokens: int
    prompt_rev: str

    def cache_domain(self) -> dict[str, Any]:
        """Return exactly the fields the spec §8.3 key is computed over.

        ``max_tokens`` is included even though the spec's listing omits it: it
        bounds the response, so two requests differing only in ``max_tokens``
        can legitimately produce different completions, and sharing a key
        between them would serve a truncated response as if it were whole.
        """
        return {
            "model": self.model,
            "system": _strip_cache_control(self.system),
            "messages": _strip_cache_control(self.messages),
            "tools": _strip_cache_control(self.tools),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_rev": self.prompt_rev,
        }


class LLMResult(_Frozen):
    """One completed call, whether it came from the network or the cache."""

    text: str
    model: str
    stop_reason: str | None
    usage: Usage
    latency_ms: float
    served_from_cache: bool
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class CachedCall(_Frozen):
    """The on-disk record of one call.

    ``raw_response`` is the provider's own JSON body, not a projection of it.
    Replay reconstructs the SDK object from this, so a parser change is caught
    by the existing corpus of recordings rather than silently reinterpreted.
    """

    key: str
    request_digest: dict[str, Any]
    raw_response: dict[str, Any]
    usage: Usage
    latency_ms: float
    recorded_at: str
    cache_format: Literal[1] = 1


def _strip_cache_control(value: Any) -> Any:
    """Remove ``cache_control`` markers from a request fragment.

    Preserves the invariant that prompt-cache tuning is a pure cost change
    (ADR-0007): marking a prefix cacheable does not alter the tokens the model
    sees, so it must not alter the key. Without this, every cache-boundary
    experiment would force a paid re-record of the entire corpus.
    """
    if isinstance(value, dict):
        return {
            key: _strip_cache_control(sub)
            for key, sub in sorted(value.items())
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value
