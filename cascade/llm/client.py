"""The single LLM call site (invariant 5, spec §8.3).

This is the only module in ``cascade/`` permitted to import the Anthropic SDK.
``tests/unit/test_invariants.py`` greps the tree and fails CI on a second
importer, because determinism, caching, cost metering and tracing all assume
there is exactly one door.

Three modes, and the difference between them is the whole point:

``record``
    A miss calls the API and persists ``(key, response, usage, latency)``.
``replay``
    A miss raises :class:`CacheMiss`. It never falls back to the network, and
    it never even constructs an SDK client -- so a replay run works with the
    API key unset, which is how the M0 acceptance test proves it.
``live``
    Bypasses the cache entirely. Used only by the M3 latency benchmark, where
    the thing being measured is the provider's own round trip.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cascade.config import Settings
from cascade.llm.cache import CallCache, cache_key
from cascade.llm.meter import CostMeter
from cascade.llm.tracing import Tracer, null_tracer
from cascade.llm.types import (
    CachedCall,
    CacheMiss,
    LLMError,
    LLMRequest,
    LLMResult,
    PromptTooShortToCache,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

__all__ = ["LLMClient", "assert_cacheable_prefix", "estimate_tokens"]

# Claude tokenises English prose at roughly 3.6 characters per token. The
# estimator is only used to gate the cacheable prefix (ADR-0001); it is
# deliberately *not* used for billing, which always uses the provider's own
# reported counts.
_CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text`` without a network round trip.

    Preserves the invariant that the prefix-length gate is checkable offline
    and in CI. This is an estimate and is never used for cost: every dollar in
    the ledger comes from the provider's reported ``usage``.
    """
    return int(len(text) / _CHARS_PER_TOKEN)


def assert_cacheable_prefix(
    prefix: str,
    settings: Settings,
    *,
    exact_tokens: int | None = None,
) -> int:
    """Fail loudly if a cacheable prefix sits below the provider's cache floor.

    Preserves the cost model in spec §12.1, which assumes the 1,900-token
    persona prefix is billed at the 10% cached-read rate. Anthropic does not
    error on a short prefix -- it silently declines to cache, and the only
    visible symptom is a ledger that is ~4.8x over. See ADR-0001.

    Returns the token count used for the decision.
    """
    tokens = exact_tokens if exact_tokens is not None else estimate_tokens(prefix)
    if not settings.prompt_cache.enabled:
        return tokens
    floor = settings.prompt_cache.min_prefix_tokens
    if tokens < floor and settings.prompt_cache.enforce_min_prefix:
        raise PromptTooShortToCache(measured_tokens=tokens, minimum_tokens=floor)
    return tokens


class LLMClient:
    """Mediates every model call: cache, meter, tracer, then maybe the network."""

    def __init__(
        self,
        settings: Settings,
        *,
        phase: str,
        meter: CostMeter | None = None,
        cache: CallCache | None = None,
        tracer: Tracer | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._phase = phase
        self._mode = settings.llm.mode
        self.cache = cache if cache is not None else CallCache(settings.cache_path())
        self.meter = meter if meter is not None else CostMeter(settings, phase)
        self._tracer = tracer if tracer is not None else null_tracer()
        self._http_client = http_client
        self._sdk: Any | None = None

    # -- properties ---------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def constructed_sdk_client(self) -> bool:
        """Whether an SDK client has ever been built in this process.

        The M0 acceptance criterion is that replay makes zero network calls.
        A transport that raises proves no call was *sent*; this proves none
        could have been, because no client exists to send one.
        """
        return self._sdk is not None

    # -- the one door -------------------------------------------------------

    def complete(
        self,
        request: LLMRequest,
        *,
        batch: bool = False,
        trace_name: str = "llm.complete",
    ) -> LLMResult:
        """Execute ``request`` under the configured mode and book its cost.

        Preserves invariant 5 (one call site) and the record/replay contract:
        in ``replay`` this function is total with respect to the recorded
        corpus -- it either returns a recorded response or raises, and there
        is no third outcome that reaches the network.
        """
        if batch:
            # The meter and the price table already model the 50% discount, so
            # landing this at M6 changes the submission path and nothing about
            # the accounting.
            raise NotImplementedError(
                "batch submission lands at M6; the cost model already prices it"
            )

        key = cache_key(request)

        if self._mode == "live":
            return self._call_api(request, key=key, batch=batch, trace_name=trace_name, store=False)

        recorded = self.cache.get(key)
        if recorded is not None:
            return self._from_recording(recorded, trace_name=trace_name)

        if self._mode == "replay":
            raise CacheMiss(
                f"no recorded response for key {key} (model={request.model!r}, "
                f"prompt_rev={request.prompt_rev!r}) in {self.cache.root}. "
                "Replay never falls back to the network -- re-record with "
                "CASCADE_LLM__MODE=record if this call is new."
            )

        return self._call_api(request, key=key, batch=batch, trace_name=trace_name, store=True)

    # -- internals ----------------------------------------------------------

    def _from_recording(self, recorded: CachedCall, *, trace_name: str) -> LLMResult:
        """Rebuild a result from disk and book it as a zero-cost generation.

        Cache hits are traced rather than skipped (spec §12.2): hit rate and
        spend have to appear on the same dashboard, or the biggest cost lever
        in the study is invisible.
        """
        result = _result_from_payload(
            recorded.raw_response,
            usage=recorded.usage,
            latency_ms=recorded.latency_ms,
            served_from_cache=True,
        )
        self.meter.record_cache_hit(model=result.model, usage=result.usage)
        self._tracer.generation(
            name=trace_name,
            model=result.model,
            usage=result.usage,
            cost_usd=None,
            cached=True,
            latency_ms=recorded.latency_ms,
        )
        return result

    def _client(self) -> Any:
        """Construct the SDK client lazily, exactly once.

        Lazy so that ``replay`` never builds one -- see
        :attr:`constructed_sdk_client`.
        """
        if self._sdk is not None:
            return self._sdk
        import anthropic  # the single SDK import in the codebase (invariant 5)

        api_key = self._settings.anthropic_api_key
        if api_key is None:
            raise LLMError(
                f"CASCADE_ANTHROPIC_API_KEY is not set but llm.mode={self._mode!r} needs it. "
                "Only replay runs without a key."
            )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "max_retries": self._settings.llm.max_retries,
            "timeout": self._settings.llm.timeout_s,
        }
        if self._http_client is not None:
            kwargs["http_client"] = self._http_client
        self._sdk = anthropic.Anthropic(**kwargs)
        return self._sdk

    def _call_api(
        self,
        request: LLMRequest,
        *,
        key: str,
        batch: bool,
        trace_name: str,
        store: bool,
    ) -> LLMResult:
        """Send one request, price it, and persist it when recording."""
        client = self._client()
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.system is not None:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = request.tools

        started = time.perf_counter()
        message = client.messages.create(**payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw: dict[str, Any] = message.model_dump(mode="json")
        usage = _usage_from_payload(raw)
        result = _result_from_payload(
            raw, usage=usage, latency_ms=latency_ms, served_from_cache=False
        )

        cost = self.meter.record(model=result.model, usage=usage, batch=batch)
        self._tracer.generation(
            name=trace_name,
            model=result.model,
            usage=usage,
            cost_usd=cost,
            cached=False,
            latency_ms=latency_ms,
        )

        if store:
            self.cache.put(
                CachedCall(
                    key=key,
                    request_digest=request.cache_domain(),
                    raw_response=raw,
                    usage=usage,
                    latency_ms=latency_ms,
                    recorded_at=datetime.now(UTC).isoformat(),
                )
            )
        return result


def _usage_from_payload(raw: dict[str, Any]) -> Usage:
    """Extract token counts from a Messages response body.

    Missing counts are an error, not a zero: a silently zero-token call would
    under-report spend, and the ledger reconciliation at M8 (±2%) is the only
    thing that would notice.
    """
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise LLMError(f"response carries no usage block; cannot price it: {raw.get('id')!r}")
    return Usage(
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
    )


def _result_from_payload(
    raw: dict[str, Any],
    *,
    usage: Usage,
    latency_ms: float,
    served_from_cache: bool,
) -> LLMResult:
    """Project a stored response body onto the boundary type.

    Reads the persisted body directly rather than through the SDK so that
    replay has no dependency on the SDK being installed or configured.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in raw.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            tool_calls.append(block)
    return LLMResult(
        text="".join(text_parts),
        model=str(raw.get("model", "")),
        stop_reason=raw.get("stop_reason"),
        usage=usage,
        latency_ms=latency_ms,
        served_from_cache=served_from_cache,
        tool_calls=tool_calls,
    )
