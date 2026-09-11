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
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cascade.canonical import canonical_json
from cascade.config import Settings
from cascade.llm.cache import CallCache, cache_key
from cascade.llm.meter import CostMeter
from cascade.llm.tracing import Tracer, null_tracer
from cascade.llm.types import (
    BatchFailed,
    BatchItem,
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

__all__ = [
    "BATCH_MAX_REQUESTS",
    "LLMClient",
    "assert_cacheable_prefix",
    "estimate_tokens",
]

# Provider limits on one batch submission: 100,000 requests or 256 MB. A wave
# of the M6 fan-out is 36,000 runs x ~4.86 active actors, so a step can exceed
# the request limit and has to be chunked; the byte limit is checked as the
# payload is assembled rather than assumed.
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 256 * 1024 * 1024

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
            # One request is not a batch. The Batches API is asynchronous with
            # an SLA measured in hours, so a single-request submission would
            # trade minutes of latency for half a cent; `complete_batch` is the
            # entry point that makes the discount worth having.
            raise LLMError(
                "complete(batch=True) submits one request and waits hours for it. "
                "Use complete_batch() with a whole wave of requests instead."
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

    # -- the batch door (spec §12.2) ----------------------------------------

    def complete_batch(
        self,
        items: Sequence[BatchItem],
        *,
        trace_name: str = "llm.batch",
        poll_interval_s: float = 5.0,
        timeout_s: float = 86_400.0,
    ) -> dict[str, LLMResult]:
        """Serve a whole wave of requests, batching whatever the cache misses.

        Returns ``custom_id -> LLMResult`` for every item. Cache hits are
        served locally and never reach the network, which is what makes the
        88% hit rate the M6 criterion measures a *saving* rather than a
        statistic -- only the misses are submitted, and they are submitted at
        the 50% batch rate the §12.1 cost model assumes.

        Identical requests inside one wave are submitted **once**. Two
        replicates whose trajectories have not diverged produce byte-identical
        requests, and paying twice for them would spend the study's budget on
        the very duplication the cache exists to exploit.

        In ``replay`` a miss raises :class:`CacheMiss`, exactly as
        :meth:`complete` does: replay never reaches the network by any door.
        """
        if self._mode == "live":
            # `complete` bypasses the cache in live mode, so a batched answer
            # would be re-requested one call at a time at DECIDE and the wave
            # would be paid for twice. Live is for the latency benchmark, where
            # the provider's own round trip is the thing being measured.
            raise LLMError(
                "complete_batch is not available in live mode: live bypasses the cache, "
                "so every batched answer would be requested again at decide time. "
                "Use record (to pay once and keep the responses) or replay."
            )

        results: dict[str, LLMResult] = {}
        pending: dict[str, list[str]] = {}
        by_key: dict[str, LLMRequest] = {}

        for item in sorted(items, key=lambda entry: entry.custom_id):
            key = cache_key(item.request)
            recorded = self.cache.get(key)
            if recorded is not None:
                results[item.custom_id] = self._from_recording(recorded, trace_name=trace_name)
                continue
            if self._mode == "replay":
                raise CacheMiss(
                    f"no recorded response for key {key} (custom_id={item.custom_id!r}, "
                    f"model={item.request.model!r}, prompt_rev={item.request.prompt_rev!r}) "
                    f"in {self.cache.root}. Replay never falls back to the network -- "
                    "re-record with CASCADE_LLM__MODE=record if this call is new."
                )
            pending.setdefault(key, []).append(item.custom_id)
            by_key[key] = item.request

        if not pending:
            return results

        for chunk in _chunked(sorted(pending), BATCH_MAX_REQUESTS):
            served = self._run_batch(
                {key: by_key[key] for key in chunk},
                trace_name=trace_name,
                poll_interval_s=poll_interval_s,
                timeout_s=timeout_s,
            )
            for key, result in sorted(served.items()):
                for custom_id in pending[key]:
                    results[custom_id] = result

        missing = sorted(set(_flatten(pending)) - set(results))
        if missing:
            raise BatchFailed({custom_id: "no result returned" for custom_id in missing})
        return results

    def _run_batch(
        self,
        requests: Mapping[str, LLMRequest],
        *,
        trace_name: str,
        poll_interval_s: float,
        timeout_s: float,
    ) -> dict[str, LLMResult]:
        """Submit one batch keyed by cache key, wait for it, and price the results."""
        client = self._client()
        payload = [
            {"custom_id": _batch_id(key), "params": _batch_params(requests[key])}
            for key in sorted(requests)
        ]
        size = len(canonical_json(payload).encode("utf-8"))
        if size > BATCH_MAX_BYTES:
            raise LLMError(
                f"batch payload is {size} bytes, over the provider's "
                f"{BATCH_MAX_BYTES}-byte limit; submit fewer requests per wave"
            )

        started = time.perf_counter()
        batch = client.messages.batches.create(requests=payload)
        batch_id = str(getattr(batch, "id", ""))
        status = str(getattr(batch, "processing_status", ""))
        while status != "ended":
            if time.perf_counter() - started > timeout_s:
                raise LLMError(
                    f"batch {batch_id} still {status!r} after {timeout_s:.0f}s; "
                    "the provider's own SLA is 24h -- raise the timeout or cancel it"
                )
            self._sleep(poll_interval_s)
            batch = client.messages.batches.retrieve(batch_id)
            status = str(getattr(batch, "processing_status", ""))

        out: dict[str, LLMResult] = {}
        failures: dict[str, str] = {}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        by_batch_id = {_batch_id(key): key for key in sorted(requests)}

        for entry in client.messages.batches.results(batch_id):
            key = by_batch_id.get(str(getattr(entry, "custom_id", "")))
            if key is None:
                continue
            outcome = getattr(entry, "result", None)
            kind = str(getattr(outcome, "type", "")) if outcome is not None else ""
            if kind != "succeeded":
                failures[key] = kind or "unknown"
                continue
            raw = _message_payload(outcome)
            usage = _usage_from_payload(raw)
            result = _result_from_payload(
                raw, usage=usage, latency_ms=elapsed_ms, served_from_cache=False
            )
            # Persist before metering. The meter raises when the phase ceiling
            # breaks, and a response that has already been paid for must not be
            # discarded by the abort -- the resumed phase would buy it again.
            if self._mode == "record":
                self.cache.put(
                    CachedCall(
                        key=key,
                        request_digest=requests[key].cache_domain(),
                        raw_response=raw,
                        usage=usage,
                        latency_ms=elapsed_ms,
                        recorded_at=datetime.now(UTC).isoformat(),
                    )
                )
            cost = self.meter.record(model=result.model, usage=usage, batch=True)
            self._tracer.generation(
                name=trace_name,
                model=result.model,
                usage=usage,
                cost_usd=cost,
                cached=False,
                latency_ms=elapsed_ms,
            )
            out[key] = result

        if failures:
            raise BatchFailed(failures)
        return out

    def _sleep(self, seconds: float) -> None:
        """Wait between polls. Isolated so a test can drive the loop instantly."""
        time.sleep(seconds)

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
        # An *empty* variable is not an absent one. `CASCADE_ANTHROPIC_API_KEY=`
        # in a .env file parses to SecretStr("") rather than None, so a bare
        # `is None` check passes it through to the SDK, which rejects it with
        # `TypeError: Could not resolve authentication method` -- an error that
        # names neither the variable nor this project. Measured on a checkout
        # whose .env carried the key with no value.
        if api_key is None or not api_key.get_secret_value().strip():
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
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

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


def _batch_id(key: str) -> str:
    """The provider-facing id for a request, derived from its cache key.

    Deriving rather than counting means the id is stable across resubmissions
    and identical across processes, so a resumed wave that re-submits the same
    misses lines its results up with the same requests.
    """
    return f"k{key[:56]}"


def _batch_params(request: LLMRequest) -> dict[str, Any]:
    """Project a request onto the Batches API's per-item ``params`` object."""
    params: dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.system is not None:
        params["system"] = request.system
    if request.tools:
        params["tools"] = request.tools
    if request.tool_choice is not None:
        params["tool_choice"] = request.tool_choice
    return params


def _message_payload(outcome: Any) -> dict[str, Any]:
    """Extract the Messages body from a succeeded batch result entry."""
    message = getattr(outcome, "message", None)
    if message is None:
        raise LLMError("batch result reported success with no message body")
    if isinstance(message, dict):
        return dict(message)
    dumped = message.model_dump(mode="json")
    return dict(dumped)


def _chunked(keys: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(keys), size):
        yield list(keys[start : start + size])


def _flatten(pending: Mapping[str, list[str]]) -> list[str]:
    return [custom_id for key in sorted(pending) for custom_id in pending[key]]


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
