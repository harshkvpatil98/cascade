"""Record -> replay round trip, and the zero-network guarantee (spec §8.3).

This is M0 acceptance criterion 2. The claim being tested is not "replay is
fast" or "replay usually avoids the network" -- it is that replay *cannot*
reach the network. Two independent instruments prove it:

1. a transport that raises on any request, so a call that was sent would fail
   the test rather than pass it silently;
2. an assertion that no SDK client was ever constructed, with the API key
   unset -- so there was nothing that *could* have sent one.

The second is the stronger claim and is the reason a replay run needs no
credentials at all.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cascade.config import Settings
from cascade.llm.client import LLMClient
from cascade.llm.types import CacheMiss, LLMError, LLMRequest
from tests.conftest import CallCounter, NetworkForbidden

MODEL = "claude-haiku-4-5-20251001"
CALL_COUNT = 25


def in_mode(settings: Settings, mode: str) -> Settings:
    return settings.model_copy(update={"llm": settings.llm.model_copy(update={"mode": mode})})


def request_for(index: int) -> LLMRequest:
    """A distinct request per unit, so 25 recordings means 25 distinct keys."""
    return LLMRequest(
        model=MODEL,
        system="You are actor A. Rules follow.",
        messages=[{"role": "user", "content": f"step {index}"}],
        tools=None,
        temperature=0.7,
        max_tokens=512,
        prompt_rev="r1",
    )


@pytest.fixture
def recorded(
    settings: Settings, recording_transport: tuple[httpx.Client, CallCounter]
) -> tuple[Settings, CallCounter]:
    """Record ``CALL_COUNT`` calls against a mock transport and return the state."""
    http_client, counter = recording_transport
    client = LLMClient(in_mode(settings, "record"), phase="bench", http_client=http_client)
    for index in range(CALL_COUNT):
        result = client.complete(request_for(index))
        assert not result.served_from_cache
    assert counter.count == CALL_COUNT
    return settings, counter


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_record_serves_every_call_and_persists_it(
    recorded: tuple[Settings, CallCounter],
) -> None:
    settings, counter = recorded
    assert counter.count == CALL_COUNT
    assert all(url.endswith("/v1/messages") for url in counter.urls)

    from cascade.llm.cache import CallCache

    assert CallCache(settings.cache_path()).count() == CALL_COUNT


def test_record_is_idempotent_on_a_repeated_call(
    settings: Settings, recording_transport: tuple[httpx.Client, CallCounter]
) -> None:
    """A second identical request in record mode is served from disk, not billed."""
    http_client, counter = recording_transport
    client = LLMClient(in_mode(settings, "record"), phase="bench", http_client=http_client)
    first = client.complete(request_for(0))
    second = client.complete(request_for(0))

    assert counter.count == 1
    assert not first.served_from_cache
    assert second.served_from_cache
    assert second.text == first.text
    assert client.meter.calls == 1
    assert client.meter.cached_calls == 1


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_makes_zero_network_calls(
    recorded: tuple[Settings, CallCounter], forbidding_transport: httpx.Client
) -> None:
    """M0 acceptance: the replay pass makes **zero** network calls."""
    settings, _ = recorded
    client = LLMClient(in_mode(settings, "replay"), phase="bench", http_client=forbidding_transport)
    for index in range(CALL_COUNT):
        result = client.complete(request_for(index))
        assert result.served_from_cache
        assert result.text == "ok"

    assert client.meter.cached_calls == CALL_COUNT
    assert client.meter.calls == 0
    assert client.meter.total_usd == 0


def test_replay_never_constructs_an_sdk_client(
    recorded: tuple[Settings, CallCounter],
) -> None:
    """Stronger than 'sent no request': there was nothing able to send one.

    Runs with the API key unset, which is also the operational claim -- a
    replay of the study needs no credentials.
    """
    settings, _ = recorded
    keyless = in_mode(settings, "replay").model_copy(update={"anthropic_api_key": None})
    client = LLMClient(keyless, phase="bench")

    for index in range(CALL_COUNT):
        client.complete(request_for(index))

    assert not client.constructed_sdk_client


def test_replay_reproduces_the_recorded_payload_exactly(
    recorded: tuple[Settings, CallCounter], forbidding_transport: httpx.Client
) -> None:
    """Replay is only useful if it returns what was recorded, field for field."""
    settings, _ = recorded
    record_client = LLMClient(in_mode(settings, "record"), phase="bench")
    replay_client = LLMClient(
        in_mode(settings, "replay"), phase="bench", http_client=forbidding_transport
    )
    for index in range(CALL_COUNT):
        from_disk = record_client.complete(request_for(index))
        replayed = replay_client.complete(request_for(index))
        assert replayed.text == from_disk.text
        assert replayed.model == from_disk.model
        assert replayed.stop_reason == from_disk.stop_reason
        assert replayed.usage == from_disk.usage


def test_replay_miss_raises_and_does_not_fall_back(
    recorded: tuple[Settings, CallCounter], forbidding_transport: httpx.Client
) -> None:
    """A miss must stop the run. Falling back would make it non-reproducible."""
    settings, _ = recorded
    client = LLMClient(in_mode(settings, "replay"), phase="bench", http_client=forbidding_transport)
    with pytest.raises(CacheMiss, match="never falls back to the network"):
        client.complete(request_for(CALL_COUNT + 1))
    assert not client.constructed_sdk_client


def test_the_forbidding_transport_would_actually_fail_the_test(
    settings: Settings, forbidding_transport: httpx.Client
) -> None:
    """Guard the instrument: prove the transport raises when a call is sent.

    Without this, a transport that silently returned would make every
    zero-network assertion above vacuous.

    The SDK wraps transport failures in its own ``APIConnectionError``, so the
    proof is that ``NetworkForbidden`` appears in the cause chain -- asserting
    on the SDK's wrapper type would couple this test to an SDK detail.
    """
    client = LLMClient(in_mode(settings, "record"), phase="bench", http_client=forbidding_transport)
    with pytest.raises(Exception) as caught:  # the SDK wraps it; see cause chain below
        client.complete(request_for(0))

    chain: list[type[BaseException]] = []
    error: BaseException | None = caught.value
    while error is not None:
        chain.append(type(error))
        error = error.__cause__
    assert NetworkForbidden in chain, f"expected a forbidden network call, got {chain}"


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


def test_live_bypasses_the_cache_entirely(
    recorded: tuple[Settings, CallCounter],
    recording_transport: tuple[httpx.Client, CallCounter],
) -> None:
    """`live` exists for the M3 latency bench, where a cache hit measures nothing.

    ``request_for(0)`` is already on disk from the ``recorded`` fixture, so a
    cache-consulting mode would serve it without touching the transport. The
    assertion is that the counter advances anyway.
    """
    settings, _ = recorded
    http_client, counter = recording_transport  # same instance the fixture recorded with
    before = counter.count
    client = LLMClient(in_mode(settings, "live"), phase="bench", http_client=http_client)

    result = client.complete(request_for(0))

    assert not result.served_from_cache
    assert counter.count == before + 1
    assert client.meter.calls == 1


def test_live_without_a_key_fails_loudly(settings: Settings) -> None:
    """Only replay runs without credentials; the others must say so."""
    keyless = in_mode(settings, "live").model_copy(update={"anthropic_api_key": None})
    client = LLMClient(keyless, phase="bench")
    with pytest.raises(LLMError, match="CASCADE_ANTHROPIC_API_KEY is not set"):
        client.complete(request_for(0))


# ---------------------------------------------------------------------------
# Deferred surface
# ---------------------------------------------------------------------------


def test_batch_submission_is_explicitly_deferred(settings: Settings) -> None:
    """Deferred to M6, and it raises rather than silently costing 2x.

    The meter and the price table already model the 50% discount, so landing
    it changes the submission path and nothing about the accounting.
    """
    client = LLMClient(in_mode(settings, "replay"), phase="bench")
    with pytest.raises(NotImplementedError, match="lands at M6"):
        client.complete(request_for(0), batch=True)


def test_usage_is_taken_from_the_provider_not_estimated(
    recorded: tuple[Settings, CallCounter],
) -> None:
    """Every dollar in the ledger comes from reported usage, never a guess."""
    settings, _ = recorded
    client = LLMClient(in_mode(settings, "replay"), phase="bench")
    result = client.complete(request_for(0))
    # The canned response body in conftest declares these counts.
    assert result.usage.input_tokens == 1900
    assert result.usage.output_tokens == 45


def test_missing_usage_block_is_an_error(settings: Settings) -> None:
    """A response we cannot price must not be booked at zero."""
    from cascade.llm.client import _usage_from_payload

    payload: dict[str, Any] = {"id": "msg_01TEST", "content": []}
    with pytest.raises(LLMError, match="carries no usage block"):
        _usage_from_payload(payload)
