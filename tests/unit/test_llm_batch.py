"""Batch submission through the one call site (spec §12.2, M6).

The Batch API is where half the study's model spend goes away, and it is also
where a mistake is most expensive: results come back in arbitrary order, so a
harness that matched them positionally would attribute answers to the wrong
decisions and produce a study that looks fine. These tests drive the real SDK
against a mock transport, so the matching, the polling and the parsing are the
provider's own code paths.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from cascade.config import Settings
from cascade.llm.cache import CallCache, cache_key
from cascade.llm.client import LLMClient
from cascade.llm.types import BatchFailed, BatchItem, CacheMiss, LLMRequest
from tests.conftest import message_payload


def request_for(index: int, *, model: str = "claude-haiku-4-5-20251001") -> LLMRequest:
    return LLMRequest(
        model=model,
        system="rules",
        messages=[{"role": "user", "content": f"turn {index}"}],
        temperature=0.7,
        max_tokens=512,
        prompt_rev="r2",
    )


class BatchTransport:
    """Serves create / retrieve / results, recording what was submitted."""

    def __init__(
        self,
        *,
        ends_after: int = 0,
        outcomes: dict[str, str] | None = None,
    ) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.polls = 0
        self.message_calls = 0
        self._ends_after = ends_after
        self._outcomes = outcomes or {}
        self._custom_ids: list[str] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if request.method == "POST" and url.endswith("/v1/messages/batches"):
                body = json.loads(request.content)
                self.submitted.append(body)
                self._custom_ids = [item["custom_id"] for item in body["requests"]]
                return httpx.Response(
                    200,
                    json={
                        "id": "msgbatch_01",
                        "type": "message_batch",
                        "processing_status": "in_progress",
                    },
                )
            if request.method == "GET" and url.endswith("/results"):
                lines = []
                # Deliberately reversed: the API makes no ordering promise, and
                # a harness that matched positionally would pass this test only
                # by accident if the order were the submitted one.
                for custom_id in reversed(self._custom_ids):
                    kind = self._outcomes.get(custom_id, "succeeded")
                    if kind == "succeeded":
                        result = {
                            "type": "succeeded",
                            "message": message_payload(
                                model="claude-haiku-4-5-20251001", text=f"answer:{custom_id}"
                            ),
                        }
                    else:
                        result = {"type": kind, "error": {"type": kind}}
                    lines.append(json.dumps({"custom_id": custom_id, "result": result}))
                return httpx.Response(
                    200,
                    text="\n".join(lines),
                    headers={"content-type": "application/x-jsonl"},
                )
            if request.method == "GET" and "/v1/messages/batches/" in url:
                self.polls += 1
                status = "ended" if self.polls > self._ends_after else "in_progress"
                body: dict[str, Any] = {
                    "id": "msgbatch_01",
                    "type": "message_batch",
                    "processing_status": status,
                }
                if status == "ended":
                    # The SDK fetches `results_url` off the batch before
                    # streaming results; a mock without it fails in the SDK
                    # rather than in our code, which is the kind of shape
                    # detail only a real-SDK test catches.
                    body["results_url"] = (
                        "https://api.anthropic.com/v1/messages/batches/msgbatch_01/results"
                    )
                return httpx.Response(200, json=body)
            if request.method == "POST" and url.endswith("/v1/messages"):
                self.message_calls += 1
                body = json.loads(request.content)
                return httpx.Response(200, json=message_payload(model=body["model"]))
            raise AssertionError(f"unexpected request: {request.method} {url}")

        return httpx.Client(transport=httpx.MockTransport(handler))


def recording_client(
    settings: Settings, tmp_path: Path, transport: BatchTransport, *, mode: str = "record"
) -> LLMClient:
    client = LLMClient(
        settings.model_copy(update={"llm": settings.llm.model_copy(update={"mode": mode})}),
        phase="simulate",
        cache=CallCache(tmp_path / "llm"),
        http_client=transport.client(),
    )
    client._sleep = lambda _seconds: None  # type: ignore[method-assign] # drive the poll loop instantly
    return client


# ---------------------------------------------------------------------------


def test_a_wave_is_submitted_as_one_batch(settings: Settings, tmp_path: Path) -> None:
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport)
    items = [BatchItem(custom_id=f"t{i}", request=request_for(i)) for i in range(25)]

    results = client.complete_batch(items)

    assert len(transport.submitted) == 1
    assert len(transport.submitted[0]["requests"]) == 25
    assert sorted(results) == sorted(item.custom_id for item in items)
    assert transport.message_calls == 0, "a batch must not fall back to single calls"


def test_results_are_matched_by_custom_id_not_by_position(
    settings: Settings, tmp_path: Path
) -> None:
    """The API returns results in any order; the transport returns them reversed."""
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport)
    items = [BatchItem(custom_id=f"t{i}", request=request_for(i)) for i in range(5)]

    results = client.complete_batch(items)

    for item in items:
        key = cache_key(item.request)
        assert results[item.custom_id].text.endswith(
            key[:56]
        ), "a result was attributed to the wrong request"


def test_cache_hits_are_served_locally_and_never_submitted(
    settings: Settings, tmp_path: Path
) -> None:
    """This is the 88% criterion doing its job: only the misses cost anything."""
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport)
    items = [BatchItem(custom_id=f"t{i}", request=request_for(i)) for i in range(4)]
    client.complete_batch(items)
    assert len(transport.submitted[0]["requests"]) == 4

    again = client.complete_batch(items)
    assert len(transport.submitted) == 1, "a fully cached wave must submit nothing"
    assert all(result.served_from_cache for result in again.values())


def test_duplicate_requests_in_one_wave_are_submitted_once(
    settings: Settings, tmp_path: Path
) -> None:
    """Replicates that have not diverged produce identical bytes -- pay once.

    Paying twice for them would spend the budget on exactly the duplication the
    cache exists to exploit.
    """
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport)
    shared = request_for(0)
    items = [BatchItem(custom_id=f"run{i}", request=shared) for i in range(10)]

    results = client.complete_batch(items)

    assert len(transport.submitted[0]["requests"]) == 1
    assert len(results) == 10
    assert len({result.text for result in results.values()}) == 1


def test_the_poll_loop_waits_for_the_batch_to_end(settings: Settings, tmp_path: Path) -> None:
    transport = BatchTransport(ends_after=3)
    client = recording_client(settings, tmp_path, transport)
    client.complete_batch([BatchItem(custom_id="t0", request=request_for(0))])
    # Three polls report "in_progress", the fourth reports "ended", and the SDK
    # retrieves once more to read `results_url` off the finished batch.
    assert transport.polls == 5


def test_a_failed_item_is_reported_with_its_reason(settings: Settings, tmp_path: Path) -> None:
    """`expired` and `invalid_request` need different responses from the operator."""
    items = [BatchItem(custom_id=f"t{i}", request=request_for(i)) for i in range(3)]
    custom_ids = [f"k{cache_key(item.request)[:56]}" for item in items]
    transport = BatchTransport(outcomes={custom_ids[1]: "expired"})
    client = recording_client(settings, tmp_path, transport)

    with pytest.raises(BatchFailed) as caught:
        client.complete_batch(items)
    assert "expired" in str(caught.value)


def test_replay_never_batches_a_miss(settings: Settings, tmp_path: Path) -> None:
    """Replay reaches the network by no door, including this one."""
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport, mode="replay")
    with pytest.raises(CacheMiss, match="Replay never falls back"):
        client.complete_batch([BatchItem(custom_id="t0", request=request_for(0))])
    assert transport.submitted == []


def test_batched_calls_are_priced_at_the_batch_rate(settings: Settings, tmp_path: Path) -> None:
    """§12.1 applies a 0.5 multiplier to the whole backtest; this is where."""
    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport)
    client.complete_batch([BatchItem(custom_id="t0", request=request_for(0))])
    batched = client.meter.total_usd

    other = recording_client(settings, tmp_path / "b", BatchTransport())
    other.complete(request_for(0))
    single = other.meter.total_usd

    assert batched == single / 2


def test_one_request_batching_is_refused(settings: Settings, tmp_path: Path) -> None:
    from cascade.llm.types import LLMError

    client = recording_client(settings, tmp_path, BatchTransport())
    with pytest.raises(LLMError, match="complete_batch"):
        client.complete(request_for(0), batch=True)


def test_live_mode_refuses_the_batch_door(settings: Settings, tmp_path: Path) -> None:
    """Live bypasses the cache, so a batched answer would be bought twice.

    The wavefront batches a step and then decides from the cache; in live mode
    `complete` ignores the cache, so every decision would re-request what the
    batch already paid for.
    """
    from cascade.llm.types import LLMError

    transport = BatchTransport()
    client = recording_client(settings, tmp_path, transport, mode="live")
    with pytest.raises(LLMError, match="not available in live mode"):
        client.complete_batch([BatchItem(custom_id="t0", request=request_for(0))])
    assert transport.submitted == []


def test_a_paid_response_is_cached_before_the_meter_can_abort(
    settings: Settings, tmp_path: Path
) -> None:
    """A ceiling breach must not throw away responses the study has already paid for.

    The meter raises when the phase ceiling breaks; if that happened before the
    response reached the cache, the resumed phase would buy it again.
    """
    from cascade.llm.types import BudgetExceeded

    tight = settings.model_copy(
        update={
            "budget": settings.budget.model_copy(
                update={
                    "phase_ceiling_usd": {
                        **settings.budget.phase_ceiling_usd,
                        "simulate": Decimal("0.0000001"),
                    }
                }
            )
        }
    )
    transport = BatchTransport()
    client = recording_client(tight, tmp_path, transport)
    items = [BatchItem(custom_id=f"t{i}", request=request_for(i)) for i in range(3)]

    with pytest.raises(BudgetExceeded):
        client.complete_batch(items)

    # Whichever result the API returned first was paid for and is on disk, so
    # the resumed phase serves it from the cache instead of buying it again.
    # (Which one that is depends on the provider's arbitrary result order --
    # asserting a particular item would be asserting the mock's ordering.)
    cached = [item for item in items if client.cache.get(cache_key(item.request))]
    assert cached, "a response was paid for and then discarded by the abort"
