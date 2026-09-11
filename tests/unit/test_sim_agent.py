"""One agent turn, through the one LLM call site (spec §7.2 stage 5, §8.3).

The mock transport is the real SDK's transport, so the response is parsed by
the code that will parse the provider's, not by a hand-built double. What these
pin down is the request shape the cost model depends on -- two system blocks,
tool-enforced output, a cacheable prefix marked only when it is actually
cacheable -- and that an identical turn is served from the cache, which is the
whole of the "action cache" in §12.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from cascade.aperture.projection import Observation
from cascade.config import Settings
from cascade.llm.cache import CallCache
from cascade.llm.client import LLMClient
from cascade.sim.actions import ActionSpace, Commit, Wait
from cascade.sim.agent import LLMAgents, prepare_actor
from cascade.sim.prompts import ACTION_TOOL_NAME, RULES, brief_from
from tests.conftest import make_actor

SPACE = ActionSpace(actor_id="actor_0", levers=("factor_0",), counterparties=("actor_1", "actor_2"))


def observation() -> Observation:
    return Observation(
        actor_id="actor_0",
        step=3,
        factors={"factor_0": 0.42, "factor_1": 0.17},
        banded=("factor_1",),
        unobserved=("factor_2",),
        budget=0.55,
        pooled=0.0,
        trust={"actor_1": 0.25},
    )


def brief(evidence_chunks: int = 6, chunk_chars: int = 1200) -> Any:
    return brief_from(
        make_actor(0),
        levers={"factor_0": 0.4},
        counterparties=("actor_1", "actor_2"),
        horizon=24,
        question_context="Will the coalition hold through the vote?",
        evidence=tuple(
            (f"2019-0{index + 1}-01T00:00:00+00:00", "govpr", "e" * chunk_chars)
            for index in range(evidence_chunks)
        ),
    )


def tool_response(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_01TEST",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": ACTION_TOOL_NAME,
                "input": {"action": payload},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 4200,
            "output_tokens": 45,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


class Transport:
    """Serves one canned tool call and records every request body."""

    def __init__(self, payload: dict[str, Any] | None = None, *, blocks: Any = None) -> None:
        self.payload = payload or {
            "type": "COMMIT",
            "target_factor": "factor_0",
            "magnitude": 0.4,
            "resource_spend": 0.2,
            "rationale": "hold the line",
        }
        self.blocks = blocks
        self.requests: list[dict[str, Any]] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            response = tool_response(body["model"], self.payload)
            if self.blocks is not None:
                response["content"] = self.blocks
            return httpx.Response(200, json=response, headers={"content-type": "application/json"})

        return httpx.Client(transport=httpx.MockTransport(handler))


def recording(settings: Settings, tmp_path: Path, transport: Transport) -> LLMAgents:
    client = LLMClient(
        settings.model_copy(update={"llm": settings.llm.model_copy(update={"mode": "record"})}),
        phase="simulate",
        cache=CallCache(tmp_path / "llm"),
        http_client=transport.client(),
    )
    return LLMAgents(
        settings=settings,
        client=client,
        prepared={("scenario-1", "actor_0"): prepare_actor(brief(), settings)},
    )


def test_the_prefix_is_marked_cacheable_only_above_the_floor(settings: Settings) -> None:
    """ADR-0001: below the floor the provider silently declines and still charges.

    Marking a short prefix is strictly worse than not marking it, so the
    decision is measured per actor rather than assumed for all of them.
    """
    rich = prepare_actor(brief(evidence_chunks=6, chunk_chars=1200), settings)
    thin = prepare_actor(brief(evidence_chunks=0), settings)

    assert rich.prefix_tokens >= settings.prompt_cache.min_prefix_tokens
    assert rich.cacheable
    assert thin.prefix_tokens < settings.prompt_cache.min_prefix_tokens
    assert not thin.cacheable


def test_evidence_is_what_carries_the_prefix_over_the_floor(settings: Settings) -> None:
    """Measured, because the §12.1 cost model turns on it (ADR-0019).

    Rules plus tool schema plus persona sit well under 4,096 tokens on their
    own; the per-actor evidence block is what makes the prefix cacheable at
    all, and it is static for the whole run because `as_of` is.
    """
    from cascade.llm.client import estimate_tokens

    assert estimate_tokens(RULES) < settings.prompt_cache.min_prefix_tokens
    assert prepare_actor(brief(), settings).cacheable


def test_the_request_is_tool_enforced_with_two_system_blocks(
    settings: Settings, tmp_path: Path
) -> None:
    transport = Transport()
    agents = recording(settings, tmp_path, transport)
    agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )

    sent = transport.requests[0]
    assert sent["model"] == settings.models.agent
    assert sent["temperature"] == settings.models.temperature
    assert sent["max_tokens"] == settings.models.agent_max_tokens
    assert sent["tool_choice"] == {"type": "tool", "name": ACTION_TOOL_NAME}
    assert [block["type"] for block in sent["system"]] == ["text", "text"]
    assert sent["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "Step 3 of 24" in sent["messages"][0]["content"]


def test_a_tool_call_becomes_an_action(settings: Settings, tmp_path: Path) -> None:
    agents = recording(settings, tmp_path, Transport())
    decision = agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )
    assert isinstance(decision.action, Commit)
    assert decision.action.target_factor == "factor_0"
    assert decision.coercion is None
    assert decision.tokens_out == 45


def test_an_answer_with_no_tool_call_costs_one_turn_not_the_run(
    settings: Settings, tmp_path: Path
) -> None:
    transport = Transport(blocks=[{"type": "text", "text": "I would prefer to negotiate."}])
    agents = recording(settings, tmp_path, transport)
    decision = agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )
    assert isinstance(decision.action, Wait)
    assert decision.coercion == "no_tool_call"


def test_an_inadmissible_tool_call_is_coerced_and_recorded(
    settings: Settings, tmp_path: Path
) -> None:
    transport = Transport({"type": "ESCALATE", "target_factor": "factor_9", "magnitude": 0.8})
    agents = recording(settings, tmp_path, transport)
    decision = agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )
    assert isinstance(decision.action, Wait)
    assert decision.coercion == "no_lever:factor_9"


def test_an_identical_turn_is_served_from_the_cache(settings: Settings, tmp_path: Path) -> None:
    """§12.1's "action cache" is the record/replay cache, keyed on the prompt.

    Two replicates whose observations round to the same two decimals produce
    identical request bytes, so the second costs nothing. That is why the
    projection rounds.
    """
    transport = Transport()
    agents = recording(settings, tmp_path, transport)
    first = agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )
    second = agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert len(transport.requests) == 1
    assert agents.hit_rate == 0.5
    assert first.action == second.action


def test_a_different_observation_is_a_different_call(settings: Settings, tmp_path: Path) -> None:
    transport = Transport()
    agents = recording(settings, tmp_path, transport)
    agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation(),
        memory="",
        space=SPACE,
    )
    agents.decide(
        scenario_id="scenario-1",
        actor_id="actor_0",
        observation=observation().model_copy(update={"factors": {"factor_0": 0.99}}),
        memory="",
        space=SPACE,
    )
    assert len(transport.requests) == 2


def test_an_unprepared_actor_is_an_error_not_a_default(settings: Settings, tmp_path: Path) -> None:
    """A missing persona would otherwise become an agent with no objective."""
    agents = recording(settings, tmp_path, Transport())
    with pytest.raises(KeyError, match="no prepared prompt"):
        agents.decide(
            scenario_id="scenario-1",
            actor_id="ghost",
            observation=observation(),
            memory="",
            space=SPACE,
        )
