"""The three-pass compiler (spec §5.2).

Driven by a scripted client rather than a model. What is under test is the
*orchestration*: that critique cannot rewrite, that the repair loop stops where
§5.2 says it stops, that a failure is returned and logged rather than raised,
and that retrieval happens inside the time lock.

A real model is never involved, so these tests say nothing about graph quality.
Quality is the audit protocol's job (§5.4), and it needs an API key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cascade.decompose.compiler import MAX_REPAIR_RETRIES, Lathe, NoToolCall
from cascade.decompose.prompts import CRITIQUE_TOOL, DRAFT_TOOL
from cascade.ledger.schema import Scenario
from tests.conftest import fake_embed, make_graph

CUTOFF = datetime(2019, 6, 1, tzinfo=UTC)


def scenario(scenario_id: str = "s1") -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        question="Will the regulator block the merger before the deadline?",
        resolution_criterion="Resolves YES if the merger is formally blocked.",
        cutoff_ts=CUTOFF,
        resolve_ts=datetime(2020, 1, 1, tzinfo=UTC),
        domain="corporate",
        source="polymarket",
        source_ref="ref",
        party_rule="named_parties",
        party_names=("Acme", "Regulator"),
        event_group=None,
    )


def graph_payload(**overrides: Any) -> dict[str, Any]:
    """A valid DraftGraph payload: the baseline graph minus its scenario id."""
    canonical = make_graph(**overrides).canonical()
    canonical.pop("scenario_id")
    return canonical


class ScriptedResult:
    def __init__(self, tool_name: str, payload: Any, text: str = "") -> None:
        self.tool_calls = (
            [{"type": "tool_use", "name": tool_name, "input": payload}] if tool_name else []
        )
        self.text = text
        self.stop_reason = "tool_use" if tool_name else "end_turn"


class ScriptedClient:
    """Returns canned tool payloads in order, recording what it was asked."""

    def __init__(self, script: list[ScriptedResult]) -> None:
        self._script = list(script)
        self.requests: list[Any] = []

    def complete(self, request: Any, *, trace_name: str = "", **_: Any) -> Any:
        self.requests.append((trace_name, request))
        if not self._script:
            raise AssertionError(f"scripted client exhausted at call {len(self.requests)}")
        return self._script.pop(0)


def no_evidence(question: str, as_of: datetime, k: int) -> list[tuple[str, str, str]]:
    return []


def lathe(settings: Any, script: list[ScriptedResult], retrieve: Any = no_evidence) -> Lathe:
    return Lathe(
        settings=settings,
        client=ScriptedClient(script),
        embed=fake_embed(),
        retrieve=retrieve,
    )


def critique_payload(*defects: dict[str, str]) -> dict[str, Any]:
    return {"defects": list(defects)}


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_clean_compile_uses_three_calls(settings: Any) -> None:
    """draft + critique + repair. §5.2: "~3 calls per scenario"."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario())

    assert outcome.ok
    assert outcome.llm_calls == 3
    assert outcome.repair_retries == 0
    assert outcome.violations == ()
    assert outcome.graph is not None
    assert outcome.graph.scenario_id == "s1"


def test_the_three_passes_run_in_the_specified_order(settings: Any) -> None:
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    compiler.compile_scenario(scenario())
    assert [name for name, _ in compiler.client.requests] == [  # type: ignore[attr-defined]
        "lathe.draft",
        "lathe.critique",
        "lathe.repair",
    ]


def test_the_scenario_id_is_attached_by_the_compiler_not_the_model(settings: Any) -> None:
    """A hallucinated id would key the graph to the wrong scenario."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario("a-different-id"))
    assert outcome.graph is not None
    assert outcome.graph.scenario_id == "a-different-id"
    # The payload the model emitted carries no scenario_id at all.
    assert "scenario_id" not in DRAFT_TOOL["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# Tool-use enforcement
# ---------------------------------------------------------------------------


def test_every_call_pins_the_model_to_its_tool(settings: Any) -> None:
    """Without tool_choice the model may answer in prose containing JSON."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    compiler.compile_scenario(scenario())
    for name, request in compiler.client.requests:  # type: ignore[attr-defined]
        assert request.tool_choice is not None, f"{name} did not enforce a tool"
        assert request.tool_choice["type"] == "tool"
        assert request.tool_choice["name"] == request.tools[0]["name"]


def test_an_answer_without_a_tool_call_raises(settings: Any) -> None:
    """Nothing to repair: enforcement did not hold and there is no payload."""
    compiler = lathe(settings, [ScriptedResult("", None, text="Here is a graph...")])
    with pytest.raises(NoToolCall, match="emit_causal_graph"):
        compiler.compile_scenario(scenario())


def test_the_compiler_model_is_used_not_the_agent_model(settings: Any) -> None:
    """§5.2 specifies Sonnet for the draft; Haiku is the agent model."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    compiler.compile_scenario(scenario())
    for _, request in compiler.client.requests:  # type: ignore[attr-defined]
        assert request.model == settings.models.compiler
        assert request.temperature == settings.models.compiler_temperature


# ---------------------------------------------------------------------------
# Critique describes; it does not rewrite
# ---------------------------------------------------------------------------


def test_the_critique_pass_cannot_emit_a_graph(settings: Any) -> None:
    """Its tool has no graph fields, so papering over a defect is impossible."""
    assert set(CRITIQUE_TOOL["input_schema"]["properties"]) == {"defects"}
    assert "actors" not in CRITIQUE_TOOL["input_schema"]["properties"]


def test_defects_are_passed_verbatim_into_the_repair_prompt(settings: Any) -> None:
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(
                CRITIQUE_TOOL["name"],
                critique_payload(
                    {
                        "kind": "missing_party",
                        "subject": "financing_bank",
                        "detail": "The lead financier can withdraw funding and is absent.",
                    }
                ),
            ),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario())
    repair_request = compiler.client.requests[-1][1]  # type: ignore[attr-defined]
    body = repair_request.messages[0]["content"]

    assert "financing_bank" in body
    assert "lead financier can withdraw funding" in body
    assert outcome.defects == (
        "missing_party [financing_bank]: The lead financier can withdraw funding and is absent.",
    )


def test_an_unparseable_critique_degrades_rather_than_failing(settings: Any) -> None:
    """Losing the critique costs quality; treating it as fatal costs the scenario."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], {"nonsense": True}),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario())
    assert outcome.ok
    assert any("critique_unparseable" in defect for defect in outcome.defects)


# ---------------------------------------------------------------------------
# The repair loop
# ---------------------------------------------------------------------------


def _self_loop_payload() -> dict[str, Any]:
    """A graph that parses but fails the validator's edge-sanity rule."""
    payload = graph_payload()
    payload["edges"] = [
        *payload["edges"],
        {"src": "factor_2", "dst": "factor_2", "sign": 1, "weight": 0.2, "lag": 0},
    ]
    return payload


def test_a_validator_failure_drives_another_repair(settings: Any) -> None:
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload()),  # repair 1: still bad
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),  # repair 2: good
        ],
    )
    outcome = compiler.compile_scenario(scenario())
    assert outcome.ok
    assert outcome.repair_retries == 1
    assert outcome.llm_calls == 4


def test_violations_reach_the_repair_prompt(settings: Any) -> None:
    """The repair pass is driven by the violation text, so it must arrive."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    compiler.compile_scenario(scenario())
    first_repair = compiler.client.requests[2][1]  # type: ignore[attr-defined]
    assert "edge_sanity" in first_repair.messages[0]["content"]


def test_the_loop_stops_after_the_specified_retries(settings: Any) -> None:
    """§5.2: "maximum two retries, then hard-fail the scenario and log it"."""
    script = [
        ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload()),
        ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
    ]
    script += [
        ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload())
        for _ in range(MAX_REPAIR_RETRIES + 1)
    ]
    compiler = lathe(settings, script)
    outcome = compiler.compile_scenario(scenario())

    assert not outcome.ok
    assert outcome.status == "failed"
    assert outcome.repair_retries == MAX_REPAIR_RETRIES
    assert any("edge_sanity" in violation for violation in outcome.violations)


def test_a_hard_failure_is_returned_not_raised(settings: Any) -> None:
    """One intractable scenario must not cost the other 179."""
    script = [
        ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload()),
        ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
    ]
    script += [
        ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload())
        for _ in range(MAX_REPAIR_RETRIES + 1)
    ]
    outcome = lathe(settings, script).compile_scenario(scenario())
    assert outcome.status == "failed"
    assert outcome.graph is not None, "the last attempt is kept for diagnosis"


def test_a_draft_that_fails_to_parse_is_repaired_not_abandoned(settings: Any) -> None:
    """Bounds errors are exactly what a repair pass can fix."""
    too_few = graph_payload()
    too_few["actors"] = too_few["actors"][:3]

    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], too_few),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario())
    assert outcome.ok
    repair_body = compiler.client.requests[2][1].messages[0]["content"]  # type: ignore[attr-defined]
    assert "schema:" in repair_body
    assert "actors" in repair_body


def test_the_original_critique_is_not_resent_on_later_retries(settings: Any) -> None:
    """Re-sending applied defects makes the model relitigate its own fixes."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(
                CRITIQUE_TOOL["name"],
                critique_payload(
                    {"kind": "missing_party", "subject": "bank", "detail": "absent financier"}
                ),
            ),
            ScriptedResult(DRAFT_TOOL["name"], _self_loop_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    compiler.compile_scenario(scenario())
    first_repair = compiler.client.requests[2][1].messages[0]["content"]  # type: ignore[attr-defined]
    second_repair = compiler.client.requests[3][1].messages[0]["content"]  # type: ignore[attr-defined]
    assert "absent financier" in first_repair
    assert "absent financier" not in second_repair
    assert "edge_sanity" in second_repair


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_is_retrieved_at_the_scenario_cutoff(settings: Any) -> None:
    """The compiler sits inside the time lock exactly like the simulation."""
    seen: list[tuple[str, datetime, int]] = []

    def spy(question: str, as_of: datetime, k: int) -> list[tuple[str, str, str]]:
        seen.append((question, as_of, k))
        return [("2019-01-02T00:00:00Z", "ccnews", "Regulator opened a phase 2 review.")]

    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
        retrieve=spy,
    )
    outcome = compiler.compile_scenario(scenario())

    assert len(seen) == 1
    _question, as_of, k = seen[0]
    assert as_of == CUTOFF
    assert k == settings.retrieval.k_compiler
    assert outcome.evidence_chunks == 1

    draft_body = compiler.client.requests[0][1].messages[0]["content"]  # type: ignore[attr-defined]
    assert "phase 2 review" in draft_body


def test_a_scenario_with_no_evidence_still_compiles(settings: Any) -> None:
    """Corpus coverage is uneven; an empty result must not lose the scenario."""
    compiler = lathe(
        settings,
        [
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
            ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
        ],
    )
    outcome = compiler.compile_scenario(scenario())
    assert outcome.ok
    assert outcome.evidence_chunks == 0
    draft_body = compiler.client.requests[0][1].messages[0]["content"]  # type: ignore[attr-defined]
    assert "No pre-cutoff evidence" in draft_body


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_recompiling_an_unchanged_input_reproduces_the_hash(settings: Any) -> None:
    """§5.3's determinism rule, at the compiler boundary."""

    def run() -> str:
        compiler = lathe(
            settings,
            [
                ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
                ScriptedResult(CRITIQUE_TOOL["name"], critique_payload()),
                ScriptedResult(DRAFT_TOOL["name"], graph_payload()),
            ],
        )
        outcome = compiler.compile_scenario(scenario())
        assert outcome.graph_sha256 is not None
        return outcome.graph_sha256

    assert run() == run()
