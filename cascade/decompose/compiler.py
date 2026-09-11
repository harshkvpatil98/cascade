"""Lathe: the three-pass causal decomposition compiler (spec §5.2).

draft -> critique -> repair -> validate, looping back to repair on a validator
failure, at most twice, then hard-failing the scenario and logging it.

The three passes are separate LLM calls on purpose. Folding critique into
repair -- "find the problems and fix them" -- lets the model quietly paper over
a defect it just identified, and the defect list is then unfalsifiable because
nothing recorded what it said before the rewrite. Keeping them apart costs one
call per scenario and makes the critique auditable.

Every model call goes through :class:`~cascade.llm.client.LLMClient`
(invariant 5), so the whole pipeline is recorded, replayable, metered against
the ``compile`` phase ceiling, and traced.

**Prompt caching is deliberately not used here.** ADR-0001 requires a cacheable
prefix to clear the provider's 4,096-token floor, below which Anthropic
silently declines to cache while still charging the cache-write premium.
Measured: the draft prefix (tool schema + system prompt) is ~3,384 tokens and
the critique prefix ~1,998. Padding the prompt with filler to reach the floor
would buy roughly $2 of cache reads across the whole milestone at the cost of
inflating every call with text that does not help the model. The agent prompts
at M5 are called 36,000 times and are where caching actually pays.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from pydantic import ValidationError

from cascade.canonical import canonical_json, canonical_timestamp
from cascade.config import Settings
from cascade.decompose.prompts import (
    CRITIQUE_TOOL,
    DRAFT_TOOL,
    SYSTEM_PROMPT,
    CritiqueReport,
    DraftGraph,
    critique_user_prompt,
    draft_user_prompt,
    repair_user_prompt,
)
from cascade.decompose.schema import CausalGraph, graph_hash
from cascade.decompose.validator import EmbedFn, ValidationReport, validate
from cascade.ledger.schema import Scenario
from cascade.llm.types import LLMRequest

__all__ = [
    "MAX_REPAIR_RETRIES",
    "CompileOutcome",
    "Lathe",
    "NoToolCall",
]

# Spec §5.2: "maximum two retries, then hard-fail the scenario and log it."
# The retry count is what the M4 acceptance histogram is over.
MAX_REPAIR_RETRIES = 2


class NoToolCall(RuntimeError):
    """The model answered in prose where a tool call was required.

    Distinct from a schema violation: a graph that fails validation is
    repairable, while an answer with no structured payload at all means the
    tool-use enforcement did not hold and there is nothing to repair.
    """


@dataclass(frozen=True)
class CompileOutcome:
    """What compiling one scenario produced, whether or not it succeeded."""

    scenario_id: str
    status: Literal["compiled", "failed"]
    graph: CausalGraph | None
    graph_sha256: str | None
    repair_retries: int
    """Repair calls beyond the first. The histogram the acceptance asks for."""
    llm_calls: int
    evidence_chunks: int
    defects: tuple[str, ...]
    violations: tuple[str, ...]
    """Outstanding violations. Empty on success, the reason on failure."""
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.status == "compiled"


@dataclass
class _Budget:
    """Tracks calls so a runaway loop is visible in the outcome, not just the meter."""

    calls: int = 0

    def spent(self) -> int:
        return self.calls


@dataclass
class Lathe:
    """The compiler. One instance per run; safe to reuse across scenarios.

    ``embed`` and ``retrieve`` are injected rather than constructed so the
    whole pipeline can be exercised against recorded responses and a stub
    corpus, which is what makes the three-pass logic testable without a model
    or a database.
    """

    settings: Settings
    client: Any
    embed: EmbedFn
    retrieve: Any
    """Callable(question, as_of, k) -> list[(published_iso, source, body)]."""

    _budget: _Budget = field(default_factory=_Budget)

    # -- one pass ----------------------------------------------------------

    def _call(
        self,
        *,
        messages: list[dict[str, Any]],
        tool: dict[str, Any],
        temperature: float,
        trace_name: str,
    ) -> dict[str, Any]:
        """Issue one tool-enforced call and return the tool input.

        ``tool_choice`` pins the model to the tool. Without it the model may
        answer in prose that happens to contain JSON, and parsing prose is how
        a schema stops being enforced.
        """
        request = LLMRequest(
            model=self.settings.models.compiler,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            temperature=temperature,
            max_tokens=self.settings.models.compiler_max_tokens,
            prompt_rev=self.settings.llm.prompt_rev,
        )
        result = self.client.complete(request, trace_name=trace_name)
        self._budget.calls += 1

        for block in result.tool_calls:
            if block.get("name") == tool["name"]:
                payload = block.get("input")
                if isinstance(payload, dict):
                    return payload
        raise NoToolCall(
            f"{trace_name}: model returned no {tool['name']!r} tool call "
            f"(stop_reason={result.stop_reason!r}, text={result.text[:160]!r})"
        )

    def _evidence(self, scenario: Scenario) -> list[tuple[str, str, str]]:
        """Top-k pre-cutoff chunks for this scenario (spec §5.2).

        ``k_compiler`` is 60 per config. Retrieval goes through Chronofence, so
        nothing published at or after the cutoff can reach the draft -- the
        compiler is inside the time lock exactly like the simulation.
        """
        return list(
            self.retrieve(
                scenario.question,
                scenario.cutoff_ts,
                self.settings.retrieval.k_compiler,
            )
        )

    def _draft(self, scenario: Scenario, chunks: list[tuple[str, str, str]]) -> dict[str, Any]:
        prompt = draft_user_prompt(
            question=scenario.question,
            resolution_criterion=scenario.resolution_criterion,
            cutoff_iso=canonical_timestamp(scenario.cutoff_ts),
            party_names=scenario.party_names,
            chunks=chunks,
        )
        return self._call(
            messages=[{"role": "user", "content": prompt}],
            tool=DRAFT_TOOL,
            temperature=self.settings.models.compiler_temperature,
            trace_name="lathe.draft",
        )

    def _critique(self, scenario: Scenario, payload: dict[str, Any]) -> tuple[str, ...]:
        prompt = critique_user_prompt(
            question=scenario.question,
            graph_json=canonical_json(payload),
        )
        raw = self._call(
            messages=[{"role": "user", "content": prompt}],
            tool=CRITIQUE_TOOL,
            temperature=self.settings.models.compiler_temperature,
            trace_name="lathe.critique",
        )
        try:
            report = CritiqueReport.model_validate(raw)
        except ValidationError as exc:
            # A malformed critique is not fatal: the validator still runs and
            # the repair pass still has its violations. Losing the critique
            # degrades quality; treating it as fatal would lose the scenario.
            return (f"critique_unparseable: {exc.error_count()} schema errors",)
        return tuple(
            f"{defect.kind} [{defect.subject}]: {defect.detail}" for defect in report.defects
        )

    def _repair(
        self,
        scenario: Scenario,
        payload: dict[str, Any],
        *,
        defects: Sequence[str],
        violations: Sequence[str],
    ) -> dict[str, Any]:
        prompt = repair_user_prompt(
            question=scenario.question,
            graph_json=canonical_json(payload),
            defects=list(defects),
            violations=list(violations),
        )
        return self._call(
            messages=[{"role": "user", "content": prompt}],
            tool=DRAFT_TOOL,
            temperature=self.settings.models.compiler_temperature,
            trace_name="lathe.repair",
        )

    # -- assembly ----------------------------------------------------------

    def _build(
        self, scenario: Scenario, payload: dict[str, Any]
    ) -> tuple[CausalGraph | None, tuple[str, ...]]:
        """Construct a `CausalGraph`, or return the parse errors as violations.

        A payload that fails to construct is handed back to the repair loop
        rather than raised. Bounds and cross-reference errors are exactly the
        kind of thing a repair pass can fix, and the pydantic messages are
        already phrased as instructions.
        """
        try:
            draft = DraftGraph.model_validate(payload)
        except ValidationError as exc:
            return None, _format_errors(exc)
        try:
            graph = CausalGraph(
                scenario_id=scenario.scenario_id,
                actors=tuple(draft.actors),
                factors=tuple(draft.factors),
                edges=tuple(draft.edges),
                outcome_rule=draft.outcome_rule,
            )
        except ValidationError as exc:
            return None, _format_errors(exc)
        return graph, ()

    def _validate(self, scenario: Scenario, graph: CausalGraph) -> ValidationReport:
        outcome_text = f"{scenario.question} {scenario.resolution_criterion}".strip()
        return validate(graph, outcome_text=outcome_text, embed=self.embed)

    def compile_scenario(self, scenario: Scenario) -> CompileOutcome:
        """Compile one scenario. Never raises for a model-quality failure.

        A scenario that cannot be compiled within the retry budget is returned
        as a ``failed`` outcome carrying its outstanding violations, because
        §5.2 requires the failure to be *logged* rather than to abort the run:
        one intractable scenario must not cost the other 179.
        """
        started = time.monotonic()
        before = self._budget.spent()
        chunks = self._evidence(scenario)

        payload = self._draft(scenario, chunks)
        graph, parse_violations = self._build(scenario, payload)

        # Critique the draft as emitted. A draft that failed to parse is still
        # worth critiquing -- the defects are about content, and the repair
        # pass gets both lists.
        defects = self._critique(scenario, payload)

        violations: tuple[str, ...] = parse_violations
        if graph is not None:
            report = self._validate(scenario, graph)
            violations = tuple(v.render() for v in report.violations)

        retries = -1
        while True:
            retries += 1
            payload = self._repair(scenario, payload, defects=defects, violations=violations)
            graph, parse_violations = self._build(scenario, payload)
            if graph is None:
                violations = parse_violations
            else:
                report = self._validate(scenario, graph)
                violations = tuple(v.render() for v in report.violations)
                if report.passed_checks:
                    return CompileOutcome(
                        scenario_id=scenario.scenario_id,
                        status="compiled",
                        graph=graph,
                        graph_sha256=graph_hash(graph),
                        repair_retries=retries,
                        llm_calls=self._budget.spent() - before,
                        evidence_chunks=len(chunks),
                        defects=defects,
                        violations=(),
                        elapsed_s=time.monotonic() - started,
                    )
            # Only the first repair is free; `retries` counts the rest.
            if retries >= MAX_REPAIR_RETRIES:
                return CompileOutcome(
                    scenario_id=scenario.scenario_id,
                    status="failed",
                    graph=graph,
                    graph_sha256=graph_hash(graph) if graph is not None else None,
                    repair_retries=retries,
                    llm_calls=self._budget.spent() - before,
                    evidence_chunks=len(chunks),
                    defects=defects,
                    violations=violations,
                    elapsed_s=time.monotonic() - started,
                )
            # Defects were addressed by the first repair; subsequent rounds are
            # driven by the validator alone. Re-sending the original critique
            # would have the model re-litigate points it already applied.
            defects = ()


def _format_errors(exc: ValidationError) -> tuple[str, ...]:
    """Render pydantic errors as repair instructions, deterministically ordered."""
    rendered = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        rendered.append(f"schema: {location or '<root>'}: {error.get('msg', 'invalid')}")
    return tuple(sorted(set(rendered)))


def usd_spent(client: Any) -> Decimal:
    """Total spend booked by a client's meter, for reporting.

    Tolerates a client without a meter -- the test doubles do not carry one --
    rather than making every caller guard the attribute access.
    """
    meter = getattr(client, "meter", None)
    if meter is None:
        return Decimal("0")
    return Decimal(str(getattr(meter, "total_usd", 0)))
