"""The parametric probe: how much the agent model already knows (spec §4.2).

Asks the agent model each scenario's question with **zero context** and records
how confidently it answers. That confidence is the floor the study cannot get
below: if the model already knows that a 2018 merger was blocked, no amount of
time-locked retrieval makes its forecast an honest 2018 forecast.

This is the one leakage control that cannot be engineered away. Chronofence
can keep post-cutoff *documents* out of the context window; it cannot keep
post-cutoff *facts* out of the weights. §4.2 is explicit that this is measured
and disclosed rather than fixed, and the `memorization_score` belongs next to
the headline Brier in the report, not in a footnote.

**The score is not accuracy.** A model that answers every question at 0.5 is
uninformative and scores 0. A model that answers confidently and *correctly*
scores high; one that answers confidently and *wrongly* also scores high on
confidence but is separately reported as miscalibrated, because a confidently
wrong prior is still a prior that will steer the simulation.

Outcomes are read with the ``eval`` role. This module is scored offline and is
never imported by the simulation path -- invariant 2 is a grant, and the grant
is what stops the simulation reaching a label, but there is no reason to put a
label-reading import anywhere near the kernel either.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from cascade.config import Settings
from cascade.ledger.schema import ScenarioRecord
from cascade.llm.types import LLMRequest

__all__ = [
    "MemorizationReport",
    "MemorizationScore",
    "parse_probability",
    "probe_prompt",
    "run_probe",
    "summarise",
]

# No retrieval, no context, no hints about the cutoff era. The point is to
# measure what the weights hold, so anything else in the prompt contaminates
# the measurement.
_SYSTEM = (
    "You are a forecasting assistant. You will be asked a question that was "
    "resolved at some point in the past. Answer only with a JSON object of the "
    'form {"p": <number between 0 and 1>}, giving the probability that the '
    "question resolved YES. Do not explain. If you do not know, answer 0.5."
)


def probe_prompt(record: ScenarioRecord) -> str:
    """The zero-context question put to the model.

    Carries the resolution criterion because without it "will the deal close"
    is ambiguous enough that a low confidence would measure the ambiguity
    rather than the absence of memorised knowledge.
    """
    scenario = record.scenario
    return (
        f"Question: {scenario.question}\n"
        f"Resolution criterion: {scenario.resolution_criterion}\n"
        f"Respond with JSON only."
    )


_NUMBER_RE = re.compile(r"-?\d*\.?\d+")


def parse_probability(text: str) -> float | None:
    """Extract ``p`` from a model response. Pure. ``None`` when unparseable.

    Returns ``None`` rather than a default: a probe that silently scored an
    unparseable answer as 0.5 would report memorisation as *absent* exactly
    when the measurement failed, and for this number that failure direction
    reads as reassurance.

    When the response contains a JSON object, only its ``p`` key is trusted --
    the bare-number fallback is not applied. Scavenging the first in-range
    number out of structured output is how ``{"scenario": 1, "p_est": 0.7}``
    becomes a confident 1.0. The fallback exists for models that drop the
    wrapper entirely and answer ``0.65``, which is common enough that refusing
    it would waste calls.
    """
    stripped = text.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            if "p" not in payload:
                return None
            try:
                value = float(payload["p"])
            except (TypeError, ValueError):
                return None
            return value if 0.0 <= value <= 1.0 else None

    for match in _NUMBER_RE.finditer(stripped):
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


@dataclass(frozen=True)
class MemorizationScore:
    """One scenario's parametric probe result."""

    scenario_id: str
    probability: float | None
    outcome: int

    @property
    def measured(self) -> bool:
        return self.probability is not None

    @property
    def confidence(self) -> float:
        """Distance from ignorance, in [0, 1]. ``2 * |p - 0.5|``.

        0.5 -> 0.0 (knows nothing), 0.0 or 1.0 -> 1.0 (certain either way).
        Direction-free on purpose: a confidently wrong prior distorts the
        simulation exactly as much as a confidently right one.
        """
        if self.probability is None:
            return 0.0
        return 2.0 * abs(self.probability - 0.5)

    @property
    def correct_direction(self) -> bool | None:
        """Whether the prior leans toward the true outcome. ``None`` at 0.5."""
        if self.probability is None or self.probability == 0.5:
            return None
        return (self.probability > 0.5) == (self.outcome == 1)

    @property
    def brier(self) -> float | None:
        """The probe's own Brier score, for comparison against the study's."""
        if self.probability is None:
            return None
        return (self.probability - self.outcome) ** 2


@dataclass(frozen=True)
class MemorizationReport:
    """The distribution §4.2 asks to be reported for all 180 scenarios."""

    scores: tuple[MemorizationScore, ...]
    scored: int
    unparseable: int
    mean_confidence: float
    median_confidence: float
    brier: float | None
    directionally_correct: int
    confident_and_correct: int
    confidence_deciles: tuple[tuple[float, int], ...]

    @property
    def total(self) -> int:
        return len(self.scores)


def summarise(scores: Sequence[MemorizationScore]) -> MemorizationReport:
    """Reduce probe results to the reported distribution. Pure."""
    from cascade.retrieval.metrics import percentile

    measured = [score for score in scores if score.measured]
    confidences = [score.confidence for score in measured]
    briers = [score.brier for score in measured if score.brier is not None]

    deciles: list[tuple[float, int]] = []
    for step in range(10):
        low, high = step / 10.0, (step + 1) / 10.0
        # The top bucket is closed so a confidence of exactly 1.0 is counted.
        count = sum(
            1 for value in confidences if (low <= value < high) or (step == 9 and value == 1.0)
        )
        deciles.append((low, count))

    return MemorizationReport(
        scores=tuple(scores),
        scored=len(measured),
        unparseable=len(scores) - len(measured),
        mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        median_confidence=percentile(confidences, 50.0) if confidences else 0.0,
        brier=sum(briers) / len(briers) if briers else None,
        directionally_correct=sum(1 for score in measured if score.correct_direction),
        confident_and_correct=sum(
            1 for score in measured if score.confidence >= 0.5 and score.correct_direction
        ),
        confidence_deciles=tuple(deciles),
    )


def run_probe(
    settings: Settings, records: Sequence[ScenarioRecord], *, client: object = None
) -> MemorizationReport:
    """Ask the agent model every question with no context, and score it.

    Goes through :class:`~cascade.llm.client.LLMClient` like every other model
    call (invariant 5), so the probe is recorded, replayable, metered against
    the ``bench`` phase ceiling and traced. In ``replay`` mode with no
    recording this raises ``CacheMiss``, which is the correct behaviour: a
    probe that quietly returned nothing would report zero memorisation.
    """
    from cascade.llm.client import LLMClient

    llm = client if client is not None else LLMClient(settings, phase="bench")

    scores: list[MemorizationScore] = []
    for record in sorted(records, key=lambda item: item.scenario.scenario_id):
        request = LLMRequest(
            model=settings.models.agent,
            system=_SYSTEM,
            messages=[{"role": "user", "content": probe_prompt(record)}],
            max_tokens=64,
            # Zero temperature: this measures what the weights hold, and
            # sampling noise would be indistinguishable from uncertainty.
            temperature=0.0,
            prompt_rev=settings.llm.prompt_rev,
        )
        result = llm.complete(request, trace_name="memorization.probe")  # type: ignore[attr-defined]
        scores.append(
            MemorizationScore(
                scenario_id=record.scenario.scenario_id,
                probability=parse_probability(result.text),
                outcome=record.label.outcome,
            )
        )
    return summarise(scores)
