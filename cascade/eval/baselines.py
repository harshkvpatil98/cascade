"""The five baselines (spec §10.2). Prompt construction is pure; the calls are not.

"Five, and all five appear in the report. Omitting the fair-compute baseline is
the most common way a multi-agent result gets dismissed."

Two of the five are configurations of the system itself -- the multi-agent
panel without decomposition is ablation cell C09 and the full system is C01 --
so they are scored from the forecasts those cells already produce rather than
re-run under another name. Running them twice would put two differently-seeded
copies of the same experiment in the report and invite the reader to wonder
which one is quoted.

The three that are not cells are built here:

* **Climatology** predicts the sealed set's own base rate for every scenario.
  Pure, no model, no evidence. It is the floor: beating it is necessary and
  not impressive.
* **Single model, direct** is one call per scenario with the *same* Chronofence
  evidence an agent gets. Same time lock, same k, same cutoff. Anything less
  and the comparison measures retrieval rather than architecture.
* **Single model, self-consistency** is the same prompt drawn 200 times and
  averaged. §10.2 calls it critical, and it is: it matches Cascade's sample
  budget, so a gain over it cannot be dismissed as "you just sampled more".

The self-consistency draws are distinguished by ``LLMRequest.sample_index``.
The call cache is content-addressed, so without it 200 identical requests would
be one recording replayed 200 times -- an ensemble with sigma exactly zero,
which would make the fair-compute baseline look structurally incapable of
dispersion for a reason that is an artefact of the cache rather than a fact
about single models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cascade.config import Settings
from cascade.ledger.schema import Scenario
from cascade.llm.types import BatchItem, LLMRequest

__all__ = [
    "BASELINES",
    "CLIMATOLOGY_CONFIG_ID",
    "DIRECT_CONFIG_ID",
    "SELF_CONSISTENCY_CONFIG_ID",
    "BaselineRun",
    "BaselineSpec",
    "SampleCollapse",
    "baseline_prompt",
    "climatology_forecasts",
    "collapse_samples",
    "evidence_for",
    "run_single_model",
    "sample_requests",
    "system_prompt",
]

CLIMATOLOGY_CONFIG_ID = "B1_climatology"
DIRECT_CONFIG_ID = "B2_single_direct"
SELF_CONSISTENCY_CONFIG_ID = "B3_self_consistency"


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """One of §10.2's five, with the construction and purpose it states."""

    baseline_id: str
    config_id: str
    name: str
    construction: str
    purpose: str
    from_grid: bool
    """True when the baseline *is* an ablation cell and is scored from it."""


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        baseline_id="climatology",
        config_id=CLIMATOLOGY_CONFIG_ID,
        name="Climatology",
        construction="Always predict the sealed set's base rate",
        purpose="Floor. Beating it is necessary, not impressive.",
        from_grid=False,
    ),
    BaselineSpec(
        baseline_id="single_direct",
        config_id=DIRECT_CONFIG_ID,
        name="Single model, direct",
        construction="One call, same Chronofence context, asked for a probability",
        purpose="The reference for the headline Brier skill score.",
        from_grid=False,
    ),
    BaselineSpec(
        baseline_id="self_consistency",
        config_id=SELF_CONSISTENCY_CONFIG_ID,
        name="Single model, self-consistency",
        construction="Same prompt, N samples, averaged",
        purpose=(
            "Fair compute. Matches Cascade's sample budget, so a gain cannot be "
            "dismissed as sampling more."
        ),
        from_grid=False,
    ),
    BaselineSpec(
        baseline_id="mas_no_decomposition",
        config_id="C09",
        name="Multi-agent, no decomposition",
        construction="Generic persona panel, no causal graph, no asymmetry",
        purpose="Isolates the value of structure over mere multiplicity.",
        from_grid=True,
    ),
    BaselineSpec(
        baseline_id="cascade_full",
        config_id="C01",
        name="Cascade, full",
        construction="Decomposition + asymmetry + ensemble",
        purpose="The system under test.",
        from_grid=True,
    ),
)


def climatology_forecasts(
    scenario_ids: Sequence[str], *, base_rate: float
) -> tuple[tuple[str, float], ...]:
    """``(scenario_id, base_rate)`` for every scenario. Pure.

    The base rate comes from the sealed manifest, never recomputed from the
    scenarios that happen to have forecasts. §10.2 calls climatology the floor,
    and a floor recomputed against a subset moves in whichever direction
    flatters the system being measured.
    """
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError(f"base rate must lie in [0, 1], got {base_rate}")
    return tuple((scenario_id, base_rate) for scenario_id in sorted(scenario_ids))


def system_prompt() -> str:
    """The single-model baseline's system prompt.

    Deliberately minimal and deliberately *not* the agent's. The agent prompt
    describes a world model, an action space and an arbiter; handing that to a
    single model asked for one number would be measuring a differently-prompted
    single model rather than the baseline §10.2 defines.
    """
    return (
        "You are a forecasting analyst. You are given a question, its "
        "resolution criterion, and evidence published strictly before the "
        "forecast cutoff. Estimate the probability that the question resolved "
        'YES. Answer with a JSON object of the form {"p": <number between 0 '
        "and 1>} and nothing else. Do not explain."
    )


def baseline_prompt(
    scenario: Scenario,
    evidence: Sequence[tuple[str, str, str]],
    *,
    evidence_chars: int,
) -> str:
    """Render the user turn: question, criterion, cutoff, and the evidence. Pure.

    ``evidence`` is ``(published_iso, source, excerpt)`` -- the same triple the
    agent prefix carries, truncated to the same width, so the only difference
    between this baseline and the system under test is the architecture.

    The cutoff is stated explicitly. Without it a model reasons from its own
    sense of "now", which for a 2018 question means reasoning from what it
    knows in 2026 -- and that is the parametric leakage §4.4 measures, imported
    into the baseline by a prompt that forgot to mention the date.
    """
    lines = [
        f"Question: {scenario.question}",
        f"Resolution criterion: {scenario.resolution_criterion}",
        f"Forecast cutoff: {scenario.cutoff_ts.isoformat()}",
        f"Resolution date: {scenario.resolve_ts.isoformat()}",
        "",
        "Reason only from what was knowable at the cutoff.",
        "",
        "# Evidence published before the cutoff",
        "",
    ]
    if evidence:
        lines.extend(
            f"[{published}] {source}\n{excerpt[:evidence_chars]}\n"
            for published, source, excerpt in evidence
        )
    else:
        lines.append("(no admissible evidence was found before the cutoff)\n")
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


def sample_requests(
    settings: Settings,
    scenario: Scenario,
    evidence: Sequence[tuple[str, str, str]],
    *,
    samples: int,
    temperature: float,
    config_id: str,
) -> list[BatchItem]:
    """Build ``samples`` independent draws of one scenario's prompt. Pure.

    ``custom_id`` carries the config and the draw, so a wave mixing several
    scenarios and several baselines matches results to requests by identity
    rather than by position -- the Batches API returns results in arbitrary
    order (§12.2).
    """
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    system = system_prompt()
    user = baseline_prompt(scenario, evidence, evidence_chars=settings.kernel.evidence_chars)
    return [
        BatchItem(
            custom_id=f"{config_id}|{scenario.scenario_id}|{index}",
            request=LLMRequest(
                model=settings.models.agent,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=64,
                temperature=temperature,
                prompt_rev=settings.llm.prompt_rev,
                sample_index=index,
            ),
        )
        for index in range(samples)
    ]


@dataclass(frozen=True, slots=True)
class SampleCollapse:
    """What a set of draws for one scenario collapsed to."""

    scenario_id: str
    p_hat: float
    sigma: float
    n_parsed: int
    n_requested: int

    @property
    def unparseable(self) -> int:
        return self.n_requested - self.n_parsed


def collapse_samples(
    scenario_id: str, probabilities: Sequence[float | None], *, requested: int
) -> SampleCollapse | None:
    """Average the parseable draws. ``None`` when none parsed.

    An unparseable answer is dropped and counted, never scored as 0.5. For the
    memorisation probe that substitution reads as reassurance (§4.4); here it
    reads as a *well-calibrated* baseline, which is the direction that makes
    the system under test look worse and is no more honest for it.
    """
    parsed = [value for value in probabilities if value is not None]
    if not parsed:
        return None
    mean = sum(parsed) / len(parsed)
    if len(parsed) > 1:
        variance = sum((value - mean) ** 2 for value in parsed) / (len(parsed) - 1)
        sigma = variance**0.5
    else:
        sigma = 0.0
    return SampleCollapse(
        scenario_id=scenario_id,
        p_hat=mean,
        sigma=sigma,
        n_parsed=len(parsed),
        n_requested=requested,
    )


def evidence_for(
    scenario: Scenario, retrieved: Mapping[str, Sequence[tuple[str, str, str]]]
) -> tuple[tuple[str, str, str], ...]:
    """The evidence triples for one scenario, or an empty tuple.

    Empty is a legitimate outcome -- an early-cutoff scenario can have nothing
    admissible -- and it renders as an explicit statement in the prompt rather
    than as a missing section, so the model is told the absence rather than
    left to infer it.
    """
    return tuple(retrieved.get(scenario.scenario_id, ()))


# ---------------------------------------------------------------------------
# The shell: the only part of this module that makes a call
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """What one single-model baseline produced, measured."""

    config_id: str
    samples: int
    collapsed: tuple[SampleCollapse, ...]
    scenarios_requested: int
    unparseable: int
    calls: int

    @property
    def scenarios_scored(self) -> int:
        return len(self.collapsed)


def run_single_model(
    settings: Settings,
    scenarios: Sequence[Scenario],
    retrieved: Mapping[str, Sequence[tuple[str, str, str]]],
    *,
    config_id: str,
    samples: int,
    temperature: float,
    client: object = None,
    batch_size: int = 2000,
) -> BaselineRun:
    """Draw ``samples`` forecasts per scenario and collapse them.

    Goes through :class:`~cascade.llm.client.LLMClient` like every other model
    call (invariant 5), and through ``complete_batch`` rather than ``complete``
    so the 50% discount §12.1 prices the study at actually applies: the
    self-consistency baseline is 36,000 calls on its own, which at interactive
    rates is most of the ``baseline`` phase ceiling.

    Requests are submitted in bounded groups rather than as one 36,000-item
    batch. The provider caps a batch's size, and a group that fails is a group
    to resubmit rather than the whole baseline.

    In ``replay`` mode with no recording this raises ``CacheMiss``, which is
    correct: a baseline that silently returned nothing would be reported as
    unmeasurable rather than as unrecorded.
    """
    from cascade.llm.client import LLMClient
    from cascade.retrieval.memorization import parse_probability

    llm = client if client is not None else LLMClient(settings, phase="baseline")

    items: list[BatchItem] = []
    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        items.extend(
            sample_requests(
                settings,
                scenario,
                evidence_for(scenario, retrieved),
                samples=samples,
                temperature=temperature,
                config_id=config_id,
            )
        )

    answers: dict[str, str] = {}
    for start in range(0, len(items), max(1, batch_size)):
        group = items[start : start + max(1, batch_size)]
        results = llm.complete_batch(group, trace_name=f"baseline.{config_id}")  # type: ignore[attr-defined]
        for custom_id in sorted(results):
            answers[custom_id] = results[custom_id].text

    collapsed: list[SampleCollapse] = []
    unparseable = 0
    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        probabilities = [
            parse_probability(answers.get(f"{config_id}|{scenario.scenario_id}|{index}", ""))
            for index in range(samples)
        ]
        unparseable += sum(1 for value in probabilities if value is None)
        outcome = collapse_samples(scenario.scenario_id, probabilities, requested=samples)
        if outcome is not None:
            collapsed.append(outcome)

    return BaselineRun(
        config_id=config_id,
        samples=samples,
        collapsed=tuple(collapsed),
        scenarios_requested=len(scenarios),
        unparseable=unparseable,
        calls=len(items),
    )
