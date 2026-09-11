"""Prompts and tool schemas for the three compiler passes (spec §5.2).

Kept apart from :mod:`cascade.decompose.compiler` because prompt text is the
one part of this subsystem that will be revised on evidence, and §1.3 requires
every revision to be logged in ``prompt_revisions`` with the before/after
Brier. Isolating the text makes a revision a diff of this file rather than a
diff of the orchestration.

The tool schemas are **generated from the Pydantic models**, never hand-written
alongside them. A hand-copied JSON schema drifts from the model it mirrors, and
the drift shows up as a model emitting a field the validator then rejects --
burning two repair calls to rediscover a mismatch that only ever existed in
this file.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cascade.decompose.schema import (
    MAX_ACTORS,
    MAX_FACTORS,
    MIN_ACTORS,
    MIN_FACTORS,
    Actor,
    Edge,
    Factor,
    OutcomeRule,
)
from cascade.decompose.validator import (
    FACTOR_SIMILARITY_MAX,
    MAX_INBOUND_WEIGHT,
    OBJECTIVE_SIMILARITY_MAX,
)

# Spec §7.2: the simulation advances (resolve_ts - cutoff_ts) / 24 per step.
# Quoted into the prompt so the compiler can size `volatility` against the
# horizon it will actually be simulated over, rather than against an unstated
# scale (see the M6 build-log entry and prompt revision r2).
SIM_STEPS = 24
SIM_SQRT_STEPS = 5

__all__ = [
    "CRITIQUE_TOOL",
    "DEFECT_KINDS",
    "DRAFT_TOOL",
    "SYSTEM_PROMPT",
    "CritiqueReport",
    "Defect",
    "DefectKind",
    "DraftGraph",
    "critique_user_prompt",
    "draft_user_prompt",
    "repair_user_prompt",
]


class DraftGraph(BaseModel):
    """What a model is asked to emit: a graph without its scenario id.

    ``scenario_id`` is deliberately absent. The compiler already knows it, and
    asking for it invites a hallucinated or subtly reformatted id that would
    key the graph to the wrong row -- a failure that surfaces at M7 as a
    scenario scored against someone else's decomposition.
    """

    model_config = ConfigDict(extra="forbid")

    actors: Annotated[list[Actor], Field(min_length=MIN_ACTORS, max_length=MAX_ACTORS)]
    factors: Annotated[list[Factor], Field(min_length=MIN_FACTORS, max_length=MAX_FACTORS)]
    edges: Annotated[list[Edge], Field(min_length=1)]
    outcome_rule: OutcomeRule


# The fixed checklist from spec §5.2. Fixed because an open-ended "what is
# wrong with this graph" invites the model to relitigate style; these four are
# the failures that actually corrupt a forecast.
DefectKind = Literal[
    "missing_party",
    "objective_restates_outcome",
    "wrong_edge_sign",
    "conflated_factor",
]

DEFECT_KINDS: tuple[str, ...] = (
    "missing_party",
    "objective_restates_outcome",
    "wrong_edge_sign",
    "conflated_factor",
)


class Defect(BaseModel):
    """One finding from the critique pass. It describes; it does not rewrite."""

    model_config = ConfigDict(extra="forbid")

    kind: DefectKind
    subject: Annotated[str, Field(min_length=1, max_length=200)]
    """The actor id, factor id, or ``src->dst`` edge the defect concerns."""
    detail: Annotated[str, Field(min_length=1, max_length=600)]
    """What is wrong and what would be right. Read verbatim by the repair pass."""


class CritiqueReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defects: Annotated[list[Defect], Field(max_length=40)]


def _tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """Build an Anthropic tool definition from a Pydantic model."""
    return {
        "name": name,
        "description": description,
        "input_schema": model.model_json_schema(),
    }


DRAFT_TOOL = _tool(
    "emit_causal_graph",
    "Emit the causal decomposition of the scenario as a typed graph. "
    "This is the only way to answer; do not describe the graph in prose.",
    DraftGraph,
)

CRITIQUE_TOOL = _tool(
    "report_defects",
    "Report defects found in the candidate graph. Describe each defect "
    "precisely; do NOT emit a corrected graph.",
    CritiqueReport,
)


# ---------------------------------------------------------------------------
# System prompt
#
# Shared verbatim by all three passes. It carries the whole contract -- field
# semantics, bounds, the validator's rules, and the failure modes -- so the
# per-scenario messages stay short and the model is never guessing at what will
# be checked. A model that knows the validator's rules can satisfy them on the
# first pass; one that does not burns repair attempts rediscovering them.
#
# It is NOT marked cacheable. Measured at ~1,737 tokens, and ~3,384 with the
# draft tool schema ahead of it, both below the provider's 4,096-token cache
# floor -- and below that floor Anthropic silently declines to cache while
# still charging the write premium (ADR-0001). Padding to reach the floor would
# buy roughly $2 across the whole milestone in exchange for inflating every
# call with text that does not help the model. See the note in compiler.py.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are Lathe, the causal decomposition compiler for a strategic forecasting
system. You convert a resolved-in-the-past forecasting question into a typed
causal graph that a multi-agent simulation will execute for 24 discrete steps.

Your output is consumed mechanically. Nothing downstream reads prose: agent
construction, the information-asymmetry policy, and the arbiter's dynamics are
all derived from the fields you emit. A graph that is vague is not a softer
answer, it is an unusable one.

# What a graph is

ACTORS are parties with independent interests and real leverage over the
situation. Between {MIN_ACTORS} and {MAX_ACTORS} of them.

  id               stable lowercase slug, e.g. "european_commission"
  name             how a domain expert would refer to the party
  objective        ONE sentence: what this party is trying to maximise, stated
                   as an interest it would hold regardless of how this
                   particular question resolves
  utility_terms    weighted references to factor ids. The weight is SIGNED and
                   non-zero, in [-1, 1]. Negative means the actor wants that
                   factor LOW. This is how preference direction is expressed --
                   never invert a factor to make a weight positive.
  resources        named budgets in [0, 1], e.g. {{"political_capital": 0.6}}
  constraints      hard limits on what this actor may do, as short phrases
  risk_posture     "averse", "neutral" or "seeking"
  initial_beliefs  factor id -> prior in [0, 1], what this actor believes the
                   factor's value is at the start

FACTORS are world variables that move over time. Between {MIN_FACTORS} and
{MAX_FACTORS} of them.

  id                     stable lowercase slug
  name                   a short noun phrase naming the variable
  state                  its value at the cutoff, in [0, 1]
  volatility             sigma of its per-step exogenous random walk, [0, 1]
  inertia                mean-reversion strength, [0, 1]. High inertia means
                         the factor resists change and is pulled back toward
                         its cutoff value.
  observable_by_default  true for public information (published prices,
                         official statements); false for private state

  On the scale of volatility. The simulation runs {SIM_STEPS} steps between the
  cutoff and the resolution date, so one step is a {SIM_STEPS}th of that
  interval -- days or weeks, not months. `volatility` is the standard
  deviation of ONE step's drift, in the same [0, 1] units as `state`, where 0
  and 1 are the variable's extremes. Over the whole horizon an undriven factor
  therefore wanders about sqrt({SIM_STEPS}) x volatility ~ {SIM_SQRT_STEPS} x
  volatility. Work backwards from how far the real quantity plausibly moves on
  its own between now and resolution: a factor that could credibly cross most
  of its range unaided needs a volatility around 0.2, and one a domain expert
  would call broadly stable over the horizon is well under 0.05. Pick the
  number this arithmetic implies for each factor rather than a default.

EDGES are signed, lagged influences. src is an actor id or a factor id; dst is
always a factor id.

  sign    -1 or +1, the direction of influence
  weight  in (0, 1], how strongly src moves dst
  lag     0 to 6 steps before the influence lands

OUTCOME_RULE maps terminal factor state to a probability.

  terms      at least TWO distinct factors, each with a signed non-zero weight
  threshold  the pivot in factor space, [0, 1]
  steepness  how sharply the score responds, > 0

  It is evaluated as sigmoid(steepness * sum(weight_i * (state_i - threshold))).
  Pick the sign of each weight so that the factor moving UP pushes the score in
  the direction that factor genuinely implies for a YES resolution.

# Rules your graph will be checked against

These are applied by a deterministic validator after you answer. A graph that
fails is sent back to you with the violations. Satisfy them the first time.

  1. Actor count between {MIN_ACTORS} and {MAX_ACTORS}; factor count between
     {MIN_FACTORS} and {MAX_FACTORS}.
  2. Every actor must have at least one outbound edge onto a factor that has a
     directed path to a factor the outcome rule reads. An actor that influences
     nothing relevant is a spectator and does not belong in the graph.
  3. No actor objective may paraphrase the outcome. Semantic similarity to the
     question is measured and must stay below {OBJECTIVE_SIMILARITY_MAX}.
  4. Factor names must be semantically distinct from one another, below
     {FACTOR_SIMILARITY_MAX} pairwise similarity.
  5. No factor may influence itself at lag 0. Total inbound edge weight on any
     single factor must not exceed {MAX_INBOUND_WEIGHT}.
  6. The outcome rule must depend on at least two distinct factors.

# The four failures that matter most

THE OBJECTIVE FAILURE. The single most common way this task is done badly is
giving every actor the objective "make the outcome happen". If the question is
whether a merger is blocked, do NOT write "wants the merger blocked" for the
regulator and "wants the merger approved" for the firms. Those are positions on
the question, not interests. Write what each party actually pursues:

  BAD   regulator:  "Wants the merger to be blocked."
  GOOD  regulator:  "Maintain credible deterrence against market concentration
                     while avoiding litigation it might lose."
  BAD   acquirer:   "Wants the merger approved."
  GOOD  acquirer:   "Expand distribution capacity at the lowest achievable cost
                     in capital and regulatory concession."

Parties with genuinely independent interests will sometimes agree and sometimes
conflict, which is what makes the simulation informative. Parties defined by
their position on the question can only ever restate it.

THE MISSING PARTY. Ask who could change this situation but is not in the graph:
financiers, courts, unions, coalition partners, regional regulators, major
customers, the party that must ratify. Someone with real leverage is usually
absent from a first draft.

THE WRONG SIGN. Trace each edge and ask which direction the influence actually
runs. Rising public opposition usually LOWERS the probability of approval, not
raises it. Signs copied from intuition about the outcome rather than about the
mechanism are the most common subtle error.

THE CONFLATED FACTOR. One driver split across near-duplicate variables
("public opposition", "political pressure", "media criticism") inflates the
factor count and lets the same influence be counted three times. Conversely, a
single factor that bundles two independent drivers ("regulatory and legal risk")
cannot move in the two directions those drivers would move it. Each factor
should be one thing that can move on its own.

# Grounding

You are given evidence retrieved from documents published strictly BEFORE the
question's cutoff. Ground the graph in it: the actors named, the pressures
described, the state of play as of the cutoff. You will never be shown what
happened afterwards, and you must not reason from anything you happen to recall
about how this situation turned out. The graph describes the situation AS IT
STOOD, not the path to a known ending.
"""


def _render_evidence(chunks: list[tuple[str, str, str]]) -> str:
    """Render retrieved evidence as ``[n] date source: body`` blocks."""
    if not chunks:
        return (
            "(No pre-cutoff evidence was retrievable for this scenario. Build the "
            "graph from the question and resolution criterion alone, and prefer "
            "generic structural actors over specific named ones you cannot verify.)"
        )
    lines = []
    for index, (published, source, body) in enumerate(chunks, start=1):
        lines.append(f"[{index}] {published} · {source}\n{body.strip()}")
    return "\n\n".join(lines)


def draft_user_prompt(
    *,
    question: str,
    resolution_criterion: str,
    cutoff_iso: str,
    party_names: tuple[str, ...],
    chunks: list[tuple[str, str, str]],
) -> str:
    """The per-scenario message for the draft pass (spec §5.2)."""
    parties = ", ".join(party_names) if party_names else "(none recorded)"
    return (
        f"# Question\n{question}\n\n"
        f"# Resolution criterion\n{resolution_criterion}\n\n"
        f"# Cutoff\n{cutoff_iso} — all evidence below predates this instant.\n\n"
        f"# Parties named in the registry\n{parties}\n\n"
        f"# Evidence\n{_render_evidence(chunks)}\n\n"
        "Emit the causal graph with the emit_causal_graph tool."
    )


def critique_user_prompt(*, question: str, graph_json: str) -> str:
    """The per-scenario message for the adversarial critique pass.

    The checklist is restated here rather than left to the system prompt, so
    the critique is anchored to four specific questions rather than a general
    invitation to find fault.
    """
    return (
        f"# Question under forecast\n{question}\n\n"
        f"# Candidate graph\n```json\n{graph_json}\n```\n\n"
        "Audit this graph against exactly four questions:\n\n"
        "1. missing_party — which party with real leverage over these factors is "
        "absent from the actor list?\n"
        "2. objective_restates_outcome — which actor's objective is a restatement "
        "of the question's outcome rather than an independent interest it would "
        "hold either way?\n"
        "3. wrong_edge_sign — which edge has its sign backwards, judged by the "
        "mechanism rather than by the expected answer?\n"
        "4. conflated_factor — which factor is really two independent drivers, or "
        "which pair of factors is really one?\n\n"
        "Report each defect with report_defects. Be specific: name the actor id, "
        "factor id, or src->dst edge, and say what would be correct. "
        "Do NOT emit a corrected graph — you are auditing, not rewriting. "
        "If a category has no defect, report nothing for it rather than "
        "inventing one."
    )


def repair_user_prompt(
    *,
    question: str,
    graph_json: str,
    defects: list[str],
    violations: list[str],
) -> str:
    """The per-scenario message for the repair pass.

    Carries both sources of fault: the critique's semantic defects and the
    validator's structural violations. They are listed separately because they
    have different authority -- a validator violation is a fact about the graph,
    while a critique defect is a judgement that the repair pass may reasonably
    decline if it is wrong.
    """
    sections = [
        f"# Question under forecast\n{question}",
        f"# Current graph\n```json\n{graph_json}\n```",
    ]

    if violations:
        sections.append(
            "# Validator violations (these are facts; the graph will be rejected "
            "again until each is resolved)\n" + "\n".join(f"- {item}" for item in violations)
        )
    if defects:
        sections.append(
            "# Reviewer defects (these are judgements; apply each unless it is "
            "clearly mistaken, and if you decline one, fix the underlying issue "
            "another way)\n" + "\n".join(f"- {item}" for item in defects)
        )

    sections.append(
        "Apply these corrections and emit the complete repaired graph with the "
        "emit_causal_graph tool. Emit the whole graph, not a patch. Preserve "
        "everything that was not criticised — renaming or renumbering unaffected "
        "actors and factors makes the change impossible to review."
    )
    return "\n\n".join(sections)
