"""The graph audit protocol (spec §5.4).

Automated validation catches structural defects, not wrong ones. A graph can
satisfy every §5.3 rule and still name the wrong actors, get every influence
sign backwards, and omit the party that actually decided the outcome. §5.4's
answer is a human read of a seeded sample, scored against a published rubric.

This module does the parts a machine can do honestly:

* **select** the sample, seeded and reproducible, so nobody picks the twenty
  graphs that flatter the compiler;
* **write** the rubric and a worksheet carrying everything a reviewer needs;
* **score** a completed worksheet and compute the mean §5.4 gates on.

It does not review the graphs. A model scoring its own output would produce a
number with no information in it, and §5.4's threshold -- mean >= 1.5 out of 2
-- exists precisely to catch the case where the compiler prompt is wrong, which
is the case a self-review is least able to see.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cascade.decompose.schema import CausalGraph

__all__ = [
    "AUDIT_AXES",
    "PASS_THRESHOLD",
    "RUBRIC",
    "SAMPLE_SIZE",
    "AuditScore",
    "AuditSummary",
    "sample_scenarios",
    "score_worksheet",
    "summarise",
    "write_worksheet",
]

# §5.4: "Sample 20 of the 180 graphs (seeded, reproducible)".
SAMPLE_SIZE = 20
# "publish the mean ... If the mean is below 1.5, the compiler prompt is the
# problem and no amount of simulation tuning will fix it."
PASS_THRESHOLD = 1.5

AUDIT_AXES: tuple[str, ...] = (
    "actor_completeness",
    "edge_sign_correctness",
    "missing_leverage",
)

RUBRIC = """\
# Cascade graph audit rubric (spec §5.4)

Score each sampled graph on three axes, 0 to 2. Write the score and a one-line
justification. The mean across all axes and all graphs must be at least 1.5;
below that, the compiler prompt is the problem and simulation tuning will not
fix it.

Judge the graph against the situation **as it stood at the cutoff**, using only
what was knowable then. You may know how the question resolved. That knowledge
makes a graph look wrong when it is merely uncertain, so set it aside: the
question is whether a domain-literate person at the cutoff would recognise this
decomposition, not whether it points at the answer.

## actor_completeness

Are the actors the ones a domain-literate person would name?

  0  The actor list is generic or wrong. Parties are placeholders ("regulator",
     "company") where specific institutions were identifiable, or several
     listed parties had no role in this situation.
  1  The principal parties are present but the list is thin or padded: an
     obvious secondary party is missing, or several actors are duplicates of
     one another under different names.
  2  The list is one a domain expert would recognise, with the principals
     present and each listed party plausibly involved.

## edge_sign_correctness

Are the influence signs right, judged by mechanism rather than by outcome?

  0  Multiple signs are backwards, or the signs appear to have been chosen to
     make the outcome rule point at the known answer.
  1  Mostly right, with one or two edges whose direction is questionable or
     which depend on an unstated assumption.
  2  Every edge's direction is defensible from the mechanism it represents.

## missing_leverage

Is anything with real leverage absent from the graph entirely -- not just as an
actor, but as a factor or an influence path?

  0  A decisive lever is missing. The graph could not produce the range of
     outcomes the situation actually admitted.
  1  The main levers are present but a material one is absent or is folded into
     another factor where it should move independently.
  2  Nothing with real leverage is missing; the graph spans the situation's
     plausible dynamics.

## Recording scores

Fill in `scores` in the worksheet JSON. Each entry needs an integer 0-2 for
each axis and a short `note`. Leave `note` empty only when the score is 2 and
there is genuinely nothing to say.
"""


@dataclass(frozen=True)
class AuditScore:
    """One reviewer's scores for one graph."""

    scenario_id: str
    actor_completeness: int
    edge_sign_correctness: int
    missing_leverage: int
    note: str = ""

    def __post_init__(self) -> None:
        for axis in AUDIT_AXES:
            value = getattr(self, axis)
            if value not in (0, 1, 2):
                raise ValueError(
                    f"{axis} for {self.scenario_id!r} must be 0, 1 or 2; got {value!r}"
                )

    @property
    def mean(self) -> float:
        return sum(int(getattr(self, axis)) for axis in AUDIT_AXES) / len(AUDIT_AXES)


@dataclass(frozen=True)
class AuditSummary:
    """The published result of an audit (spec §5.4)."""

    scored: int
    mean: float
    per_axis: tuple[tuple[str, float], ...]
    worst: tuple[tuple[str, float], ...]
    threshold: float = PASS_THRESHOLD

    @property
    def passed(self) -> bool:
        return self.scored > 0 and self.mean >= self.threshold


def sample_scenarios(
    scenario_ids: Sequence[str], *, salt: str, size: int = SAMPLE_SIZE
) -> tuple[str, ...]:
    """Choose the audit sample, seeded and reproducible (§5.4).

    Seeded from the study salt and the *full* scenario id list, so the sample
    is fixed for a given registry and cannot be re-rolled until it looks
    favourable. Re-running the audit on the same registry reviews the same
    twenty graphs; adding a scenario changes the population and therefore the
    sample, which is correct.
    """
    if not scenario_ids:
        raise ValueError("cannot sample an audit from an empty scenario set")
    ordered = sorted(scenario_ids)
    digest = hashlib.blake2b(
        "|".join(ordered).encode("utf-8"), key=salt.encode("utf-8"), digest_size=8
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))  # noqa: S311 -- reproducibility, not secrecy
    return tuple(sorted(rng.sample(ordered, min(size, len(ordered)))))


def _graph_brief(graph: CausalGraph) -> dict[str, Any]:
    """Everything a reviewer needs, without making them read raw JSON."""
    return {
        "actors": [
            {
                "id": actor.id,
                "name": actor.name,
                "objective": actor.objective,
                "influences": sorted({edge.dst for edge in graph.edges if edge.src == actor.id}),
            }
            for actor in sorted(graph.actors, key=lambda a: a.id)
        ],
        "factors": [
            {"id": factor.id, "name": factor.name, "state": factor.state}
            for factor in sorted(graph.factors, key=lambda f: f.id)
        ],
        "edges": [
            f"{edge.src} --({'+' if edge.sign > 0 else '-'}{edge.weight:.2f}, lag {edge.lag})--> {edge.dst}"
            for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag))
        ],
        "outcome_rule": {
            "terms": [
                f"{term.factor_id} {term.weight:+.2f}"
                for term in sorted(graph.outcome_rule.terms, key=lambda t: t.factor_id)
            ],
            "threshold": graph.outcome_rule.threshold,
            "steepness": graph.outcome_rule.steepness,
        },
    }


def write_worksheet(
    directory: Path,
    *,
    graphs: Sequence[tuple[str, str, CausalGraph]],
    salt: str,
) -> tuple[Path, Path]:
    """Write the rubric and a scoring worksheet. Returns both paths.

    ``graphs`` is ``(scenario_id, question, graph)`` for the sampled scenarios.
    The question travels with the graph because the reviewer cannot judge actor
    completeness without knowing what is being forecast -- and the *outcome*
    deliberately does not, because a reviewer who knows the answer scores the
    graph against the answer rather than against the situation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rubric_path = directory / "rubric.md"
    rubric_path.write_text(RUBRIC, encoding="utf-8")

    worksheet = {
        "generated_at": datetime.now(UTC).isoformat(),
        "salt": salt,
        "threshold": PASS_THRESHOLD,
        "axes": list(AUDIT_AXES),
        "instructions": "See rubric.md. Fill in every score with an integer 0-2.",
        "graphs": [
            {
                "scenario_id": scenario_id,
                "question": question,
                "graph": _graph_brief(graph),
            }
            for scenario_id, question, graph in graphs
        ],
        "scores": [
            {
                "scenario_id": scenario_id,
                "actor_completeness": None,
                "edge_sign_correctness": None,
                "missing_leverage": None,
                "note": "",
            }
            for scenario_id, _, _ in graphs
        ],
    }
    worksheet_path = directory / "worksheet.json"
    worksheet_path.write_text(json.dumps(worksheet, indent=2), encoding="utf-8")
    return rubric_path, worksheet_path


def score_worksheet(path: Path) -> tuple[AuditScore, ...]:
    """Read completed scores from a worksheet.

    Unscored entries -- any axis left null -- are skipped rather than treated as
    zero. A half-finished audit must report as half-finished, not as a failing
    one: the two are different facts and only one of them is about the compiler.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[AuditScore] = []
    for entry in payload.get("scores", []):
        values = [entry.get(axis) for axis in AUDIT_AXES]
        if any(value is None for value in values):
            continue
        out.append(
            AuditScore(
                scenario_id=str(entry.get("scenario_id", "")),
                actor_completeness=int(entry["actor_completeness"]),
                edge_sign_correctness=int(entry["edge_sign_correctness"]),
                missing_leverage=int(entry["missing_leverage"]),
                note=str(entry.get("note", "")),
            )
        )
    return tuple(out)


def summarise(scores: Sequence[AuditScore]) -> AuditSummary:
    """Compute the published mean and the per-axis breakdown. Pure.

    The per-axis breakdown is reported alongside the headline because a mean of
    1.5 made of (2.0, 2.0, 0.5) and one made of (1.5, 1.5, 1.5) call for
    entirely different prompt changes.
    """
    if not scores:
        return AuditSummary(scored=0, mean=0.0, per_axis=(), worst=())

    per_axis = tuple(
        (axis, sum(int(getattr(score, axis)) for score in scores) / len(scores))
        for axis in AUDIT_AXES
    )
    overall = sum(score.mean for score in scores) / len(scores)
    worst = tuple(
        (score.scenario_id, score.mean)
        for score in sorted(scores, key=lambda s: (s.mean, s.scenario_id))[:5]
    )
    return AuditSummary(scored=len(scores), mean=overall, per_axis=per_axis, worst=worst)
