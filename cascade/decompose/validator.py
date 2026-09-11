"""Graph validation (spec §5.3). Pure Python, no LLM, no I/O.

Six rules, each a function of the graph and nothing else. That purity is the
point: the validator is what stands between a plausible-looking graph and 200
simulation runs built on it, so it must be property-testable without a
database, a network, or a model.

Two of the six rules are *semantic* -- objective independence and factor
orthogonality -- and need sentence embeddings to evaluate. Rather than reach
for the embedder (which would make this module do I/O), :func:`validate` takes
an ``embed`` callable. When it is absent the semantic rules are reported as
**skipped**, never as passed: a check that did not run must not be
indistinguishable in the output from a check that succeeded, because the
difference is exactly whether anyone verified the failure §5.3 calls the most
common one -- every actor handed the objective "make the outcome happen".

The output is a list of violations phrased for two readers at once: a human
reading a build log, and the repair pass, which receives them verbatim as the
instruction set for its next attempt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cascade.decompose.schema import (
    MAX_ACTORS,
    MAX_FACTORS,
    MIN_ACTORS,
    MIN_FACTORS,
    CausalGraph,
)

__all__ = [
    "FACTOR_SIMILARITY_MAX",
    "MAX_INBOUND_WEIGHT",
    "OBJECTIVE_SIMILARITY_MAX",
    "EmbedFn",
    "ValidationReport",
    "Violation",
    "cosine",
    "validate",
]

# Spec §5.3. Named rather than inlined so the repair prompt can quote the
# number it has to satisfy.
OBJECTIVE_SIMILARITY_MAX = 0.85
FACTOR_SIMILARITY_MAX = 0.80
MAX_INBOUND_WEIGHT = 3.0

# Points at which each factor is held while another is swept, for the numeric
# monotonicity check. Three interior points rather than one: a rule that is
# monotone through the middle of the domain but not at the edges is still not
# monotone, and the edges are where a saturating form misbehaves.
_MONOTONE_HOLD_POINTS = (0.1, 0.5, 0.9)
_MONOTONE_SWEEP = tuple(index / 10.0 for index in range(11))

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Violation:
    """One broken rule, phrased as an instruction the repair pass can act on."""

    rule: str
    message: str
    subjects: tuple[str, ...] = ()

    def render(self) -> str:
        subjects = f" [{', '.join(self.subjects)}]" if self.subjects else ""
        return f"{self.rule}: {self.message}{subjects}"


@dataclass(frozen=True)
class ValidationReport:
    """The verdict, plus which rules were actually evaluated."""

    violations: tuple[Violation, ...]
    skipped_rules: tuple[str, ...] = ()
    checked_rules: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True only when every rule ran and none was violated.

        A skipped rule makes the verdict unknown, not favourable. Treating
        "could not check" as "passed" is how the objective-independence rule
        would quietly stop catching the failure it exists for.
        """
        return not self.violations and not self.skipped_rules

    @property
    def passed_checks(self) -> bool:
        """True when nothing that *ran* was violated, whatever was skipped."""
        return not self.violations

    def render(self) -> str:
        return "\n".join(violation.render() for violation in self.violations)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity. Pure.

    Implemented here rather than assuming unit-norm inputs: the corpus
    embedder normalises, but this function is also handed vectors by tests and
    by any future caller, and a silently wrong similarity would relax a
    leakage-adjacent threshold rather than tighten it.
    """
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _check_actor_count(graph: CausalGraph) -> list[Violation]:
    """§5.3: 8 <= |actors| <= 20.

    Below 8 the asymmetry mechanism has nothing to work with; above 20 the cost
    model breaks. The schema also enforces this, so reaching here means a graph
    was built another way -- but the repair pass needs a message, not an
    exception.
    """
    count = len(graph.actors)
    if count < MIN_ACTORS:
        return [
            Violation(
                rule="actor_count",
                message=(
                    f"{count} actors, minimum {MIN_ACTORS}. Add parties with genuine "
                    "leverage over the factors, not spectators."
                ),
            )
        ]
    if count > MAX_ACTORS:
        return [
            Violation(
                rule="actor_count",
                message=f"{count} actors, maximum {MAX_ACTORS}. Merge or drop the least decisive.",
            )
        ]
    return []


def _check_factor_count(graph: CausalGraph) -> list[Violation]:
    count = len(graph.factors)
    if not MIN_FACTORS <= count <= MAX_FACTORS:
        return [
            Violation(
                rule="factor_count",
                message=f"{count} factors, must be between {MIN_FACTORS} and {MAX_FACTORS}.",
            )
        ]
    return []


def _factors_reaching_outcome(graph: CausalGraph) -> frozenset[str]:
    """Factors with a directed path to any factor the outcome rule reads.

    Reverse BFS from the outcome factors over factor -> factor edges. Actor
    edges are excluded here because an actor is a source, never an
    intermediate: a path that leaves the factor graph through an actor and
    comes back is two influences, not one.
    """
    targets = set(graph.outcome_rule.factor_ids)
    factor_ids = graph.factor_ids
    incoming: dict[str, list[str]] = {factor: [] for factor in factor_ids}
    for edge in graph.edges:
        if edge.src in factor_ids:
            incoming[edge.dst].append(edge.src)

    reached = set(targets)
    frontier = sorted(targets)
    while frontier:
        current = frontier.pop()
        for parent in sorted(incoming.get(current, ())):
            if parent not in reached:
                reached.add(parent)
                frontier.append(parent)
    return frozenset(reached)


def _check_reachability(graph: CausalGraph) -> list[Violation]:
    """§5.3: every actor influences a factor on a path to the outcome.

    Decorative actors inflate the agent count -- and the per-run cost -- without
    informing the forecast.
    """
    live = _factors_reaching_outcome(graph)
    outbound: dict[str, set[str]] = {actor.id: set() for actor in graph.actors}
    for edge in graph.edges:
        if edge.src in outbound:
            outbound[edge.src].add(edge.dst)

    # Sorted at the iteration as well as the result: invariant 7's static check
    # is deliberately blunt, and uniform compliance is what keeps it meaningful.
    stranded = sorted(
        actor_id for actor_id, targets in sorted(outbound.items()) if not (targets & live)
    )
    if not stranded:
        return []
    return [
        Violation(
            rule="reachability",
            message=(
                f"{len(stranded)} actor(s) have no outbound edge onto a factor that "
                "reaches the outcome rule. Either give each real influence or remove it."
            ),
            subjects=tuple(stranded),
        )
    ]


def _check_edge_sanity(graph: CausalGraph) -> list[Violation]:
    """§5.3: no zero-lag factor self-loops; inbound weight per factor <= 3.0.

    A zero-lag self-loop is an instantaneous algebraic loop -- the factor's next
    value depends on itself with no step in between, which has no fixed point
    the arbiter can resolve. The weight cap keeps the dynamics bounded.
    """
    violations: list[Violation] = []

    loops = sorted({edge.src for edge in graph.edges if edge.src == edge.dst and edge.lag == 0})
    if loops:
        violations.append(
            Violation(
                rule="edge_sanity",
                message=(
                    "factor(s) influence themselves at lag 0, which is an instantaneous "
                    "loop with no fixed point. Use lag >= 1 or drop the edge."
                ),
                subjects=tuple(loops),
            )
        )

    inbound: dict[str, float] = {factor.id: 0.0 for factor in graph.factors}
    for edge in graph.edges:
        inbound[edge.dst] += edge.weight
    overloaded = sorted(
        f"{factor}={total:.2f}"
        for factor, total in sorted(inbound.items())
        if total > MAX_INBOUND_WEIGHT
    )
    if overloaded:
        violations.append(
            Violation(
                rule="edge_sanity",
                message=(
                    f"total inbound edge weight exceeds {MAX_INBOUND_WEIGHT} on these factors, "
                    "which makes the dynamics explosive. Reduce weights or remove edges."
                ),
                subjects=tuple(overloaded),
            )
        )
    return violations


def _check_outcome_rule(graph: CausalGraph) -> list[Violation]:
    """§5.3: depends on >= 2 factors and is monotone in each.

    Monotonicity holds by construction for the rule form in ADR-0014, so this
    is a redundant numeric check -- deliberately. The redundancy is what would
    catch a future rule form added without re-reading the ADR, and the failure
    it guards against silently invalidates every forecast the rule produces.
    """
    rule = graph.outcome_rule
    violations: list[Violation] = []

    referenced = set(rule.factor_ids)
    if len(referenced) < 2:
        violations.append(
            Violation(
                rule="outcome_rule",
                message=(
                    f"outcome rule depends on {len(referenced)} distinct factor(s); it must "
                    "depend on at least 2, or the simulation reduces to a random walk."
                ),
                subjects=tuple(sorted(referenced)),
            )
        )
        return violations

    others = [factor.id for factor in graph.factors]
    for term in rule.terms:
        expected = 1 if term.weight > 0 else -1
        for hold in _MONOTONE_HOLD_POINTS:
            state = {factor_id: hold for factor_id in others}
            previous: float | None = None
            for value in _MONOTONE_SWEEP:
                state[term.factor_id] = value
                score = rule(state)
                if previous is not None:
                    delta = score - previous
                    # Exactly-flat steps are permitted only where the logistic
                    # has saturated to the limit of float resolution; a sign
                    # reversal never is.
                    if delta * expected < 0.0:
                        violations.append(
                            Violation(
                                rule="outcome_rule",
                                message=(
                                    f"outcome rule is not monotone in {term.factor_id!r}: "
                                    f"weight {term.weight:+.3f} implies a "
                                    f"{'rising' if expected > 0 else 'falling'} score, but the "
                                    f"score moved {delta:+.6f} at state {value:.1f}."
                                ),
                                subjects=(term.factor_id,),
                            )
                        )
                        break
                previous = score
            else:
                continue
            break
    return violations


def _check_objective_independence(
    graph: CausalGraph, outcome_text: str, embed: EmbedFn
) -> list[Violation]:
    """§5.3: no actor objective may restate the outcome.

    Spec §5.2 names this the most common failure of the whole milestone --
    every actor given the goal "make the outcome happen". An actor whose
    objective *is* the outcome has no independent interest, so it cannot
    disagree with anyone, and a simulation of agents that cannot disagree
    measures nothing.
    """
    objectives = [actor.objective for actor in graph.actors]
    vectors = embed([outcome_text, *objectives])
    outcome_vector = vectors[0]

    offenders: list[str] = []
    for actor, vector in zip(graph.actors, vectors[1:], strict=True):
        similarity = cosine(outcome_vector, vector)
        if similarity > OBJECTIVE_SIMILARITY_MAX:
            offenders.append(f"{actor.id}={similarity:.3f}")

    if not offenders:
        return []
    return [
        Violation(
            rule="objective_independence",
            message=(
                f"actor objective(s) restate the outcome (cosine > "
                f"{OBJECTIVE_SIMILARITY_MAX}). Give each party an independent interest it "
                "would hold regardless of how this question resolves."
            ),
            subjects=tuple(offenders),
        )
    ]


def _check_factor_orthogonality(graph: CausalGraph, embed: EmbedFn) -> list[Violation]:
    """§5.3: pairwise factor-name similarity < 0.80.

    Prevents one real driver being split across three near-duplicate
    variables, which inflates the factor count and lets the same influence be
    counted several times in the inbound-weight budget.
    """
    names = [factor.name for factor in graph.factors]
    vectors = embed(names)

    offenders: list[str] = []
    for i in range(len(graph.factors)):
        for j in range(i + 1, len(graph.factors)):
            similarity = cosine(vectors[i], vectors[j])
            if similarity >= FACTOR_SIMILARITY_MAX:
                offenders.append(f"{graph.factors[i].id}~{graph.factors[j].id}={similarity:.3f}")

    if not offenders:
        return []
    return [
        Violation(
            rule="factor_orthogonality",
            message=(
                f"factor pair(s) are near-duplicates (cosine >= {FACTOR_SIMILARITY_MAX}). "
                "Merge them into one factor, or redefine them so they move independently."
            ),
            subjects=tuple(offenders),
        )
    ]


_STRUCTURAL_RULES = (
    "actor_count",
    "factor_count",
    "reachability",
    "edge_sanity",
    "outcome_rule",
)
_SEMANTIC_RULES = ("objective_independence", "factor_orthogonality")


@dataclass
class _Accumulator:
    violations: list[Violation] = field(default_factory=list)

    def extend(self, found: list[Violation]) -> None:
        self.violations.extend(found)


def validate(
    graph: CausalGraph,
    *,
    outcome_text: str,
    embed: EmbedFn | None = None,
) -> ValidationReport:
    """Apply every §5.3 rule. Pure given ``embed``.

    ``outcome_text`` is the scenario's question and resolution criterion -- the
    thing an actor objective must not restate. It is required rather than
    derived from the graph, because a graph that already contains the outcome
    text would be grading itself.

    Passing ``embed=None`` runs the five structural rules and reports the two
    semantic ones as skipped. ``ValidationReport.ok`` is then False, so a
    caller cannot mistake an unchecked graph for a valid one.
    """
    accumulator = _Accumulator()
    accumulator.extend(_check_actor_count(graph))
    accumulator.extend(_check_factor_count(graph))
    accumulator.extend(_check_reachability(graph))
    accumulator.extend(_check_edge_sanity(graph))
    accumulator.extend(_check_outcome_rule(graph))

    if embed is None:
        return ValidationReport(
            violations=tuple(accumulator.violations),
            skipped_rules=_SEMANTIC_RULES,
            checked_rules=_STRUCTURAL_RULES,
        )

    accumulator.extend(_check_objective_independence(graph, outcome_text, embed))
    accumulator.extend(_check_factor_orthogonality(graph, embed))
    return ValidationReport(
        violations=tuple(accumulator.violations),
        skipped_rules=(),
        checked_rules=_STRUCTURAL_RULES + _SEMANTIC_RULES,
    )
