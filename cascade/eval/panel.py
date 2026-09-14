"""The ``causal_decomposition = off`` arm of the grid: a generic persona panel.

Appendix C's factor A reads "off => generic persona panel", and §10.2 describes
the corresponding baseline as "N generic personas debating, no causal graph, no
asymmetry". Before this module existed, ``flags.causal_decomposition`` was a
configuration field nothing read: cells C09-C12 would have run the *compiled*
graph and produced forecasts indistinguishable from C05-C08, and the headline
"+0.035 from decomposition" would have been a measurement of nothing. Factor C
had the same defect and is fixed in the same milestone (see ADR-0025).

**Why a panel graph rather than a separate debate engine.** An ablation has to
vary one factor. Running the A=off arm through a different execution engine
would confound "no causal decomposition" with "different kernel, different
step count, different arbiter, different event log" -- and the difference
measured would be uninterpretable. So the panel is expressed as a
:class:`CausalGraph` that the *same* kernel runs: same 24 steps, same arbiter,
same seeded draw plan, same append-only log. The only thing that changes is
where the structure came from.

**What makes it generic.** Nothing in the panel is derived from the scenario's
causal content. Actors are forecasting archetypes with fixed objectives and
risk postures; factors are four content-free drivers; the outcome rule is a
fixed signed logistic. The scenario id enters only as the graph's key and as
the seed for the small amount of variation that keeps fourteen actors from
being fourteen copies of one actor -- a panel of identical agents is not a
panel, it is one agent sampled fourteen times, which is the self-consistency
baseline wearing a different name.

Pure: no I/O, no clock, no RNG from global state. The per-scenario variation
is a keyed hash, so a panel is byte-identical across processes and runs.
"""

from __future__ import annotations

import hashlib

from cascade.decompose.schema import (
    MAX_ACTORS,
    MIN_ACTORS,
    Actor,
    CausalGraph,
    Edge,
    Factor,
    OutcomeRule,
    OutcomeTerm,
    RiskPosture,
    UtilityTerm,
)

__all__ = [
    "PANEL_ARCHETYPES",
    "PANEL_FACTORS",
    "PANEL_FACTOR_SKELETON",
    "panel_graph",
]

# Four content-free drivers. Deliberately not "the causes of this question":
# that is what decomposition produces, and this is the arm that does not have
# it. They are named so an agent prompt reads coherently, and nothing about
# them is scenario-specific.
PANEL_FACTORS: tuple[tuple[str, str, float, float, float], ...] = (
    ("momentum", "Momentum toward the outcome", 0.5, 0.02, 0.55),
    ("resistance", "Resistance to the outcome", 0.5, 0.02, 0.55),
    ("external_pressure", "External pressure on the parties", 0.4, 0.03, 0.45),
    ("commitment", "Public commitment already made", 0.5, 0.015, 0.7),
)

# Forecasting archetypes: (suffix, display name, objective, risk posture, lean).
# `lean` is the sign of the archetype's utility on `momentum`; the panel is
# balanced so the ensemble's prior is not built in.
PANEL_ARCHETYPES: tuple[tuple[str, str, str, RiskPosture, int], ...] = (
    ("advocate", "Advocate", "Argue that the outcome occurs", "seeking", 1),
    ("skeptic", "Skeptic", "Argue that the outcome does not occur", "averse", -1),
    (
        "base_rater",
        "Base-rate analyst",
        "Anchor on how often this kind of thing happens",
        "neutral",
        -1,
    ),
    ("insider", "Domain specialist", "Reason from how this domain usually behaves", "neutral", 1),
    ("contrarian", "Contrarian", "Look for what the consensus is missing", "seeking", -1),
    (
        "institutionalist",
        "Institutionalist",
        "Reason from process, rules and precedent",
        "averse",
        1,
    ),
    ("quant", "Quantitative analyst", "Reason from measurable indicators only", "neutral", 1),
    ("journalist", "Reporter", "Reason from what has been publicly stated", "neutral", -1),
    ("operator", "Operator", "Reason from what the parties can actually execute", "neutral", 1),
    ("regulator", "Regulator", "Reason from what would be permitted", "averse", -1),
    ("investor", "Investor", "Reason from where capital is being committed", "seeking", 1),
    ("historian", "Historian", "Reason from the closest historical analogue", "neutral", -1),
    ("forecaster", "Generalist forecaster", "Aggregate the other views and adjust", "neutral", 1),
    ("devils_advocate", "Devil's advocate", "Stress the strongest case against", "averse", -1),
    ("pragmatist", "Pragmatist", "Reason from incentives as they stand", "neutral", 1),
    ("theorist", "Theorist", "Reason from the mechanism that would have to hold", "neutral", -1),
    ("insurer", "Risk analyst", "Price the downside of being wrong", "averse", -1),
    ("strategist", "Strategist", "Reason from what each party would do next", "seeking", 1),
    ("auditor", "Auditor", "Check whether the stated criterion is actually met", "neutral", -1),
    ("outsider", "Outsider", "Reason with no stake in the question", "neutral", 1),
)


# A minimal factor-to-factor skeleton, so the panel's world is not four
# independent random walks wearing a graph's clothes. Fixed weights: this is
# the part of the panel that does not scale with its size.
PANEL_FACTOR_SKELETON: tuple[tuple[str, str, int, float, int], ...] = (
    ("external_pressure", "momentum", 1, 0.3, 1),
    ("commitment", "resistance", -1, 0.25, 1),
    ("momentum", "resistance", -1, 0.2, 2),
)

# How much of §5.3's per-factor inbound cap the panelists may spend. The
# remainder covers the skeleton above (at most 0.45 on one factor) and leaves
# the graph comfortably inside the cap at every panel size, rather than one
# jitter draw away from it.
_ACTOR_INBOUND_BUDGET = 1.8


def _jitter(scenario_id: str, actor_id: str, span: float) -> float:
    """A deterministic offset in ``[-span, span]``, keyed by the pair.

    Not an RNG: the run's RNG is seeded once and drawn in a fixed order
    (invariant 4), and constructing a graph is not one of those draws. A keyed
    hash gives per-actor variation that is byte-identical across processes,
    which is what §8.4's replay requires of everything upstream of the seed.
    """
    digest = hashlib.blake2b(
        f"{scenario_id}|{actor_id}".encode(), digest_size=8, key=b"cascade-panel"
    ).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    return (unit * 2.0 - 1.0) * span


def panel_graph(scenario_id: str, *, actors: int) -> CausalGraph:
    """Build the generic panel for one scenario. Pure and deterministic.

    ``actors`` is the panel size. It is configuration rather than a constant so
    it can be set to the decomposition arm's *measured* mean actor count --
    otherwise the A=off arm differs from A=on in panel size as well as in
    structure, and the delta the study reports would contain both.

    The graph satisfies every §5.1 bound and §5.3 rule that applies to it: at
    least two distinct factors in the outcome rule, no zero weights, every
    reference resolving, and signed weights that make the rule monotone by
    construction (ADR-0014).
    """
    if not MIN_ACTORS <= actors <= MAX_ACTORS:
        raise ValueError(
            f"panel size must lie in [{MIN_ACTORS}, {MAX_ACTORS}] to satisfy the "
            f"CausalGraph bounds of spec §5.1, got {actors}"
        )
    if actors > len(PANEL_ARCHETYPES):
        raise ValueError(
            f"only {len(PANEL_ARCHETYPES)} archetypes are defined; a panel of "
            f"{actors} would have to repeat one, and two identical personas are "
            "one persona sampled twice"
        )

    # §5.3 caps total inbound edge weight per factor at MAX_INBOUND_WEIGHT, and
    # every panelist adds two inbound edges. A fixed per-actor weight therefore
    # passes at eight panelists and makes the dynamics explosive at twenty --
    # so the weight is a function of the panel size, leaving headroom for the
    # factor-to-factor skeleton below. Verified for every size in [8, 20] by
    # running the real §5.3 validator over the built graph in the test suite.
    per_factor_inbound = 1.5 * actors / len(PANEL_FACTORS)
    base_weight = min(1.0, _ACTOR_INBOUND_BUDGET / per_factor_inbound)
    jitter_span = base_weight * 0.25

    factors = tuple(
        Factor(
            id=factor_id,
            name=name,
            state=state,
            volatility=volatility,
            inertia=inertia,
            observable_by_default=True,
        )
        for factor_id, name, state, volatility, inertia in PANEL_FACTORS
    )

    built: list[Actor] = []
    edges: list[Edge] = []
    for index in range(actors):
        suffix, name, objective, posture, lean = PANEL_ARCHETYPES[index]
        actor_id = f"panelist_{suffix}"
        # Each panelist holds a lever on two factors, rotating through the set
        # so no factor is unowned and no panelist owns everything.
        primary = PANEL_FACTORS[index % len(PANEL_FACTORS)][0]
        secondary = PANEL_FACTORS[(index + 1) % len(PANEL_FACTORS)][0]
        weight = round(base_weight + _jitter(scenario_id, actor_id, jitter_span), 4)
        built.append(
            Actor(
                id=actor_id,
                name=name,
                objective=objective,
                utility_terms=(
                    UtilityTerm(factor_id=primary, weight=round(lean * 0.6, 4)),
                    UtilityTerm(factor_id=secondary, weight=round(-lean * 0.3, 4)),
                ),
                resources={"attention": 0.5},
                constraints=(
                    "You have no privileged information; you reason from the "
                    "question and from what other panelists say.",
                ),
                risk_posture=posture,
                initial_beliefs={},
            )
        )
        edges.append(
            Edge(src=actor_id, dst=primary, sign=1 if lean > 0 else -1, weight=weight, lag=0)
        )
        edges.append(
            Edge(
                src=actor_id,
                dst=secondary,
                sign=-1 if lean > 0 else 1,
                weight=round(weight / 2.0, 4),
                lag=1,
            )
        )

    edges.extend(
        Edge(src=src, dst=dst, sign=sign, weight=weight, lag=lag)
        for src, dst, sign, weight, lag in PANEL_FACTOR_SKELETON
    )

    return CausalGraph(
        scenario_id=scenario_id,
        actors=tuple(built),
        factors=factors,
        edges=tuple(edges),
        outcome_rule=OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="momentum", weight=1.0),
                OutcomeTerm(factor_id="resistance", weight=-1.0),
                OutcomeTerm(factor_id="commitment", weight=0.5),
            ),
            threshold=0.5,
            steepness=6.0,
        ),
    )
