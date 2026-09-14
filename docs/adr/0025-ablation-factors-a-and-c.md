# ADR-0025 — Ablation factors A (decomposition) and C (grounding) are mechanisms, not configuration fields

**Status:** accepted · **Milestone:** M7

## Context

Appendix C's grid turns on four factors:

| | factor | values |
|---|---|---|
| A | `causal_decomposition` | on / off — "off => generic persona panel" |
| B | `information_asymmetry` | on / off, requires A=on |
| C | `grounding` | `chronofence` / `parametric_only` |
| D | `ensemble` | n=200 / n=1 |

`configs/ablations/C01.yaml … C12.yaml` have encoded all four since M0, and
`tests/unit/test_ablation_cells.py` asserts every overlay against Appendix C.

Measured at the start of M7, by grepping the package for each flag:

- **B** was read: `derive_policies(graph, aperture, asymmetry=flags.information_asymmetry)`.
- **D** was read: `ensemble.replicates` drives the fan-out's replicate count.
- **A** appeared **only** in `cascade/config.py`. Nothing else in the package
  referenced it.
- **C** appeared **only** in `cascade/config.py`. Nothing else referenced it.

So the twelve-cell grid would have executed as **four distinct
configurations**. C09–C12 would have run the *compiled* graph and produced
forecasts identical to C05–C08, and every `parametric_only` cell would have run
with full Chronofence retrieval. The headline ablation deltas — `+0.035` for
decomposition and the grounding contribution — would have been measurements of
nothing, reported with confidence intervals and Holm-adjusted p-values, and
nothing downstream could have detected it. The paired bootstrap on two
identical columns returns `[0, 0]`, which reads as a precise null rather than
as a broken switch.

This was not a missing feature so much as a missing *mechanism* behind a
feature that had been configured, documented, tested and shipped since M0.

## Decision

**Factor A off runs a generic persona panel through the same kernel.**
`cascade/eval/panel.py` builds a `CausalGraph` deterministically from the
scenario id alone: N forecasting archetypes (advocate, skeptic, base-rate
analyst, …) over four content-free drivers (`momentum`, `resistance`,
`external_pressure`, `commitment`) with a fixed signed-logistic outcome rule.
`_graph_for()` returns it whenever `flags.causal_decomposition` is false.

The alternative was a separate debate engine — N personas exchanging arguments
over R rounds, which is what "generic personas debating" most literally
describes. It was rejected because **an ablation has to vary one factor**. That
arm would have differed from the full system in its structure *and* its step
count *and* its arbiter *and* its event-log shape, and the resulting delta
would not have been attributable to decomposition. Running the panel through
the same 24-step kernel, the same deterministic arbiter and the same seeded
draw plan makes "where did the structure come from" the only thing that moves.

Panel size is `flags.panel_actors` (default 14), set to §3.1's design mean
actor count so the two arms do not differ in how many agents there are.

The panel satisfies every §5.3 structural rule at every size in [8, 20],
asserted by running the real validator over the built graph in the test suite.
Per-actor edge weight is a function of panel size for exactly that reason: a
fixed weight passes at eight panelists and exceeds §5.3's inbound-weight cap at
fourteen.

**Factor C off withholds evidence and says so.** Under `parametric_only` no
embedder is loaded and no Chronofence connection is opened, and the actor
prefix carries an explicit statement that retrieval was disabled. The
distinction matters: an empty evidence block otherwise reads as "retrieval ran
and found nothing admissible before the cutoff", which is a fact about the
world, when the truth is "retrieval was switched off for this cell", which is a
property of the exercise. An agent told the wrong one reasons differently about
its own ignorance.

**Factor C governs the agents' evidence, not the compiler's.** The 180 compiled
graphs are shared across all twelve cells. Recompiling them ungrounded would
spend the `compile` ceiling a second time and would change what the A=on arm
*is*, not just what it reads. Appendix C's own gloss for C03 — "isolates the
value of grounded retrieval" — is about what the simulation reads. This is an
interpretive choice with a budget consequence and it is stated in the report
rather than left to be inferred.

## Consequences

- The `parametric_only` cells lose prompt caching. The evidence block is what
  carries the static prefix over the provider's 4,096-token floor (ADR-0019),
  so those cells are legitimately more expensive per call. `_agent_policy`
  already measures and warns; the cost difference is reported, not hidden.
- C09–C12 no longer require a compiled graph, so the multi-agent baseline of
  §10.2 is runnable without the `compile` phase having succeeded.
- The panel graph's hash is stable and scenario-specific, so `runs.graph_sha256`
  identifies which arm produced a run without consulting the config.
- A regression test asserts that A and C are *read*: two configurations
  differing only in a flag must not produce identical wiring. A switch that is
  configured, documented and inert is the failure this ADR exists to record,
  and a test is the only thing that keeps it fixed.
