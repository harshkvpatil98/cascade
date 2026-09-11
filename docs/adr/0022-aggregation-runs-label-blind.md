# ADR-0022 — Aggregation runs as `cascade_sim`, so it cannot see an outcome

- **Status:** accepted
- **Milestone:** M6
- **Completes:** spec §2.2 (PHASE 3 AGGREGATE / PHASE 4 EVALUATE) and invariant 2

## Context

§2.2 separates collapsing replicates into forecasts (PHASE 3) from scoring
those forecasts against labels (PHASE 4). The spec does not say what enforces
the separation, and nothing about the code makes it obvious: `chorus.collapse`
is arithmetic over outcome scores, and it would run perfectly well as any role.

Invariant 2 already says the simulation never reads `scenario_labels`, enforced
by a Postgres grant rather than by code review (ADR-0005). The question is
which side of that line aggregation sits on.

## Decision

**PHASE 3 runs under `cascade_sim`.** The collapse reads `runs` and writes
`forecasts` as the role that has no grant on `scenario_labels`, so a forecast
cannot be conditioned on the answer it is about to be scored against — not
because nobody wrote that code, but because the query would fail.

That requires two grants that look like exceptions and are not:

- `cascade_sim` gets **UPDATE** on `forecasts`. A forecast is derived data:
  adding replicates and re-collapsing must replace the row, not accumulate a
  second one. Invariant 6 is about the event log, and `events` remains
  INSERT-only for every role.
- `cascade_eval` gets SELECT only. Scoring reads forecasts and labels together;
  it has no reason to write either.

The alternative — aggregating as `cascade_eval` because it is "analysis" —
would put the labels within reach of the step that decides what number to
report, which is exactly the hole the grant separation exists to close.

## Consequence

`cascade ensemble collapse` connects as `sim` and will fail loudly if a future
change makes the collapse want a label. That failure is the feature: it means
the phase boundary is a property of the deployment, not of the current
implementation.

## Verified by

`tests/integration/test_ensemble_store.py::test_the_simulation_role_can_collapse_but_never_sees_an_outcome`
— the same connection reads the forecast it just wrote and gets
`InsufficientPrivilege` on `scenario_labels`.
