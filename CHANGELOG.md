# Changelog

One entry per milestone, recording what shipped and what the acceptance numbers
**actually were**. The full record, including every defect found and fixed
along the way, is the build log in [`CLAUDE.md`](CLAUDE.md).

Versions track milestones rather than releases; nothing here is published to a
package index.

---

## M8 — Strata: determinism and provenance

Replay verification, provenance chains, and the cost-ledger gate.

- **25/25 runs replay byte-identically** across fresh interpreters under a
  differing `PYTHONHASHSEED`, in 24.3 s. A mismatch reports the first divergent
  step with both state hashes rather than only "the hashes differ".
- **Provenance chains complete in 0.27 s.** The specification's recursive CTE
  recurses over all six antecedents and expands 6^depth — measured on a real
  run it did not return. Its own example output is a single path
  ([ADR-0027](docs/adr/0027-provenance-chain-is-a-path.md)).
- **The cost ledger gate refuses to pass vacuously.** Two defects made it
  report "0.0000% — reconciles" over 452,328 calls costing nothing:
  `runs.llm_calls` counted decisions rather than model calls, and zero agreed
  with zero. Both fixed; the gate now has three verdicts.
- **Retrieval regression closed on recall.** recall@20 **0.8950 → 0.9675**
  (criterion met); p95 **176.00 → 90.92 ms** (criterion still missed, still
  reported). Two independent causes: a parameterised `LIMIT` in a
  non-inlinable function cost 4×, and IVFFlat does not scale past ~2M chunks
  ([ADR-0026](docs/adr/0026-hnsw-and-bounded-candidate-pool.md)). HNSW also
  retires the index-drift class that had interrupted M4 through M7 in turn.

## M7 — Assay: the evaluation harness

Scoring, the ablation grid, statistics, and the report artifact.

- Brier, Brier skill score, Murphy decomposition with its binning residual,
  clipped log loss, mid-rank AUC, Wilson intervals, ten-bin calibration,
  isotonic recalibration — all written from their definitions, because the
  declared dependency set has no scipy or scikit-learn.
- Paired bootstrap over **scenarios** (B = 10,000) with Holm–Bonferroni across
  the family, seeded from the study salt so intervals reproduce.
- **Two of four ablation factors were configuration fields nothing read.** Six
  of the twelve cells would have run as duplicates and the headline deltas
  would have come back as nulls with confidence intervals attached
  ([ADR-0025](docs/adr/0025-ablation-factors-a-and-c.md)).
- Climatology scored over all 180 scenarios: Brier **0.250000**, AUC
  **0.5000**, ECE **0.0000** — the exact closed form for a constant base-rate
  forecast, which is what makes it a validation of the metric stack.
- Report artifact writes all ten Appendix D files plus four SVG figures. A
  quantity that could not be produced is `null` and named, never substituted.

## M6 — Chorus: ensemble at scale

The fan-out, batch submission, and forecast aggregation.

- Wavefront batching: one batch per step across the whole wave, **24
  submissions** for the study instead of 864,000
  ([ADR-0020](docs/adr/0020-wavefront-batching.md)).
- Hartigan's dip test implemented from its definition and validated against
  closed-form cases rather than transcribed
  ([ADR-0021](docs/adr/0021-dip-test-from-definition.md)).
- Resumable by set difference: kill the pool mid-phase and it resumes with
  exactly the runs that did not finish, adding zero duplicate events.

## M5 — Loom + Aperture

The simulation kernel and the information-asymmetry layer.

- Four arbiter property tests pass under Hypothesis: bounded, conserving,
  permutation-invariant, monotone.
- One scenario runs 24 steps end to end; `WorldState` round-trips exactly,
  byte-identically across processes.
- The whole step's randomness is drawn at once, so an ablation runs the
  identical stream through a different policy
  ([ADR-0015](docs/adr/0015-run-rng-draw-plan.md)).

## M4 — Lathe: the decomposition compiler

Draft → critique → repair → validate, into a typed `CausalGraph`.

- All six validator rules, with `OutcomeRule` monotone by construction
  ([ADR-0014](docs/adr/0014-outcome-rule-and-utility-term.md)).
- Acceptance numbers need 540 Sonnet calls and remain **unmeasured**; the
  pipeline is verified end to end against a live database with only the model
  stubbed.

## M3 — Chronofence: time-locked retrieval

- p95 **13.43 ms**, recall@20 **0.9372** at 409,899 chunks. *(Both superseded
  at M8 — see above.)*
- **0 of 500** planted post-cutoff documents retrieved across all 180 cutoffs,
  with a positive control asserting the poison *is* reachable without the time
  filter.
- Date-monotonicity property test over the full retrieval trace.

## M2 — Evidence corpus

- **1,950,912 chunks** across **492,270 documents**; zero NULL, future or naive
  dates over the full table; 100% embedding coverage.
- Ingest order turned out to be a correctness concern: chronological units left
  a 1.76M-chunk corpus with 99.6% of its evidence before 2018 against 169 of
  180 cutoffs after 2024 ([ADR-0023](docs/adr/0023-corpus-unit-granularity-and-demand-ordering.md)).

## M1 — Scenario registry

- Exactly **180** scenarios, YES rate **0.5000**, no domain above **25.0%**.
- `manifest.sha256` sealed and asserted by every later phase; a mutated label
  is caught in both directions.
- `cascade_sim` gets `InsufficientPrivilege` on `scenario_labels` — the
  separation is a grant, not a convention.

## M0 — Foundations

- Repo scaffold, typed settings, the single LLM call site with
  record/replay/live, exact-`Decimal` cost metering with per-phase ceilings,
  Langfuse tracing that degrades to a no-op.
- Record → replay round trip makes **zero** network calls, asserted against a
  transport that raises on any request.
