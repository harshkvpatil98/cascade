# Architecture decision records

Twenty-seven records. Each states the decision, the evidence behind it, and
what it costs. They exist because this project's specification is detailed
enough to be wrong in specific, checkable ways — and when it was, the record
says so with the measurement that showed it.

Fifteen correct a defect in the specification. Three correct a defect found in
this build. The rest record a choice the specification left open.

| # | Decision | Milestone |
|---|---|---|
| [0001](0001-prompt-cache-minimum-prefix.md) | The static prompt prefix must clear the model's 4,096-token cache floor; the spec's 1,900-token prefix silently does not cache | M0 |
| [0002](0002-chronofence-security-definer.md) | `chronofence_search` needs `SECURITY DEFINER` and a pinned `search_path`; as specified the app role gets a permission error on its own function | M3 |
| [0003](0003-worldstate-relation-keys.md) | `WorldState.relations` uses a canonical string key — a tuple key cannot round-trip through JSON | M5 |
| [0004](0004-quarterly-chunk-partitions.md) | `chunks` partitions are quarterly, not monthly — ~108 monthly partitions is ~100 index scans against a 15 ms budget | M3 |
| [0005](0005-role-grants-deny-by-default.md) | Role grants deny by default; the spec's `REVOKE … FROM cascade_sim` is a no-op | M0 |
| [0006](0006-langfuse-v2.md) | Langfuse pinned to v2 — v3 needs ClickHouse, Redis and MinIO | M0 |
| [0007](0007-cache-key-excludes-cache-control.md) | The cache key excludes `cache_control` markers, so prompt-cache tuning is a pure cost change | M0 |
| [0008](0008-scenario-constraint-precedence.md) | Constraint precedence when the pool cannot satisfy every inclusion rule: eligibility > base rate > domain cap > N | M1 |
| [0009](0009-party-screen-and-metaculus.md) | `named_parties` counts recognised institutional actors, not proper nouns | M1 |
| [0010](0010-corpus-dates-and-window.md) | Corpus dates come from the document, never the crawl; Wikipedia is anchored per scenario | M2 |
| [0011](0011-chunk-cap-verified.md) | The 512-token cap is verified per chunk, not estimated | M2 |
| [0012](0012-ivfflat-index-lifecycle.md) | Index lifecycle is an operational command, not a migration | M3 |
| [0013](0013-ivfflat-probes-under-partition-fanout.md) | `probes` is 40, not 10: partitioning applies probes per scan *(superseded by 0026)* | M3 |
| [0014](0014-outcome-rule-and-utility-term.md) | `OutcomeRule` is monotone by construction, so §5.3's rule holds for any rule that parses | M4 |
| [0015](0015-run-rng-draw-plan.md) | The run RNG draws a whole step at once, so an ablation runs the identical stream through a different policy | M5 |
| [0016](0016-per-target-action-visibility.md) | Action visibility is per target; one enum per policy cannot express the spec's own derivation rule | M5 |
| [0017](0017-worldstate-scheduler-and-pool-ledgers.md) | `WorldState` carries the alliance pool ledger and the scheduler's per-actor history | M5 |
| [0018](0018-action-admissibility.md) | An action the actor cannot take is coerced to WAIT and the reason recorded on the event | M5 |
| [0019](0019-evidence-in-the-cached-prefix.md) | Evidence is retrieved once per (scenario, actor), not per step — 4.2M searches become 2,520 | M5 |
| [0020](0020-wavefront-batching.md) | The fan-out advances runs in lockstep so one batch serves a whole step — 24 submissions instead of 864,000 | M6 |
| [0021](0021-dip-test-from-definition.md) | The dip test is implemented from its definition and validated against closed forms, not transcribed | M6 |
| [0022](0022-aggregation-runs-label-blind.md) | Aggregation runs as `cascade_sim`, so a forecast cannot be conditioned on the outcome it will be scored against | M6 |
| [0023](0023-corpus-unit-granularity-and-demand-ordering.md) | **Found here.** Ingest order is a correctness concern: chronological units left a 1.76M-chunk corpus covering the wrong decade | M2 |
| [0024](0024-report-figures-are-svg.md) | Report figures are SVG from the pinned stack; the declared dependency set has no plotting library | M7 |
| [0025](0025-ablation-factors-a-and-c.md) | **Found here.** Two of four ablation factors were configuration fields nothing read | M7 |
| [0026](0026-hnsw-and-bounded-candidate-pool.md) | **Found here.** HNSW replaces IVFFlat, and a parameterised `LIMIT` in a non-inlinable function cost 4× | M8 |
| [0027](0027-provenance-chain-is-a-path.md) | The provenance walk is a path, not an ancestor set: the spec's CTE expands 6^depth and does not return | M8 |
