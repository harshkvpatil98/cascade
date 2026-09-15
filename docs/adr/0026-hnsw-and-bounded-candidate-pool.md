# ADR-0026 — HNSW replaces IVFFlat, and the caller's `k` leaves the plan

**Status:** accepted · **Milestone:** M8 · **Amends ADR-0012, ADR-0013**

## Context

M3's acceptance figures were p95 **13.43 ms** and recall@20 **0.9372**, measured
against a 409,899-chunk corpus across 37 partitions. Re-measured at M7 against
**1,950,912 chunks across 47 partitions**, the same code produced p95 **176.00
ms** and recall@20 **0.8950** — 11.7× over the latency budget and below the
recall floor. M7 recorded the regression, named the levers and deliberately
tuned nothing, because `probes` trades the two criteria against each other and
moving it to make either number pass is the steering §1 forbids.

Measuring instead of tuning found **two independent defects**, one of which had
nothing to do with the index at all.

### Defect 1 — the parameterised `LIMIT k` cost 4×

`chronofence_search` is `SECURITY DEFINER` with a pinned `search_path`
(ADR-0002 requires both). Each of those independently disqualifies a SQL
function from inlining, which the M3 entry already noted when it fixed the
provenance join. What it did not notice is what non-inlining does to the *main*
scan: the body runs as its own statement with `k` as a runtime parameter, so
the planner cannot know how far the Merge Append across 47 partitions will be
driven, and drives every partition's index scan far past what the caller asked
for.

Measured — same body, same corpus, same `probes`, only the function attributes
varying:

| function attributes | latency |
|---|---|
| none (inlinable) | 36.2 ms |
| `SECURITY DEFINER` | 138.0 ms |
| `SET search_path` | 142.9 ms |
| `SET ivfflat.probes` | 142.5 ms |
| all three | 143.5 ms |

The attributes are not additive and not individually to blame: **any one of them
loses inlining, and losing inlining is the cost.** Holding the attributes fixed
and varying only the limit isolates it:

| body | latency |
|---|---|
| `LIMIT k` (parameter), `as_of` parameter | 138.5 ms |
| `LIMIT 20` (literal), `as_of` parameter | **35.4 ms** |
| `LIMIT k` (parameter), `as_of` literal | 1281.2 ms |
| `LIMIT 20` (literal), `as_of` literal | 35.2 ms |

### Defect 2 — IVFFlat does not scale to this corpus

IVFFlat with `lists = sqrt(rows)` reads about `probes × sqrt(n_p)` rows per
partition, so a query spanning the corpus reads about `probes × Σ_p sqrt(n_p)`.
That sum measured **5,727** against roughly 1,840 at M3's distribution — about
229,000 rows scanned per query against 74,000. The growth is structural, not
incidental: `√` is concave, so spreading a corpus over more populated
partitions raises the sum even at constant total rows, and M7's ingest-order
fix deliberately populated 2018–2026.

Measured on `chunks_2017q4` (605,083 rows), same query, same warm cache:

| index | latency | recall@20 |
|---|---|---|
| IVFFlat, probes = 40 | 6.57 ms | — |
| HNSW, ef_search = 20 | 0.63 ms | 0.9500 |
| HNSW, ef_search = 40 | **0.60 ms** | **1.0000** |
| HNSW, ef_search = 100 | 0.92 ms | 1.0000 |

## Decision

**1. The function draws a bounded candidate pool with a constant limit.**
`chronofence_search` selects `LIMIT 200` — a literal — and applies the caller's
`k` to that pool. For any `k ≤ 200` the result is identical: the top-*k* of the
top-*N* under one total order is the top-*k*. `retrieval.max_k` is 200 and the
client refuses a larger `k` with a message naming the constant, so the
equivalence is enforced on both sides rather than assumed on one.

This is also *more* deterministic than what it replaces. The old body applied
`ORDER BY distance, chunk_id` after taking `k` rows by distance alone, so a tie
at the k-th position was broken arbitrarily; the pool applies the total order
over a set larger than `k`. M8's replay hash depends on that ordering being a
function of the data.

**2. HNSW replaces IVFFlat**, with pgvector's default `m = 16` and
`ef_construction = 64`, and `hnsw.ef_search = 100` pinned into the function by
migration 014 the way `ivfflat.probes` was by 007. 100 rather than the 40 that
measured perfect recall on one partition, because `ef_search` bounds how many
candidates a *single* partition can contribute to the merge and
`retrieval.k_compiler` is 60.

This is not a stack substitution. HNSW is a feature of pgvector 0.8, which is
the pinned dependency; ADR-0013 changed `probes` within the same extension for
the same kind of measured reason.

**3. The rebuild-on-drift pass is retired.** `lists` is a function of the row
count, so an IVFFlat index silently degraded as its partition grew and needed a
periodic rebuild. That drift interrupted **M4, M5, M6 and M7** in turn — four
consecutive milestones, each losing an integration test to it. HNSW has no
row-dependent build parameter: an index that exists stays correct as rows
arrive. `cascade retrieval index` now plans `rebuild` only when `hnsw_m` or
`hnsw_ef_construction` changes, which is an operator action rather than a
background process.

## Consequences

- Index names carry the access method (`..._embedding_hnsw_idx`), so an
  IVFFlat index left by an older database is not mistaken for the current one;
  the partition simply plans as `create`. `cascade retrieval index
  --drop-legacy` removes them **after** the build, because dropping first would
  leave the corpus unindexed for its duration.
- Builds are slower than IVFFlat's — 605k rows took 95 s single-threaded
  against a few seconds — and run with `max_parallel_maintenance_workers = 0`
  on purpose: parallel workers size a shared-memory segment from
  `maintenance_work_mem`, and this container's `/dev/shm` is 1 GB, so
  requesting them turns a working build into `could not resize shared memory
  segment`.
- HNSW indexes are larger than IVFFlat's and are maintained on insert, so
  corpus ingest pays a little more per row. The corpus is loaded once.
- **ADR-0013's `probes = 40` is superseded**, and with it the recall curve that
  justified it. ADR-0012's argument for why the index lifecycle is a command
  rather than a migration survives unchanged; only its *sizing* half is retired.
