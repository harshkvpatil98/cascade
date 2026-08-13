# ADR-0004 — `chunks` partitions start quarterly, not monthly

- **Status:** accepted
- **Milestone:** M3
- **Amends:** spec §4.2

## Context

Spec §4.2 specifies `chunks` partitioned by `RANGE (published_at)` at
**monthly** granularity with one IVFFlat index per partition. The corpus spans
2016–2024: **108 monthly partitions**.

Partition pruning is the mechanism — the planner drops post-cutoff partitions
before touching a vector, so the time lock is structural rather than a
predicate someone can forget. That part is right and is not in question here.

The cost is on the *other* side of the cutoff. A scenario with a late-2024
cutoff must scan every partition *before* it: ~100 separate IVFFlat index
scans, each with its own probe cost, all merged and re-sorted to produce the
global top-k. Against a **15 ms p95** budget that is roughly 0.15 ms per
partition for scan, heap fetch and merge. IVFFlat with `probes = 10` does not
do that.

There is a second effect. `lists ≈ sqrt(rows_in_partition)`: 1.3M chunks over
108 partitions is ~12k rows each, so `lists ≈ 110` and `probes = 10` reads
~9% of each partition. The index barely narrows anything, and recall@20
degrades at the same time as latency — the worst of both.

## Decision

Start at **quarterly** granularity: 36 partitions over 2016–2024, ~36k rows
each, `lists ≈ 190`.

`retrieval.partition_granularity` is config (`month | quarter`) so the M3
bench can measure both rather than argue about them. Quarterly is the
*starting point*, not the answer; the acceptance criteria are p95 < 15 ms and
recall@20 > 0.92, and whichever granularity meets both wins.

## Cost of being wrong

Quarterly widens the pruning boundary: a cutoff mid-quarter leaves its own
quarter partially post-cutoff, so the residual `published_at < as_of`
predicate inside `chronofence_search` does real filtering rather than none.
**That predicate is required either way** — it is what makes the lock correct.
Partitioning is a performance mechanism layered on top of a correct filter,
never a substitute for it. A leak here would be a bug in the function, not in
the granularity.

## Verified by

`cascade/retrieval/bench.py` — 10,000 queries across the full cutoff range,
reporting p50/p95/p99 **and** recall@20 versus exact search on a 5% sample. A
latency number without a recall number is meaningless.
