# ADR-0013 — `ivfflat.probes` is 40, not 10, because partitioning fans the query out

- **Status:** accepted
- **Milestone:** M3
- **Amends:** spec §4.2

## Context

Spec §4.2 specifies one IVFFlat index per partition with
`lists ≈ sqrt(rows_in_partition)` and `probes = 10`. Both numbers were adopted
verbatim in migration 004 and `configs/base.yaml`.

`probes = 10` is a sound default **for a single index**. It is not the same
setting once the same query is answered by 36 of them.

`chunks` is partitioned by `published_at` (migration 003, quarterly per
ADR-0004). A time-locked query is a `Merge Append` over every pre-cutoff
partition, and `ivfflat.probes` applies *per index scan*. Each partition
independently returns an approximate top-k from `probes` of its own lists, and
the global top-20 is merged from those partial answers. A chunk that belongs
in the true top-20 is missed whenever its own partition failed to probe the
list holding it — so the per-partition recall required to hit a given *global*
recall rises with the number of partitions.

The corpus makes this concrete. It is heavily skewed: 88.8% of the 409,899
chunks sit in 2016q3–2017q2, where `lists` is 210–361. At `probes = 10` the
largest partition scans 10/361 ≈ 2.8% of its vectors.

## Measurement

80 queries from the bench distribution, k = 20, exact ground truth from
`chronofence_search_exact`:

| probes | recall@20 | p50 | p95 |
|---:|---:|---:|---:|
| 10 | 0.8331 | 3.74 ms | 4.44 ms |
| 20 | 0.9012 | 6.18 ms | 7.20 ms |
| **40** | **0.9450** | **10.04 ms** | **11.86 ms** |
| 80 | 0.9662 | 19.58 ms | 21.55 ms |
| 160 | 0.9812 | 35.54 ms | 38.62 ms |

`lists` was varied too, in case the sizing rule rather than `probes` was the
lever. It is not — the spec's rule is the best of the four tried, at every
recall level that clears the criterion:

| lists | probes | recall@20 | p95 |
|---|---:|---:|---:|
| `sqrt(n)` | 40 | 0.9450 | 11.86 ms |
| `sqrt(n)/2` | 20 | 0.9325 | 12.99 ms |
| `sqrt(n)/4` | 15 | 0.9319 | 15.52 ms |
| `sqrt(n)/8` | 8 | 0.9150 | 15.96 ms |

The intuition that fewer, larger lists would buy the same recall for fewer
index seeks is measurably wrong here: coarser quantisation loses more recall
than the saved seeks buy back.

## Decision

Keep `lists = round(sqrt(rows_in_partition))` exactly as specified. Raise
`probes` from 10 to **40**, pinned at the function boundary by migration 007
and mirrored in `retrieval.ivfflat_probes`.

40 is the smallest value on the measured curve that clears recall@20 > 0.92.
Confirmed on the full bench path with the complete row payload, 400 queries:
**p95 12.97 ms, p99 14.67 ms, recall@20 0.9350.**

`chronofence_search_exact` keeps `probes = 32768` (pgvector's maximum, and its
maximum `lists`), which is what guarantees it visits every list and is
therefore exhaustive.

## Cost of being wrong

The margins are real but not generous: 2.03 ms of latency headroom and 0.015
of recall. Two things move them.

**Corpus growth is the risk that matters.** The corpus is at 31.5% of the
1.30M target. With `lists = sqrt(n)`, the fraction of a partition scanned at
fixed `probes` is `probes/sqrt(n)`, which *shrinks* as a partition grows —
so recall drifts down as ingest continues, silently. This is not a setting
that can be tuned once. `cascade retrieval bench` exits 3 when either
criterion is missed and must be re-run after any material ingest; ADR-0012's
`cascade retrieval index` keeps `lists` correct as rows arrive, but nothing
keeps `probes` correct except re-measuring.

**If growth breaks the budget**, the levers in preference order are: coarser
partitions (annual rather than quarterly, reducing fan-out — the same argument
ADR-0004 used to move monthly to quarterly); then HNSW, which offers a better
recall/latency frontier than IVFFlat but is a substitution requiring its own
ADR and a rebuild of every index.

## Verified by

`cascade retrieval bench`, which reports p50/p95/p99, the full histogram and
recall@20 in one report and exits 3 when either criterion is missed —
`tests/integration/test_chronofence.py` asserts the deployed function's pinned
probes equals `retrieval.ivfflat_probes`, so config and schema cannot drift
apart without a test failing.
