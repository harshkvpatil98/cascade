# ADR-0012 — IVFFlat index lifecycle is an operational command, not a migration

- **Status:** accepted
- **Milestone:** M3
- **Amends:** spec §4.2

## Context

Spec §4.2 asks for "one IVFFlat index per partition
(`lists ≈ sqrt(rows_in_partition)`, `probes = 10`)", and §13 requires
forward-only SQL migrations with no ORM autogeneration because the partition
strategy is load-bearing. Read together, the obvious implementation is a
migration that creates every index.

That does not work, for a reason specific to IVFFlat.

An IVFFlat index is a k-means quantiser. Building one **assigns the rows
present at build time** to `lists` centroids; rows inserted afterwards are
added to whichever existing list is nearest. The index therefore encodes the
data distribution at the moment it was built. Two consequences:

1. **An index built on an empty partition is permanently degenerate.** With no
   rows, `lists` can only be 1, and every vector inserted later lands in that
   single list. The scan is then exhaustive regardless of `probes` — the index
   costs storage and maintenance and narrows nothing. A migration applied to a
   fresh database, which is the normal case in CI and on any new checkout,
   creates exactly this.
2. **Correct sizing is not knowable when the migration is authored.**
   `lists ≈ sqrt(rows_in_partition)` is a function of measured row counts. The
   corpus is built incrementally by a resumable pipeline (invariant 8) and is
   currently at 409,899 chunks against a 1.30M target, so every partition's row
   count is still moving. A number hardcoded today is wrong tomorrow, and
   wrong in the direction that quietly degrades recall rather than failing.

The distribution also turns out to be extremely skewed — 88.8% of chunks sit in
2016q3–2017q2 — so a single global `lists` would be wrong for nearly every
partition even if the totals were final. Per-partition sizing is doing real
work here, not satisfying a formula.

## Decision

Split the two concerns by what they depend on:

- **Migration 004 creates structure only** — `chronofence_search`,
  `chronofence_search_exact`, the `chronofence_partitions` view and the grants.
  All of it is data-independent, so it is deterministic, fast, and identical in
  every environment. It stays forward-only and hand-written, as §13 requires.

- **`cascade retrieval index` owns index lifecycle.** It reads exact row counts
  per partition, computes `lists = clamp(round(sqrt(rows)), 1, 2000)`, skips
  empty partitions entirely, and creates or rebuilds an index only where the
  configured `lists` differs from the target by more than
  `retrieval.index_rebuild_tolerance`. It is idempotent and reports what it
  did.

- **`cascade retrieval verify` fails on drift.** It exits 3 when a non-empty
  partition is unindexed, when `lists` has drifted outside tolerance, or when
  the pinned `probes` in the deployed function disagrees with
  `retrieval.ivfflat_probes`. Index staleness becomes a loud precondition
  failure rather than a silent recall regression.

`probes` stays in the migration, pinned at the function boundary, because it
is a constant from config rather than a function of the data — and because a
caller who forgets to set it would otherwise get pgvector's default of 1 with
no error.

## Why not build indexes concurrently in the migration

`CREATE INDEX CONCURRENTLY` cannot run inside a transaction block, and every
migration here is wrapped in `BEGIN`/`COMMIT` so that a partial migration
cannot be recorded as applied. Non-concurrent index builds take an
`ACCESS EXCLUSIVE` lock; against the 130k-row 2016q4 partition that is a
multi-second stall, and the migration runner's timeout is 300 s for the whole
file. The operational command has neither constraint: it runs outside a
transaction, one partition at a time, and can report progress.

## Cost of being wrong

If the index lifecycle command is never run, retrieval is *correct* and
*slow* — a sequential scan over the pruned partitions, which is exact. It
fails the p95 criterion, not the leakage criteria. That is the right direction
for this failure to point: a missing index cannot cause a leak, and
`cascade retrieval verify` refuses to pass while one is missing.

## Verified by

`tests/integration/test_chronofence.py` — asserts sizing against seeded
partitions of known row counts, that empty partitions are skipped, that a
second run is a no-op, and that `verify` exits 3 on an unindexed non-empty
partition and on a `probes` mismatch between the deployed function and config.
