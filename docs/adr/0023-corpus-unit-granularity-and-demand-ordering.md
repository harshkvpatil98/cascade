# ADR-0023 — CC-NEWS units are one WARC file, and the queue is ordered by scenario demand

**Status:** accepted · **Milestone:** M2 (found at M7) · **Supersedes nothing**

## Context

At the close of M6 the corpus stood at 1,172,495 chunks and the build log
carried an open item: "the corpus is concentrated where the scenarios are not."
By the start of M7 ingest had taken it to **1,764,890 chunks — 136% of the
1.30M target — and the concentration had become worse, not better**:

| | measured |
|---|---|
| chunks published before 2018-04 | 1,757,140 of 1,764,890 (**99.6%**) |
| scenario cutoffs in 2024 or later | 169 of 180 (**94%**) |
| chunks in 2025q2–2026q3 | 2,054 |

Every M2 criterion held. Every M3 criterion held. The corpus was over its size
target, had zero bad dates, full embedding coverage, and time-locked retrieval
that provably never returns a post-cutoff document — over a body of evidence
that, for the scenarios the study actually forecasts, was six to eight years
stale. A nearest-neighbour search over 1.77M chunks of which 99.6% predate
2018 returns 2017 news for a 2026 question, correctly and quickly.

Two mechanisms produced this, and neither is a bug in any single function.

**Units were walked in key order, which for every date-keyed source is
chronological.** `pending = [unit for unit in units if unit not in done]`
preserves the generated order, and every source generates ascending months. A
multi-day ingest is always interrupted, so the corpus it leaves is complete at
the start of the window and empty at the end. The ingest was not behind
schedule; it was building the wrong corpus in the right order.

**A CC-NEWS unit was a whole month, and a month was marked done after a
bounded, shallow visit.** `ccnews_max_files` files of several hundred were
read, then the month was recorded `done` and became permanently unreachable —
completed units are skipped (invariant 8), so no later pass could deepen it.
Depth was therefore a property frozen at first visit rather than a coordinate
anything could schedule over.

`corpus.end_year` was also 2024 while cutoffs run to 2026-09, so the last two
years of the study had no source months generated at all.

## Decision

**1. A CC-NEWS unit is one WARC file: `YYYY/MM#k`.**

The file ordinal makes depth explicit. A month can now be visited repeatedly at
increasing depth, and resumability gets finer rather than coarser — an
interruption costs one file, not one month.

A month-level `done` row written before this change is read as covering
`#0 … #(LEGACY_MONTH_FILES-1)`, where `LEGACY_MONTH_FILES = 12` is the
`ccnews_max_files` in force when those rows were written. This is bookkeeping,
not a data rewrite: the stored row is untouched and read as what it meant. The
alternative — treating a legacy key as unrecognised — would re-fetch 18 months
of completed work.

**2. The pending queue is sorted `(depth, −demand, unit_key)`.**

*Depth-major* so the queue sweeps the whole span before deepening any month.
This is the property the change exists for: the prefix of an interrupted
ingest is now a thin layer over the entire study window rather than a complete
copy of its first quarter.

*Demand* is a pure function of the scenario cutoffs: a month earns weight from
every scenario whose cutoff falls within `coverage_lookback_months` after it,
decaying linearly with the gap. Months at or after a cutoff earn nothing from
it — they are inadmissible for that scenario by construction, and weighting
them would ask the ingest to fetch documents Chronofence will refuse to return.

*Key last* so the order is total and reproducible (invariant 7).

**3. `cascade corpus coverage` measures evidence at each scenario's own
cutoff** and exits 3 when a scenario falls below `corpus.coverage_min_chunks`
within its window. It reports chunks in window, chunks before cutoff, the most
recent admissible document, and staleness in days.

**4. `corpus.end_year` is 2026**, matching the measured cutoff span.

## Why this is not steering

§1 forbids building a harness that can be aimed at a target number. Ordering
ingest by cutoff proximity reads `scenarios.cutoff_ts` and nothing else. A
cutoff is not an outcome: it lives in the registry table the simulation role is
granted (invariant 2 fences `scenario_labels`, not `scenarios`), it is sealed
into `manifest.sha256` before any evidence is fetched, and it is fixed
independently of how the scenario resolved. The precedent is ADR-0010, which
already anchors each Wikipedia snapshot at its own scenario's cutoff for the
same reason.

What would be steering is selecting *which documents* to keep by how they bear
on an outcome. Nothing here touches document content; the ordering is over
calendar months.

## Consequences

- The documented M2 throughput improvement — several WARC files streamed
  concurrently *inside* one unit — is no longer available, because a unit is
  now one file. The overlap moves up a level: `_prefetch` fetches ahead by
  `corpus.fetch_workers` units while the CPU-bound stages drain, preserving
  submission order so units commit in exactly the sequence they would have run
  in sequentially. An interruption leaves the same state either way.
- Raising `ccnews_max_files` adds a later sweep rather than deepening months
  already visited, so depth is now a budget decision made per run.
- `cascade corpus verify` (size, dates, embeddings) and `cascade corpus
  coverage` (span) are separate gates and both must pass. The first cannot
  detect this failure and the second cannot detect an empty corpus.
- `coverage_min_chunks` and `coverage_lookback_months` are **this project's
  thresholds, not the spec's**. The spec states no coverage criterion. They are
  reported as measured values with the threshold named, never folded into an
  M2 acceptance claim.
