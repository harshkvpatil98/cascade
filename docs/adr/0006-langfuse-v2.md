# ADR-0006 — Langfuse pinned to v2

- **Status:** accepted
- **Milestone:** M0
- **Records a choice the spec left open**

## Context

Spec §11.3 requires self-hosted Langfuse and does not name a version. Langfuse
v3 is the current major line. Its self-hosted deployment requires, in addition
to Postgres: **ClickHouse** (event store), **Redis** (queue), and **S3 or
MinIO** (blob storage). Langfuse v2 self-hosts on Postgres alone.

## Decision

Pin `langfuse/langfuse:2` in `docker-compose.yml` and `langfuse>=2.50,<3` in
`pyproject.toml`. `cascade doctor` asserts the SDK major version, so a
transitive bump to v3 fails a check instead of failing a run.

## Rationale

- **Four services versus one.** The demo criterion at M9 is a clean clone to a
  rendered causal trace on a second machine. Three additional stateful
  services is a materially worse first-run experience, and this project has
  already lost a working tree to a full disk once.
- **Nothing in the spec needs v3.** The requirement is trace → span →
  generation with token counts and cost. v2 serves that. The v3 additions
  (high-cardinality analytics over hundreds of millions of events) address a
  scale this study does not reach: ~378k generations total.
- **Langfuse is not authoritative.** The cost ledger of record is the `runs`
  table in Postgres. Langfuse is reconciled *against* it at M8 with a 2%
  threshold. A telemetry backend that is by design never read back during a
  run does not warrant three extra services.

## Consequences

- Upgrading to v3 later is a data-migration decision, not a code one:
  `cascade/llm/tracing.py` touches `trace()`, `span()`, `generation()` and
  `flush()`, all of which persist across the major version.
- v2 receives security and maintenance updates but no new features. If M9
  wants a dashboard capability that only v3 has, build it against the study's
  own Parquet exports rather than adopting three services for a chart.

## Interaction with the sanctioned broad guard

`tracing.py` degrades to a no-op on any Langfuse failure (CLAUDE.md §4). That
guard is what makes this pin low-risk: if v2 proves inadequate mid-study, the
run continues untraced rather than aborting, and the ledger is unaffected.
