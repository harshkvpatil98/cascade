# ADR-0005 — Role grants deny by default; roles are explicitly non-superuser

- **Status:** accepted
- **Milestone:** M0
- **Corrects:** spec §3.3 / Appendix A

## Context

Invariant 2 — the simulation never reads `scenario_labels` — is enforced by a
Postgres grant rather than by code review, because code review is not a
mechanism. The spec expresses that as:

```sql
REVOKE ALL ON scenario_labels FROM cascade_sim;
```

That statement is a **no-op**. `REVOKE` removes privileges that were granted
*directly to the named role*. `cascade_sim` was never granted anything on
`scenario_labels` directly, so there is nothing to revoke — and the access it
actually has comes from somewhere else entirely.

Two somewhere-elses:

1. **`PUBLIC`.** Every role inherits `PUBLIC`. In a default database `PUBLIC`
   holds `CREATE` and `USAGE` on schema `public` (through PostgreSQL 14;
   `CREATE` was dropped in 15, `USAGE` was not). Table privileges are not
   granted to `PUBLIC` by default, but *function* `EXECUTE` is — which matters
   directly for ADR-0002.
2. **Superuser.** A superuser bypasses every privilege check. If the roles are
   created without `NOSUPERUSER` in an environment whose bootstrap role is a
   superuser, the grant test passes trivially while enforcing nothing.

The failure mode is the dangerous one: a test that goes green against an
absent mechanism.

## Decision

Migration 001 does three things, in this order:

1. **Create roles explicitly restricted.**
   ```sql
   CREATE ROLE cascade_sim LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS ...
   ```
   `NOBYPASSRLS` matters for the same reason as `NOSUPERUSER` once row-level
   security lands on the event log.

2. **Strip `PUBLIC` first, then grant back the minimum.**
   ```sql
   REVOKE ALL ON SCHEMA public FROM PUBLIC;
   REVOKE ALL ON ALL TABLES    IN SCHEMA public FROM PUBLIC;
   REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC;
   GRANT USAGE ON SCHEMA public TO cascade_sim, cascade_eval;
   ```
   Deny by default. `cascade_sim` then receives table-level `SELECT` on
   exactly the tables it needs, and `scenario_labels` is not among them —
   which is enforced by the *absence* of a grant, the only form of enforcement
   that survives a later refactor.

3. **Set default privileges** so tables created by future migrations inherit
   no `PUBLIC` grant either. Without this, migration 007 creating a table
   silently re-opens what 001 closed.

## Verified by

`tests/integration/test_infrastructure.py`:

- both roles exist and `rolsuper` is **false** for each — asserted directly,
  because every other assertion is vacuous if this one fails;
- `PUBLIC` holds no privilege on schema `public`;
- migration re-run is idempotent.

And at M1, the criterion that matters: connecting as `cascade_sim` and
selecting from `scenario_labels` raises `InsufficientPrivilege`.
