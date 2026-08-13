# ADR-0002 — `chronofence_search` needs `SECURITY DEFINER` and a pinned `search_path`

- **Status:** accepted
- **Milestone:** M3
- **Corrects:** spec §4.2

## Context

Spec §4.2 specifies the retrieval boundary as:

> The app role has `EXECUTE` on the function and **no `SELECT` on `chunks`**.

A plain (`SECURITY INVOKER`) SQL function executes with the *caller's*
privileges. A role holding `EXECUTE` on `chronofence_search` but no `SELECT`
on `chunks` therefore gets a permission error on the first row the function
touches — the function is unusable by exactly the role it exists for.

## Decision

Declare the function `SECURITY DEFINER`, owned by `cascade_admin`:

```sql
CREATE FUNCTION chronofence_search(q halfvec(384), as_of timestamptz, k int)
RETURNS TABLE (...)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$ ... $$;

REVOKE ALL ON FUNCTION chronofence_search FROM PUBLIC;
GRANT EXECUTE ON FUNCTION chronofence_search TO cascade_sim;
```

Three details are load-bearing:

- **`SET search_path = public, pg_temp`.** A `SECURITY DEFINER` function
  without a pinned `search_path` is a privilege-escalation vector: the caller
  controls name resolution and can shadow a referenced object with one in a
  schema they own. `pg_temp` is listed last so a temporary table cannot
  shadow anything.
- **`REVOKE ... FROM PUBLIC` first.** `CREATE FUNCTION` grants `EXECUTE` to
  `PUBLIC` by default. Granting to `cascade_sim` without revoking first leaves
  the function callable by every role, which makes the grant decorative.
- **`STABLE`, not `VOLATILE`.** Lets the planner inline and cache within a
  statement; a volatile function in a lateral join re-executes per row.

## The invariant this does *not* weaken

`as_of` stays a **required parameter with no default** (invariant 1). A
`SECURITY DEFINER` function that defaulted `as_of` would be strictly worse
than direct table access: it would read post-cutoff rows *with elevated
privileges* and look authoritative doing it. The signature has three required
arguments and no `DEFAULT` clause, so an omitted `as_of` is a function-
resolution error at parse time.

## Verified by

`tests/leakage/` — connecting as `cascade_sim`, asserting `SELECT` on `chunks`
raises `InsufficientPrivilege` while `chronofence_search(...)` succeeds, and
that calling it without `as_of` fails to resolve.
