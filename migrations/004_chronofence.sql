-- 004: Chronofence -- the time-locked retrieval boundary (spec §4.2).
--
-- This migration is the one that decides whether the study is valid. Every
-- other leakage control is downstream of it: if a simulation agent can read a
-- chunk published after its scenario's cutoff, the headline Brier is measuring
-- hindsight and nothing else in the harness can detect that.
--
-- The lock is built from three mechanisms, in decreasing order of how much
-- they are trusted:
--
--   1. **A grant.** `cascade_sim` has no SELECT on `chunks` or `documents`
--      (migration 003) and gets EXECUTE on `chronofence_search` only. There
--      is no code path from the simulation to a raw vector.
--   2. **A required parameter.** `as_of` has no DEFAULT, so an omitted
--      argument is a function-resolution error at parse time rather than a
--      silent read of the present (invariant 1).
--   3. **Partition pruning.** `chunks` is partitioned on `published_at`
--      (migration 003), so the planner drops wholly-post-cutoff partitions
--      before touching a vector.
--
-- Mechanism 3 is a performance property, never the correctness property.
-- Quarterly granularity (ADR-0004) means the partition containing `as_of` is
-- partially post-cutoff, so the residual `published_at < as_of` predicate in
-- the function body does real filtering. That predicate is what makes the
-- lock correct; pruning only makes it fast. Removing it because "partitioning
-- already handles it" would leak every row in the boundary quarter.
--
-- Index creation is deliberately NOT here. `lists ~= sqrt(rows_in_partition)`
-- is a function of measured row counts, which are unknown when a migration is
-- authored and change as the corpus grows; an index sized against an empty
-- table stays wrong forever. See ADR-0012 -- `cascade retrieval index` owns
-- the index lifecycle and `cascade retrieval verify` fails when sizing drifts.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- chronofence_search -- the only path from the simulation to the corpus
--
-- SECURITY DEFINER is required, not stylistic (ADR-0002): a SECURITY INVOKER
-- function executes with the caller's privileges, so a role holding EXECUTE
-- here but no SELECT on `chunks` would get a permission error on the first
-- row the function touches. The function would be unusable by exactly the
-- role it exists for.
--
-- `SET search_path = public, pg_temp` is the other half. A SECURITY DEFINER
-- function without a pinned search_path is a privilege-escalation vector: the
-- caller controls name resolution and can shadow a referenced object with one
-- in a schema they own. `pg_temp` is listed last so a temporary table cannot
-- shadow anything.
--
-- `SET ivfflat.probes` pins recall at the function boundary. Left to the
-- session, probes defaults to 1 -- a caller who forgets to set it gets a
-- silently worse recall profile and no error, which is the same class of
-- defect as a forgettable time predicate. The value must agree with
-- `retrieval.ivfflat_probes` in configs/base.yaml; an integration test asserts
-- the two match and directs a change to a new migration rather than an edit.
-- --------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION chronofence_search(
    q     halfvec(384),
    as_of timestamptz,
    k     int
)
RETURNS TABLE (
    chunk_id     text,
    document_id  text,
    ordinal      integer,
    body         text,
    published_at timestamptz,
    source       text,
    url          text,
    title        text,
    distance     real
)
LANGUAGE sql
STABLE
PARALLEL SAFE
SECURITY DEFINER
SET search_path = public, pg_temp
SET ivfflat.probes = 10
AS $$
    -- MATERIALIZED is load-bearing. Inlined, the planner is free to join
    -- `documents` before taking the top k, which turns an index scan over the
    -- k nearest vectors into a scan over every joined row. Materialising pins
    -- the plan to: prune partitions -> IVFFlat top-k -> join k rows.
    WITH hits AS MATERIALIZED (
        SELECT
            c.chunk_id,
            c.document_id,
            c.ordinal,
            c.body,
            c.published_at,
            (c.embedding <-> q)::real AS distance
        FROM chunks c
        WHERE c.published_at < as_of
        ORDER BY c.embedding <-> q
        LIMIT k
    )
    SELECT
        h.chunk_id,
        h.document_id,
        h.ordinal,
        h.body,
        h.published_at,
        d.source,
        d.url,
        d.title,
        h.distance
    FROM hits h
    JOIN documents d
      ON  d.document_id  = h.document_id
      AND d.published_at = h.published_at
    -- chunk_id breaks distance ties deterministically. Without it two chunks
    -- at the same distance may come back in either order, and the event log
    -- at M8 has to hash byte-identically across replays. The sort is over k
    -- rows, not the corpus, so it costs nothing measurable.
    ORDER BY h.distance, h.chunk_id;
$$;

COMMENT ON FUNCTION chronofence_search(halfvec, timestamptz, int) IS
    'Time-locked vector search. Returns the k chunks nearest q among those '
    'published strictly before as_of. as_of has no default by design '
    '(invariant 1). The sole corpus read path granted to cascade_sim.';

-- --------------------------------------------------------------------------
-- chronofence_search_exact -- the recall oracle
--
-- Identical semantics, exhaustive search. `ivfflat.probes` is set to 32768,
-- which is pgvector's maximum for both `probes` and `lists` -- so probes can
-- never be less than a partition's list count, IVFFlat visits every list, and
-- every vector belongs to exactly one list. The result is therefore exact. A
-- partition with no index falls back to a sequential scan, exact for the same
-- reason.
--
-- This exists so recall@20 has a ground truth. A latency number without a
-- recall number is meaningless (spec §4.2), and recall cannot be measured
-- against the same approximate index whose recall is in question.
--
-- Granted to `cascade_eval` only. `cascade_sim` must have exactly one corpus
-- read path; a second one that happens to be slower is still a second one.
-- --------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION chronofence_search_exact(
    q     halfvec(384),
    as_of timestamptz,
    k     int
)
RETURNS TABLE (
    chunk_id     text,
    document_id  text,
    ordinal      integer,
    body         text,
    published_at timestamptz,
    source       text,
    url          text,
    title        text,
    distance     real
)
LANGUAGE sql
STABLE
PARALLEL SAFE
SECURITY DEFINER
SET search_path = public, pg_temp
SET ivfflat.probes = 32768
AS $$
    WITH hits AS MATERIALIZED (
        SELECT
            c.chunk_id,
            c.document_id,
            c.ordinal,
            c.body,
            c.published_at,
            (c.embedding <-> q)::real AS distance
        FROM chunks c
        WHERE c.published_at < as_of
        ORDER BY c.embedding <-> q
        LIMIT k
    )
    SELECT
        h.chunk_id,
        h.document_id,
        h.ordinal,
        h.body,
        h.published_at,
        d.source,
        d.url,
        d.title,
        h.distance
    FROM hits h
    JOIN documents d
      ON  d.document_id  = h.document_id
      AND d.published_at = h.published_at
    ORDER BY h.distance, h.chunk_id;
$$;

COMMENT ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int) IS
    'Exhaustive time-locked vector search. Ground truth for recall@k. '
    'cascade_eval only -- the simulation gets exactly one corpus read path.';

-- --------------------------------------------------------------------------
-- chronofence_partitions -- what `cascade retrieval index` measures against
--
-- Exposes live row counts and the IVFFlat index (if any) per `chunks`
-- partition, so index sizing is computed from what is actually stored rather
-- than from a number written down when the migration was authored.
--
-- `pg_class.reltuples` is deliberately not used: it is an estimate that reads
-- -1 until the partition is analysed, and sizing an index off -1 produces
-- lists = 1. The count is exact and this view is not on the hot path.
-- --------------------------------------------------------------------------

CREATE OR REPLACE VIEW chronofence_partitions AS
SELECT
    part.relname::text                              AS partition,
    pg_get_expr(part.relpartbound, part.oid)::text  AS bounds,
    idx.relname::text                               AS index_name,
    substring(
        pg_get_indexdef(idx.oid) from 'lists\s*=\s*''?([0-9]+)'
    )::int                                          AS lists
FROM pg_class parent
JOIN pg_inherits inh   ON inh.inhparent = parent.oid
JOIN pg_class part     ON part.oid = inh.inhrelid
LEFT JOIN pg_index pgi ON pgi.indrelid = part.oid
LEFT JOIN pg_class idx ON idx.oid = pgi.indexrelid
                      AND idx.relam = (SELECT oid FROM pg_am WHERE amname = 'ivfflat')
WHERE parent.relname = 'chunks';

COMMENT ON VIEW chronofence_partitions IS
    'One row per chunks partition with its IVFFlat index and configured '
    'lists, or NULLs when unindexed. Read by `cascade retrieval index`.';

-- --------------------------------------------------------------------------
-- Grants
--
-- REVOKE FROM PUBLIC first: CREATE FUNCTION grants EXECUTE to PUBLIC by
-- default, and every role inherits PUBLIC, so granting to cascade_sim without
-- revoking first leaves the function callable by everyone and makes the grant
-- decorative (ADR-0005 records the same failure mode for tables).
-- --------------------------------------------------------------------------

REVOKE ALL ON FUNCTION chronofence_search(halfvec, timestamptz, int) FROM PUBLIC;
REVOKE ALL ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION chronofence_search(halfvec, timestamptz, int)
    TO cascade_sim, cascade_eval;

-- Deliberately not granted to cascade_sim. See the note on the function.
GRANT EXECUTE ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int)
    TO cascade_eval;

GRANT SELECT ON chronofence_partitions TO cascade_eval;

COMMIT;
