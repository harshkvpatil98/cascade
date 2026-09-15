-- 014: HNSW in place of IVFFlat, and a bounded candidate pool so the top-k
-- limit stops poisoning the plan (spec §4.2, §4.3; M8).
--
-- Two independent defects, both measured against the 1,950,912-chunk corpus
-- where M3's acceptance figures (p95 13.43 ms, recall@20 0.9372) had decayed
-- to p95 176.00 ms and recall@20 0.8950. See ADR-0026.
--
-- 1. THE PARAMETERISED LIMIT. `chronofence_search` is SECURITY DEFINER with a
--    pinned search_path, and each of those independently disqualifies a SQL
--    function from inlining (ADR-0002 needs the first, ADR-0002 the second).
--    Non-inlined, the body runs as its own statement with `k` as a runtime
--    parameter -- so the planner cannot know how many rows the Merge Append
--    across 47 partitions will be asked for, and drives every partition's
--    index scan far past what the caller wants. Measured, same body, same
--    corpus, same probes:
--
--        function attributes            latency
--        none (inlinable)                36.2 ms
--        SECURITY DEFINER                138.0 ms
--        SET search_path                 142.9 ms
--        SET ivfflat.probes              142.5 ms
--        LIMIT 20 literal, SECURITY DEF   35.4 ms
--
--    The attribute is not the cause; losing inlining is, and a literal limit
--    fixes it while the attributes stay. So the body now takes a bounded pool
--    with a *constant* limit and applies the caller's `k` to that pool. For
--    any k <= CHRONOFENCE_POOL the result is identical -- the top-k of the
--    top-N under one total order is the top-k -- and it is *more*
--    deterministic, because the (distance, chunk_id) tie-break is now applied
--    over a pool larger than k rather than at the k-th boundary.
--
-- 2. IVFFLAT DOES NOT SCALE HERE. IVFFlat with lists = sqrt(rows) and probes
--    per scan reads about `probes x sqrt(rows)` rows per partition, so a query
--    spanning the corpus reads about `probes x sum_p sqrt(n_p)` -- measured
--    229,000 rows against 74,000 at M3. HNSW is logarithmic in the partition
--    and has no row-dependent build parameter at all. Measured on
--    chunks_2017q4 (605,083 rows), same query, same page cache:
--
--        IVFFlat probes=40   6.57 ms
--        HNSW ef_search=20   0.63 ms   recall@20 0.9500
--        HNSW ef_search=40   0.60 ms   recall@20 1.0000
--        HNSW ef_search=100  0.92 ms   recall@20 1.0000
--
--    HNSW also ends the drift class ADR-0012 describes: `lists` is a function
--    of measured rows, so a growing partition silently degrades its own index
--    and needed a rebuild pass. That has interrupted M4, M5, M6 and M7. HNSW
--    has no such parameter; an index that exists stays correct as rows arrive.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- The partition view now reports HNSW build parameters.
--
-- `m` and `ef_construction` are read from the index definition the same way
-- `lists` was, so `cascade retrieval index` can tell an index built under the
-- current settings from one built under older ones.
-- --------------------------------------------------------------------------

DROP VIEW IF EXISTS chronofence_partitions;

CREATE VIEW chronofence_partitions AS
SELECT
    part.relname::text                             AS partition,
    pg_get_expr(part.relpartbound, part.oid)::text AS bounds,
    idx.index_name,
    idx.m,
    idx.ef_construction
FROM pg_class parent
JOIN pg_inherits inh ON inh.inhparent = parent.oid
JOIN pg_class part   ON part.oid = inh.inhrelid
LEFT JOIN LATERAL (
    SELECT
        i.relname::text AS index_name,
        COALESCE(
            substring(pg_get_indexdef(i.oid) from 'm\s*=\s*''?([0-9]+)')::int,
            16  -- pgvector's default when the option is not spelled out
        ) AS m,
        COALESCE(
            substring(pg_get_indexdef(i.oid) from 'ef_construction\s*=\s*''?([0-9]+)')::int,
            64
        ) AS ef_construction
    FROM pg_index pgi
    JOIN pg_class i ON i.oid = pgi.indexrelid
    WHERE pgi.indrelid = part.oid
      AND i.relam = (SELECT oid FROM pg_am WHERE amname = 'hnsw')
    -- A partition should never carry two HNSW indexes on `embedding`;
    -- ordering makes the choice deterministic if one ever does, rather than
    -- letting the row count depend on the planner.
    ORDER BY i.relname
    LIMIT 1
) idx ON true
WHERE parent.relname = 'chunks';

COMMENT ON VIEW chronofence_partitions IS
    'Exactly one row per chunks partition, with its HNSW index and build '
    'parameters, or NULLs when unindexed. Read by `cascade retrieval index`.';

GRANT SELECT ON chronofence_partitions TO cascade_eval;

-- --------------------------------------------------------------------------
-- chronofence_search: bounded pool, HNSW ef_search.
--
-- The pool constant is 200 and the client refuses k above it, so the
-- equivalence the pool rests on (top-k of top-N == top-k, for k <= N) is
-- enforced on both sides rather than assumed on one.
--
-- `as_of` still has no default, by design (invariant 1).
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
SET hnsw.ef_search = 100
AS $$
    WITH pool AS MATERIALIZED (
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
        LIMIT 200
    ),
    hits AS (
        SELECT * FROM pool ORDER BY distance, chunk_id LIMIT k
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
    CROSS JOIN LATERAL (
        SELECT doc.source, doc.url, doc.title
        FROM documents doc
        WHERE doc.document_id  = h.document_id
          AND doc.published_at = h.published_at
        LIMIT 1
    ) d
    ORDER BY h.distance, h.chunk_id;
$$;

COMMENT ON FUNCTION chronofence_search(halfvec, timestamptz, int) IS
    'Time-locked vector search. Returns the k chunks nearest q among those '
    'published strictly before as_of, drawn from a bounded candidate pool so '
    'the caller''s k does not enter the plan. as_of has no default by design '
    '(invariant 1). The sole corpus read path granted to cascade_sim.';

-- --------------------------------------------------------------------------
-- chronofence_search_exact: the recall oracle. Same bounded pool, and
-- deliberately no index -- it is the ground truth the approximate path is
-- measured against, so it must not share the approximate path's parameters.
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
SET enable_indexscan = off
SET enable_bitmapscan = off
AS $$
    WITH pool AS MATERIALIZED (
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
        LIMIT 200
    ),
    hits AS (
        SELECT * FROM pool ORDER BY distance, chunk_id LIMIT k
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
    CROSS JOIN LATERAL (
        SELECT doc.source, doc.url, doc.title
        FROM documents doc
        WHERE doc.document_id  = h.document_id
          AND doc.published_at = h.published_at
        LIMIT 1
    ) d
    ORDER BY h.distance, h.chunk_id;
$$;

COMMENT ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int) IS
    'Exhaustive ground truth for recall measurement. Granted to cascade_eval '
    'only, so the simulation keeps exactly one corpus read path.';

REVOKE ALL ON FUNCTION chronofence_search(halfvec, timestamptz, int) FROM PUBLIC;
REVOKE ALL ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION chronofence_search(halfvec, timestamptz, int)
    TO cascade_sim, cascade_eval;
GRANT EXECUTE ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int)
    TO cascade_eval;

COMMIT;
