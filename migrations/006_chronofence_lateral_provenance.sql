-- 006: make the provenance lookup in chronofence_search prunable.
--
-- Performance defect in migration 004, measured against the M3 budget.
--
-- The function joined `documents` to carry source/url/title out with each
-- chunk:
--
--     FROM hits h
--     JOIN documents d ON  d.document_id  = h.document_id
--                     AND  d.published_at = h.published_at
--
-- `documents` is partitioned by `published_at` into 52 partitions. Written as
-- a plain join inside a function body that **cannot be inlined** -- SECURITY
-- DEFINER and a SET clause each independently disqualify a SQL function from
-- inlining -- the planner has no constant for `h.published_at` at plan time
-- and produces a join that touches every partition rather than the one the
-- equality identifies.
--
-- Measured on the 409,899-chunk corpus, k = 20, probes = 10:
--
--     function body without the join .................  2.95 ms
--     with the join (migration 004) .................. 15.37 ms
--     with the join + a redundant `d.published_at < as_of` .. 16.08 ms
--     with a correlated scalar subquery .............. 1191.34 ms
--     with CROSS JOIN LATERAL ... LIMIT 1 ............  3.34 ms
--
-- The same body inlined costs 4.25 ms, which is why this never showed up as a
-- slow *query* -- only as a slow *function*.
--
-- LATERAL wins because the subquery is evaluated per hit row, so
-- `published_at = h.published_at` is a runtime constant and executor partition
-- pruning selects exactly one partition per lookup: 20 single-partition index
-- probes instead of a join against all 52. `LIMIT 1` makes that explicit and
-- is semantically free -- (document_id, published_at) is the primary key of
-- `documents`, so at most one row can match.
--
-- Semantics are unchanged. CROSS JOIN LATERAL over a subquery that returns at
-- most one row drops a hit with no parent document, exactly as the inner join
-- did; `store.write_batch` commits documents and their chunks in one
-- transaction, so that case is a data error either way and is asserted
-- against by `cascade retrieval verify`.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

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
    'published strictly before as_of. as_of has no default by design '
    '(invariant 1). The sole corpus read path granted to cascade_sim.';

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
    'Exhaustive time-locked vector search. Ground truth for recall@k. '
    'cascade_eval only -- the simulation gets exactly one corpus read path.';

-- CREATE OR REPLACE preserves existing grants, but re-asserting them keeps
-- this migration correct if it is ever applied to a database where 004 was
-- rolled forward differently.
REVOKE ALL ON FUNCTION chronofence_search(halfvec, timestamptz, int) FROM PUBLIC;
REVOKE ALL ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION chronofence_search(halfvec, timestamptz, int)
    TO cascade_sim, cascade_eval;
GRANT EXECUTE ON FUNCTION chronofence_search_exact(halfvec, timestamptz, int)
    TO cascade_eval;

COMMIT;
