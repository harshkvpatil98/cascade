-- 005: correct `chronofence_partitions` to one row per partition.
--
-- Defect in migration 004. The view joined `pg_index` and then filtered the
-- *joined* `pg_class` row to the IVFFlat access method:
--
--     LEFT JOIN pg_index pgi ON pgi.indrelid = part.oid
--     LEFT JOIN pg_class idx ON idx.oid = pgi.indexrelid
--                           AND idx.relam = (ivfflat)
--
-- The first join fans out to one row per index on the partition, and the
-- opclass predicate sits on the second join, so it nulls the non-matching
-- rows instead of removing them. Every `chunks` partition carries at least the
-- primary key and `chunks_document_idx`, so the view returned 104 rows for 52
-- partitions -- measured, before any IVFFlat index existed at all.
--
-- Consequences had it shipped: `cascade retrieval index` would have planned
-- each partition two or three times, `CREATE INDEX` would have raised
-- "relation already exists" on the second attempt, and `retrieval verify`
-- would have reported "0/104 correctly sized indexes" against a real 52.
--
-- The fix is a LATERAL subquery that picks the IVFFlat index if there is one,
-- so cardinality is one row per partition by construction rather than by a
-- predicate that has to be written in the right place.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

CREATE OR REPLACE VIEW chronofence_partitions AS
SELECT
    part.relname::text                             AS partition,
    pg_get_expr(part.relpartbound, part.oid)::text AS bounds,
    idx.index_name,
    idx.lists
FROM pg_class parent
JOIN pg_inherits inh ON inh.inhparent = parent.oid
JOIN pg_class part   ON part.oid = inh.inhrelid
LEFT JOIN LATERAL (
    SELECT
        i.relname::text AS index_name,
        substring(
            pg_get_indexdef(i.oid) from 'lists\s*=\s*''?([0-9]+)'
        )::int          AS lists
    FROM pg_index pgi
    JOIN pg_class i ON i.oid = pgi.indexrelid
    WHERE pgi.indrelid = part.oid
      AND i.relam = (SELECT oid FROM pg_am WHERE amname = 'ivfflat')
    -- A partition should never carry two IVFFlat indexes on `embedding`;
    -- ordering makes the choice deterministic if one ever does, rather than
    -- letting the row count depend on the planner.
    ORDER BY i.relname
    LIMIT 1
) idx ON true
WHERE parent.relname = 'chunks';

COMMENT ON VIEW chronofence_partitions IS
    'Exactly one row per chunks partition, with its IVFFlat index and '
    'configured lists, or NULLs when unindexed. Read by `cascade retrieval index`.';

GRANT SELECT ON chronofence_partitions TO cascade_eval;

COMMIT;
