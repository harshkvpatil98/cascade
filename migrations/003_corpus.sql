-- 003: the evidence corpus -- documents and time-partitioned chunks.
--
-- This migration builds the structure the Chronofence time-lock rests on
-- (spec §3.2, §4.2). `chunks` is partitioned by RANGE (published_at) so the
-- planner can prune post-cutoff partitions *before* touching a vector: the
-- time filter is structural, not a predicate a caller can forget.
--
-- Granularity is quarterly, not monthly, per ADR-0004: ~108 monthly partitions
-- means ~100 index scans for a late-cutoff query against a 15 ms budget.
-- `documents` uses the same boundaries for operational symmetry -- spec §3.3
-- says monthly, but documents are not on the retrieval hot path, so the only
-- effect of monthly there would be three times the partition count.
--
-- There is deliberately **no DEFAULT partition**. A default partition can
-- never be pruned, so every time-locked query would have to scan it; and a
-- document whose date falls outside the study range is a data error that
-- should fail loudly rather than land in a bucket the time-lock cannot prune.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- documents -- one row per source document
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    document_id  text        NOT NULL,
    source       text        NOT NULL,
    source_ref   text        NOT NULL,
    url          text        NOT NULL DEFAULT '',
    title        text        NOT NULL DEFAULT '',
    -- NOT NULL is half of the date-validate rule; the other half (no future
    -- dates, no naive timestamps) is enforced in Python because a CHECK
    -- cannot call now(), and timestamptz has already absorbed the offset by
    -- the time Postgres sees it. `cascade corpus verify` re-asserts both over
    -- the full table rather than a sample.
    published_at timestamptz NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    -- 64-bit SimHash stored signed: Postgres has no unsigned bigint, and the
    -- Hamming distance is computed on the bit pattern, which round-trips.
    simhash      bigint      NOT NULL,
    n_chunks     integer     NOT NULL DEFAULT 0,
    PRIMARY KEY (document_id, published_at)
) PARTITION BY RANGE (published_at);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
CREATE INDEX IF NOT EXISTS documents_simhash_idx ON documents (simhash);

-- --------------------------------------------------------------------------
-- chunks -- the retrieval unit
--
-- `published_at` is denormalised from the parent document so the partition
-- key travels with the vector (spec §3.3). Without it, pruning would require
-- a join and the time-lock would stop being structural.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text        NOT NULL,
    document_id  text        NOT NULL,
    ordinal      integer     NOT NULL,
    body         text        NOT NULL,
    token_count  integer     NOT NULL,
    published_at timestamptz NOT NULL,
    embedding    halfvec(384),
    PRIMARY KEY (chunk_id, published_at)
) PARTITION BY RANGE (published_at);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- --------------------------------------------------------------------------
-- Quarterly partitions across the study range.
--
-- Generated rather than hand-written so the boundaries cannot drift between
-- the two tables; still explicit SQL, executed once, with no ORM involved.
-- --------------------------------------------------------------------------

DO $$
DECLARE
    quarter_start date := date '2015-01-01';
    quarter_end   date;
    horizon       date := date '2028-01-01';
    suffix        text;
BEGIN
    WHILE quarter_start < horizon LOOP
        quarter_end := quarter_start + interval '3 months';
        suffix := to_char(quarter_start, 'YYYY') || 'q' ||
                  to_char(quarter_start, 'Q');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS documents_%s PARTITION OF documents '
            'FOR VALUES FROM (%L) TO (%L)', suffix, quarter_start, quarter_end);

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS chunks_%s PARTITION OF chunks '
            'FOR VALUES FROM (%L) TO (%L)', suffix, quarter_start, quarter_end);

        quarter_start := quarter_end;
    END LOOP;
END $$;

-- --------------------------------------------------------------------------
-- Ingest bookkeeping -- invariant 8, every phase resumable
--
-- One row per fetch unit (an EDGAR quarter index, a Wikipedia article, a
-- GDELT window). A restart skips whatever is already `done`, so a 1.3M-chunk
-- ingest that dies at 80% resumes rather than restarting.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS corpus_ingest_state (
    source      text        NOT NULL,
    unit_key    text        NOT NULL,
    state       text        NOT NULL CHECK (state IN ('done', 'failed')),
    n_documents integer     NOT NULL DEFAULT 0,
    n_chunks    integer     NOT NULL DEFAULT 0,
    detail      text        NOT NULL DEFAULT '',
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, unit_key)
);

-- --------------------------------------------------------------------------
-- Grants
--
-- `cascade_sim` gets **nothing** on documents or chunks. Retrieval reaches the
-- corpus only through `chronofence_search`, which lands at M3 as a
-- SECURITY DEFINER function (ADR-0002). Granting SELECT here would make that
-- function's whole point -- no direct table access for the app role --
-- unenforceable.
-- --------------------------------------------------------------------------

GRANT SELECT ON documents TO cascade_eval;
GRANT SELECT ON chunks    TO cascade_eval;
GRANT SELECT ON corpus_ingest_state TO cascade_eval, cascade_sim;

COMMIT;
