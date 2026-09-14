-- 011: the report provenance row, and the grants PHASE 4 needs (spec §1.3,
-- §10, Appendix D; M7).
--
-- Appendix D puts the whole report on disk, and that is where it stays: this
-- table does not duplicate `metrics.json`. It records one fact per written
-- report -- which sealed split it was computed against -- because that is the
-- fact a reader cannot recover from a directory that was copied, renamed, or
-- written before the registry moved.
--
-- The second half of this migration is the §1.3 prompt audit. `prompt_revisions`
-- landed at M6 with both Brier columns NULL, because the revision it recorded
-- predates the first backtest. Filling them in is an evaluation output, so
-- `cascade_eval` gets UPDATE on exactly those two columns' table and nothing
-- else -- the summary and rationale are still write-once in practice, since
-- the only writer is a migration.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

CREATE TABLE IF NOT EXISTS study_reports (
    report_id       text        PRIMARY KEY,
    written_at      timestamptz NOT NULL DEFAULT now(),
    -- The seal in force when the report was written. A report whose hash does
    -- not match the current `scenario_manifest` was computed against a
    -- different set, and that must be visible without opening the directory.
    manifest_sha256 text        NOT NULL,
    git_sha         text        NOT NULL,
    headline_config text        NOT NULL,
    n_scenarios     integer     NOT NULL CHECK (n_scenarios >= 0),
    -- NULL when the headline configuration had no scoreable forecasts. A
    -- report can legitimately be written with nothing to score -- that is what
    -- a blocked milestone looks like -- and a 0.0 in this column would be a
    -- measurement nobody made.
    headline_brier  double precision CHECK (headline_brier IS NULL
                                            OR (headline_brier >= 0 AND headline_brier <= 1)),
    notes           text        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS study_reports_manifest_idx ON study_reports (manifest_sha256);

GRANT SELECT, INSERT, UPDATE ON study_reports TO cascade_eval;
GRANT SELECT ON study_reports TO cascade_sim;

-- §1.3's audit: the before/after Brier of a prompt change is measured in
-- PHASE 4, so PHASE 4 is what writes it.
GRANT UPDATE ON prompt_revisions TO cascade_eval;

COMMIT;
