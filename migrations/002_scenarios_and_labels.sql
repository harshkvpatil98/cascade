-- 002: the scenario registry, and the label table the simulation cannot read.
--
-- This migration is where invariant 2 becomes real. `scenarios` and
-- `scenario_labels` are separate tables with separate grants: `cascade_sim`
-- receives SELECT on the former and **nothing** on the latter. The enforcement
-- is the *absence* of a grant, not a REVOKE -- see ADR-0005, and note that the
-- spec's `REVOKE ALL ON scenario_labels FROM cascade_sim` is a no-op because
-- the role was never granted anything directly.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- scenarios -- readable by everyone that runs the study
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id          text        PRIMARY KEY,
    question             text        NOT NULL,
    resolution_criterion text        NOT NULL,
    cutoff_ts            timestamptz NOT NULL,
    resolve_ts           timestamptz NOT NULL,
    domain               text        NOT NULL,
    source               text        NOT NULL,
    source_ref           text        NOT NULL,
    -- Why this question passed the >= 3 distinguishable parties rule. Kept so
    -- the screen is auditable after the fact rather than a number nobody can
    -- reconstruct; M4's validator is the authoritative actor count.
    party_rule           text        NOT NULL,
    party_names          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- Markets that belong to one real-world event (one row per candidate) are
    -- not independent scenarios. At most one row per group survives selection;
    -- the key is kept so the constraint is visible in the data.
    event_group          text,
    ingested_at          timestamptz NOT NULL DEFAULT now(),

    -- Spec Appendix A writes this as strictly greater. §3.1 prose says "at
    -- least 21 days"; the DDL is the executable form, so it wins and the
    -- Python rule in rules.py matches it exactly.
    CONSTRAINT scenarios_horizon_meaningful
        CHECK (resolve_ts > cutoff_ts + interval '21 days')
);

CREATE INDEX IF NOT EXISTS scenarios_domain_idx ON scenarios (domain);
CREATE INDEX IF NOT EXISTS scenarios_cutoff_idx ON scenarios (cutoff_ts);

-- --------------------------------------------------------------------------
-- scenario_labels -- the outcome. cascade_sim must never be able to read this.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS scenario_labels (
    scenario_id text        PRIMARY KEY REFERENCES scenarios (scenario_id),
    outcome     smallint    NOT NULL CHECK (outcome IN (0, 1)),
    resolved_at timestamptz NOT NULL
);

-- --------------------------------------------------------------------------
-- scenario_manifest -- the frozen split (spec §1.3)
--
-- One row per seal. `manifest_sha256` is taken over
-- (scenario_id, cutoff_ts, resolve_ts, outcome) for every scenario, sorted.
-- Every later phase asserts this hash on startup, which is what makes the
-- split frozen rather than merely documented.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS scenario_manifest (
    manifest_sha256   text        PRIMARY KEY,
    sealed_at         timestamptz NOT NULL DEFAULT now(),
    n_scenarios       integer     NOT NULL,
    n_yes             integer     NOT NULL,
    yes_rate          double precision NOT NULL,
    -- Climatology = always predict the set base rate. A system that does not
    -- beat this has produced nothing (spec §3.1), so it is stored with the
    -- split it describes rather than recomputed later from a possibly
    -- different set.
    climatology_brier double precision NOT NULL,
    study_salt        text        NOT NULL,
    notes             text        NOT NULL DEFAULT ''
);

-- --------------------------------------------------------------------------
-- Grants: invariant 2
-- --------------------------------------------------------------------------

GRANT SELECT ON scenarios         TO cascade_sim, cascade_eval;
GRANT SELECT ON scenario_manifest TO cascade_sim, cascade_eval;

-- Only the evaluation role sees outcomes. cascade_sim is deliberately absent:
-- a permission error is the mechanism, and a later `GRANT ... TO PUBLIC` would
-- be caught by the deny-by-default default privileges set in migration 001.
GRANT SELECT ON scenario_labels TO cascade_eval;

COMMIT;
