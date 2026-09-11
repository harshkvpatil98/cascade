-- 008: compiled causal graphs (spec §5, M4).
--
-- A scenario compiles **once** for the whole study (§5.3, determinism rule).
-- That is why `scenario_id` is the primary key rather than one key per
-- compilation attempt: there is no history to keep, and a table that could
-- hold two graphs for one scenario would make "which graph did run 4,118 use"
-- a question the schema cannot answer.
--
-- `graph_sha256` is the content hash of the canonical JSON. It is stored
-- alongside the graph rather than derived on read so that a corrupted or
-- hand-edited row is detectable: `cascade compile verify` recomputes the hash
-- and compares.
--
-- Failures get their own table. Folding them into `causal_graphs` with a
-- nullable `graph` would mean every downstream reader has to remember to
-- filter, and the one that forgets builds a simulation on a NULL.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

CREATE TABLE IF NOT EXISTS causal_graphs (
    scenario_id     text        PRIMARY KEY REFERENCES scenarios (scenario_id),
    graph_sha256    text        NOT NULL,
    graph           jsonb       NOT NULL,
    -- Denormalised from the graph so the M4 acceptance figures (mean actor
    -- count, factor count range) are an aggregate query rather than a scan
    -- that parses 180 JSON documents.
    n_actors        integer     NOT NULL CHECK (n_actors BETWEEN 8 AND 20),
    n_factors       integer     NOT NULL CHECK (n_factors BETWEEN 4 AND 12),
    n_edges         integer     NOT NULL CHECK (n_edges > 0),
    -- The histogram the acceptance criterion asks for.
    repair_retries  integer     NOT NULL CHECK (repair_retries >= 0),
    llm_calls       integer     NOT NULL CHECK (llm_calls > 0),
    evidence_chunks integer     NOT NULL CHECK (evidence_chunks >= 0),
    -- Provenance. A graph compiled under a different prompt revision or a
    -- different model is a different artifact even when the hash of the JSON
    -- happens to match, and §1.3 requires prompt changes to be auditable.
    compiler_model  text        NOT NULL,
    prompt_rev      text        NOT NULL,
    compiled_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS causal_graphs_hash_idx ON causal_graphs (graph_sha256);

-- --------------------------------------------------------------------------
-- Hard failures (spec §5.2: "then hard-fail the scenario and log it")
--
-- Logged rather than raised: one intractable scenario must not cost the other
-- 179, and a study that silently compiled 176 of 180 would report a headline
-- metric over a set nobody chose.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS causal_graph_failures (
    scenario_id    text        PRIMARY KEY REFERENCES scenarios (scenario_id),
    violations     jsonb       NOT NULL DEFAULT '[]'::jsonb,
    defects        jsonb       NOT NULL DEFAULT '[]'::jsonb,
    repair_retries integer     NOT NULL CHECK (repair_retries >= 0),
    compiler_model text        NOT NULL,
    prompt_rev     text        NOT NULL,
    failed_at      timestamptz NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- Grants
--
-- `cascade_sim` reads graphs: M5 constructs one agent per actor and derives
-- the Aperture visibility policy from the edge topology, and it does that as
-- the simulation role. It gets SELECT on the graphs and **nothing** on the
-- failures table -- a failed compile is an operational fact, not study input.
--
-- Neither table carries an outcome, so invariant 2 is untouched: the graph is
-- built from the question, the resolution criterion and pre-cutoff evidence,
-- and `scenario_labels` is reachable by neither role through here.
-- --------------------------------------------------------------------------

GRANT SELECT ON causal_graphs TO cascade_sim, cascade_eval;
GRANT SELECT ON causal_graph_failures TO cascade_eval;

COMMIT;
