-- 009: the simulation's output -- runs, per-step records, and the event log
-- (spec §7, §11.1, M5/M6).
--
-- Three tables, and the split between them is load-bearing.
--
-- `events` is one row per **agent decision**, as §11.1 specifies: ~4.2M rows
-- for the main study. Nothing else goes in it. Exogenous shocks and arriving
-- effects are world movements, not decisions, and writing them here would add
-- 864,000 rows -- 20% -- to a count the M6 criterion states as 4.2M +/- 5%.
--
-- `run_steps` holds those world movements, one row per (run, step). It is also
-- where §11.2's provenance chain terminates: a trace that walks `caused_by`
-- back through decisions ends at the step whose exogenous shock started it.
--
-- `runs` is one row per completed run, inserted once, at completion. There is
-- no "started" row to update later: a run that crashed leaves no row, which is
-- exactly what makes M6's resume rule ("skip completed units") checkable with
-- a single SELECT rather than a status column that can lie.
--
-- ## Invariant 6 -- the event log is append-only
--
-- Enforced by the grant, not by convention (the same mechanism as invariant 2,
-- ADR-0005). `cascade_sim` receives SELECT and INSERT on all three tables and
-- **no UPDATE and no DELETE**, so an append-only violation is a permission
-- error rather than a code review. Re-inserting a replayed run is idempotent
-- through ON CONFLICT DO NOTHING, which modifies no existing row and so does
-- not need UPDATE.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- runs
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS runs (
    run_id          uuid        PRIMARY KEY,
    scenario_id     text        NOT NULL REFERENCES scenarios (scenario_id),
    config_id       text        NOT NULL,
    replicate       integer     NOT NULL CHECK (replicate >= 0),
    -- 'agent' for the pinned model, or the name of a deterministic stand-in
    -- (cascade/sim/policies.py). A run that did not involve the agent model
    -- must never be mistaken for one that did, and a footnote is not a
    -- mechanism -- this column is.
    policy          text        NOT NULL DEFAULT 'agent',
    agent_model     text        NOT NULL,
    prompt_rev      text        NOT NULL,
    graph_sha256    text        NOT NULL,
    -- The 64-bit seed is unsigned; bigint would overflow for half of them.
    run_seed        numeric(20,0) NOT NULL,
    -- np.random.Generator makes no cross-version stream promise (NEP 19 covers
    -- RandomState). Replay under a different numpy is a hard error at M8, so
    -- the version has to be recorded with the run that used it (ADR-0015).
    numpy_version   text        NOT NULL,
    steps_run       integer     NOT NULL CHECK (steps_run >= 0),
    termination     text        NOT NULL CHECK (termination IN ('horizon', 'absorbed')),
    outcome_score   double precision NOT NULL CHECK (outcome_score >= 0 AND outcome_score <= 1),
    activation_rate double precision NOT NULL,
    decisions       integer     NOT NULL CHECK (decisions >= 0),
    llm_calls       integer     NOT NULL CHECK (llm_calls >= 0),
    cache_hits      integer     NOT NULL CHECK (cache_hits >= 0),
    tokens_in       bigint      NOT NULL CHECK (tokens_in >= 0),
    tokens_out      bigint      NOT NULL CHECK (tokens_out >= 0),
    cost_usd        numeric(12,6) NOT NULL DEFAULT 0,
    -- The M8 replay criterion is byte-identical across processes; this is the
    -- byte in question.
    event_log_hash  text        NOT NULL,
    started_at      timestamptz NOT NULL,
    completed_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (scenario_id, config_id, replicate)
);

CREATE INDEX IF NOT EXISTS runs_scenario_config_idx ON runs (scenario_id, config_id);

-- --------------------------------------------------------------------------
-- run_steps
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS run_steps (
    run_id          uuid        NOT NULL REFERENCES runs (run_id),
    step            smallint    NOT NULL CHECK (step >= 0),
    exogenous_delta jsonb       NOT NULL,
    arrivals        jsonb       NOT NULL,
    contest_delta   jsonb       NOT NULL,
    active          text[]      NOT NULL,
    eligible        smallint    NOT NULL CHECK (eligible >= 0),
    -- Position in the seeded stream. A function of the step and the graph
    -- (ADR-0015), so a value that does not match is a draw nobody planned.
    rng_counter     integer     NOT NULL CHECK (rng_counter >= 0),
    state_hash      text        NOT NULL,
    absorbed        text[]      NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, step)
);

-- --------------------------------------------------------------------------
-- events (spec §11.1)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS events (
    run_id        uuid        NOT NULL,
    step          smallint    NOT NULL,
    seq           smallint    NOT NULL,   -- ordering within a step
    actor_id      text        NOT NULL,
    obs_hash      bytea       NOT NULL,   -- canonical hash of the observation
    action        jsonb       NOT NULL,
    caused_by     jsonb,                  -- [{run_id,step,seq}, ...]
    factor_delta  jsonb       NOT NULL,   -- what this decision moved
    cache_hit     boolean     NOT NULL,
    tokens_in     integer     NOT NULL,
    tokens_out    integer     NOT NULL,
    latency_ms    integer,
    -- Why the action the model emitted was not the action executed: the actor
    -- has no lever on that factor, no channel to that party, or the payload
    -- did not parse (ADR-0018). NULL when it was executed as emitted. Recorded
    -- rather than repaired, so "how often did agents reach for something they
    -- do not have" is a query instead of a guess.
    coercion      text,
    PRIMARY KEY (run_id, step, seq)
) PARTITION BY HASH (run_id);           -- 32 partitions

-- Hash partitioning on run_id keeps a single run's events in one partition,
-- which is what the trace query in §11.2 walks, and spreads the write load of
-- a 36,000-run fan-out across 32 relations instead of one.
DO $$
DECLARE
    i integer;
BEGIN
    FOR i IN 0..31 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS events_p%s PARTITION OF events '
            'FOR VALUES WITH (MODULUS 32, REMAINDER %s)', i, i
        );
    END LOOP;
END $$;

-- The provenance walk starts from an actor's decisions within one run; the
-- primary key already serves (run_id, step, seq) lookups, so this index is for
-- the other direction -- every decision one actor took.
CREATE INDEX IF NOT EXISTS events_actor_idx ON events (run_id, actor_id);

-- --------------------------------------------------------------------------
-- Grants (invariant 6; deny-by-default per ADR-0005)
-- --------------------------------------------------------------------------

GRANT SELECT, INSERT ON runs       TO cascade_sim;
GRANT SELECT, INSERT ON run_steps  TO cascade_sim;
GRANT SELECT, INSERT ON events     TO cascade_sim;

GRANT SELECT ON runs      TO cascade_eval;
GRANT SELECT ON run_steps TO cascade_eval;
GRANT SELECT ON events    TO cascade_eval;

COMMENT ON TABLE events IS
    'Append-only decision log (spec §11.1, invariant 6). No role other than the '
    'owner holds UPDATE or DELETE; that absence is the enforcement.';

COMMIT;
