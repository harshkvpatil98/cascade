-- 010: collapsed ensemble forecasts, and the prompt-change audit (spec §9.1,
-- §3.3, §1.3; M6).
--
-- `forecasts` is PHASE 3's artifact: 200 replicates of one (scenario, config)
-- collapse to a single row carrying p_hat, the dispersion statistics and a
-- confidence interval. One row per (scenario, config) -- the same key the M7
-- ablation grid joins on.
--
-- Aggregation runs as `cascade_sim`, not `cascade_eval`. That is deliberate
-- and it is invariant 2 expressed as a phase boundary: collapsing runs into a
-- forecast must not be able to see an outcome, and the way to guarantee that
-- is to do it under the role that has no grant on `scenario_labels`. The
-- scoring in PHASE 4 is where labels enter, and that phase runs as `eval`.
--
-- `prompt_revisions` is the third integrity mechanism named in §1.3, alongside
-- the frozen split and the label grant: "any edit to an agent, decomposition,
-- or arbiter prompt after the first full backtest is recorded ... with the
-- Brier before and after, so tuning-on-test is visible in the record rather
-- than hidden." A revision made *before* the first backtest -- as r2 below is
-- -- has no before/after Brier to record, and says so.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- forecasts
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS forecasts (
    scenario_id   text    NOT NULL REFERENCES scenarios (scenario_id),
    config_id     text    NOT NULL,
    -- The ensemble mean of the terminal outcome scores (§9.1). Not the median
    -- and not the mode: the mean is the quantity the Brier score is defined
    -- against.
    p_hat         double precision NOT NULL CHECK (p_hat >= 0 AND p_hat <= 1),
    sigma         double precision NOT NULL CHECK (sigma >= 0),
    ci_lo         double precision NOT NULL,
    ci_hi         double precision NOT NULL,
    -- §9.2 asks for two independent corroborating statistics, not one
    -- threshold, so both travel with the flag they justify.
    modality      text    NOT NULL CHECK (modality IN ('single', 'multi')),
    dip_p         double precision,
    bimodality    double precision,
    n_replicates  integer NOT NULL CHECK (n_replicates > 0),
    mean_events   double precision NOT NULL,
    mean_steps    double precision NOT NULL,
    absorbed_runs integer NOT NULL DEFAULT 0,
    collapsed_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (scenario_id, config_id),
    CHECK (ci_lo <= ci_hi)
);

CREATE INDEX IF NOT EXISTS forecasts_config_idx ON forecasts (config_id);

-- --------------------------------------------------------------------------
-- prompt_revisions (spec §1.3)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS prompt_revisions (
    prompt_rev   text        NOT NULL,
    subsystem    text        NOT NULL CHECK (subsystem IN ('compiler', 'agent', 'arbiter')),
    changed_at   timestamptz NOT NULL DEFAULT now(),
    summary      text        NOT NULL,
    rationale    text        NOT NULL,
    -- NULL until there is a backtest on each side of the change. A revision
    -- made before the first full backtest has no before; one whose after has
    -- not been measured yet has no after. Neither is an excuse to omit the row.
    brier_before double precision,
    brier_after  double precision,
    PRIMARY KEY (prompt_rev, subsystem)
);

-- The revision that lands with this migration. Recorded here rather than by a
-- command so the audit trail cannot be skipped by forgetting to run one.
INSERT INTO prompt_revisions (prompt_rev, subsystem, summary, rationale)
VALUES (
    'r2',
    'compiler',
    'Give `volatility` an explicit scale: one step is 1/24 of the '
    'cutoff-to-resolution interval, sigma is per step in the factor''s own '
    '[0, 1] units, and an undriven factor wanders ~sqrt(24) x sigma over the '
    'horizon.',
    'r1 asked for "sigma of its per-step exogenous random walk, [0, 1]" with '
    'no scale anchor, so the number was uninterpretable and a model had no '
    'basis to choose it. Measured at M5: the activation rate of §7.3 is '
    'essentially a function of factor volatility against the 0.06 salience '
    'threshold -- at volatility >= 0.06 the scheduler pins to its cap (8 of 14 '
    'actors) and a run logs ~184 decisions instead of the 116.6 the §12.1 cost '
    'model assumes. The revision states the arithmetic relating one step to '
    'the horizon; it does not prescribe a value, and the resulting activation '
    'rate is whatever it measures. No Brier exists on either side: this '
    'predates the first backtest.'
)
ON CONFLICT (prompt_rev, subsystem) DO NOTHING;

-- --------------------------------------------------------------------------
-- Grants
--
-- `cascade_sim` collapses runs into forecasts, so it needs INSERT and the
-- UPDATE half of an upsert: a forecast is derived data and is recomputed when
-- more replicates land. That is not in tension with invariant 6, which is
-- about the event log -- `events` remains INSERT-only for every role.
-- --------------------------------------------------------------------------

GRANT SELECT, INSERT, UPDATE ON forecasts TO cascade_sim;
GRANT SELECT ON forecasts TO cascade_eval;
GRANT SELECT ON prompt_revisions TO cascade_sim, cascade_eval;

COMMIT;
