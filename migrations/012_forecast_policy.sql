-- 012: carry the decider policy onto the forecast (M7).
--
-- M5 recorded the decision that a run made with a stand-in decider is stamped
-- `policy = 'heuristic'` in the `runs` table, with the reasoning that "a
-- footnote is not a mechanism; the column is". The same argument applies one
-- level up and had not been applied: a forecast collapsed from heuristic runs
-- was indistinguishable from one collapsed from model-backed agents, so a
-- Brier computed over it would enter the report as a study result.
--
-- The column is derived, like everything else in `forecasts`: it is the policy
-- of the runs that produced it. A (scenario, config) whose runs disagree is a
-- defect -- half a mechanism and half a study -- and `collapse_inputs` reports
-- the mix rather than picking one.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

ALTER TABLE forecasts
    ADD COLUMN IF NOT EXISTS policy text NOT NULL DEFAULT 'agent';

-- Rows written before this column existed came from the agent path -- it was
-- the only path that reached `ensemble collapse`. The default records that
-- rather than leaving it NULL, which would be a third state meaning neither.
ALTER TABLE forecasts
    DROP CONSTRAINT IF EXISTS forecasts_policy_check;
ALTER TABLE forecasts
    ADD CONSTRAINT forecasts_policy_check
    CHECK (policy IN ('agent', 'heuristic', 'mixed'));

CREATE INDEX IF NOT EXISTS forecasts_policy_idx ON forecasts (policy);

COMMENT ON COLUMN forecasts.policy IS
    'Decider that produced the runs behind this forecast. Only ''agent'' is '
    'study data; ''heuristic'' is a mechanism check and ''mixed'' is a defect.';

COMMIT;
