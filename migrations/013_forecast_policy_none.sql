-- 013: a forecast can have no decider (M7).
--
-- Migration 012 added `forecasts.policy` with the vocabulary 'agent' /
-- 'heuristic' / 'mixed', which covers every forecast collapsed from runs. It
-- does not cover §10.2's climatology baseline, which is arithmetic over the
-- sealed base rate: no runs, no kernel, no model. Labelling it 'agent' would
-- claim a decider that never existed, and it is the one baseline for which
-- that claim is checkable and false.
--
-- 'none' rather than 'climatology' because the column describes the decider,
-- not the baseline: any forecast produced without one takes it.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

ALTER TABLE forecasts DROP CONSTRAINT IF EXISTS forecasts_policy_check;
ALTER TABLE forecasts
    ADD CONSTRAINT forecasts_policy_check
    CHECK (policy IN ('agent', 'heuristic', 'mixed', 'none'));

COMMENT ON COLUMN forecasts.policy IS
    'Decider that produced the runs behind this forecast. ''agent'' is study '
    'data; ''none'' is a forecast with no decider (climatology); ''heuristic'' '
    'is a mechanism check and ''mixed'' is a defect.';

COMMIT;
