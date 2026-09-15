-- 016: correct `runs.llm_calls` for runs that never reached the model (M8).
--
-- The kernel incremented `llm_calls` once per decision rather than once per
-- decision that came from the model. For a run made with the stand-in decider
-- -- stamped `policy = 'heuristic'` since M5, precisely so these runs are
-- distinguishable -- that recorded one model call per decision while zero were
-- made.
--
-- It is not a cosmetic miscount. §12.4 reconciles `runs` against Langfuse and
-- blocks the report on a discrepancy above 2%; summing a decision count as a
-- call count made the local side report 452,328 calls costing $0.000000
-- against a Langfuse total of $0.000000, which agrees to 0.0000% and passes.
-- A gate that compares zero with zero is a gate that cannot fail, and this one
-- guards the study's cost claim.
--
-- The kernel is fixed (`Decision.from_model`), so new runs record it
-- correctly. This corrects the rows already written. It touches `runs`, which
-- is a ledger of completed work -- **not** `events`, which is append-only and
-- stays untouched (invariant 6). The correction is narrow by construction: it
-- can only affect rows whose policy says no model was involved, so a run that
-- did call the model cannot be altered by it.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

UPDATE runs
   SET llm_calls = 0
 WHERE policy = 'heuristic'
   AND llm_calls <> 0
   AND cost_usd = 0
   AND tokens_in = 0
   AND tokens_out = 0;

COMMIT;
