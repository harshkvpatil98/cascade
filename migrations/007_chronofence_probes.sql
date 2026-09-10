-- 007: raise the pinned `ivfflat.probes` from 10 to 40.
--
-- Spec §4.2 specifies `probes = 10`. Measured on the 409,899-chunk corpus with
-- `lists = round(sqrt(rows_in_partition))` per ADR-0012, that yields
-- recall@20 = 0.8331 against the >0.92 acceptance criterion. See ADR-0013 for
-- the measurement and the reasoning; the short version is that the spec's
-- value assumes one index, while quarterly partitioning (ADR-0004) fans a
-- single query across 36 of them and each probes independently.
--
-- Measured trade-off curve (80 queries, exact ground truth, k = 20):
--
--     probes    recall@20    p95
--       10       0.8331     4.44 ms
--       20       0.9012     7.20 ms
--       40       0.9450    11.86 ms
--       80       0.9662    21.55 ms
--      160       0.9812    38.62 ms
--
-- 40 is the smallest value on this curve that clears recall; 80 and above
-- break the 15 ms budget. Confirmed end to end on the full bench path at
-- 400 queries: p95 12.97 ms, p99 14.67 ms, recall@20 0.9350.
--
-- `chronofence_search_exact` is untouched -- it stays at pgvector's maximum so
-- it remains exhaustive, which is what makes it a valid oracle.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

ALTER FUNCTION chronofence_search(halfvec, timestamptz, int)
    SET ivfflat.probes = 40;

COMMIT;
