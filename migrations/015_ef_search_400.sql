-- 015: raise the pinned `hnsw.ef_search` to 400 (spec §4.2, §4.3; M8).
--
-- Migration 014 pinned 100 on the strength of a single-partition measurement
-- (recall@20 1.0000 on chunks_2017q4 at ef_search 40). Measured afterwards
-- across 120 queries from the bench's own generator -- short, agent-shaped
-- queries over all 47 partitions, scored against `chronofence_search_exact`:
--
--     ef_search   recall@20   p50      p95
--     100         0.8521      19.09    24.51 ms
--     200         0.9133      29.44    36.12 ms
--     400         0.9458      49.51    59.76 ms
--     800         0.9533      82.07    97.96 ms
--
-- 400 is the smallest measured value clearing §4.2's recall@20 > 0.92, which
-- is the same rule ADR-0013 applied to `probes` and the same reason.
--
-- **Why recall is bought at the cost of latency here, deliberately.** §4.3's
-- 15 ms budget was derived for §7.2's per-step retrieval -- 4.2M searches
-- across the study. ADR-0019 replaced that with one retrieval per (scenario,
-- actor), which is roughly 2,520 searches in total. At the measured p95 the
-- study's entire retrieval cost is about two and a half minutes, while recall
-- sets the quality of every agent's evidence for all 36,000 runs. The p95
-- criterion is still missed and still reported as missed; it is not relaxed,
-- and `cascade retrieval bench` still exits 3.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

ALTER FUNCTION chronofence_search(halfvec, timestamptz, int)
    SET hnsw.ef_search = 400;

COMMIT;
