"""The Chronofence bench: latency and recall (spec §4.2).

Reports p50/p95/p99 **and** recall@20 against exhaustive search. Both, always,
in the same report. A latency number without a recall number says only that
the index returned something quickly, and the cheapest way to return something
quickly is to return the wrong thing -- `probes = 1` would halve the p95 and
quietly cost most of the recall the study depends on.

Recall is measured on a sample (`retrieval.bench_recall_sample`, 5%) because
the oracle is exhaustive: on this corpus one exact query costs roughly two
orders of magnitude more than one indexed query, so scoring all 10,000 would
turn a two-minute bench into a multi-hour one for a number that is already
tight at 500 samples.

Query embedding is excluded from the measured interval. The 15 ms budget in
§4.2 is a budget for the *retrieval*, and an agent's query vector at M5 comes
from the same embedder amortised across a batch; folding a cold single-text
encode into every sample would measure the model, not the index.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from cascade.config import Settings
from cascade.corpus.embed import Embedder
from cascade.retrieval.metrics import RecallSummary, recall_at_k, summarise_latency
from cascade.retrieval.queries import BenchQuery, build_queries
from cascade.retrieval.schema import LatencySummary
from cascade.retrieval.search import Chronofence

__all__ = ["MIN_RECALL_SAMPLE", "BenchResult", "run_bench"]

# Below this many scored queries, recall@k is too noisy to decide a criterion
# stated to two decimal places. At `--queries 100` the 5% sample is 5 queries,
# where a single imperfect result moves the mean by 0.04 -- twice the margin
# the criterion is judged on. The bench still refuses to pass in that case, but
# it says the sample was too small rather than reporting a verdict it cannot
# support.
MIN_RECALL_SAMPLE = 30


@dataclass(frozen=True)
class BenchResult:
    """Everything the M3 acceptance criteria are read off.

    `empty_results` is carried alongside the percentiles because a query that
    matched nothing is fast, and a bench that met its p95 by retrieving
    nothing would otherwise look like a pass.
    """

    queries: int
    k: int
    latency: LatencySummary
    recall: RecallSummary
    empty_results: int
    distinct_cutoffs: int
    earliest_cutoff: str
    latest_cutoff: str
    probes: int
    elapsed_s: float

    def meets(self, settings: Settings) -> tuple[bool, tuple[str, ...]]:
        """Verdict against the configured targets, with the reasons it failed.

        Returns measured failures, never a tolerance. If p95 lands at 15.4 ms
        against a 15 ms budget the answer is that it failed at 15.4 ms.
        """
        target = settings.retrieval
        failures: list[str] = []
        if self.latency.n == 0:
            failures.append("no queries were executed")
        if self.latency.p95 >= target.target_p95_ms:
            failures.append(
                f"p95 {self.latency.p95:.2f} ms is not below the "
                f"{target.target_p95_ms:.0f} ms budget"
            )
        if not self.recall.measured:
            failures.append(
                f"recall@{self.recall.k} could not be measured: no sampled query "
                "had any admissible evidence at its cutoff"
            )
        elif self.recall.scored < MIN_RECALL_SAMPLE:
            failures.append(
                f"recall@{self.recall.k} measured {self.recall.mean:.4f} on only "
                f"{self.recall.scored} sampled queries, too few to decide a criterion "
                f"stated to two decimals -- run the configured "
                f"{settings.retrieval.bench_queries:,}-query bench rather than a subset"
            )
        elif self.recall.mean <= target.target_recall_at_k:
            failures.append(
                f"recall@{self.recall.k} {self.recall.mean:.4f} is not above "
                f"{target.target_recall_at_k:.2f}"
            )
        return (not failures, tuple(failures))


def _sample_indices(total: int, fraction: float) -> frozenset[int]:
    """Evenly spaced sample positions.

    Evenly spaced rather than random: the query set is built round-robin over
    scenarios ordered by id, so a stride samples every scenario roughly
    equally, while a random draw of 5% would leave whole cutoffs unmeasured
    and make the recall number depend on the seed.
    """
    if total <= 0:
        return frozenset()
    wanted = max(1, round(total * fraction))
    if wanted >= total:
        return frozenset(range(total))
    stride = total / wanted
    return frozenset(min(total - 1, round(index * stride)) for index in range(wanted))


def run_bench(
    settings: Settings,
    *,
    queries: Sequence[BenchQuery] | None = None,
    count: int | None = None,
    progress: object = None,
) -> BenchResult:
    """Run the bench and return measured values.

    ``progress`` is any object with an ``advance(int)`` method, or None. Typed
    loosely on purpose: the pure measurement path must not acquire a Rich
    dependency to report itself.
    """
    from cascade.ledger.store import load_scenarios

    retrieval = settings.retrieval
    started = time.monotonic()

    if queries is None:
        scenarios = load_scenarios(settings, role="admin")
        queries = build_queries(
            scenarios,
            count=count if count is not None else retrieval.bench_queries,
            salt=settings.study.salt,
        )
    if not queries:
        raise ValueError("bench requires at least one query")

    embedder = Embedder(
        model_name=settings.models.embedding,
        batch_size=settings.corpus.embed_batch_size,
    )
    # Embedded up front, in one batch, so the measured interval contains only
    # the database round trip. Also amortises model load across the whole run.
    vectors = embedder.encode([query.text for query in queries])

    sampled = _sample_indices(len(queries), retrieval.bench_recall_sample)
    k = retrieval.bench_recall_k

    latencies: list[float] = []
    empty = 0
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    # `eval` rather than `sim`: the exact oracle is granted only to eval, and
    # running both arms on one connection keeps the comparison honest about
    # cache state. The indexed arm is identical under either role -- the same
    # SECURITY DEFINER function with the same pinned probes.
    with Chronofence(settings, role="eval") as fence:
        probes = fence.probes()
        for index, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
            result = fence.search(vector, as_of=query.as_of, k=k)
            latencies.append(result.elapsed_ms)
            if not result.chunks:
                empty += 1

            if index in sampled:
                truth = fence.search_exact(vector, as_of=query.as_of, k=k)
                pairs.append((result.chunk_ids, truth.chunk_ids))

            if progress is not None:
                advance = getattr(progress, "advance", None)
                if callable(advance):
                    advance(1)

    cutoffs = sorted({query.as_of for query in queries})
    return BenchResult(
        queries=len(queries),
        k=k,
        latency=summarise_latency(latencies),
        recall=recall_at_k(pairs, k=k),
        empty_results=empty,
        distinct_cutoffs=len(cutoffs),
        earliest_cutoff=cutoffs[0].isoformat(),
        latest_cutoff=cutoffs[-1].isoformat(),
        probes=probes,
        elapsed_s=time.monotonic() - started,
    )
