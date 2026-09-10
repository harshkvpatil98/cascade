"""Chronofence: time-locked retrieval (M3, spec §4).

``as_of`` is a required parameter everywhere in this package -- no default in
Python, no default in SQL (invariant 1). The time filter is structural: the
planner prunes post-cutoff partitions before touching a vector, so the lock
does not depend on a caller remembering a predicate.

The public surface is deliberately small. Everything that reads the corpus
goes through :class:`Chronofence`, which binds one database role for its
lifetime; opened as ``sim`` -- what the simulation actually runs as -- the
exhaustive oracle is not reachable, because ``cascade_sim`` is not granted it.
"""

from __future__ import annotations

from cascade.retrieval.bench import BenchResult, run_bench
from cascade.retrieval.index import apply_plans, measure, plan_all, target_lists
from cascade.retrieval.metrics import (
    RecallSummary,
    histogram,
    percentile,
    recall_at_k,
    summarise_latency,
)
from cascade.retrieval.queries import BenchQuery, build_queries
from cascade.retrieval.schema import (
    IndexPlan,
    IndexReport,
    LatencySummary,
    PartitionIndex,
    RetrievedChunk,
    SearchResult,
)
from cascade.retrieval.search import Chronofence

__all__ = [
    "BenchQuery",
    "BenchResult",
    "Chronofence",
    "IndexPlan",
    "IndexReport",
    "LatencySummary",
    "PartitionIndex",
    "RecallSummary",
    "RetrievedChunk",
    "SearchResult",
    "apply_plans",
    "build_queries",
    "histogram",
    "measure",
    "percentile",
    "plan_all",
    "recall_at_k",
    "run_bench",
    "summarise_latency",
    "target_lists",
]
