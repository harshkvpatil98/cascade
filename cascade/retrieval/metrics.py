"""Pure measurement functions for the retrieval bench (spec §4.2).

No clock, no RNG from global state, no I/O -- the whole module is a function
of its arguments, which is what makes the percentile and recall definitions
property-testable rather than merely plausible.

Two definitions are pinned here because the acceptance criteria are stated to
three significant figures and an unstated definition is not reproducible:

* **Percentiles** use linear interpolation between the two closest ranks on
  the sorted sample -- the same definition as ``numpy.percentile`` with its
  default ``method='linear'``. Nearest-rank would report a p95 up to one
  sample's width lower on a 10,000-point set, which is enough to move a 15 ms
  verdict.
* **recall@k** is ``|approx_k ∩ exact_k| / |exact_k|`` per query, averaged over
  queries that returned anything. Queries whose cutoff admits no evidence at
  all have no ground truth to recall and are counted separately rather than
  scored as 1.0 -- scoring them as perfect would let an empty corpus report
  perfect recall, which is precisely backwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from cascade.retrieval.schema import LatencySummary

__all__ = [
    "RecallSummary",
    "histogram",
    "percentile",
    "recall_at_k",
    "summarise_latency",
]

# Bucket edges in milliseconds. Dense below the 15 ms budget because that is
# where the verdict is decided, then coarse -- a run whose tail is past 100 ms
# has failed and the exact shape of the failure is not what the reader needs.
_HISTOGRAM_EDGES: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
    30.0,
    50.0,
    100.0,
    float("inf"),
)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``values`` (``q`` in [0, 100]).

    Preserves the definition the acceptance criteria are read against: linear
    interpolation between closest ranks, identical to ``numpy.percentile``'s
    default. Implemented here rather than delegated so the pure core stays
    free of an array dependency and the definition is visible at the point it
    is relied on.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    low, high = ordered[lower], ordered[upper]
    # `low + (high - low) * weight`, not `low * (1 - weight) + high * weight`.
    # The two are equivalent in exact arithmetic and not in floating point:
    # when low == high the second form computes `a*0.11 + a*0.89`, which can
    # land one ULP below `a`. A property test caught it as p99 < p95 on a
    # sample whose top two values were equal -- a monotonicity violation in
    # the reported percentiles, which is exactly the kind of defect that would
    # be dismissed as noise if it ever surfaced in a bench report.
    # This is the same form numpy uses.
    return float(low + (high - low) * weight)


def histogram(
    values: Sequence[float], edges: Sequence[float] = _HISTOGRAM_EDGES
) -> tuple[tuple[float, float, int], ...]:
    """Bucket ``values`` into ``[lower, upper)`` bins.

    Half-open upwards so a value exactly on an edge lands in one bucket only
    and the counts sum to ``len(values)``; the final bucket is unbounded so
    nothing is silently dropped off the top.
    """
    if len(edges) < 2:
        raise ValueError("a histogram needs at least two edges")
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index in range(len(counts)):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    return tuple(
        (float(edges[index]), float(edges[index + 1]), counts[index])
        for index in range(len(counts))
    )


def summarise_latency(samples: Sequence[float]) -> LatencySummary:
    """Percentiles and a histogram over measured latencies, in milliseconds."""
    if not samples:
        return LatencySummary(
            n=0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            minimum=0.0,
            maximum=0.0,
            histogram=histogram([]),
        )
    return LatencySummary(
        n=len(samples),
        p50=percentile(samples, 50.0),
        p95=percentile(samples, 95.0),
        p99=percentile(samples, 99.0),
        minimum=float(min(samples)),
        maximum=float(max(samples)),
        histogram=histogram(samples),
    )


@dataclass(frozen=True)
class RecallSummary:
    """recall@k over a sample of queries.

    ``empty_ground_truth`` is reported rather than folded into the mean: those
    queries had no admissible evidence at their cutoff, which is a fact about
    corpus coverage and belongs in the report next to the recall number, not
    inside it.
    """

    k: int
    scored: int
    empty_ground_truth: int
    mean: float
    minimum: float
    perfect: int

    @property
    def measured(self) -> bool:
        """False when nothing could be scored, so a caller cannot read 0.0 as a result."""
        return self.scored > 0


def recall_at_k(pairs: Iterable[tuple[Sequence[str], Sequence[str]]], *, k: int) -> RecallSummary:
    """Mean recall@k of approximate results against exact ground truth.

    ``pairs`` yields ``(approximate_ids, exact_ids)`` per query, each already
    ordered nearest-first. Both are truncated to ``k`` before comparison: an
    approximate search asked for more than it is scored on would be credited
    for hits outside the window the criterion is about.

    Compared as sets. Rank order within the top k is not part of the recall@20
    criterion, and requiring it would be a stricter, different measurement
    reported under the criterion's name.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores: list[float] = []
    empty = 0
    perfect = 0
    for approximate, exact in pairs:
        truth = set(exact[:k])
        if not truth:
            empty += 1
            continue
        found = set(approximate[:k]) & truth
        score = len(found) / len(truth)
        scores.append(score)
        if score == 1.0:
            perfect += 1
    return RecallSummary(
        k=k,
        scored=len(scores),
        empty_ground_truth=empty,
        mean=sum(scores) / len(scores) if scores else 0.0,
        minimum=min(scores) if scores else 0.0,
        perfect=perfect,
    )
