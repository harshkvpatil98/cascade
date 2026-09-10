"""Date monotonicity over the retrieval trace (spec §4.2, criterion 4).

The other leakage probes test the *mechanism*: the grant, the signature, the
poison. This one tests the **output** -- every chunk that any query actually
returned, checked against the cutoff it was returned under.

That distinction is the point. A regression in `chronofence_search` that the
SQL's own tests miss still has to produce rows, and those rows are what this
sees. Hypothesis drives it over the real registry so the cutoffs are the
study's own, not invented ones: a property that only holds for hand-picked
dates is not the property the study needs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from cascade.config import Settings
from cascade.corpus.embed import EMBEDDING_DIM
from cascade.retrieval.leakage import violations
from cascade.retrieval.search import Chronofence

pytestmark = pytest.mark.leakage


@pytest.fixture(scope="module")
def fence(live_settings: Settings) -> Any:
    with Chronofence(live_settings, role="eval") as bound:
        yield bound


@pytest.fixture(scope="module")
def cutoffs(records: tuple[Any, ...]) -> list[datetime]:
    return sorted({record.scenario.cutoff_ts for record in records})


@pytest.fixture(scope="module")
def query_vectors(live_settings: Settings, records: tuple[Any, ...], embedder: Any) -> Any:
    """A handful of real query vectors, reused across examples.

    Embedding inside a Hypothesis example would make the model load part of
    the shrinking loop and the test would take hours.
    """
    texts = [record.scenario.question for record in records[:24]]
    return embedder.encode(texts)


@given(
    cutoff_index=st.integers(min_value=0, max_value=179),
    vector_index=st.integers(min_value=0, max_value=23),
    k=st.integers(min_value=1, max_value=60),
)
@hyp_settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_no_returned_chunk_is_dated_at_or_after_its_cutoff(
    fence: Any,
    cutoffs: list[datetime],
    query_vectors: Any,
    cutoff_index: int,
    vector_index: int,
    k: int,
) -> None:
    """The invariant, over arbitrary (cutoff, query, k) triples from the study."""
    as_of = cutoffs[cutoff_index % len(cutoffs)]
    vector = query_vectors[vector_index % len(query_vectors)]

    result = fence.search(vector, as_of=as_of, k=k)
    offenders = violations(result.chunks, as_of=as_of)

    assert offenders == (), (
        f"{len(offenders)} chunks dated at or after {as_of.isoformat()}: "
        f"{[(chunk.chunk_id, chunk.published_at.isoformat()) for chunk in offenders[:3]]}"
    )


@given(
    cutoff_index=st.integers(min_value=0, max_value=179),
    vector_index=st.integers(min_value=0, max_value=23),
)
@hyp_settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_the_exact_oracle_obeys_the_same_lock(
    fence: Any,
    cutoffs: list[datetime],
    query_vectors: Any,
    cutoff_index: int,
    vector_index: int,
) -> None:
    """The oracle is a second implementation and could leak independently.

    It is the ground truth recall is measured against, so a leak here would
    make the *approximate* path look like it was missing rows it should never
    have returned.
    """
    as_of = cutoffs[cutoff_index % len(cutoffs)]
    vector = query_vectors[vector_index % len(query_vectors)]

    result = fence.search_exact(vector, as_of=as_of, k=20)
    assert violations(result.chunks, as_of=as_of) == ()


@given(cutoff_index=st.integers(min_value=0, max_value=179))
@hyp_settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_shrinking_the_cutoff_never_adds_evidence(
    fence: Any,
    cutoffs: list[datetime],
    query_vectors: Any,
    cutoff_index: int,
) -> None:
    """Monotonicity in ``as_of``: an earlier cutoff can only see less.

    Catches a filter applied to the wrong side of the comparison, which would
    still pass a fixed-date assertion.
    """
    as_of = cutoffs[cutoff_index % len(cutoffs)]
    earlier = as_of - timedelta(days=365)
    vector = query_vectors[0]

    later_result = fence.search_exact(vector, as_of=as_of, k=20)
    earlier_result = fence.search_exact(vector, as_of=earlier, k=20)

    latest_early = earlier_result.latest_published_at()
    if latest_early is not None:
        assert latest_early < earlier
    # Everything visible at the earlier cutoff is visible at the later one.
    assert (
        set(earlier_result.chunk_ids) <= set(later_result.chunk_ids)
        or len(later_result.chunks) == 20
    )


def test_a_cutoff_before_the_corpus_returns_nothing(fence: Any) -> None:
    """The degenerate end of the range: no evidence exists, so none is returned.

    Confirms the lock does not fall open when there is nothing to filter.
    """
    result = fence.search([0.0] * EMBEDDING_DIM, as_of=datetime(2000, 1, 1, tzinfo=UTC), k=20)
    assert result.chunks == ()
