"""Pure leakage helpers (spec §4.2).

The database-backed probes live in ``tests/leakage/``. What is tested here is
the logic those probes depend on: the boundary condition on ``as_of``, the
construction of poison documents, and the signature scanner's discrimination.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.ledger.schema import Scenario, ScenarioLabel, ScenarioRecord
from cascade.retrieval.leakage import (
    POISON_MARKER,
    build_poison_documents,
    resolution_signature,
    scan_signatures,
    violations,
)
from cascade.retrieval.schema import RetrievedChunk

CUTOFF = datetime(2019, 6, 1, tzinfo=UTC)


def chunk(chunk_id: str, published_at: datetime, body: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        ordinal=0,
        body=body,
        published_at=published_at,
        source="ccnews",
        url="",
        title="",
        distance=0.1,
    )


def record(
    scenario_id: str = "s1",
    outcome: int = 1,
    question: str = "Will the merger be blocked?",
) -> ScenarioRecord:
    return ScenarioRecord(
        scenario=Scenario(
            scenario_id=scenario_id,
            question=question,
            resolution_criterion="Resolves YES if blocked.",
            cutoff_ts=CUTOFF,
            resolve_ts=datetime(2020, 1, 1, tzinfo=UTC),
            domain="corporate",
            source="polymarket",
            source_ref="ref",
            party_rule="named_parties",
            party_names=("Acme", "Regulator"),
            event_group=None,
        ),
        label=ScenarioLabel(
            scenario_id=scenario_id,
            outcome=outcome,  # type: ignore[arg-type]
            resolved_at=datetime(2020, 1, 1, tzinfo=UTC),
        ),
    )


# ---------------------------------------------------------------------------
# violations -- the boundary condition
# ---------------------------------------------------------------------------


def test_a_chunk_before_the_cutoff_is_not_a_violation() -> None:
    assert violations([chunk("c1", CUTOFF - timedelta(seconds=1))], as_of=CUTOFF) == ()


def test_a_chunk_exactly_at_the_cutoff_is_a_violation() -> None:
    """`chronofence_search` promises *strictly* before.

    A document published at the cutoff instant existed when the forecast was
    made, so admitting it would quietly widen the lock by one timestamp.
    """
    assert len(violations([chunk("c1", CUTOFF)], as_of=CUTOFF)) == 1


def test_a_chunk_after_the_cutoff_is_a_violation() -> None:
    assert len(violations([chunk("c1", CUTOFF + timedelta(seconds=1))], as_of=CUTOFF)) == 1


@given(st.integers(min_value=1, max_value=10**7))
def test_any_chunk_at_or_after_the_cutoff_is_reported(offset_seconds: int) -> None:
    late = chunk("c1", CUTOFF + timedelta(seconds=offset_seconds))
    assert violations([late], as_of=CUTOFF) == (late,)


# ---------------------------------------------------------------------------
# Poison documents
# ---------------------------------------------------------------------------


def test_poison_documents_are_dated_after_resolution() -> None:
    """Post-cutoff by a wide margin, so the probe tests the lock not rounding."""
    documents = build_poison_documents([record()], count=3)
    for document in documents:
        assert document.published_at > document.cutoff_ts
        assert document.published_at > datetime(2020, 1, 1, tzinfo=UTC)


def test_poison_documents_state_the_outcome() -> None:
    """The probe must be the easiest possible thing for the retriever to return."""
    yes = build_poison_documents([record(outcome=1)], count=1)[0]
    no = build_poison_documents([record(scenario_id="s2", outcome=0)], count=1)[0]
    assert "resolved YES" in yes.body
    assert "resolved NO" in no.body


def test_poison_documents_restate_the_question_for_similarity() -> None:
    document = build_poison_documents([record()], count=1)[0]
    assert "Will the merger be blocked?" in document.body


def test_every_poison_document_is_identifiable() -> None:
    """A stray row that cannot be told from real evidence is worse than a leak."""
    for document in build_poison_documents([record()], count=5):
        assert POISON_MARKER in document.document_id


def test_poison_document_ids_are_unique() -> None:
    documents = build_poison_documents([record()], count=50)
    assert len({document.document_id for document in documents}) == 50


def test_poison_documents_spread_over_all_scenarios() -> None:
    records = [record(f"s{i}") for i in range(10)]
    documents = build_poison_documents(records, count=30)
    assert len({document.scenario_id for document in documents}) == 10


def test_poison_generation_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        build_poison_documents([record()], count=0)
    with pytest.raises(ValueError, match="without scenarios"):
        build_poison_documents([], count=5)


# ---------------------------------------------------------------------------
# Signature scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The question resolved YES.",
        "Final determination: the deal is blocked.",
        "The outcome is now confirmed.",
        "The matter has been settled.",
        "Officially concluded last week.",
    ],
)
def test_retrospective_phrases_are_flagged(text: str) -> None:
    assert resolution_signature(text)


@pytest.mark.parametrize(
    "text",
    [
        "The regulator approved the merger today.",
        "Analysts expect the coalition to hold.",
        "The company wins a contract in Berlin.",
        "A decision is expected next quarter.",
    ],
)
def test_ordinary_pre_cutoff_reporting_is_not_flagged(text: str) -> None:
    """Words like "approved" or "wins" saturate live coverage of a question.

    Flagging them would mark most of the corpus and make the reviewed count
    meaningless.
    """
    assert resolution_signature(text) == ()


def test_signature_matching_is_case_insensitive() -> None:
    assert resolution_signature("THE QUESTION RESOLVED YES")


def test_scan_flags_on_similarity_even_without_a_phrase() -> None:
    """Regex alone misses paraphrase."""
    findings = scan_signatures(
        [chunk("c1", CUTOFF, body="entirely innocuous prose")],
        scenario_id="s1",
        cutoff_ts=CUTOFF,
        similarities=[0.95],
        threshold=0.80,
    )
    assert len(findings) == 1
    assert findings[0].phrases == ()


def test_scan_flags_on_a_phrase_even_at_low_similarity() -> None:
    """Similarity alone flags every on-topic document; the criterion describes the topic."""
    findings = scan_signatures(
        [chunk("c1", CUTOFF, body="The question resolved NO.")],
        scenario_id="s1",
        cutoff_ts=CUTOFF,
        similarities=[0.10],
        threshold=0.80,
    )
    assert len(findings) == 1
    assert findings[0].phrases == ("resolved no",)


def test_scan_returns_nothing_for_ordinary_evidence() -> None:
    assert (
        scan_signatures(
            [chunk("c1", CUTOFF, body="The regulator opened a review.")],
            scenario_id="s1",
            cutoff_ts=CUTOFF,
            similarities=[0.2],
            threshold=0.80,
        )
        == ()
    )


def test_scan_refuses_mismatched_similarity_counts() -> None:
    """Pairing a chunk with another chunk's score would misattribute a finding."""
    with pytest.raises(ValueError, match="cannot pair"):
        scan_signatures(
            [chunk("c1", CUTOFF), chunk("c2", CUTOFF)],
            scenario_id="s1",
            cutoff_ts=CUTOFF,
            similarities=[0.5],
            threshold=0.80,
        )
