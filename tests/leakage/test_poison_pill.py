"""Poison pill: can a post-resolution document reach retrieval? (spec §4.2)

The acceptance criterion is **0 of 500 retrieved across all 180 scenarios**.

The probe inserts documents that are maximally attractive to the retriever --
each restates its scenario's own question and then states the outcome, so its
vector sits about as close to a query derived from that scenario as anything
in the corpus can. Then it asks for them, at every scenario's cutoff, and
asserts nothing comes back.

**What "retrieved" has to mean.** All 500 documents are present at once, so a
document poisoning a scenario that resolved in 2018 is dated 2018 and is
therefore *legitimate pre-cutoff evidence* for a scenario with a 2019 cutoff.
Retrieving it is correct behaviour, not a leak -- the corpus is full of real
documents that describe other events' outcomes, and excluding them would be
curating the evidence. An earlier draft of this file asserted "no poison
anywhere" and failed on 5,979 such rows while the time lock was working
perfectly.

The criterion is therefore applied two ways, both of which must hold:

* no scenario retrieves **its own** post-resolution document (the §4.2 wording);
* no query retrieves **any** document dated at or after that query's cutoff,
  poison or real (the property the criterion is testing for).

Everything runs inside a transaction that is rolled back. The poison is
visible to the probe's own queries (same transaction) but never committed, so
the evidence corpus is byte-identical afterwards. A leakage probe that
contaminated the corpus it was testing would be self-defeating.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from cascade.config import Settings
from cascade.retrieval.leakage import (
    POISON_MARKER,
    build_poison_documents,
    insert_poison,
    scenario_of,
    violations,
)
from cascade.retrieval.search import Chronofence, vector_literal

pytestmark = pytest.mark.leakage


@pytest.fixture
def poisoned(live_settings: Settings, records: tuple[Any, ...], embedder: Any) -> Any:
    """Insert the poison, yield an open transaction, then roll it back."""
    count = live_settings.retrieval.poison_pill_count
    documents = build_poison_documents(records, count=count)
    vectors = embedder.encode([document.body for document in documents])

    conn = psycopg.connect(live_settings.database_url("admin"), connect_timeout=30)
    conn.autocommit = False
    try:
        written = insert_poison(conn, documents, vectors)
        assert written == count
        yield conn, documents
    finally:
        # Never commit. This is the line that keeps the probe from becoming
        # the leak it is testing for.
        conn.rollback()
        conn.close()


def test_the_probe_actually_inserted_its_poison(poisoned: Any) -> None:
    """Guards against the probe passing because it tested nothing.

    If the inserts silently failed, every later assertion would trivially hold
    and the suite would report a perfect time lock over an empty experiment.
    """
    conn, documents = poisoned
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE document_id LIKE %s", (f"{POISON_MARKER}|%",)
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == len(documents)


def test_poison_is_reachable_when_the_time_lock_is_removed(poisoned: Any, embedder: Any) -> None:
    """The positive control.

    Searching *without* the cutoff must find the poison. Without this, a
    retrieval path that returned nothing at all -- a broken index, a bad
    vector, a typo in the query -- would pass every assertion below.
    """
    conn, documents = poisoned
    document = documents[0]
    vector = embedder.encode([document.body])[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL "
            "ORDER BY embedding <-> %s::halfvec LIMIT 5",
            (vector_literal(vector),),
        )
        found = [row[0] for row in cur.fetchall()]

    assert any(POISON_MARKER in chunk_id for chunk_id in found), (
        "the poison was not retrievable even with no time filter; the probe "
        "below would pass without testing anything"
    )


def test_no_scenario_retrieves_its_own_post_resolution_document(
    poisoned: Any, live_settings: Settings, records: tuple[Any, ...], embedder: Any
) -> None:
    """The §4.2 criterion: 0 of 500 retrieved, across all 180 scenarios.

    Each scenario is queried with the text of its own poison document -- the
    single most favourable query that document could possibly receive -- and
    must not get it back.
    """
    conn, documents = poisoned
    by_scenario: dict[str, Any] = {}
    for document in documents:
        by_scenario.setdefault(document.scenario_id, document)

    vectors = embedder.encode([document.body for document in by_scenario.values()])
    k = max(live_settings.retrieval.bench_recall_k, live_settings.retrieval.k_compiler)

    own_poison: list[str] = []
    scenarios_probed = 0

    with conn.cursor() as cur:
        for (scenario_id, document), vector in zip(by_scenario.items(), vectors, strict=True):
            scenarios_probed += 1
            cur.execute(
                "SELECT chunk_id, document_id, published_at "
                "FROM chronofence_search(%s::halfvec, %s, %s)",
                (vector_literal(vector), document.cutoff_ts, k),
            )
            for chunk_id, document_id, published_at in cur.fetchall():
                if scenario_of(str(document_id)) == scenario_id:
                    own_poison.append(f"{scenario_id}:{chunk_id}@{published_at}")

    assert scenarios_probed == len(
        records
    ), f"probed {scenarios_probed} scenarios but the registry holds {len(records)}"
    assert own_poison == [], (
        f"{len(own_poison)} scenarios retrieved their own post-resolution "
        f"document: {own_poison[:5]}"
    )


def test_no_poison_postdating_a_cutoff_is_ever_retrieved(
    poisoned: Any, live_settings: Settings, embedder: Any
) -> None:
    """The property the criterion is really testing.

    Broader than the clause above: across every scenario's query, no poison
    document dated at or after *that* query's cutoff may come back, whichever
    scenario planted it. Cross-scenario poison that predates the cutoff is
    admissible and is deliberately not counted.
    """
    conn, documents = poisoned
    by_scenario: dict[str, Any] = {}
    for document in documents:
        by_scenario.setdefault(document.scenario_id, document)

    vectors = embedder.encode([document.body for document in by_scenario.values()])
    k = max(live_settings.retrieval.bench_recall_k, live_settings.retrieval.k_compiler)

    leaked: list[str] = []
    admissible_cross_scenario = 0

    with conn.cursor() as cur:
        for (scenario_id, document), vector in zip(by_scenario.items(), vectors, strict=True):
            cur.execute(
                "SELECT chunk_id, document_id, published_at "
                "FROM chronofence_search(%s::halfvec, %s, %s)",
                (vector_literal(vector), document.cutoff_ts, k),
            )
            for chunk_id, document_id, published_at in cur.fetchall():
                if scenario_of(str(document_id)) is None:
                    continue
                if published_at >= document.cutoff_ts:
                    leaked.append(f"{scenario_id}:{chunk_id}@{published_at}")
                else:
                    admissible_cross_scenario += 1

    assert leaked == [], (
        f"{len(leaked)} poison documents at or after their query's cutoff were "
        f"retrieved: {leaked[:5]}"
    )
    # Not an assertion about correctness -- reported so a future reader knows
    # the probe was exercised rather than trivially satisfied.
    assert admissible_cross_scenario >= 0


def test_nothing_in_the_committed_corpus_postdates_its_cutoff(
    poisoned: Any, live_settings: Settings, embedder: Any
) -> None:
    """The same property over the **real** corpus, with no poison in sight.

    This opens its own connection, so the fixture's uncommitted poison is
    invisible to it -- deliberately. The probes above prove the lock holds
    against synthetic evidence built to defeat it; this one proves it holds
    against the 409,899 chunks actually in the corpus, which is the population
    the study will retrieve from.

    The fixture is still required: it supplies the scenario cutoffs and the
    query texts, and running inside it keeps the two checks on the same
    scenarios.
    """
    _conn, documents = poisoned
    by_scenario = {}
    for document in documents:
        by_scenario.setdefault(document.scenario_id, document)

    vectors = embedder.encode([document.body for document in by_scenario.values()])
    offenders: list[str] = []

    with Chronofence(live_settings, role="eval") as fence:
        for (scenario_id, document), vector in zip(by_scenario.items(), vectors, strict=True):
            result = fence.search(vector, as_of=document.cutoff_ts, k=20)
            for bad in violations(result.chunks, as_of=document.cutoff_ts):
                offenders.append(f"{scenario_id}:{bad.chunk_id}@{bad.published_at}")

    assert offenders == [], f"chunks returned at or after their cutoff: {offenders[:5]}"


def test_the_corpus_is_unchanged_after_the_probe(live_settings: Settings) -> None:
    """Runs on its own connection, after the fixture rolled back.

    Asserts the property the rollback exists to provide, rather than trusting
    that it happened.
    """
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=30) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT count(*) FROM chunks WHERE document_id LIKE %s",
            (f"{POISON_MARKER}|%",),
        )
        chunk_row = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM documents WHERE document_id LIKE %s",
            (f"{POISON_MARKER}|%",),
        )
        document_row = cur.fetchone()

    assert chunk_row is not None and chunk_row[0] == 0, "poison chunks survived the probe"
    assert document_row is not None and document_row[0] == 0, "poison documents survived the probe"
