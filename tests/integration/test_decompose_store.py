"""Compiled-graph persistence and the end-to-end compile path (M4, migration 008).

Everything here runs against the live database. The model is the one component
stubbed out, so what is exercised is the whole integration the acceptance run
depends on: Chronofence retrieval at the cutoff, the real embedder driving the
semantic validator rules, the upsert, the hash round-trip, and the grant that
lets M5 read a graph as ``cascade_sim``.

Graphs written here are removed afterwards. A test that left rows behind would
make `cascade compile status` report a compiled scenario nobody compiled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from cascade.config import Settings
from cascade.decompose.compiler import CompileOutcome, Lathe
from cascade.decompose.prompts import CRITIQUE_TOOL, DRAFT_TOOL
from cascade.decompose.schema import graph_hash
from cascade.decompose.store import (
    compile_stats,
    completed_scenarios,
    load_graph,
    record_failure,
    verify_hashes,
    write_graph,
)
from tests.conftest import make_graph

pytestmark = pytest.mark.integration

TEST_PREFIX = "itest-m4"


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))
    settings = Settings()
    try:
        import psycopg

        with (
            psycopg.connect(settings.database_url("admin"), connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT to_regclass('public.causal_graphs')")
            row = cur.fetchone()
            if row is None or row[0] is None:
                pytest.skip("causal_graphs is absent; run `cascade db migrate`")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 -- a skip needs its reason
        pytest.skip(f"postgres not reachable: {type(exc).__name__}")
    return settings


@pytest.fixture
def scenario_id(live_settings: Settings) -> Any:
    """Borrow a real scenario id, and clean up any graph written against it."""
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT scenario_id FROM scenarios ORDER BY scenario_id LIMIT 1")
        row = cur.fetchone()
    if row is None:
        pytest.skip("scenario registry is empty; run `cascade ledger build`")
    borrowed = str(row[0])

    def cleanup() -> None:
        with (
            psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("DELETE FROM causal_graphs WHERE scenario_id = %s", (borrowed,))
            cur.execute("DELETE FROM causal_graph_failures WHERE scenario_id = %s", (borrowed,))
            conn.commit()

    cleanup()
    yield borrowed
    cleanup()


def outcome_for(scenario_id: str, **graph_kwargs: Any) -> CompileOutcome:
    graph = make_graph(scenario_id=scenario_id, **graph_kwargs)
    return CompileOutcome(
        scenario_id=scenario_id,
        status="compiled",
        graph=graph,
        graph_sha256=graph_hash(graph),
        repair_retries=1,
        llm_calls=4,
        evidence_chunks=60,
        defects=("missing_party [bank]: absent financier",),
        violations=(),
        elapsed_s=1.0,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_a_graph_round_trips_through_the_database(
    live_settings: Settings, scenario_id: str
) -> None:
    original = outcome_for(scenario_id)
    write_graph(live_settings, original)

    loaded = load_graph(live_settings, scenario_id, role="admin")
    assert loaded is not None
    assert graph_hash(loaded) == original.graph_sha256
    assert loaded == original.graph, "the model must round-trip field for field"


def test_the_stored_hash_matches_the_stored_content(
    live_settings: Settings, scenario_id: str
) -> None:
    """§5.3's determinism rule, asserted against the database."""
    write_graph(live_settings, outcome_for(scenario_id))
    assert verify_hashes(live_settings) == ()


def test_a_tampered_row_is_detected(live_settings: Settings, scenario_id: str) -> None:
    """A hand-edited graph keeps its old hash and passes every other check."""
    import psycopg

    write_graph(live_settings, outcome_for(scenario_id))
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "UPDATE causal_graphs SET graph = jsonb_set(graph, "
            "'{outcome_rule,threshold}', '0.25') WHERE scenario_id = %s",
            (scenario_id,),
        )
        conn.commit()

    mismatches = verify_hashes(live_settings)
    assert [item.scenario_id for item in mismatches] == [scenario_id]
    assert mismatches[0].recorded != mismatches[0].recomputed


def test_writing_twice_is_an_upsert_not_a_duplicate(
    live_settings: Settings, scenario_id: str
) -> None:
    """A scenario compiles once; a rebuild replaces rather than accumulates."""
    write_graph(live_settings, outcome_for(scenario_id))
    write_graph(live_settings, outcome_for(scenario_id, n_actors=9))

    loaded = load_graph(live_settings, scenario_id, role="admin")
    assert loaded is not None
    assert len(loaded.actors) == 9
    assert completed_scenarios(live_settings) >= {scenario_id}


def test_storing_a_failed_outcome_is_refused(live_settings: Settings, scenario_id: str) -> None:
    """A failed attempt in the graphs table is a graph nobody accepted."""
    failed = outcome_for(scenario_id)
    broken = CompileOutcome(**{**failed.__dict__, "status": "failed"})
    with pytest.raises(ValueError, match="refusing to store"):
        write_graph(live_settings, broken)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_failure_is_logged_and_cleared_on_a_later_success(
    live_settings: Settings, scenario_id: str
) -> None:
    """Otherwise `compile status` reports 180 compiled and 4 failed."""
    failed = CompileOutcome(
        scenario_id=scenario_id,
        status="failed",
        graph=None,
        graph_sha256=None,
        repair_retries=2,
        llm_calls=5,
        evidence_chunks=60,
        defects=(),
        violations=("edge_sanity: self loop",),
        elapsed_s=1.0,
    )
    record_failure(live_settings, failed)
    assert compile_stats(live_settings).failed >= 1

    write_graph(live_settings, outcome_for(scenario_id))
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT count(*) FROM causal_graph_failures WHERE scenario_id = %s", (scenario_id,)
        )
        row = cur.fetchone()
    assert row is not None and row[0] == 0


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------


def test_the_simulation_role_can_read_graphs(live_settings: Settings, scenario_id: str) -> None:
    """M5 builds one agent per actor as `cascade_sim`; the grant must allow it."""
    write_graph(live_settings, outcome_for(scenario_id))
    loaded = load_graph(live_settings, scenario_id, role="sim")
    assert loaded is not None
    assert len(loaded.actors) >= 8


def test_the_simulation_role_cannot_read_compile_failures(live_settings: Settings) -> None:
    """A failed compile is an operational fact, not study input."""
    import psycopg

    with (
        psycopg.connect(live_settings.database_url("sim"), connect_timeout=10) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        cur.execute("SELECT 1 FROM causal_graph_failures LIMIT 1")


def test_graphs_carry_no_outcome(live_settings: Settings, scenario_id: str) -> None:
    """Invariant 2: nothing reachable through this table is a label."""
    import psycopg

    write_graph(live_settings, outcome_for(scenario_id))
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'causal_graphs'"
        )
        columns = {str(row[0]) for row in cur.fetchall()}
    assert not columns & {"outcome", "resolved_at", "label"}


# ---------------------------------------------------------------------------
# The full compile path, with only the model stubbed
# ---------------------------------------------------------------------------


class _Scripted:
    def __init__(self, payloads: list[tuple[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def complete(self, request: Any, *, trace_name: str = "", **_: Any) -> Any:
        self.calls += 1
        name, payload = self._payloads.pop(0)

        class _Result:
            tool_calls: ClassVar[list[dict[str, Any]]] = [
                {"type": "tool_use", "name": name, "input": payload}
            ]
            text = ""
            stop_reason = "tool_use"

        return _Result()


def test_the_compiler_runs_against_live_retrieval_and_the_real_embedder(
    live_settings: Settings, scenario_id: str
) -> None:
    """Everything but the model: Chronofence, the embedder, the validator, the write.

    This is the integration the acceptance run rests on. It cannot say anything
    about graph *quality* -- that needs a real model and the §5.4 audit -- but it
    proves the pipeline is wired end to end.
    """
    from cascade.corpus.embed import Embedder, EmbeddingUnavailable
    from cascade.ledger.store import load_scenarios
    from cascade.retrieval.search import Chronofence

    scenario = next(
        item
        for item in load_scenarios(live_settings, role="admin")
        if item.scenario_id == scenario_id
    )

    embedder = Embedder(
        model_name=live_settings.models.embedding,
        batch_size=live_settings.corpus.embed_batch_size,
    )
    try:
        embedder.load()
    except EmbeddingUnavailable as exc:
        pytest.skip(str(exc))

    payload = make_graph(scenario_id=scenario_id).canonical()
    payload.pop("scenario_id")

    retrieved: list[int] = []

    with Chronofence(live_settings, role="eval") as fence:

        def retrieve(question: str, as_of: Any, k: int) -> list[tuple[str, str, str]]:
            vector = embedder.encode([question])[0]
            result = fence.search(vector, as_of=as_of, k=k)
            retrieved.append(len(result.chunks))
            # Every chunk the compiler sees must predate the cutoff.
            for chunk in result.chunks:
                assert chunk.published_at < as_of
            return [
                (chunk.published_at.isoformat(), chunk.source, chunk.body)
                for chunk in result.chunks
            ]

        compiler = Lathe(
            settings=live_settings,
            client=_Scripted(
                [
                    (DRAFT_TOOL["name"], payload),
                    (CRITIQUE_TOOL["name"], {"defects": []}),
                    (DRAFT_TOOL["name"], payload),
                ]
            ),
            embed=embedder.encode,
            retrieve=retrieve,
        )
        outcome = compiler.compile_scenario(scenario)

    assert outcome.ok, outcome.violations
    assert retrieved and retrieved[0] <= live_settings.retrieval.k_compiler
    assert outcome.evidence_chunks == retrieved[0]

    write_graph(live_settings, outcome)
    stored = load_graph(live_settings, scenario_id, role="sim")
    assert stored is not None
    assert graph_hash(stored) == outcome.graph_sha256
    assert verify_hashes(live_settings) == ()


def test_the_real_embedder_separates_a_restated_objective(
    live_settings: Settings, scenario_id: str
) -> None:
    """The semantic rules must work with the pinned model, not only a stub.

    A fake embedder can be made to separate anything. This asserts that the
    actual 384-dimension model scores a verbatim restatement of the outcome
    above the 0.85 threshold, which is what the rule depends on.
    """
    from cascade.corpus.embed import Embedder, EmbeddingUnavailable
    from cascade.decompose.validator import OBJECTIVE_SIMILARITY_MAX, cosine

    embedder = Embedder(
        model_name=live_settings.models.embedding,
        batch_size=live_settings.corpus.embed_batch_size,
    )
    try:
        embedder.load()
    except EmbeddingUnavailable as exc:
        pytest.skip(str(exc))

    outcome_text = "Will the European Commission block the merger before the deadline?"
    restatement = "Wants the European Commission to block the merger before the deadline."
    independent = "Expand distribution capacity at the lowest achievable cost in capital."

    vectors = embedder.encode([outcome_text, restatement, independent])
    restated_similarity = cosine(vectors[0], vectors[1])
    independent_similarity = cosine(vectors[0], vectors[2])

    assert restated_similarity > OBJECTIVE_SIMILARITY_MAX, (
        f"a verbatim restatement scored only {restated_similarity:.3f}; the "
        "objective-independence rule would not catch it"
    )
    assert independent_similarity < OBJECTIVE_SIMILARITY_MAX, (
        f"an independent interest scored {independent_similarity:.3f}, which would "
        "be flagged as a restatement"
    )
