"""The corpus schema against a live database (spec §3.2, §4.2, ADR-0004).

The partition layout is the mechanism the M3 time-lock rests on, so it is
asserted against the real catalogue rather than the migration text: a
partition that failed to create, or a range that drifted from the Python
window, would only surface at M3 as a query that reads rows it should have
pruned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cascade.config import Settings
from cascade.corpus.schema import WINDOW_END, WINDOW_START

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))
    settings = Settings()
    try:
        import psycopg

        with psycopg.connect(settings.database_url("admin"), connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001 -- a skip needs the reason
        pytest.skip(f"postgres not reachable: {type(exc).__name__}")
    return settings


def query(settings: Settings, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    import psycopg

    with (
        psycopg.connect(settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(sql, params)
        return list(cur.fetchall())


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["documents", "chunks"])
def test_table_is_range_partitioned_on_published_at(live_settings: Settings, table: str) -> None:
    """Pruning is what makes the time filter structural rather than a predicate."""
    rows = query(
        live_settings,
        "SELECT partstrat, pg_get_partkeydef(partrelid) FROM pg_partitioned_table "
        "WHERE partrelid = %s::regclass",
        (table,),
    )
    assert rows, f"{table} is not partitioned"
    assert rows[0][0] == "r"
    assert "published_at" in str(rows[0][1])


@pytest.mark.parametrize("table", ["documents", "chunks"])
def test_partitions_are_quarterly(live_settings: Settings, table: str) -> None:
    """ADR-0004: ~108 monthly partitions means ~100 index scans per query."""
    rows = query(
        live_settings,
        "SELECT count(*) FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = %s::regclass",
        (table,),
    )
    quarters = (WINDOW_END.year - WINDOW_START.year) * 4
    assert int(rows[0][0]) == quarters


@pytest.mark.parametrize("table", ["documents", "chunks"])
def test_there_is_no_default_partition(live_settings: Settings, table: str) -> None:
    """A DEFAULT partition can never be pruned.

    Every time-locked query would have to scan it, which would quietly undo
    the structural half of the Chronofence guarantee.
    """
    rows = query(
        live_settings,
        "SELECT count(*) FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = %s::regclass "
        "AND pg_get_expr(c.relpartbound, c.oid) LIKE '%%DEFAULT%%'",
        (table,),
    )
    assert int(rows[0][0]) == 0


def test_partitions_cover_exactly_the_python_window(live_settings: Settings) -> None:
    """If the DDL and ``schema.WINDOW_*`` drift, a valid document has nowhere
    to go and aborts the COPY of its whole batch."""
    import re

    bounds = query(
        live_settings,
        "SELECT pg_get_expr(c.relpartbound, c.oid) FROM pg_inherits i "
        "JOIN pg_class c ON c.oid = i.inhrelid WHERE i.inhparent = 'chunks'::regclass",
    )
    dates: list[str] = []
    for (expression,) in bounds:
        dates.extend(re.findall(r"'(\d{4}-\d{2}-\d{2})", str(expression)))
    assert dates, "no partition bounds found"
    assert min(dates) == str(WINDOW_START.date())
    assert max(dates) == str(WINDOW_END.date())


def test_chunks_carry_a_denormalised_published_at(live_settings: Settings) -> None:
    """The partition key must travel with the vector (spec §3.3).

    Without it, pruning would need a join and the lock would stop being
    structural.
    """
    columns = {
        str(row[0])
        for row in query(
            live_settings,
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks'",
        )
    }
    assert {"published_at", "embedding", "body", "token_count"} <= columns


def test_embedding_is_a_384_dimension_halfvec(live_settings: Settings) -> None:
    """The pinned model is 384-d; a mismatch breaks every distance at M3."""
    rows = query(
        live_settings,
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'",
    )
    assert str(rows[0][0]) == "halfvec(384)"


# ---------------------------------------------------------------------------
# Grants -- ADR-0002 preparation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["documents", "chunks"])
def test_sim_role_has_no_direct_access_to_the_corpus(live_settings: Settings, table: str) -> None:
    """Retrieval reaches the corpus only through ``chronofence_search`` (M3).

    Granting SELECT here would make that function's whole point -- no direct
    table access for the app role -- unenforceable before it is even written.
    """
    rows = query(
        live_settings,
        "SELECT has_table_privilege('cascade_sim', %s, 'SELECT')",
        (table,),
    )
    assert rows[0][0] is False


def test_eval_role_can_read_the_corpus(live_settings: Settings) -> None:
    for table in ("documents", "chunks"):
        rows = query(
            live_settings, "SELECT has_table_privilege('cascade_eval', %s, 'SELECT')", (table,)
        )
        assert rows[0][0] is True, table


# ---------------------------------------------------------------------------
# Corpus invariants over the full table
# ---------------------------------------------------------------------------


def test_no_document_has_a_null_or_future_date(live_settings: Settings) -> None:
    """M2 acceptance: asserted over the full table, never a sample.

    A sampled check on a leakage invariant is one that can miss the row that
    matters.
    """
    from cascade.corpus.store import corpus_stats

    stats = corpus_stats(live_settings)
    if stats.n_documents == 0:
        pytest.skip("corpus is empty; run `cascade corpus build`")
    assert stats.null_dates == 0
    assert stats.future_dates == 0


def test_every_chunk_has_an_embedding(live_settings: Settings) -> None:
    """A chunk without a vector is unreachable by retrieval but counts as one."""
    from cascade.corpus.store import corpus_stats

    stats = corpus_stats(live_settings)
    if stats.n_chunks == 0:
        pytest.skip("corpus is empty; run `cascade corpus build`")
    assert stats.missing_embeddings == 0
    assert stats.embedding_coverage == 1.0


def test_every_stored_document_falls_inside_the_corpus_window(
    live_settings: Settings,
) -> None:
    rows = query(
        live_settings,
        "SELECT count(*) FROM documents WHERE published_at < %s OR published_at >= %s",
        (WINDOW_START, WINDOW_END),
    )
    assert int(rows[0][0]) == 0


def test_stored_chunks_respect_the_token_cap(live_settings: Settings) -> None:
    """Over the cap, the model truncates and the chunk's tail has no vector."""
    rows = query(live_settings, "SELECT count(*) FROM chunks WHERE token_count > 512")
    assert int(rows[0][0]) == 0


def test_chunk_dates_match_their_parent_document(live_settings: Settings) -> None:
    """The denormalised copy must not drift, or pruning would be wrong."""
    rows = query(
        live_settings,
        "SELECT count(*) FROM chunks c JOIN documents d USING (document_id) "
        "WHERE c.published_at <> d.published_at",
    )
    assert int(rows[0][0]) == 0
