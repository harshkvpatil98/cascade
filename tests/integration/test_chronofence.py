"""The Chronofence schema objects against a live database (spec §4.2).

Asserted against the catalogue rather than the migration text. A function that
lost its `SET search_path` to a later `CREATE OR REPLACE`, or an index that was
never built, would look identical in the migration directory and behave very
differently at query time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cascade.config import Settings
from cascade.retrieval.index import index_name_for, measure, plan_all

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

        with (
            psycopg.connect(settings.database_url("admin"), connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT to_regprocedure('chronofence_search(halfvec,timestamptz,int)')")
            row = cur.fetchone()
            if row is None or row[0] is None:
                pytest.skip("chronofence_search is absent; run `cascade db migrate`")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 -- a skip needs its reason
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
# Function attributes (ADR-0002)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["chronofence_search", "chronofence_search_exact"])
def test_function_is_security_definer_and_stable(live_settings: Settings, name: str) -> None:
    rows = query(
        live_settings,
        "SELECT prosecdef, provolatile FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND proname = %s",
        (name,),
    )
    assert rows, f"{name} is absent"
    secdef, volatility = rows[0]
    assert secdef is True, f"{name} is SECURITY INVOKER; the app role cannot use it (ADR-0002)"
    assert volatility == "s", f"{name} is not STABLE; a lateral join would re-execute it per row"


@pytest.mark.parametrize("name", ["chronofence_search", "chronofence_search_exact"])
def test_function_pins_its_search_path(live_settings: Settings, name: str) -> None:
    """A SECURITY DEFINER function without a pinned search_path is an escalation.

    The caller controls name resolution and can shadow a referenced object
    with one in a schema they own. `pg_temp` must come last so a temporary
    table cannot shadow anything either.
    """
    rows = query(
        live_settings,
        "SELECT proconfig FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND proname = %s",
        (name,),
    )
    config = rows[0][0] or []
    search_path = [entry for entry in config if entry.startswith("search_path=")]
    assert search_path, f"{name} does not pin search_path"
    assert search_path[0].endswith("pg_temp"), f"{name} must list pg_temp last: {search_path[0]}"


def test_deployed_probes_match_the_configured_value(live_settings: Settings) -> None:
    """Config and schema cannot drift apart silently.

    `probes` lives in two places by necessity -- pinned into the function so a
    caller cannot forget it, and in config so the bench can report it. This is
    the assertion that keeps them equal, and it points at the fix: migrations
    are forward-only, so a change means a new migration, not an edit.
    """
    from cascade.retrieval.search import Chronofence

    with Chronofence(live_settings, role="admin") as fence:
        deployed = fence.probes()
    assert deployed == live_settings.retrieval.ivfflat_probes, (
        f"chronofence_search pins ivfflat.probes={deployed} but "
        f"retrieval.ivfflat_probes is {live_settings.retrieval.ivfflat_probes}; "
        "add a migration rather than editing an applied one"
    )


def test_the_exact_oracle_is_exhaustive(live_settings: Settings) -> None:
    """It is the ground truth for recall, so it must visit every list.

    32768 is pgvector's maximum for both `probes` and `lists`, so probes can
    never be smaller than a partition's list count.
    """
    rows = query(
        live_settings,
        "SELECT proconfig FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND proname = 'chronofence_search_exact'",
    )
    config = rows[0][0] or []
    assert "ivfflat.probes=32768" in config


def test_as_of_has_no_default_in_the_signature(live_settings: Settings) -> None:
    """Invariant 1 at the SQL boundary: three required arguments, no DEFAULT."""
    rows = query(
        live_settings,
        "SELECT pronargdefaults, pronargs FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND proname = 'chronofence_search'",
    )
    defaults, nargs = rows[0]
    assert nargs == 3
    assert defaults == 0, "chronofence_search has a defaulted argument; as_of must be required"


# ---------------------------------------------------------------------------
# chronofence_partitions (migration 005)
# ---------------------------------------------------------------------------


def test_the_partition_view_returns_one_row_per_partition(live_settings: Settings) -> None:
    """Regression for migration 004's fan-out defect.

    The original view filtered the joined pg_class row to the IVFFlat access
    method *after* joining pg_index, which nulls non-matching rows instead of
    removing them. Every partition carries a primary key and a document index,
    so the view returned 104 rows for 52 partitions.
    """
    rows = query(
        live_settings,
        "SELECT count(*), count(DISTINCT partition) FROM chronofence_partitions",
    )
    total, distinct = rows[0]
    assert total == distinct, f"{total} rows for {distinct} partitions; the view fans out"


def test_the_partition_view_covers_every_chunks_partition(live_settings: Settings) -> None:
    rows = query(
        live_settings,
        "SELECT (SELECT count(*) FROM chronofence_partitions), "
        "(SELECT count(*) FROM pg_inherits i JOIN pg_class p ON p.oid = i.inhparent "
        " WHERE p.relname = 'chunks')",
    )
    in_view, actual = rows[0]
    assert in_view == actual


# ---------------------------------------------------------------------------
# Index state (ADR-0012)
# ---------------------------------------------------------------------------


def test_every_non_empty_partition_has_a_correctly_sized_index(
    live_settings: Settings,
) -> None:
    """What `cascade retrieval verify` asserts, asserted in CI too.

    A missing index cannot cause a leak -- retrieval degrades to an exact
    sequential scan -- but it does break the p95 criterion, silently.
    """
    retrieval = live_settings.retrieval
    plans = plan_all(
        measure(live_settings),
        max_lists=retrieval.index_max_lists,
        tolerance=retrieval.index_rebuild_tolerance,
    )
    stale = [plan for plan in plans if plan.action in {"create", "rebuild"}]
    assert stale == [], (
        f"{len(stale)} partitions need an index pass: "
        f"{[(plan.partition, plan.action) for plan in stale[:5]]} -- "
        "run `cascade retrieval index`"
    )


def test_empty_partitions_carry_no_index(live_settings: Settings) -> None:
    """An IVFFlat index built on no rows is permanently degenerate (ADR-0012)."""
    empty_with_index = [
        partition
        for partition in measure(live_settings)
        if partition.rows == 0 and partition.index_name is not None
    ]
    assert empty_with_index == []


def test_indexes_use_the_expected_name_and_operator_class(live_settings: Settings) -> None:
    """Names are derived, not stored, so `verify` can detect absence.

    `halfvec_l2_ops` is correct because the corpus stores unit-norm vectors:
    on those, L2 and cosine rank identically, so the operator the index is
    built for is the one the study reasons in.
    """
    measured = [partition for partition in measure(live_settings) if partition.rows > 0]
    assert measured, "no non-empty partitions; the corpus is empty"
    for partition in measured:
        assert partition.index_name == index_name_for(partition.partition)

    rows = query(
        live_settings,
        "SELECT count(*) FROM pg_indexes WHERE indexname LIKE '%%_embedding_ivfflat_idx' "
        "AND indexdef NOT LIKE '%%halfvec_l2_ops%%'",
    )
    assert rows[0][0] == 0, "an IVFFlat index uses an unexpected operator class"
