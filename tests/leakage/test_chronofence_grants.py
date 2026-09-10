"""The grant boundary around Chronofence (ADR-0002, ADR-0005, invariant 2).

Spec §4.2: "The app role has EXECUTE on the function and **no SELECT on
chunks**." That sentence is a claim about Postgres privileges, so it is
asserted against Postgres, not against the code that is supposed to respect
it. Code review is not a mechanism.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from cascade.config import Settings
from cascade.corpus.embed import EMBEDDING_DIM
from cascade.retrieval.search import Chronofence, vector_literal

pytestmark = pytest.mark.leakage

CUTOFF = datetime(2019, 6, 1, tzinfo=UTC)
ZERO_VECTOR = [0.0] * EMBEDDING_DIM


@pytest.mark.parametrize("table", ["chunks", "documents"])
def test_sim_cannot_select_the_corpus_directly(live_settings: Settings, table: str) -> None:
    """The whole point of the function: there is no other way in."""
    with (
        psycopg.connect(live_settings.database_url("sim"), connect_timeout=10) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        cur.execute(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608


def test_sim_can_execute_chronofence_search(live_settings: Settings) -> None:
    """SECURITY DEFINER is what makes this work despite the grant above.

    A SECURITY INVOKER function would raise InsufficientPrivilege on the first
    row it touched -- unusable by exactly the role it exists for (ADR-0002).
    """
    with Chronofence(live_settings, role="sim") as fence:
        result = fence.search(ZERO_VECTOR, as_of=CUTOFF, k=5)
    assert len(result.chunks) <= 5


def test_sim_cannot_execute_the_exact_oracle(live_settings: Settings) -> None:
    """The simulation gets exactly one corpus read path.

    A second path that happens to be slower is still a second path, and it
    would not be covered by whatever instrumentation watches the first.
    """
    with (
        Chronofence(live_settings, role="sim") as fence,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        fence.search_exact(ZERO_VECTOR, as_of=CUTOFF, k=5)


def test_eval_can_execute_the_exact_oracle(live_settings: Settings) -> None:
    """Recall needs a ground truth, and eval is the role that measures."""
    with Chronofence(live_settings, role="eval") as fence:
        result = fence.search_exact(ZERO_VECTOR, as_of=CUTOFF, k=5)
    assert len(result.chunks) <= 5


def test_neither_function_is_executable_by_public(live_settings: Settings) -> None:
    """CREATE FUNCTION grants EXECUTE to PUBLIC by default.

    Granting to cascade_sim without revoking from PUBLIC first leaves the
    function callable by every role and makes the grant decorative -- the same
    failure mode ADR-0005 records for tables.
    """
    with (
        psycopg.connect(live_settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT proname, proacl::text FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public' AND proname LIKE 'chronofence%'"
        )
        rows = cur.fetchall()

    assert rows, "chronofence functions are absent"
    for name, acl in rows:
        assert acl is not None, f"{name} has a default ACL, so PUBLIC still holds EXECUTE"
        assert "=X/" not in str(acl).replace("cascade_admin=X/", "").replace(
            "cascade_sim=X/", ""
        ).replace("cascade_eval=X/", ""), f"{name} grants EXECUTE beyond the three roles: {acl}"


def test_as_of_has_no_default_at_the_sql_boundary(live_settings: Settings) -> None:
    """Invariant 1, enforced by the signature rather than by discipline.

    Calling with two arguments must fail to *resolve* -- a parse-time error,
    not a silent read of the present.
    """
    with (
        psycopg.connect(live_settings.database_url("eval"), connect_timeout=10) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.UndefinedFunction),
    ):
        cur.execute(
            "SELECT * FROM chronofence_search(%s::halfvec, 5)",
            (vector_literal(ZERO_VECTOR),),
        )


def test_a_null_as_of_returns_nothing_rather_than_everything(live_settings: Settings) -> None:
    """The failure mode of this boundary must be empty, not leaky.

    `published_at < NULL` is NULL, never true, so a NULL cutoff that somehow
    reached the function yields zero rows. Asserted because the opposite --
    NULL meaning "no filter" -- is how this class of bug usually goes.
    """
    with (
        psycopg.connect(live_settings.database_url("eval"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT count(*) FROM chronofence_search(%s::halfvec, NULL, 20)",
            (vector_literal(ZERO_VECTOR),),
        )
        row = cur.fetchone()
    assert row is not None and row[0] == 0


def test_as_of_has_no_default_at_the_python_boundary(live_settings: Settings) -> None:
    """Enforced twice: Python raises before a connection is even opened."""
    with Chronofence(live_settings, role="sim") as fence, pytest.raises(TypeError):
        fence.search(ZERO_VECTOR, k=5)  # type: ignore[call-arg]


def test_a_naive_cutoff_is_refused_before_it_reaches_postgres(
    live_settings: Settings,
) -> None:
    """A naive timestamp is interpreted in the server's timezone.

    That silently shifts the time lock by the server's UTC offset, which is
    the kind of leak that leaves no trace in the results.
    """
    with (
        Chronofence(live_settings, role="sim") as fence,
        pytest.raises(ValueError, match="timezone-aware"),
    ):
        fence.search(ZERO_VECTOR, as_of=datetime(2019, 6, 1), k=5)


def test_a_wrong_width_query_vector_is_refused(live_settings: Settings) -> None:
    """The corpus and the query must come from the same pinned model."""
    with (
        Chronofence(live_settings, role="sim") as fence,
        pytest.raises(ValueError, match="dimensions"),
    ):
        fence.search([0.0] * 100, as_of=CUTOFF, k=5)
