"""Live infrastructure: pinned versions, role separation, migration bookkeeping.

Requires ``make up`` and ``make migrate``. Deselected by default
(``-m "not integration"``).

The role assertions here are the foundation of invariant 2 -- the simulation
never reads ``scenario_labels``. That invariant is enforced by a Postgres
grant rather than by code review, and a grant enforces nothing if the role is
a superuser. So ``rolsuper`` is asserted directly and first: every other
privilege assertion in this file and at M1 is vacuous if it fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cascade.config import Settings
from cascade.db import applied_versions, apply_all, discover, pending

pytestmark = pytest.mark.integration

CONSOLE_SCRIPT = Path(sys.executable).parent / "cascade"


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real settings against the running stack, with services reachable.

    The project's ``.env`` is re-attached explicitly. The conftest fixture
    detaches every test from ambient credentials so that unit results cannot
    depend on the machine; these tests are *about* the deployed configuration,
    so they opt back in by name rather than relying on fixture ordering to
    leave the environment intact.
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))

    settings = Settings()
    try:
        import psycopg

        with psycopg.connect(settings.database_url("admin"), connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001 -- a skip needs the reason, not a traceback
        pytest.skip(f"postgres not reachable: {type(exc).__name__}: {exc}")
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
# Pinned stack
# ---------------------------------------------------------------------------


def test_postgres_is_version_16(live_settings: Settings) -> None:
    (row,) = query(live_settings, "SHOW server_version")
    assert str(row[0]).startswith("16."), f"pinned stack requires PostgreSQL 16, found {row[0]}"


def test_pgvector_is_installed_at_0_8(live_settings: Settings) -> None:
    rows = query(live_settings, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    assert rows, "pgvector extension is not installed; migration 001 creates it"
    assert str(rows[0][0]).startswith("0.8"), f"pinned pgvector 0.8, found {rows[0][0]}"


def test_halfvec_is_usable(live_settings: Settings) -> None:
    """The corpus stores ``halfvec(384)``; a missing type breaks M2, not M3."""
    (row,) = query(live_settings, "SELECT '[1,2,3]'::halfvec(3) <-> '[1,2,4]'::halfvec(3)")
    assert float(row[0]) == pytest.approx(1.0)


def test_pgcrypto_is_installed(live_settings: Settings) -> None:
    rows = query(live_settings, "SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto'")
    assert rows, "pgcrypto provides gen_random_uuid() for run ids"


def test_collation_is_deterministic(live_settings: Settings) -> None:
    """Sort order participates in query results; a locale change diverges replay."""
    (row,) = query(
        live_settings, "SELECT datcollate FROM pg_database WHERE datname = current_database()"
    )
    assert str(row[0]) in {"C", "C.UTF-8", "POSIX"}, (
        f"expected a deterministic collation, found {row[0]!r}; "
        "docker-compose sets --locale=C for exactly this reason"
    )


# ---------------------------------------------------------------------------
# Role separation (ADR-0005) -- the foundation of invariant 2
# ---------------------------------------------------------------------------


def test_both_study_roles_exist(live_settings: Settings) -> None:
    names = {str(row[0]) for row in query(live_settings, "SELECT rolname FROM pg_roles")}
    assert {"cascade_sim", "cascade_eval"} <= names


@pytest.mark.parametrize("role", ["cascade_sim", "cascade_eval"])
def test_study_roles_are_not_superusers(live_settings: Settings, role: str) -> None:
    """Assert this first: a superuser bypasses every grant below it.

    Without this check the M1 grant test would pass while enforcing nothing --
    a green test over an absent mechanism, which is worse than no test.
    """
    rows = query(
        live_settings, "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s", (role,)
    )
    assert rows, f"role {role} does not exist"
    is_super, bypasses_rls = rows[0]
    assert is_super is False, f"{role} is a superuser and bypasses every grant"
    assert bypasses_rls is False, f"{role} bypasses row-level security"


def test_public_holds_no_privilege_on_the_public_schema(live_settings: Settings) -> None:
    """Every role inherits PUBLIC, so a PUBLIC grant defeats role separation."""
    (row,) = query(
        live_settings,
        "SELECT has_schema_privilege('public', 'public', 'CREATE') "
        "OR has_schema_privilege('public', 'public', 'USAGE')",
    )
    assert row[0] is False, "PUBLIC still holds privileges on schema public (ADR-0005)"


@pytest.mark.parametrize("role", ["cascade_sim", "cascade_eval"])
def test_study_roles_can_use_the_schema(live_settings: Settings, role: str) -> None:
    """Deny by default, then grant back the minimum -- USAGE and nothing more."""
    (row,) = query(live_settings, "SELECT has_schema_privilege(%s, 'public', 'USAGE')", (role,))
    assert row[0] is True, f"{role} cannot use schema public; migration 001 grants this"


def test_default_privileges_deny_public_on_future_tables(live_settings: Settings) -> None:
    """Without this, a later migration silently re-opens what 001 closed."""
    rows = query(
        live_settings,
        "SELECT unnest(defaclacl)::text FROM pg_default_acl WHERE defaclobjtype = 'r'",
    )
    granted_to_public = [str(row[0]) for row in rows if str(row[0]).startswith("=")]
    assert granted_to_public == [], f"PUBLIC holds default table privileges: {granted_to_public}"


# ---------------------------------------------------------------------------
# Migration bookkeeping
# ---------------------------------------------------------------------------


def test_migration_001_is_recorded_with_its_checksum(live_settings: Settings) -> None:
    recorded = applied_versions(live_settings)
    on_disk = {migration.version: migration.checksum for migration in discover()}
    assert "001" in recorded, "migration 001 has not been applied; run `make migrate`"
    assert recorded["001"] == on_disk["001"], (
        "migration 001 has changed since it was applied. Migrations are "
        "forward-only: add a new one instead of editing this."
    )


def test_reapplying_is_a_no_op(live_settings: Settings) -> None:
    """Idempotence is what makes `make migrate` safe to run from any state."""
    assert pending(live_settings) == []
    assert apply_all(live_settings) == []


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


def test_langfuse_has_its_own_database(live_settings: Settings) -> None:
    rows = query(live_settings, "SELECT 1 FROM pg_database WHERE datname = 'langfuse'")
    assert rows, "the postgres-init script creates the langfuse database"


def test_langfuse_is_healthy(live_settings: Settings) -> None:
    if not live_settings.langfuse.enabled:
        pytest.skip("langfuse disabled in config")
    import httpx

    url = live_settings.langfuse.host.rstrip("/") + "/api/public/health"
    try:
        response = httpx.get(url, timeout=10.0)
    except Exception as exc:  # noqa: BLE001 -- a skip needs the reason, not a traceback
        pytest.skip(f"langfuse not reachable: {type(exc).__name__}")
    assert response.status_code == 200


@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="console script not installed")
def test_doctor_exits_zero_against_live_services() -> None:
    """M0 acceptance criterion 1, in full: no ``--offline`` escape hatch.

    The subprocess is handed the project's real ``.env``. The conftest fixture
    detaches the *test* process from ambient credentials so unit tests cannot
    depend on the machine, and the subprocess would inherit that detachment --
    but this test is specifically about the deployed configuration working, so
    it opts back in explicitly rather than by accident.
    """
    import os

    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")

    result = subprocess.run(
        [str(CONSOLE_SCRIPT), "doctor"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "CASCADE_ENV_FILE": str(env_file)},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "all checks passed" in result.stdout
