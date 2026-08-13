"""Forward-only SQL migration runner.

No ORM, no autogeneration: the partition strategy at M3 is load-bearing and
has to be explicit in SQL (spec §13). Migrations are executed by ``psql``
rather than by the driver, because they legitimately use psql-level constructs
(``\\gexec``, ``-v`` variables) to create roles with secrets that must not be
interpolated into a SQL string by hand.

Applied versions are tracked with a checksum, so editing an already-applied
migration is detected instead of silently diverging environments.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cascade.config import Settings, repo_root

__all__ = ["Migration", "MigrationError", "applied_versions", "apply_all", "discover", "pending"]

MIGRATIONS_DIR = repo_root() / "migrations"


class MigrationError(RuntimeError):
    """A migration failed to apply, or an applied migration was edited."""


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str

    @property
    def name(self) -> str:
        return self.path.name


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(directory: Path | None = None) -> list[Migration]:
    """Return every migration on disk, ordered by version.

    Ordering is lexical on the numeric prefix, which is why the files are
    zero-padded: ``010`` must not sort before ``002``.
    """
    root = directory or MIGRATIONS_DIR
    if not root.is_dir():
        return []
    out: list[Migration] = []
    for path in sorted(root.glob("*.sql")):
        version = path.stem.split("_", 1)[0]
        if not version.isdigit():
            raise MigrationError(
                f"migration {path.name!r} does not start with a numeric version prefix"
            )
        out.append(Migration(version=version, path=path, checksum=_checksum(path)))
    return out


def applied_versions(settings: Settings) -> dict[str, str]:
    """Map applied version -> recorded checksum.

    Returns empty when the bookkeeping table does not exist yet, which is the
    expected state before migration 001 has ever run.
    """
    import psycopg

    with (
        psycopg.connect(settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT to_regclass('public.schema_migrations') IS NOT NULL")
        row = cur.fetchone()
        if row is None or not row[0]:
            return {}
        cur.execute("SELECT version, checksum FROM schema_migrations")
        return {str(v): str(c) for v, c in cur.fetchall()}


def pending(settings: Settings, directory: Path | None = None) -> list[Migration]:
    """Return migrations not yet applied, verifying applied ones are unedited."""
    on_disk = discover(directory)
    already = applied_versions(settings)
    out: list[Migration] = []
    for migration in on_disk:
        recorded = already.get(migration.version)
        if recorded is None:
            out.append(migration)
        elif recorded != migration.checksum:
            raise MigrationError(
                f"migration {migration.name} has changed since it was applied "
                f"(recorded {recorded[:12]}, on disk {migration.checksum[:12]}). "
                "Migrations are forward-only: add a new one instead of editing this."
            )
    return out


def _psql_command(settings: Settings) -> list[str]:
    """Prefer a local ``psql``; otherwise run the one inside the container."""
    local = shutil.which("psql")
    if local is not None:
        return [local, settings.database_url("admin")]
    docker = shutil.which("docker")
    if docker is None:
        raise MigrationError(
            "neither psql nor docker is available; cannot apply migrations. "
            "Install libpq (brew install libpq) or start Docker."
        )
    return [
        docker,
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        f"postgresql://{settings.database.admin_user}:"
        f"{settings.db_admin_password.get_secret_value() if settings.db_admin_password else ''}"
        f"@localhost:5432/{settings.database.name}",
    ]


def _secret(settings: Settings, which: str) -> str:
    value = {
        "sim": settings.db_sim_password,
        "eval": settings.db_eval_password,
    }[which]
    if value is None:
        raise MigrationError(
            f"CASCADE_DB_{which.upper()}_PASSWORD is not set; migration 001 creates the "
            f"cascade_{which} role and refuses to give it an empty password"
        )
    return value.get_secret_value()


def apply_one(settings: Settings, migration: Migration) -> None:
    """Apply a single migration, then record it."""
    import psycopg

    command = [
        *_psql_command(settings),
        "-v",
        "ON_ERROR_STOP=1",
        "-v",
        f"sim_password={_secret(settings, 'sim')}",
        "-v",
        f"eval_password={_secret(settings, 'eval')}",
        "-f",
        "-",
    ]
    result = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        command,
        input=migration.path.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise MigrationError(
            f"migration {migration.name} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    with (
        psycopg.connect(settings.database_url("admin"), connect_timeout=10) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s) "
            "ON CONFLICT (version) DO UPDATE SET checksum = EXCLUDED.checksum",
            (migration.version, migration.checksum),
        )
        conn.commit()


def apply_all(settings: Settings, directory: Path | None = None) -> list[Migration]:
    """Apply every pending migration in order. Returns what was applied."""
    todo = pending(settings, directory)
    for migration in todo:
        apply_one(settings, migration)
    return todo
