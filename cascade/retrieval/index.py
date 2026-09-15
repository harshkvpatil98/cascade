"""HNSW index lifecycle for the ``chunks`` partitions (ADR-0012, ADR-0026).

Why this is a command and not a migration is argued in ADR-0012, and the
argument survives the switch to HNSW: an index on an empty partition is built
from nothing, migrations run against empty databases all the time, and the
corpus is loaded by a separate resumable phase.

What *does* change with HNSW is the failure mode the command exists to catch.
IVFFlat's ``lists`` is a function of the row count, so a partition that grew
after its index was built carried a silently degraded index until someone ran
a rebuild pass -- a drift class that interrupted M4, M5, M6 and M7 in turn.
HNSW has no row-dependent build parameter. An index that exists stays correct
as rows arrive, so ``rebuild`` now fires only when the *configured* build
parameters have changed, which is an operator action rather than a background
process.

The planning rule is pure (:func:`plan_partition`, :func:`plan_all`) so it can
be tested without a database; only :func:`measure` and :func:`apply_plans`
touch Postgres.

Nothing here can cause a leak. A missing index makes retrieval *slower* -- at
worst an exact sequential scan over the pruned partitions -- never wider. The
time predicate lives in ``chronofence_search`` and does not depend on any
index existing.
"""

from __future__ import annotations

import time
from typing import Any

from cascade.config import Settings
from cascade.retrieval.schema import IndexPlan, IndexReport, PartitionIndex

__all__ = [
    "PGVECTOR_MAX_EF_CONSTRUCTION",
    "PGVECTOR_MAX_M",
    "apply_plans",
    "drop_legacy_ivfflat",
    "index_name_for",
    "measure",
    "plan_all",
    "plan_partition",
]

# pgvector's own ceilings for the two HNSW build parameters.
PGVECTOR_MAX_M = 100
PGVECTOR_MAX_EF_CONSTRUCTION = 1000


def index_name_for(partition: str) -> str:
    """Deterministic index name for a partition.

    Derived rather than stored: `cascade retrieval verify` has to be able to
    say "this partition has no index" without consulting a registry that could
    itself be stale.

    The name carries the access method, so an IVFFlat index left behind by a
    pre-ADR-0026 database is not mistaken for this one -- it simply does not
    exist under the name the command looks for, and the partition plans as
    ``create``.
    """
    return f"{partition}_embedding_hnsw_idx"


def _validate(m: int, ef_construction: int) -> None:
    """Reject build parameters pgvector itself would refuse.

    Checked here rather than left to Postgres so a misconfiguration fails
    before the command starts dropping indexes, not midway through a pass.
    """
    if not 2 <= m <= PGVECTOR_MAX_M:
        raise ValueError(f"hnsw m must lie in [2, {PGVECTOR_MAX_M}], got {m}")
    if not 4 <= ef_construction <= PGVECTOR_MAX_EF_CONSTRUCTION:
        raise ValueError(
            f"hnsw ef_construction must lie in [4, {PGVECTOR_MAX_EF_CONSTRUCTION}], "
            f"got {ef_construction}"
        )
    if ef_construction < 2 * m:
        raise ValueError(
            f"hnsw ef_construction ({ef_construction}) must be at least 2 * m ({2 * m}); "
            "pgvector enforces this and a smaller value builds a degraded graph"
        )


def plan_partition(partition: PartitionIndex, *, m: int, ef_construction: int) -> IndexPlan:
    """Decide what one partition needs. Pure.

    An empty partition is skipped outright. Building on nothing produces an
    index that is not wrong so much as pointless, and the partition will be
    planned again -- correctly -- once it holds rows.

    A partition whose index was built with different parameters is rebuilt.
    That is the only rebuild trigger left: with IVFFlat, growth alone forced
    one, and this command spent four milestones chasing it.
    """
    _validate(m, ef_construction)
    if partition.rows == 0:
        return IndexPlan(
            partition=partition.partition,
            rows=0,
            current_m=partition.m,
            current_ef_construction=partition.ef_construction,
            target_m=m,
            target_ef_construction=ef_construction,
            action="skip-empty",
            reason="no rows; an index built here would describe an empty graph",
        )

    if partition.index_name is None:
        return IndexPlan(
            partition=partition.partition,
            rows=partition.rows,
            current_m=None,
            current_ef_construction=None,
            target_m=m,
            target_ef_construction=ef_construction,
            action="create",
            reason=f"no HNSW index on {partition.rows:,} rows",
        )

    if partition.m != m or partition.ef_construction != ef_construction:
        return IndexPlan(
            partition=partition.partition,
            rows=partition.rows,
            current_m=partition.m,
            current_ef_construction=partition.ef_construction,
            target_m=m,
            target_ef_construction=ef_construction,
            action="rebuild",
            reason=(
                f"built with m={partition.m}, ef_construction="
                f"{partition.ef_construction}; configured m={m}, "
                f"ef_construction={ef_construction}"
            ),
        )

    return IndexPlan(
        partition=partition.partition,
        rows=partition.rows,
        current_m=partition.m,
        current_ef_construction=partition.ef_construction,
        target_m=m,
        target_ef_construction=ef_construction,
        action="keep",
        reason=(
            f"m={m}, ef_construction={ef_construction} on {partition.rows:,} rows; "
            "HNSW has no row-dependent parameter to drift"
        ),
    )


def plan_all(
    partitions: list[PartitionIndex], *, m: int, ef_construction: int
) -> tuple[IndexPlan, ...]:
    """Plan every partition, largest first, ties broken by name (invariant 7).

    Largest first so an interrupted pass has already indexed what dominates
    latency -- a full build is minutes, and the partition that costs the most
    to leave unindexed is the one with the most rows. The name tie-break is
    what keeps the order total and reproducible.
    """
    ordered = sorted(partitions, key=lambda item: (-item.rows, item.partition))
    return tuple(
        plan_partition(partition, m=m, ef_construction=ef_construction) for partition in ordered
    )


def _connect(settings: Settings) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url("admin"), connect_timeout=30)


def measure(settings: Settings) -> list[PartitionIndex]:
    """Read exact row counts and current index state per ``chunks`` partition.

    Counts are exact rather than ``reltuples``. An unanalysed partition reports
    ``reltuples = -1``, and while HNSW no longer sizes anything off the count,
    the *skip-empty* decision still turns on it -- and skipping a partition
    that merely looks empty is how a partition ends up permanently unindexed.
    """
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT partition, index_name, m, ef_construction "
            "FROM chronofence_partitions ORDER BY partition"
        )
        rows = cur.fetchall()

        out: list[PartitionIndex] = []
        for partition, index_name, m, ef_construction in rows:
            # Partition names come from pg_class, not from user input; quoted
            # with an identifier quote regardless, because building DDL by
            # interpolation is a habit that has to be uniform to be safe.
            cur.execute(f'SELECT count(*) FROM "{partition}"')  # noqa: S608
            count_row = cur.fetchone()
            out.append(
                PartitionIndex(
                    partition=str(partition),
                    rows=int(count_row[0]) if count_row else 0,
                    index_name=str(index_name) if index_name is not None else None,
                    m=int(m) if m is not None else None,
                    ef_construction=int(ef_construction) if ef_construction is not None else None,
                )
            )
    return out


def apply_plans(
    settings: Settings, plans: tuple[IndexPlan, ...], *, maintenance_work_mem: str = "900MB"
) -> IndexReport:
    """Execute ``plans``, one partition per transaction.

    Autocommit per statement so an interrupted run keeps every index it has
    already finished rather than rolling the whole pass back.

    ``maintenance_work_mem`` is raised for the session because an HNSW build
    that does not fit in it falls back to an on-disk phase that is far slower.
    Parallel maintenance workers are left at zero deliberately: they allocate a
    shared-memory segment sized from ``maintenance_work_mem``, and this
    container's ``/dev/shm`` is 1 GB, so requesting them turns a working build
    into ``could not resize shared memory segment``.

    ``CREATE INDEX`` is not ``CONCURRENTLY``: this is an offline maintenance
    command on a research corpus, and the concurrent build is materially
    slower for no benefit when nothing else is writing. The tradeoff is an
    ACCESS EXCLUSIVE lock for the duration of each partition's build.
    """
    import psycopg

    started = time.monotonic()
    created = rebuilt = kept = skipped = 0

    with psycopg.connect(settings.database_url("admin"), connect_timeout=30) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
            cur.execute("SET max_parallel_maintenance_workers = 0")
            for plan in plans:
                if plan.action == "skip-empty":
                    skipped += 1
                    continue
                if plan.action == "keep":
                    kept += 1
                    continue

                name = index_name_for(plan.partition)
                if plan.action == "rebuild":
                    cur.execute(f'DROP INDEX IF EXISTS "{name}"')
                    rebuilt += 1
                else:
                    created += 1

                cur.execute(
                    f'CREATE INDEX "{name}" ON "{plan.partition}" '
                    f"USING hnsw (embedding halfvec_l2_ops) "
                    f"WITH (m = {int(plan.target_m)}, "
                    f"ef_construction = {int(plan.target_ef_construction)})"
                )
                # Without fresh statistics the planner keeps costing this
                # partition off whatever it believed before the index existed,
                # which is how a correctly built index goes unused.
                cur.execute(f'ANALYZE "{plan.partition}"')

    return IndexReport(
        plans=plans,
        created=created,
        rebuilt=rebuilt,
        kept=kept,
        skipped_empty=skipped,
        elapsed_s=time.monotonic() - started,
    )


def drop_legacy_ivfflat(settings: Settings) -> tuple[str, ...]:
    """Remove IVFFlat indexes left on ``chunks`` partitions. Returns their names.

    Run after the HNSW pass, not before: dropping first would leave the corpus
    unindexed for the length of the build, and a partition with two vector
    indexes is merely wasteful while a partition with none is a sequential
    scan. Separate from :func:`apply_plans` because it is a one-way migration
    step, and a maintenance command that silently drops indexes it did not
    plan is one nobody can reason about.
    """
    import psycopg

    dropped: list[str] = []
    with psycopg.connect(settings.database_url("admin"), connect_timeout=30) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                SELECT i.relname
                FROM pg_index pgi
                JOIN pg_class i ON i.oid = pgi.indexrelid
                JOIN pg_class t ON t.oid = pgi.indrelid
                JOIN pg_inherits inh ON inh.inhrelid = t.oid
                JOIN pg_class parent ON parent.oid = inh.inhparent
                WHERE parent.relname = 'chunks'
                  AND i.relam = (SELECT oid FROM pg_am WHERE amname = 'ivfflat')
                ORDER BY i.relname
                """)
            names = [str(row[0]) for row in cur.fetchall()]
            for name in names:
                cur.execute(f'DROP INDEX IF EXISTS "{name}"')
                dropped.append(name)
    return tuple(dropped)
