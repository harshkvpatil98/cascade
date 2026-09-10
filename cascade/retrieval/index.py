"""IVFFlat index lifecycle for the ``chunks`` partitions (ADR-0012).

Why this is a command and not a migration is argued in ADR-0012; the short
version is that ``lists ~= sqrt(rows_in_partition)`` is a function of measured
data, an IVFFlat index built on an empty partition is permanently degenerate,
and migrations run against empty databases all the time.

The sizing rule is a pure function of row counts (:func:`target_lists`,
:func:`plan_partition`) so it can be tested without a database; only
:func:`measure` and :func:`apply_plans` touch Postgres.

Nothing here can cause a leak. A missing or badly sized index makes retrieval
*slower* -- at worst an exact sequential scan over the pruned partitions --
never wider. The time predicate lives in ``chronofence_search`` and does not
depend on any index existing.
"""

from __future__ import annotations

import math
import time
from typing import Any

from cascade.config import Settings
from cascade.retrieval.schema import IndexPlan, IndexReport, PartitionIndex

__all__ = [
    "apply_plans",
    "index_name_for",
    "measure",
    "plan_all",
    "plan_partition",
    "target_lists",
]

# pgvector's own ceiling for both `lists` and `probes`. Sizing is clamped to
# the configured maximum, which must not exceed this.
PGVECTOR_MAX_LISTS = 32768


def index_name_for(partition: str) -> str:
    """Deterministic index name for a partition.

    Derived rather than stored: `cascade retrieval verify` has to be able to
    say "this partition has no index" without consulting a registry that could
    itself be stale.
    """
    return f"{partition}_embedding_ivfflat_idx"


def target_lists(rows: int, *, max_lists: int) -> int:
    """``clamp(round(sqrt(rows)), 1, max_lists)`` -- spec §4.2's sizing rule.

    Rounds rather than truncates: ``int(sqrt(2400)) == 48`` against a true
    48.99, and truncation biases every partition's list count low, which
    widens each list and costs recall at fixed probes.
    """
    if rows < 0:
        raise ValueError(f"rows cannot be negative, got {rows}")
    if max_lists < 1:
        raise ValueError(f"max_lists must be at least 1, got {max_lists}")
    if rows == 0:
        return 0
    return max(1, min(round(math.sqrt(rows)), max_lists, PGVECTOR_MAX_LISTS))


def plan_partition(partition: PartitionIndex, *, max_lists: int, tolerance: float) -> IndexPlan:
    """Decide what one partition needs. Pure.

    An empty partition is skipped outright rather than given a ``lists = 1``
    index: every row inserted later would land in that single list, so the
    index would never narrow anything and would have to be rebuilt anyway.
    """
    target = target_lists(partition.rows, max_lists=max_lists)

    if partition.rows == 0:
        return IndexPlan(
            partition=partition.partition,
            rows=0,
            current_lists=partition.lists,
            target_lists=0,
            action="skip-empty",
            reason="no rows; an index built here would be degenerate for every later insert",
        )

    if partition.index_name is None or partition.lists is None:
        return IndexPlan(
            partition=partition.partition,
            rows=partition.rows,
            current_lists=None,
            target_lists=target,
            action="create",
            reason=f"no IVFFlat index on {partition.rows:,} rows",
        )

    drift = abs(partition.lists - target) / target
    if drift > tolerance:
        return IndexPlan(
            partition=partition.partition,
            rows=partition.rows,
            current_lists=partition.lists,
            target_lists=target,
            action="rebuild",
            reason=(
                f"lists={partition.lists} vs target {target} on {partition.rows:,} rows "
                f"(drift {drift:.0%} > {tolerance:.0%})"
            ),
        )

    return IndexPlan(
        partition=partition.partition,
        rows=partition.rows,
        current_lists=partition.lists,
        target_lists=target,
        action="keep",
        reason=f"lists={partition.lists} within {tolerance:.0%} of target {target}",
    )


def plan_all(
    partitions: list[PartitionIndex], *, max_lists: int, tolerance: float
) -> tuple[IndexPlan, ...]:
    """Plan every partition, largest first.

    Ordering is by row count descending so the partitions that dominate query
    latency are rebuilt first: an interrupted run has then already fixed the
    ones that matter. Ties break on name so the order is deterministic
    (invariant 7).
    """
    ordered = sorted(partitions, key=lambda item: (-item.rows, item.partition))
    return tuple(
        plan_partition(partition, max_lists=max_lists, tolerance=tolerance) for partition in ordered
    )


def _connect(settings: Settings) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url("admin"), connect_timeout=30)


def measure(settings: Settings) -> list[PartitionIndex]:
    """Read exact row counts and current index state per ``chunks`` partition.

    Counts are exact rather than ``reltuples``. An unanalysed partition reports
    ``reltuples = -1``, and sizing an index off -1 yields ``lists = 1`` -- the
    degenerate index this module exists to avoid. Measured on this corpus: 16
    of 52 partitions read -1 before an ANALYZE.
    """
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT partition, index_name, lists FROM chronofence_partitions ORDER BY partition"
        )
        rows = cur.fetchall()

        out: list[PartitionIndex] = []
        for partition, index_name, lists in rows:
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
                    lists=int(lists) if lists is not None else None,
                )
            )
    return out


def apply_plans(settings: Settings, plans: tuple[IndexPlan, ...]) -> IndexReport:
    """Execute ``plans``, one partition per transaction.

    Autocommit per statement so an interrupted run keeps every index it has
    already finished rather than rolling the whole pass back. Measured on the
    current corpus the full pass is 5.8 s for 36 partitions, but the cost
    scales with rows and the study target is 3x the present corpus.

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
                    f"USING ivfflat (embedding halfvec_l2_ops) "
                    f"WITH (lists = {int(plan.target_lists)})"
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
