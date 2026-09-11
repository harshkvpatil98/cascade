"""Persistence for compiled graphs (migration 008, spec §5.3).

A scenario compiles once for the whole study, so writes are upserts keyed on
``scenario_id`` and a re-compile of unchanged input overwrites a row with
identical content. The stored ``graph_sha256`` is what makes that checkable:
:func:`verify_hashes` recomputes it from the stored JSON, so a row that was
hand-edited or corrupted is detectable rather than merely unlikely.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from cascade.canonical import canonical_json
from cascade.config import Settings
from cascade.decompose.compiler import CompileOutcome
from cascade.decompose.schema import CausalGraph, graph_hash

__all__ = [
    "CompileStats",
    "HashMismatch",
    "compile_stats",
    "completed_scenarios",
    "load_graph",
    "load_graphs",
    "record_failure",
    "verify_hashes",
    "write_graph",
]

Role = Literal["admin", "sim", "eval"]


@dataclass(frozen=True)
class HashMismatch:
    """A stored graph whose content no longer matches its recorded hash."""

    scenario_id: str
    recorded: str
    recomputed: str


@dataclass(frozen=True)
class CompileStats:
    """The figures the M4 acceptance criteria are read off."""

    scenarios: int
    compiled: int
    failed: int
    mean_actors: float
    min_actors: int
    max_actors: int
    mean_factors: float
    min_factors: int
    max_factors: int
    mean_edges: float
    retry_histogram: tuple[tuple[int, int], ...]
    """(repair_retries, count) ascending -- the histogram §M4 asks for."""
    distinct_hashes: int
    total_llm_calls: int
    mean_evidence_chunks: float


def _connect(settings: Settings, role: Role = "admin") -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


def write_graph(settings: Settings, outcome: CompileOutcome) -> None:
    """Persist one successful compile. Idempotent on ``scenario_id``.

    Refuses an outcome that did not compile: writing a failed attempt into the
    graphs table is how a downstream reader ends up simulating a graph nobody
    accepted.
    """
    if outcome.graph is None or outcome.status != "compiled":
        raise ValueError(
            f"refusing to store scenario {outcome.scenario_id!r} with status "
            f"{outcome.status!r}; failures belong in causal_graph_failures"
        )
    graph = outcome.graph
    payload = canonical_json(graph.canonical())

    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO causal_graphs (
                scenario_id, graph_sha256, graph, n_actors, n_factors, n_edges,
                repair_retries, llm_calls, evidence_chunks, compiler_model, prompt_rev
            ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scenario_id) DO UPDATE SET
                graph_sha256 = EXCLUDED.graph_sha256,
                graph = EXCLUDED.graph,
                n_actors = EXCLUDED.n_actors,
                n_factors = EXCLUDED.n_factors,
                n_edges = EXCLUDED.n_edges,
                repair_retries = EXCLUDED.repair_retries,
                llm_calls = EXCLUDED.llm_calls,
                evidence_chunks = EXCLUDED.evidence_chunks,
                compiler_model = EXCLUDED.compiler_model,
                prompt_rev = EXCLUDED.prompt_rev,
                compiled_at = now()
            """,
            (
                graph.scenario_id,
                outcome.graph_sha256,
                payload,
                len(graph.actors),
                len(graph.factors),
                len(graph.edges),
                outcome.repair_retries,
                outcome.llm_calls,
                outcome.evidence_chunks,
                settings.models.compiler,
                settings.llm.prompt_rev,
            ),
        )
        # A scenario that previously failed and now compiles must not keep its
        # failure row, or `compile status` reports 180 compiled and 4 failed.
        cur.execute(
            "DELETE FROM causal_graph_failures WHERE scenario_id = %s", (graph.scenario_id,)
        )
        conn.commit()


def record_failure(settings: Settings, outcome: CompileOutcome) -> None:
    """Log a hard failure (spec §5.2). Idempotent on ``scenario_id``."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO causal_graph_failures (
                scenario_id, violations, defects, repair_retries, compiler_model, prompt_rev
            ) VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s)
            ON CONFLICT (scenario_id) DO UPDATE SET
                violations = EXCLUDED.violations,
                defects = EXCLUDED.defects,
                repair_retries = EXCLUDED.repair_retries,
                compiler_model = EXCLUDED.compiler_model,
                prompt_rev = EXCLUDED.prompt_rev,
                failed_at = now()
            """,
            (
                outcome.scenario_id,
                json.dumps(list(outcome.violations)),
                json.dumps(list(outcome.defects)),
                outcome.repair_retries,
                settings.models.compiler,
                settings.llm.prompt_rev,
            ),
        )
        conn.commit()


def completed_scenarios(settings: Settings) -> set[str]:
    """Scenario ids already compiled, so a restart skips them (invariant 8)."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT scenario_id FROM causal_graphs")
        return {str(row[0]) for row in cur.fetchall()}


def load_graph(settings: Settings, scenario_id: str, *, role: Role = "sim") -> CausalGraph | None:
    """Load one compiled graph, or None.

    Read as ``sim`` by default: this is what M5 calls, and it must work under
    the role the simulation actually runs as.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute("SELECT graph FROM causal_graphs WHERE scenario_id = %s", (scenario_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return CausalGraph.model_validate(row[0])


def load_graphs(settings: Settings, *, role: Role = "sim") -> tuple[CausalGraph, ...]:
    """Every compiled graph, ordered by scenario id (invariant 7)."""
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute("SELECT graph FROM causal_graphs ORDER BY scenario_id")
        rows = cur.fetchall()
    return tuple(CausalGraph.model_validate(row[0]) for row in rows)


def verify_hashes(settings: Settings) -> tuple[HashMismatch, ...]:
    """Recompute every stored graph's hash and report disagreements.

    This is the determinism rule (§5.3) asserted against what is actually in
    the database. A row whose JSON was edited in place keeps its old hash and
    would otherwise pass every other check.
    """
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT scenario_id, graph_sha256, graph FROM causal_graphs ORDER BY 1")
        rows = cur.fetchall()

    mismatches: list[HashMismatch] = []
    for scenario_id, recorded, payload in rows:
        recomputed = graph_hash(CausalGraph.model_validate(payload))
        if recomputed != recorded:
            mismatches.append(
                HashMismatch(
                    scenario_id=str(scenario_id),
                    recorded=str(recorded),
                    recomputed=recomputed,
                )
            )
    return tuple(mismatches)


def compile_stats(settings: Settings, *, expected_scenarios: int | None = None) -> CompileStats:
    """Aggregate the M4 acceptance figures from the stored rows."""
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenarios")
        row = cur.fetchone()
        total = int(row[0]) if row else 0

        cur.execute(
            "SELECT count(*), coalesce(avg(n_actors), 0), coalesce(min(n_actors), 0), "
            "coalesce(max(n_actors), 0), coalesce(avg(n_factors), 0), "
            "coalesce(min(n_factors), 0), coalesce(max(n_factors), 0), "
            "coalesce(avg(n_edges), 0), count(DISTINCT graph_sha256), "
            "coalesce(sum(llm_calls), 0), coalesce(avg(evidence_chunks), 0) "
            "FROM causal_graphs"
        )
        aggregate = cur.fetchone()

        cur.execute("SELECT repair_retries, count(*) FROM causal_graphs GROUP BY 1 ORDER BY 1")
        histogram = tuple((int(retries), int(count)) for retries, count in cur.fetchall())

        cur.execute("SELECT count(*) FROM causal_graph_failures")
        failed_row = cur.fetchone()

    compiled = int(aggregate[0]) if aggregate else 0
    return CompileStats(
        scenarios=expected_scenarios if expected_scenarios is not None else total,
        compiled=compiled,
        failed=int(failed_row[0]) if failed_row else 0,
        mean_actors=float(aggregate[1]) if aggregate else 0.0,
        min_actors=int(aggregate[2]) if aggregate else 0,
        max_actors=int(aggregate[3]) if aggregate else 0,
        mean_factors=float(aggregate[4]) if aggregate else 0.0,
        min_factors=int(aggregate[5]) if aggregate else 0,
        max_factors=int(aggregate[6]) if aggregate else 0,
        mean_edges=float(aggregate[7]) if aggregate else 0.0,
        retry_histogram=histogram,
        distinct_hashes=int(aggregate[8]) if aggregate else 0,
        total_llm_calls=int(aggregate[9]) if aggregate else 0,
        mean_evidence_chunks=float(aggregate[10]) if aggregate else 0.0,
    )


def store_outcomes(settings: Settings, outcomes: Sequence[CompileOutcome]) -> tuple[int, int]:
    """Persist a batch, routing each outcome to the right table. Returns (ok, failed)."""
    compiled = failed = 0
    for outcome in outcomes:
        if outcome.ok:
            write_graph(settings, outcome)
            compiled += 1
        else:
            record_failure(settings, outcome)
            failed += 1
    return compiled, failed
