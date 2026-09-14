"""Assay's persistence: the one place a forecast meets an outcome.

Everything here runs as ``cascade_eval``. That is §2.2's PHASE 3 / PHASE 4
boundary: PHASE 3 collapses runs into forecasts under ``cascade_sim``, which
has no grant on ``scenario_labels`` and therefore *cannot* condition a
forecast on the outcome it will be scored against (ADR-0022). PHASE 4 is this
module, and it is the first code in the pipeline that can see a label.

The frozen split is asserted before any label is read, not after. §1.3 names it
as one of three integrity mechanisms and the assertion is worth nothing if it
happens once the numbers are already out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cascade.config import Settings
from cascade.eval.schema import ScoredForecast

__all__ = [
    "FrozenSplit",
    "SplitNotSealed",
    "assert_frozen_split",
    "available_configs",
    "config_policies",
    "prompt_revision_audit",
    "record_prompt_revision_brier",
    "scored_forecasts",
    "write_study_report",
]

Role = Literal["admin", "eval"]


class SplitNotSealed(RuntimeError):
    """The registry has never been sealed, so there is no split to freeze."""


@dataclass(frozen=True)
class FrozenSplit:
    """The sealed set every metric in the report is computed against."""

    manifest_sha256: str
    n_scenarios: int
    n_yes: int
    base_rate: float
    climatology_brier: float
    study_salt: str


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


def prompt_revision_audit(settings: Settings, *, role: Role = "eval") -> list[dict[str, Any]]:
    """§1.3's prompt-change audit, as rows the report can serialise.

    Read here rather than in the report writer so `cascade/eval/report.py` stays
    free of database handles -- it is the module that must be testable without
    one, because it is the module that decides what the study *says*.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT prompt_rev, subsystem, changed_at, summary, brier_before, brier_after "
            "FROM prompt_revisions ORDER BY prompt_rev, subsystem"
        )
        return [
            {
                "prompt_rev": str(row[0]),
                "subsystem": str(row[1]),
                "changed_at": row[2].isoformat(),
                "summary": str(row[3]),
                "brier_before": None if row[4] is None else float(row[4]),
                "brier_after": None if row[5] is None else float(row[5]),
            }
            for row in cur.fetchall()
        ]


def assert_frozen_split(settings: Settings, *, role: Role = "eval") -> FrozenSplit:
    """Re-hash the registry and check it against the seal (spec §1.3).

    Called at the start of every evaluation path, before a single label is
    read. Raises :class:`cascade.ledger.manifest.ManifestMismatch` when the
    set has moved -- the correct response to which is never to reseal.

    Returns the sealed climatology as well, because §10.2's floor must come
    from the split the report describes rather than be recomputed from
    whatever happens to be loaded at report time.
    """
    from cascade.ledger.manifest import verify_manifest
    from cascade.ledger.store import load_records, read_manifest

    sealed = read_manifest(settings, role=role)
    if sealed is None:
        raise SplitNotSealed(
            "the scenario registry has never been sealed; run `cascade ledger seal` "
            "before scoring anything against it"
        )
    records = load_records(settings, role=role)
    verify_manifest(records, sealed.manifest_sha256)
    return FrozenSplit(
        manifest_sha256=sealed.manifest_sha256,
        n_scenarios=sealed.n_scenarios,
        n_yes=sealed.n_yes,
        base_rate=sealed.n_yes / sealed.n_scenarios if sealed.n_scenarios else 0.0,
        climatology_brier=float(sealed.climatology_brier),
        study_salt=sealed.study_salt,
    )


def scored_forecasts(
    settings: Settings, *, config_id: str, role: Role = "eval"
) -> tuple[ScoredForecast, ...]:
    """Join one configuration's forecasts to their outcomes and domains.

    An INNER join on both sides, deliberately. A forecast without a label is
    not scoreable and a label without a forecast is not a forecast -- either
    would quietly change the denominator of the headline metric, and a Brier
    over 174 scenarios reported as if it were over 180 is the kind of drift
    nothing downstream can detect. The count travels with every metric so the
    reader can see it.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.scenario_id, f.config_id, f.p_hat, l.outcome, s.domain,
                   f.sigma, f.n_replicates, f.modality, f.policy
            FROM forecasts f
            JOIN scenarios s USING (scenario_id)
            JOIN scenario_labels l USING (scenario_id)
            WHERE f.config_id = %s
            ORDER BY f.scenario_id
            """,
            (config_id,),
        )
        rows = cur.fetchall()
    return tuple(
        ScoredForecast(
            scenario_id=str(row[0]),
            config_id=str(row[1]),
            p_hat=float(row[2]),
            outcome=1 if int(row[3]) == 1 else 0,
            domain=str(row[4]),
            sigma=float(row[5]),
            n_replicates=int(row[6]),
            modality="multi" if str(row[7]) == "multi" else "single",
            policy=_policy_literal(str(row[8])),
        )
        for row in rows
    )


def _policy_literal(value: str) -> Literal["agent", "heuristic", "mixed", "none"]:
    """Narrow a stored policy string, failing loudly on an unknown one.

    The column carries a CHECK constraint, so an unrecognised value means the
    database and this code disagree about the vocabulary -- which is exactly
    the case where defaulting to ``"agent"`` would label a mechanism check as
    study data.
    """
    if value in ("agent", "heuristic", "mixed", "none"):
        return value  # type: ignore[return-value]
    raise ValueError(f"unknown forecast policy {value!r}; expected agent, heuristic, mixed or none")


def available_configs(settings: Settings, *, role: Role = "eval") -> tuple[tuple[str, int], ...]:
    """Every config with stored forecasts, and how many. Sorted (invariant 7)."""
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT config_id, count(*) FROM forecasts GROUP BY config_id ORDER BY config_id"
        )
        return tuple((str(row[0]), int(row[1])) for row in cur.fetchall())


def config_policies(settings: Settings, *, role: Role = "eval") -> dict[str, tuple[str, ...]]:
    """Which decider policies each config's forecasts came from. Sorted.

    Surfaced by `cascade eval status` so an operator can see *before* launching
    a grid that a cell already holds mechanism-check runs. Resuming into those
    produces a `mixed` forecast, which the report flags -- correct, but better
    discovered in the status table than in the report.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT config_id, policy FROM forecasts GROUP BY config_id, policy "
            "ORDER BY config_id, policy"
        )
        out: dict[str, list[str]] = {}
        for config_id, policy in cur.fetchall():
            out.setdefault(str(config_id), []).append(str(policy))
    return {key: tuple(value) for key, value in sorted(out.items())}


def write_study_report(
    settings: Settings,
    *,
    report_id: str,
    manifest_sha256: str,
    git_sha: str,
    headline_config: str,
    n_scenarios: int,
    headline_brier: float | None,
    notes: str = "",
    role: Role = "eval",
) -> None:
    """Record that a report was written, and against which sealed split.

    The report artifact on disk carries the same facts in ``manifest.json``.
    This row is what makes them queryable and what makes a report that was
    written against a *different* split visible without opening the directory.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO study_reports (
                report_id, manifest_sha256, git_sha, headline_config,
                n_scenarios, headline_brier, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (report_id) DO UPDATE SET
                manifest_sha256 = EXCLUDED.manifest_sha256,
                git_sha = EXCLUDED.git_sha,
                headline_config = EXCLUDED.headline_config,
                n_scenarios = EXCLUDED.n_scenarios,
                headline_brier = EXCLUDED.headline_brier,
                notes = EXCLUDED.notes
            """,
            (
                report_id,
                manifest_sha256,
                git_sha,
                headline_config,
                n_scenarios,
                headline_brier,
                notes,
            ),
        )
        conn.commit()


def record_prompt_revision_brier(
    settings: Settings,
    *,
    prompt_rev: str,
    subsystem: str,
    brier_before: float | None,
    brier_after: float | None,
    role: Role = "eval",
) -> int:
    """Fill in §1.3's before/after Brier for one prompt revision.

    The audit exists so that tuning on the test set is visible in the record
    rather than hidden. Writing the numbers is therefore part of evaluation,
    not a manual step someone remembers -- and the update touches only the two
    Brier columns, so the summary and rationale recorded at the time of the
    change cannot be edited afterwards to match the result.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE prompt_revisions
               SET brier_before = COALESCE(%s, brier_before),
                   brier_after  = COALESCE(%s, brier_after)
             WHERE prompt_rev = %s AND subsystem = %s
            """,
            (brier_before, brier_after, prompt_rev, subsystem),
        )
        updated = cur.rowcount
        conn.commit()
    return int(updated)
