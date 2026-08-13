"""Persistence for the scenario registry.

The write path is deliberately narrow. Scenarios and labels go into separate
tables (migration 002) and are written in one transaction, because a registry
that is half-loaded is a registry whose manifest does not describe it.

Reading is split the same way the grants are. :func:`load_scenarios` is what
the simulation calls and it cannot return an outcome -- there is no code path
here that joins the two tables. :func:`load_records` exists for the evaluation
role and for sealing, and it says so in its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from cascade.canonical import canonical_json
from cascade.config import Settings
from cascade.ledger.climatology import Climatology
from cascade.ledger.schema import Scenario, ScenarioLabel, ScenarioRecord

__all__ = [
    "SealedManifest",
    "load_records",
    "load_scenarios",
    "read_manifest",
    "write_manifest",
    "write_registry",
]

Role = Literal["admin", "sim", "eval"]


@dataclass(frozen=True)
class SealedManifest:
    """The stored seal: what was frozen, and what baseline it implies."""

    manifest_sha256: str
    sealed_at: datetime
    n_scenarios: int
    n_yes: int
    yes_rate: float
    climatology_brier: float
    study_salt: str
    notes: str


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=10)


def write_registry(
    settings: Settings, records: tuple[ScenarioRecord, ...], *, replace: bool = False
) -> int:
    """Insert scenarios and labels in one transaction.

    ``replace`` is required to overwrite an existing registry. Rebuilding a
    sealed set is a deliberate act -- every metric already reported was
    computed against the old one -- so it cannot happen as a side effect of
    re-running a build.
    """
    if not records:
        raise ValueError("refusing to write an empty registry")

    scenario_rows = [
        (
            record.scenario.scenario_id,
            record.scenario.question,
            record.scenario.resolution_criterion,
            record.scenario.cutoff_ts,
            record.scenario.resolve_ts,
            record.scenario.domain,
            record.scenario.source,
            record.scenario.source_ref,
            record.scenario.party_rule,
            canonical_json(list(record.scenario.party_names)),
            record.scenario.event_group,
        )
        for record in sorted(records, key=lambda item: item.scenario.scenario_id)
    ]
    label_rows = [
        (record.label.scenario_id, record.label.outcome, record.label.resolved_at)
        for record in sorted(records, key=lambda item: item.scenario.scenario_id)
    ]

    with _connect(settings, "admin") as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenarios")
        row = cur.fetchone()
        existing = int(row[0]) if row else 0
        if existing and not replace:
            raise RuntimeError(
                f"{existing} scenarios are already loaded. Pass --replace to rebuild: "
                "every metric already reported was computed against the current set."
            )
        if existing:
            cur.execute("DELETE FROM scenario_labels")
            cur.execute("DELETE FROM scenarios")

        cur.executemany(
            "INSERT INTO scenarios (scenario_id, question, resolution_criterion, "
            "cutoff_ts, resolve_ts, domain, source, source_ref, party_rule, "
            "party_names, event_group) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
            scenario_rows,
        )
        cur.executemany(
            "INSERT INTO scenario_labels (scenario_id, outcome, resolved_at) "
            "VALUES (%s, %s, %s)",
            label_rows,
        )
        conn.commit()
    return len(records)


def load_scenarios(settings: Settings, *, role: Role = "sim") -> tuple[Scenario, ...]:
    """Load scenarios without outcomes. This is what the simulation calls.

    Preserves invariant 2 at the code boundary as well as the grant boundary:
    there is no join here, so even a role that *could* read ``scenario_labels``
    does not get an outcome through this function.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT scenario_id, question, resolution_criterion, cutoff_ts, "
            "resolve_ts, domain, source, source_ref, party_rule, party_names, "
            "event_group FROM scenarios ORDER BY scenario_id"
        )
        rows = cur.fetchall()
    return tuple(
        Scenario(
            scenario_id=row[0],
            question=row[1],
            resolution_criterion=row[2],
            cutoff_ts=row[3],
            resolve_ts=row[4],
            domain=row[5],
            source=row[6],
            source_ref=row[7],
            party_rule=row[8],
            party_names=tuple(row[9]),
            event_group=row[10],
        )
        for row in rows
    )


def load_records(settings: Settings, *, role: Role = "eval") -> tuple[ScenarioRecord, ...]:
    """Load scenarios *with* outcomes. Evaluation and sealing only.

    Named so the call site is self-incriminating: anything in ``cascade/sim/``
    calling this is visible in review, and the grant makes it fail anyway.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT s.scenario_id, s.question, s.resolution_criterion, s.cutoff_ts, "
            "s.resolve_ts, s.domain, s.source, s.source_ref, s.party_rule, "
            "s.party_names, s.event_group, l.outcome, l.resolved_at "
            "FROM scenarios s JOIN scenario_labels l USING (scenario_id) "
            "ORDER BY s.scenario_id"
        )
        rows = cur.fetchall()
    return tuple(
        ScenarioRecord(
            scenario=Scenario(
                scenario_id=row[0],
                question=row[1],
                resolution_criterion=row[2],
                cutoff_ts=row[3],
                resolve_ts=row[4],
                domain=row[5],
                source=row[6],
                source_ref=row[7],
                party_rule=row[8],
                party_names=tuple(row[9]),
                event_group=row[10],
            ),
            label=ScenarioLabel(scenario_id=row[0], outcome=row[11], resolved_at=row[12]),
        )
        for row in rows
    )


def write_manifest(
    settings: Settings,
    *,
    manifest_sha256: str,
    climatology: Climatology,
    study_salt: str,
    notes: str = "",
) -> None:
    """Record the seal. Re-sealing the same hash is idempotent."""
    with _connect(settings, "admin") as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_manifest")
        cur.execute(
            "INSERT INTO scenario_manifest (manifest_sha256, n_scenarios, n_yes, "
            "yes_rate, climatology_brier, study_salt, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                manifest_sha256,
                climatology.n,
                climatology.n_yes,
                climatology.base_rate,
                climatology.brier,
                study_salt,
                notes,
            ),
        )
        conn.commit()


def read_manifest(settings: Settings, *, role: Role = "sim") -> SealedManifest | None:
    """Return the seal, or ``None`` when the registry has never been sealed.

    Readable by the simulation role: every phase asserts the hash on startup,
    so the row itself carries no outcome and must be visible to everyone.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT manifest_sha256, sealed_at, n_scenarios, n_yes, yes_rate, "
            "climatology_brier, study_salt, notes FROM scenario_manifest "
            "ORDER BY sealed_at DESC LIMIT 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    return SealedManifest(
        manifest_sha256=row[0],
        sealed_at=row[1],
        n_scenarios=row[2],
        n_yes=row[3],
        yes_rate=row[4],
        climatology_brier=row[5],
        study_salt=row[6],
        notes=row[7],
    )
