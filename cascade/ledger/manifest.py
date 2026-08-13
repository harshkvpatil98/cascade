"""Sealing the split (spec §1.3, §3.1).

Pure: no I/O, no clock.

The manifest is one of the three integrity mechanisms, and it is a functional
requirement rather than documentation. It hashes
``(scenario_id, cutoff_ts, resolve_ts, outcome)`` for every scenario; every
later phase asserts the hash on startup. That is what makes the split *frozen*:
a label edited after the first simulation run changes the hash, and every
downstream phase refuses to start rather than reporting a number computed
against a set that no longer exists.

The hash deliberately covers the outcome. A manifest over questions alone
would survive exactly the edit it needs to catch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cascade.canonical import canonical_json, canonical_timestamp
from cascade.ledger.schema import ScenarioRecord

__all__ = ["ManifestMismatch", "compute_manifest", "manifest_rows", "verify_manifest"]


class ManifestMismatch(RuntimeError):
    """The registry does not hash to the sealed manifest.

    Raised on startup by every phase after M1. The correct response is never
    to reseal: it is to find out what changed.
    """

    def __init__(self, expected: str, actual: str, n_scenarios: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"scenario manifest mismatch over {n_scenarios} scenarios: sealed "
            f"{expected[:16]}..., computed {actual[:16]}.... The frozen split has "
            "changed. Do not reseal -- find out what changed. Every metric "
            "already reported was computed against the sealed set."
        )


@dataclass(frozen=True)
class ManifestRow:
    """One row of the hashed projection. Deliberately only four fields."""

    scenario_id: str
    cutoff_ts: str
    resolve_ts: str
    outcome: int


def manifest_rows(records: tuple[ScenarioRecord, ...]) -> tuple[ManifestRow, ...]:
    """Project records onto the four fields the manifest covers, sorted by id.

    Sorted (invariant 7) because the hash must not depend on the order the
    loaders happened to return questions in.
    """
    return tuple(
        sorted(
            (
                ManifestRow(
                    scenario_id=record.scenario.scenario_id,
                    cutoff_ts=canonical_timestamp(record.scenario.cutoff_ts),
                    resolve_ts=canonical_timestamp(record.scenario.resolve_ts),
                    outcome=record.label.outcome,
                )
                for record in records
            ),
            key=lambda row: row.scenario_id,
        )
    )


def compute_manifest(records: tuple[ScenarioRecord, ...]) -> str:
    """Return the SHA-256 over the sealed projection of ``records``."""
    payload = [
        [row.scenario_id, row.cutoff_ts, row.resolve_ts, row.outcome]
        for row in manifest_rows(records)
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_manifest(records: tuple[ScenarioRecord, ...], expected: str) -> None:
    """Raise :class:`ManifestMismatch` unless ``records`` hash to ``expected``.

    Preserves the frozen-split guarantee at every phase boundary. Returns
    ``None`` on success so a caller cannot accidentally treat a falsy result
    as a pass.
    """
    actual = compute_manifest(records)
    if actual != expected:
        raise ManifestMismatch(expected=expected, actual=actual, n_scenarios=len(records))
