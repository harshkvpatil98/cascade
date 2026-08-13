"""Strata: event log, provenance and replay verification (M8, spec §11).

The event log is append-only -- no ``UPDATE``, no ``DELETE``, ever
(invariant 6) -- which is what lets ``caused_by`` be walked with a recursive
CTE from a terminal outcome back to its root cause.

Replay verification hashes the canonical event log from 25 recorded runs and
asserts the re-run hash is *equal*, not similar. A tolerance here would make
every downstream reproducibility claim false.
"""

from __future__ import annotations

__all__: list[str] = []
