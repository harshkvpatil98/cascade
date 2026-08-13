"""Scenario registry and manifest sealing (M1, spec §3.1).

Lands the 180 resolved binary questions, the inclusion rules as executable
predicates, and ``manifest.sha256`` over ``(scenario_id, cutoff_ts,
resolve_ts, outcome)``. Every later phase asserts that hash on startup, which
is what makes the split frozen rather than merely documented.
"""

from __future__ import annotations

__all__: list[str] = []
