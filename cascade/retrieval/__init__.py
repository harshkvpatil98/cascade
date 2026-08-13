"""Chronofence: time-locked retrieval (M3, spec §4).

``as_of`` is a required parameter everywhere in this package -- no default in
Python, no default in SQL (invariant 1). The time filter is structural: the
planner prunes post-cutoff partitions before touching a vector, so the lock
does not depend on a caller remembering a predicate.
"""

from __future__ import annotations

__all__: list[str] = []
