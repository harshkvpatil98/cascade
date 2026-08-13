"""Chorus: ensembling and uncertainty (M6, spec §9).

36,000 runs = 180 scenarios x 200 replicates, each seeded once from
``blake2b(scenario_id|config_id|replicate, key=STUDY_SALT)`` (invariant 4).
Collapses each scenario's outcome distribution to ``p_hat`` plus a dispersion
measure, and separates "uncertain" from "undecided" via the modality flag.
"""

from __future__ import annotations

__all__: list[str] = []
