"""Assay: the evaluation harness (M7, spec §10).

Brier and its Murphy decomposition, five baselines, the 12-cell ablation grid,
paired bootstrap over *scenarios* rather than runs, and Holm-Bonferroni across
the ablation family.

This package reads ``scenario_labels`` and the simulation does not
(invariant 2); the separation is enforced by a Postgres grant, so it survives
a refactor that code review would not catch.
"""

from __future__ import annotations

__all__: list[str] = []
