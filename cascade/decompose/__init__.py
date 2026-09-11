"""Lathe: the causal decomposition compiler (M4, spec §5).

Three passes -- draft, adversarial critique, repair -- kept separate so the
model cannot quietly paper over a defect it just found. ``validator.py`` is
pure Python with no LLM and no I/O, which is what makes it property-testable.

The compiler runs inside the time lock: evidence comes from Chronofence at the
scenario's cutoff, so a graph is built from what was knowable then. Nothing in
this package reads an outcome -- ``scenario_labels`` is behind the
``cascade_eval`` grant and the compiler runs without it.
"""

from __future__ import annotations

from cascade.decompose.audit import (
    AuditScore,
    AuditSummary,
    sample_scenarios,
    score_worksheet,
    summarise,
    write_worksheet,
)
from cascade.decompose.compiler import CompileOutcome, Lathe, NoToolCall
from cascade.decompose.schema import (
    Actor,
    CausalGraph,
    Edge,
    Factor,
    OutcomeRule,
    OutcomeTerm,
    UtilityTerm,
    graph_hash,
)
from cascade.decompose.store import (
    CompileStats,
    compile_stats,
    load_graph,
    load_graphs,
    verify_hashes,
    write_graph,
)
from cascade.decompose.validator import ValidationReport, Violation, validate

__all__ = [
    "Actor",
    "AuditScore",
    "AuditSummary",
    "CausalGraph",
    "CompileOutcome",
    "CompileStats",
    "Edge",
    "Factor",
    "Lathe",
    "NoToolCall",
    "OutcomeRule",
    "OutcomeTerm",
    "UtilityTerm",
    "ValidationReport",
    "Violation",
    "compile_stats",
    "graph_hash",
    "load_graph",
    "load_graphs",
    "sample_scenarios",
    "score_worksheet",
    "summarise",
    "validate",
    "verify_hashes",
    "write_graph",
    "write_worksheet",
]
