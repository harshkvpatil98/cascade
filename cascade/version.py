"""The pinned stack (spec §2.3) and the process exit codes.

Keeping the stack list in code rather than prose means ``cascade doctor`` can
assert it, and a silent substitution shows up as a failed check instead of a
surprise three milestones later.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EXIT_BUDGET_BREACH",
    "EXIT_CACHE_MISS",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_PRECONDITION",
    "PINNED_STACK",
    "StackEntry",
]

# Distinct codes so a supervising script can tell the failure modes apart.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BUDGET_BREACH = 2
EXIT_PRECONDITION = 3
EXIT_CACHE_MISS = 4


@dataclass(frozen=True)
class StackEntry:
    """One pinned dependency.

    ``required=False`` marks a component that is pinned for a later milestone
    and installed via an extra; doctor reports it as absent without failing,
    so an M0 checkout does not need the multi-gigabyte ML stack.
    """

    label: str
    distribution: str
    pin: str
    required: bool = True
    extra: str = "core"


PINNED_STACK: tuple[StackEntry, ...] = (
    StackEntry("anthropic sdk", "anthropic", ">=0.40"),
    StackEntry("pydantic", "pydantic", ">=2.7"),
    StackEntry("pydantic-settings", "pydantic-settings", ">=2.3"),
    StackEntry("typer", "typer", ">=0.12"),
    StackEntry("rich", "rich", ">=13.7"),
    StackEntry("psycopg", "psycopg", ">=3.2"),
    StackEntry("numpy", "numpy", ">=1.26"),
    StackEntry("httpx", "httpx", ">=0.27"),
    StackEntry("pyyaml", "PyYAML", ">=6.0"),
    StackEntry("langfuse", "langfuse", ">=2.50,<3"),
    StackEntry("pytest", "pytest", ">=8.2", required=False, extra="dev"),
    StackEntry("hypothesis", "hypothesis", ">=6.100", required=False, extra="dev"),
    StackEntry("mypy", "mypy", ">=1.10", required=False, extra="dev"),
    StackEntry("ruff", "ruff", ">=0.5", required=False, extra="dev"),
    StackEntry("black", "black", ">=24.4", required=False, extra="dev"),
    StackEntry("langgraph", "langgraph", ">=0.2,<0.3", required=False, extra="kernel"),
    StackEntry("duckdb", "duckdb", ">=1.0", required=False, extra="analytics"),
    StackEntry("pyarrow", "pyarrow", ">=16.0", required=False, extra="analytics"),
    StackEntry(
        "sentence-transformers", "sentence-transformers", ">=3.0", required=False, extra="embed"
    ),
    StackEntry("torch", "torch", ">=2.3", required=False, extra="embed"),
)
