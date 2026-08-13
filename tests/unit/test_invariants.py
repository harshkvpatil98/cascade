"""Static enforcement of invariants 1, 5 and 7 (CLAUDE.md §2).

These three are checkable without running the system, which means they can be
checked over code that does not exist yet. That is the point: they are the
invariants whose violations are cheap to introduce and expensive to find, so
the check has to be in place before the code that would violate it is written.

The other five invariants need runtime state and are enforced at their own
milestones: 2 by a Postgres grant (M0/M1), 3 by the arbiter's import surface
(M5), 4 and 8 by the determinism suite (M5/M6), 6 by the event-log schema (M6).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "cascade"
MIGRATIONS_ROOT = REPO_ROOT / "migrations"

# The one module permitted to import the provider SDK (invariant 5).
SOLE_LLM_CALL_SITE = PACKAGE_ROOT / "llm" / "client.py"


def python_sources() -> list[Path]:
    """Every module in the package, sorted (invariant 7 applies to the tests too)."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_package_root_is_where_we_think_it_is() -> None:
    """Guard the guard: a moved package would make every check below vacuous."""
    assert PACKAGE_ROOT.is_dir()
    assert (PACKAGE_ROOT / "config.py").is_file()
    assert len(python_sources()) >= 5


# ---------------------------------------------------------------------------
# Invariant 1 -- as_of is never defaulted
# ---------------------------------------------------------------------------


def _defaulted_as_of(tree: ast.Module) -> list[tuple[str, int]]:
    """Return ``(function_name, lineno)`` for any function defaulting ``as_of``."""
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        # Positional-or-keyword defaults align to the *tail* of the parameter list.
        positional = args.posonlyargs + args.args
        for arg, default in zip(
            positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
        ):
            if arg.arg == "as_of" and default is not None:
                offenders.append((node.name, node.lineno))
        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if arg.arg == "as_of" and kw_default is not None:
                offenders.append((node.name, node.lineno))
    return offenders


def test_as_of_is_never_defaulted_in_python() -> None:
    """Invariant 1: a missing ``as_of`` must be a TypeError, not a silent 'now'.

    A default here does not fail loudly -- it retrieves post-cutoff evidence
    and returns plausible results, which invalidates the study silently.
    """
    offenders = [
        f"{rel(path)}:{lineno} in {name}()"
        for path in python_sources()
        for name, lineno in _defaulted_as_of(parse(path))
    ]
    assert (
        offenders == []
    ), "as_of must never have a default value (invariant 1). Offenders:\n  " + "\n  ".join(
        offenders
    )


def test_as_of_is_never_defaulted_in_sql() -> None:
    """Invariant 1 at the SQL boundary: no ``DEFAULT`` on an ``as_of`` argument."""
    pattern = re.compile(r"as_of\s+[a-z_ ]*\bDEFAULT\b", re.IGNORECASE)
    offenders = [
        f"{rel(path)}:{index}"
        for path in sorted(MIGRATIONS_ROOT.glob("*.sql"))
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert (
        offenders == []
    ), "as_of must have no SQL default (invariant 1, ADR-0002). Offenders:\n  " + "\n  ".join(
        offenders
    )


# ---------------------------------------------------------------------------
# Invariant 5 -- exactly one LLM call site
# ---------------------------------------------------------------------------


def _imports_provider_sdk(tree: ast.Module) -> list[int]:
    """Return line numbers importing the Anthropic SDK, at any nesting depth."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines.extend(
                node.lineno for alias in node.names if alias.name.split(".")[0] == "anthropic"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "anthropic"
        ):
            lines.append(node.lineno)
    return lines


def test_only_one_module_imports_the_provider_sdk() -> None:
    """Invariant 5: determinism, caching, metering and tracing need one door.

    A second importer does not break anything visibly -- it breaks the cost
    ledger and the replay guarantee, both of which are only checked at M8.
    """
    importers = sorted(rel(path) for path in python_sources() if _imports_provider_sdk(parse(path)))
    assert importers == [rel(SOLE_LLM_CALL_SITE)], (
        f"the Anthropic SDK may only be imported by {rel(SOLE_LLM_CALL_SITE)} "
        f"(invariant 5). Found: {importers}"
    )


def test_the_sole_call_site_actually_imports_the_sdk() -> None:
    """Guard the guard: a client that stopped importing the SDK would pass above."""
    assert _imports_provider_sdk(parse(SOLE_LLM_CALL_SITE)), (
        f"{rel(SOLE_LLM_CALL_SITE)} is supposed to be the one SDK call site but "
        "does not import the SDK at all -- the invariant test above is vacuous"
    )


# ---------------------------------------------------------------------------
# Invariant 7 -- all iteration over collections is sorted
# ---------------------------------------------------------------------------

_MAPPING_VIEWS = {"items", "keys", "values"}


def _is_sorted_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"sorted", "enumerate", "zip", "reversed"}
    )


def _unsorted_mapping_iteration(tree: ast.Module) -> list[tuple[str, int]]:
    """Find ``for`` loops and comprehensions walking a mapping view unsorted.

    ``sorted()``, ``enumerate()``, ``zip()`` and ``reversed()`` at the top of
    the iterable are accepted: the first sorts, and the others are transparent
    wrappers whose own argument is checked by recursion.
    """
    offenders: list[tuple[str, int]] = []

    def check(iter_node: ast.expr, lineno: int) -> None:
        node = iter_node
        while _is_sorted_call(node):
            call = node
            assert isinstance(call, ast.Call)
            if isinstance(call.func, ast.Name) and call.func.id == "sorted":
                return
            if not call.args:
                return
            node = call.args[0]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MAPPING_VIEWS
            and not node.args
        ):
            offenders.append((node.func.attr, lineno))
        elif isinstance(node, ast.Set | ast.SetComp):
            offenders.append(("set literal", lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.For | ast.AsyncFor):
            check(node.iter, node.lineno)
        elif isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            for generator in node.generators:
                check(generator.iter, node.lineno)
    return offenders


def test_iteration_over_mappings_is_sorted() -> None:
    """Invariant 7: dict and set iteration order is a nondeterminism vector.

    Insertion order is reproducible within a process and *not* across a
    checkpoint resume, a different worker count, or a differently ordered
    upstream. That is precisely the M8 replay-divergence failure mode.
    """
    offenders = [
        f"{rel(path)}:{lineno} iterates {what} without sorted()"
        for path in python_sources()
        for what, lineno in _unsorted_mapping_iteration(parse(path))
    ]
    assert (
        offenders == []
    ), "iterate mappings and sets in sorted order (invariant 7). Offenders:\n  " + "\n  ".join(
        offenders
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("for k in d.items():\n    pass\n", 1),
        ("for k in sorted(d.items()):\n    pass\n", 0),
        ("for i, k in enumerate(sorted(d)):\n    pass\n", 0),
        ("for i, k in enumerate(d.keys()):\n    pass\n", 1),
        ("x = [v for v in d.values()]\n", 1),
        ("x = [v for v in sorted(d.values())]\n", 0),
        ("for x in {1, 2}:\n    pass\n", 1),
        ("for x in some_list:\n    pass\n", 0),
    ],
)
def test_the_invariant_7_detector_actually_detects(source: str, expected: int) -> None:
    """Guard the guard: a detector that finds nothing would pass silently."""
    assert len(_unsorted_mapping_iteration(ast.parse(source))) == expected
