"""Only ``cascade/config.py`` reads the process environment.

Every tunable must reach the rest of the codebase as a validated model. A bare
``os.getenv`` somewhere else is how a setting acquires a silent default that
differs between a developer's shell and the study run -- and the M8 replay
hash is the first thing that would notice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "cascade"
SOLE_ENV_READER = PACKAGE_ROOT / "config.py"

_ENV_ATTRS = {"environ", "getenv", "environb"}


def _environment_reads(tree: ast.AST) -> list[int]:
    """Line numbers of ``os.environ`` / ``os.getenv`` accesses."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _ENV_ATTRS
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            lines.extend(node.lineno for alias in node.names if alias.name in _ENV_ATTRS)
    return lines


def test_only_config_reads_the_environment() -> None:
    readers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if _environment_reads(ast.parse(path.read_text(encoding="utf-8")))
    )
    assert readers == [str(SOLE_ENV_READER.relative_to(REPO_ROOT))], (
        "only cascade/config.py may read the process environment. Found: " f"{readers}"
    )


def test_config_actually_reads_the_environment() -> None:
    """Guard the guard: an empty result set would pass the test above."""
    assert _environment_reads(ast.parse(SOLE_ENV_READER.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os\nx = os.environ['A']\n", 1),
        ("import os\nx = os.getenv('A')\n", 1),
        ("from os import getenv\nx = getenv('A')\n", 1),
        ("from cascade.config import Settings\nx = Settings()\n", 0),
        ("x = settings.database.host\n", 0),
    ],
)
def test_the_detector_actually_detects(source: str, expected: int) -> None:
    assert len(_environment_reads(ast.parse(source))) == expected
