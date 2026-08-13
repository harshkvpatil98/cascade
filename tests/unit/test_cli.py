"""The CLI surface and the process-wide error boundary.

Every phase of the study is a subcommand (spec §13) -- there are no
notebook-driven pipelines -- so this is also the test that the study is
scriptable end to end.

The budget-ceiling acceptance criterion is exercised **in a subprocess against
the real console script**, not through an in-process runner. The criterion is
about the process exit code, and an in-process test would assert on a
``typer.Exit`` object rather than on what a supervising script would actually
observe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cascade.cli import app
from cascade.version import (
    EXIT_BUDGET_BREACH,
    EXIT_OK,
    EXIT_PRECONDITION,
    PINNED_STACK,
)

runner = CliRunner()

CONSOLE_SCRIPT = Path(sys.executable).parent / "cascade"

# Phases still to land. Each must refuse loudly rather than exit 0.
# `ledger` left M0's stub list at M1 and is now a sub-app (build/seal/verify).
DEFERRED_PHASES = ["corpus", "compile", "simulate", "evaluate", "trace", "report"]
IMPLEMENTED_SUBAPPS = ["ledger", "db"]


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_offline_exits_zero() -> None:
    """M0 acceptance: doctor exits 0 and prints the pinned stack."""
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == EXIT_OK, result.output
    assert "all checks passed" in result.output


def test_doctor_prints_every_pinned_dependency() -> None:
    """A silent substitution must show as a failed check, not a surprise at M5."""
    result = runner.invoke(app, ["doctor", "--offline"])
    for entry in PINNED_STACK:
        assert entry.label in result.output, f"{entry.label} missing from doctor output"


def test_doctor_reports_the_pinned_models() -> None:
    """Rich folds long cells onto a second line at narrow widths.

    Nothing is lost -- folding is not truncation -- but a substring assertion
    would see the wrap. The console is widened so the test measures what
    doctor reports, not how the terminal happens to be sized.
    """
    result = runner.invoke(app, ["doctor", "--offline"], env={"COLUMNS": "200"})
    assert "claude-haiku-4-5-20251001" in result.output
    assert "claude-sonnet-4-6" in result.output


def test_doctor_folds_rather_than_truncates_long_values() -> None:
    """A truncated version string would defeat the point of printing it."""
    narrow = runner.invoke(app, ["doctor", "--offline"], env={"COLUMNS": "60"})
    assert narrow.exit_code == EXIT_OK
    assert "…" not in narrow.output, "doctor must not ellipsise a version string"


def test_doctor_online_checks_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--offline`` a dead Postgres must fail the check, not be skipped."""
    monkeypatch.setenv("CASCADE_DATABASE__PORT", "1")  # nothing listens here
    monkeypatch.setenv("CASCADE_LANGFUSE__ENABLED", "false")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == EXIT_PRECONDITION
    assert "postgres" in result.output


# ---------------------------------------------------------------------------
# Deferred phases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", DEFERRED_PHASES)
def test_unimplemented_phase_exits_precondition(phase: str) -> None:
    """A stub must exit non-zero. A stub that exits 0 claims success."""
    result = runner.invoke(app, [phase])
    assert result.exit_code == EXIT_PRECONDITION
    assert "not implemented yet" in result.output


def test_every_spec_subcommand_exists() -> None:
    """Spec §13: if it is a step in the study, it is reachable from the CLI."""
    result = runner.invoke(app, ["--help"])
    for phase in ["doctor", *DEFERRED_PHASES, *IMPLEMENTED_SUBAPPS]:
        assert phase in result.output


def test_ledger_exposes_the_m1_lifecycle() -> None:
    """build / seal / verify are the three states the frozen split has."""
    result = runner.invoke(app, ["ledger", "--help"])
    for command in ("build", "seal", "verify", "status"):
        assert command in result.output


def test_no_arguments_shows_help_rather_than_doing_something() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


# ---------------------------------------------------------------------------
# dev hooks
# ---------------------------------------------------------------------------


def test_dev_cost_prints_six_decimal_places() -> None:
    """The ledger tests read this; fewer digits would hide sub-cent drift."""
    result = runner.invoke(
        app,
        [
            "dev",
            "cost",
            "--model",
            "claude-haiku-4-5-20251001",
            "--input-tokens",
            "260",
            "--output-tokens",
            "45",
            "--cache-read-tokens",
            "1900",
            "--batch",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    printed = result.output.strip()
    assert printed == "0.000338"
    assert Decimal(printed) == Decimal("0.000338")


def test_dev_config_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dev config` dumps everything; a plaintext key in that dump is a leak."""
    monkeypatch.setenv("CASCADE_ANTHROPIC_API_KEY", "sk-ant-should-not-appear")
    result = runner.invoke(app, ["dev", "config"])
    assert result.exit_code == EXIT_OK, result.output
    assert "sk-ant-should-not-appear" not in result.output
    assert "cascade-v1" in result.output


def test_dev_config_accepts_an_ablation_overlay() -> None:
    result = runner.invoke(app, ["dev", "config", "--config", "C09"])
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.output)
    assert payload["flags"]["causal_decomposition"] is False


# ---------------------------------------------------------------------------
# The budget ceiling, end to end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="console script not installed")
def test_budget_ceiling_aborts_with_exit_code_two(tmp_path: Path) -> None:
    """M0 acceptance: a $0.01 ceiling aborts, checkpoints and exits non-zero.

    Run as a real process so the assertion is on the exit status a supervising
    script would see -- exit code 2 specifically, distinct from a crash (1) or
    a precondition failure (3).
    """
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CASCADE_PATHS__CHECKPOINTS": str(tmp_path / "checkpoints"),
        "CASCADE_LLM__CACHE_DIR": str(tmp_path / "llm-cache"),
    }
    result = subprocess.run(
        [str(CONSOLE_SCRIPT), "dev", "budget-probe", "--ceiling", "0.01", "--phase", "simulate"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )

    assert result.returncode == EXIT_BUDGET_BREACH, (
        f"expected exit {EXIT_BUDGET_BREACH}, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "budget ceiling breached" in result.stderr

    checkpoint = tmp_path / "checkpoints" / "simulate.checkpoint.json"
    assert checkpoint.is_file(), "a breach must leave a resumable checkpoint"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["phase"] == "simulate"
    assert payload["resume"]["probe"] is True
    assert Decimal(payload["spent_usd"]) > Decimal("0.01")


@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="console script not installed")
def test_the_probe_would_pass_under_a_ceiling_it_cannot_reach(tmp_path: Path) -> None:
    """Guard the instrument: prove the probe exits 0 when nothing breaches.

    Without this, a probe that always exited 2 -- for any reason -- would make
    the acceptance test above meaningless.
    """
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CASCADE_PATHS__CHECKPOINTS": str(tmp_path / "checkpoints"),
    }
    result = subprocess.run(
        [str(CONSOLE_SCRIPT), "dev", "budget-probe", "--ceiling", "1000", "--phase", "simulate"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == EXIT_OK, result.stderr
    assert "no breach after 1000 calls" in result.stdout


# ---------------------------------------------------------------------------
# The error boundary
# ---------------------------------------------------------------------------


def test_exit_codes_are_distinct() -> None:
    """A supervising script must be able to tell the failure modes apart."""
    from cascade.version import EXIT_CACHE_MISS, EXIT_ERROR

    codes = [EXIT_OK, EXIT_ERROR, EXIT_BUDGET_BREACH, EXIT_PRECONDITION, EXIT_CACHE_MISS]
    assert codes == [0, 1, 2, 3, 4]
    assert len(set(codes)) == len(codes)


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        ("BudgetExceeded", EXIT_BUDGET_BREACH),
        ("CacheMiss", 4),
        ("PromptTooShortToCache", EXIT_PRECONDITION),
    ],
)
def test_error_boundary_maps_each_failure_to_its_code(
    monkeypatch: pytest.MonkeyPatch, exception: str, expected_code: int
) -> None:
    """The boundary is what turns a raised type into an observable exit status."""
    from decimal import Decimal as D

    import cascade.cli as cli_module
    from cascade.llm import types as llm_types

    raised: BaseException
    if exception == "BudgetExceeded":
        raised = llm_types.BudgetExceeded("simulate", D("1"), D("0.5"), "checkpoints/cp.json")
    elif exception == "CacheMiss":
        raised = llm_types.CacheMiss("no recording")
    else:
        raised = llm_types.PromptTooShortToCache(1900, 4096)

    def boom(*args: object, **kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(cli_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["cascade", "doctor"])
    assert cli_module.main() == expected_code


def test_typer_exit_codes_survive_the_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``standalone_mode=False`` click *returns* the code instead of raising.

    Dropping that return value would turn every ``raise typer.Exit(3)`` into a
    zero exit -- a stub that claims success.
    """
    import cascade.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["cascade", "corpus"])
    assert cli_module.main() == EXIT_PRECONDITION
