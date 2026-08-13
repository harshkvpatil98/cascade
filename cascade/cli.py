"""Typer application. Every phase of the study is a subcommand.

There are no notebook-driven pipelines (spec §13): if it is a step in the
study, it is reachable from here and therefore scriptable, resumable and
CI-checkable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from decimal import Decimal
from importlib import metadata
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from cascade.config import Settings, load_settings
from cascade.llm.types import BudgetExceeded, CacheMiss, PromptTooShortToCache
from cascade.version import (
    EXIT_BUDGET_BREACH,
    EXIT_CACHE_MISS,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_PRECONDITION,
    PINNED_STACK,
)

app = typer.Typer(
    name="cascade",
    help="Multi-agent causal simulation for strategic forecasting.",
    no_args_is_help=True,
    add_completion=False,
)
dev_app = typer.Typer(help="Self-test hooks. Not part of the study pipeline.", hidden=True)
app.add_typer(dev_app, name="dev")
db_app = typer.Typer(help="Forward-only schema migrations.", no_args_is_help=True)
app.add_typer(db_app, name="db")
ledger_app = typer.Typer(
    help="Build, seal and verify the scenario registry (M1).", no_args_is_help=True
)
app.add_typer(ledger_app, name="ledger")

console = Console()
err_console = Console(stderr=True)

OverlayOpt = Annotated[
    str | None,
    typer.Option("--config", help="Ablation overlay name from configs/ablations/."),
]


def _settings(overlay: str | None) -> Settings:
    return load_settings(overlay)


def _fail(message: str, code: int) -> None:
    err_console.print(f"[bold red]error[/bold red] {message}")
    raise typer.Exit(code)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _dep_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _tool_version(binary: str, args: list[str]) -> str | None:
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [path, *args], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def _postgres_status(settings: Settings) -> tuple[bool, str]:
    try:
        import psycopg
    except ImportError:
        return False, "psycopg not installed"
    try:
        with (
            psycopg.connect(settings.database_url("admin"), connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SHOW server_version")
            server_row = cur.fetchone()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            vec_row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return False, f"unreachable: {type(exc).__name__}: {exc}".replace("\n", " ")[:120]
    server = server_row[0] if server_row else "unknown"
    vector = vec_row[0] if vec_row else "extension not installed"
    return True, f"PostgreSQL {server}, pgvector {vector}"


def _langfuse_status(settings: Settings) -> tuple[bool, str]:
    if not settings.langfuse.enabled:
        return True, "disabled in config"
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"
    url = settings.langfuse.host.rstrip("/") + "/api/public/health"
    try:
        response = httpx.get(url, timeout=5.0)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return False, f"unreachable: {type(exc).__name__}"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"
    return True, f"healthy at {settings.langfuse.host}"


@app.command()
def doctor(
    config: OverlayOpt = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip Postgres and Langfuse service checks.")
    ] = False,
) -> None:
    """Verify the toolchain and pinned stack. Exits 0 only when everything checks out."""
    ok = True
    settings = _settings(config)

    # overflow="fold" rather than Rich's default ellipsis: the acceptance
    # criterion is that doctor *prints* every pinned version, and a version
    # string silently cut to "0.12.3 (Homebrew 2026-08-0…" does not satisfy it
    # on a narrow terminal. Folding wraps instead of discarding.
    table = Table(title="Cascade doctor", show_lines=False)
    table.add_column("Component", style="cyan", overflow="fold")
    table.add_column("Pinned", style="dim", overflow="fold")
    table.add_column("Found", overflow="fold")
    table.add_column("", width=3)

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info[:2] == (3, 12)
    ok &= py_ok
    table.add_row("python", "3.12.x", py, "[green]OK[/green]" if py_ok else "[red]NO[/red]")

    for entry in PINNED_STACK:
        found = _dep_version(entry.distribution)
        present = found is not None
        if not present and entry.required:
            ok = False
        status = (
            "[green]OK[/green]"
            if present
            else ("[red]NO[/red]" if entry.required else "[yellow]--[/yellow]")
        )
        table.add_row(
            entry.label,
            entry.pin,
            found or f"not installed ({entry.extra})" if not present else found or "",
            status,
        )

    for label, binary, args in (
        ("uv", "uv", ["--version"]),
        ("docker", "docker", ["--version"]),
    ):
        found = _tool_version(binary, args)
        present = found is not None
        ok &= present
        table.add_row(
            label,
            "pinned",
            found or "not found",
            "[green]OK[/green]" if present else "[red]NO[/red]",
        )

    table.add_row("agent model", settings.models.agent, settings.models.agent, "[green]OK[/green]")
    table.add_row(
        "compiler model", settings.models.compiler, settings.models.compiler, "[green]OK[/green]"
    )
    table.add_row("llm mode", "record|replay|live", settings.llm.mode, "[green]OK[/green]")

    if not offline:
        pg_ok, pg_msg = _postgres_status(settings)
        ok &= pg_ok
        table.add_row(
            "postgres",
            "16 + pgvector 0.8",
            pg_msg,
            "[green]OK[/green]" if pg_ok else "[red]NO[/red]",
        )
        lf_ok, lf_msg = _langfuse_status(settings)
        ok &= lf_ok
        table.add_row(
            "langfuse", "self-hosted", lf_msg, "[green]OK[/green]" if lf_ok else "[red]NO[/red]"
        )

    console.print(table)

    cache = settings.cache_path()
    console.print(f"llm cache: {cache} ({'exists' if cache.is_dir() else 'not yet created'})")

    if not ok:
        _fail("one or more checks failed (see table above)", EXIT_PRECONDITION)
    console.print("[bold green]doctor: all checks passed[/bold green]")


# ---------------------------------------------------------------------------
# Phase subcommands. Stubbed at M0; each lands in its own milestone.
# ---------------------------------------------------------------------------


def _not_yet(name: str, milestone: str) -> None:
    err_console.print(
        f"[yellow]{name} is not implemented yet[/yellow] -- lands at {milestone}. "
        "Milestones are executed strictly in order (spec §14.1)."
    )
    raise typer.Exit(EXIT_PRECONDITION)


@app.command()
def corpus(config: OverlayOpt = None) -> None:
    """Ingest, dedupe, chunk and embed the evidence corpus (M2)."""
    _not_yet("corpus", "M2")


@app.command()
def compile(config: OverlayOpt = None) -> None:
    """Compile scenarios into typed causal graphs (M4)."""
    _not_yet("compile", "M4")


@app.command()
def simulate(config: OverlayOpt = None) -> None:
    """Run the seeded ensemble (M5/M6)."""
    _not_yet("simulate", "M5/M6")


@app.command()
def evaluate(config: OverlayOpt = None) -> None:
    """Score forecasts, run the ablation grid, write metrics (M7)."""
    _not_yet("evaluate", "M7")


@app.command()
def trace(config: OverlayOpt = None) -> None:
    """Walk the provenance chain from an outcome to its root cause (M8)."""
    _not_yet("trace", "M8")


@app.command()
def report(config: OverlayOpt = None) -> None:
    """Write the study report artifact (M7/M9)."""
    _not_yet("report", "M7/M9")


# ---------------------------------------------------------------------------
# db: migrations
# ---------------------------------------------------------------------------


@db_app.command("migrate")
def db_migrate(
    config: OverlayOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List pending migrations without applying them.")
    ] = False,
) -> None:
    """Apply every pending migration, in order."""
    from cascade.db import MigrationError, apply_all, pending

    settings = _settings(config)
    try:
        todo = pending(settings)
        if not todo:
            console.print("[green]schema is up to date[/green]")
            return
        if dry_run:
            for migration in todo:
                console.print(f"pending: {migration.name}")
            return
        applied = apply_all(settings)
    except MigrationError as exc:
        _fail(str(exc), EXIT_PRECONDITION)
        return
    for migration in applied:
        console.print(f"[green]applied[/green] {migration.name}")


@db_app.command("status")
def db_status(config: OverlayOpt = None) -> None:
    """Show which migrations are applied and which are pending."""
    from cascade.db import applied_versions, discover

    settings = _settings(config)
    try:
        already = applied_versions(settings)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        _fail(f"cannot reach database: {exc}", EXIT_PRECONDITION)
        return

    table = Table(title="Migrations")
    table.add_column("Version", style="cyan")
    table.add_column("File")
    table.add_column("State")
    for migration in discover():
        recorded = already.get(migration.version)
        if recorded is None:
            state = "[yellow]pending[/yellow]"
        elif recorded != migration.checksum:
            state = "[red]CHANGED SINCE APPLIED[/red]"
        else:
            state = "[green]applied[/green]"
        table.add_row(migration.version, migration.name, state)
    console.print(table)


# ---------------------------------------------------------------------------
# ledger: the scenario registry (M1)
# ---------------------------------------------------------------------------


def _print_build_report(result: Any) -> None:
    """Print the measured composition of a build. Measured, never targeted."""
    from cascade.ledger.registry import summarise

    summary = summarise(result)
    report = result.selection.report

    sources = Table(title="Sources")
    sources.add_column("Source", style="cyan")
    sources.add_column("Status")
    sources.add_column("Raw", justify="right")
    sources.add_column("Detail", overflow="fold")
    for status in result.sources:
        sources.add_row(
            status.name,
            "[green]ok[/green]" if status.available else "[red]unavailable[/red]",
            str(status.n_raw),
            status.detail,
        )
    console.print(sources)

    composition = Table(title="Selected set")
    composition.add_column("Quantity", style="cyan")
    composition.add_column("Measured", justify="right")
    composition.add_column("Required", justify="right")
    composition.add_column("", width=3)

    def row(name: str, measured: str, required: str, ok: bool) -> None:
        composition.add_row(
            name, measured, required, "[green]OK[/green]" if ok else "[red]NO[/red]"
        )

    settings_ledger = _settings(None).ledger
    row(
        "scenarios",
        str(report.n_selected),
        str(report.target_n),
        report.met_target,
    )
    row(
        "resolved-YES rate",
        f"{report.yes_rate:.4f}",
        f"[{settings_ledger.yes_rate_min:.2f}, {settings_ledger.yes_rate_max:.2f}]",
        report.n_selected > 0
        and settings_ledger.yes_rate_min <= report.yes_rate <= settings_ledger.yes_rate_max,
    )
    row(
        "max domain share",
        f"{report.max_domain_share:.4f}",
        f"<= {settings_ledger.max_domain_share:.2f}",
        report.max_domain_share <= settings_ledger.max_domain_share,
    )
    console.print(composition)

    histogram = Table(title="Domain histogram")
    histogram.add_column("Domain", style="cyan")
    histogram.add_column("n", justify="right")
    histogram.add_column("share", justify="right")
    for domain, count in report.domain_counts:
        share = count / report.n_selected if report.n_selected else 0.0
        histogram.add_row(domain, str(count), f"{share:.3f}")
    console.print(histogram)

    rejects = Table(title="Rejections by rule")
    rejects.add_column("Reason", style="cyan")
    rejects.add_column("n", justify="right")
    for reason, count in sorted(report.rejection_counts, key=lambda item: -item[1]):
        rejects.add_row(reason, str(count))
    console.print(rejects)

    console.print(f"party rules: {dict(report.party_rule_counts)}")
    console.print(f"sources in set: {dict(report.source_counts)}")
    if result.climatology is not None:
        console.print(
            f"climatology: base rate {result.climatology.base_rate:.4f}, "
            f"Brier [bold]{result.climatology.brier:.6f}[/bold]"
        )
    console.print(f"manifest sha256: {summary['manifest_sha256']}")
    if report.shortfall_reason:
        err_console.print(f"[yellow]shortfall[/yellow] {report.shortfall_reason}")


@ledger_app.command("build")
def ledger_build(
    config: OverlayOpt = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch sources. Changes the pool and the manifest."),
    ] = False,
    write: Annotated[bool, typer.Option("--write/--no-write", help="Persist to Postgres.")] = False,
    replace: Annotated[
        bool, typer.Option("--replace", help="Overwrite an existing registry.")
    ] = False,
) -> None:
    """Assemble the registry from every source and report its composition.

    Exits 3 when the set does not satisfy the inclusion rules, so a shortfall
    cannot be mistaken for success by a script. The rules are never relaxed to
    reach the target -- see cascade/ledger/select.py for the precedence.
    """
    from cascade.ledger.registry import build_registry
    from cascade.ledger.store import write_registry

    settings = _settings(config)
    result = build_registry(settings, refresh=refresh)
    _print_build_report(result)

    report = result.selection.report
    ledger_cfg = settings.ledger
    ok = (
        report.met_target
        and ledger_cfg.yes_rate_min <= report.yes_rate <= ledger_cfg.yes_rate_max
        and report.max_domain_share <= ledger_cfg.max_domain_share
    )

    if write and result.records:
        written = write_registry(settings, result.records, replace=replace)
        console.print(f"[green]wrote[/green] {written} scenarios and labels")

    if not ok:
        _fail(
            "the assembled set does not satisfy the inclusion rules; "
            "see the shortfall above. Do not relax the rules to close the gap.",
            EXIT_PRECONDITION,
        )


@ledger_app.command("seal")
def ledger_seal(config: OverlayOpt = None) -> None:
    """Hash the loaded registry and record the seal (spec §1.3).

    Every later phase asserts this hash on startup. Sealing a set that does
    not meet the inclusion rules is refused: the seal is what downstream
    phases trust.
    """
    from cascade.ledger.climatology import climatology_of
    from cascade.ledger.manifest import compute_manifest
    from cascade.ledger.store import load_records, write_manifest

    settings = _settings(config)
    records = load_records(settings, role="admin")
    if not records:
        _fail(
            "no scenarios are loaded; run `cascade ledger build --write` first", EXIT_PRECONDITION
        )
        return

    climatology = climatology_of(records)
    ledger_cfg = settings.ledger
    if not ledger_cfg.yes_rate_min <= climatology.base_rate <= ledger_cfg.yes_rate_max:
        _fail(
            f"resolved-YES rate {climatology.base_rate:.4f} is outside "
            f"[{ledger_cfg.yes_rate_min}, {ledger_cfg.yes_rate_max}]; refusing to seal",
            EXIT_PRECONDITION,
        )

    manifest = compute_manifest(records)
    write_manifest(
        settings,
        manifest_sha256=manifest,
        climatology=climatology,
        study_salt=settings.study.salt,
        notes=f"n={climatology.n}",
    )
    console.print(f"[green]sealed[/green] {climatology.n} scenarios")
    console.print(f"manifest sha256: [bold]{manifest}[/bold]")
    console.print(
        f"climatology: base rate {climatology.base_rate:.4f}, " f"Brier {climatology.brier:.6f}"
    )


@ledger_app.command("verify")
def ledger_verify(config: OverlayOpt = None) -> None:
    """Assert the loaded registry still hashes to the sealed manifest.

    This is the check every later phase runs on startup. A mismatch means the
    frozen split moved, and the answer is never to reseal.
    """
    from cascade.ledger.manifest import ManifestMismatch, verify_manifest
    from cascade.ledger.store import load_records, read_manifest

    settings = _settings(config)
    sealed = read_manifest(settings, role="admin")
    if sealed is None:
        _fail("the registry has never been sealed; run `cascade ledger seal`", EXIT_PRECONDITION)
        return
    records = load_records(settings, role="admin")
    try:
        verify_manifest(records, sealed.manifest_sha256)
    except ManifestMismatch as exc:
        _fail(str(exc), EXIT_PRECONDITION)
        return
    console.print(
        f"[green]manifest verified[/green] {sealed.n_scenarios} scenarios, "
        f"sha256 {sealed.manifest_sha256[:16]}..."
    )


@ledger_app.command("status")
def ledger_status(config: OverlayOpt = None) -> None:
    """Show what is loaded and whether it is sealed."""
    from cascade.ledger.store import load_scenarios, read_manifest

    settings = _settings(config)
    scenarios = load_scenarios(settings, role="admin")
    sealed = read_manifest(settings, role="admin")

    table = Table(title="Scenario registry")
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("scenarios loaded", str(len(scenarios)))
    table.add_row("sealed", "yes" if sealed else "[yellow]no[/yellow]")
    if sealed:
        table.add_row("manifest sha256", sealed.manifest_sha256)
        table.add_row("sealed at", str(sealed.sealed_at))
        table.add_row("resolved-YES rate", f"{sealed.yes_rate:.4f}")
        table.add_row("climatology Brier", f"{sealed.climatology_brier:.6f}")
    console.print(table)


# ---------------------------------------------------------------------------
# dev: self-test hooks exercised by the acceptance tests
# ---------------------------------------------------------------------------


@dev_app.command("budget-probe")
def budget_probe(
    ceiling: Annotated[float, typer.Option(help="Ceiling in USD for this probe.")] = 0.01,
    phase: Annotated[str, typer.Option(help="Phase name to book spend against.")] = "simulate",
) -> None:
    """Spend synthetic tokens against a real meter until the ceiling breaks.

    Exercises the production abort path end to end -- meter -> BudgetExceeded ->
    CLI boundary -> non-zero exit -- so the M0 acceptance criterion is tested
    against the real code rather than a mock.
    """
    from cascade.llm.meter import CostMeter
    from cascade.llm.types import Usage

    settings = _settings(None)
    meter = CostMeter(settings, phase, ceiling_usd=Decimal(str(ceiling)))
    meter.set_resume_state({"probe": True, "completed_units": 0})

    usage = Usage(input_tokens=100_000, output_tokens=10_000)
    for index in range(1_000):
        meter.set_resume_state({"probe": True, "completed_units": index})
        meter.record(model=settings.models.agent, usage=usage, batch=False)
    console.print(f"no breach after 1000 calls; spent ${meter.total_usd:.6f}")


@dev_app.command("cost")
def dev_cost(
    model: Annotated[str, typer.Option(help="Model id from the pricing table.")] = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> None:
    """Print the exact USD cost of a hypothetical call. Used by the ledger tests."""
    from cascade.llm.meter import compute_cost
    from cascade.llm.types import Usage

    settings = _settings(None)
    target = model or settings.models.agent
    cost = compute_cost(
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_tokens,
            cache_creation_input_tokens=cache_write_tokens,
        ),
        price=settings.price_for(target),
        multipliers=settings.pricing_multipliers,
        batch=batch,
        cache_ttl=settings.prompt_cache.ttl,
    )
    console.print(f"{cost:.6f}")


@dev_app.command("config")
def dev_config(config: OverlayOpt = None) -> None:
    """Dump the resolved configuration, secrets redacted."""
    import json

    settings = _settings(config)
    payload: dict[str, Any] = json.loads(settings.model_dump_json())
    console.print_json(data=payload)


# ---------------------------------------------------------------------------
# Error boundary
# ---------------------------------------------------------------------------


def main() -> int:
    """Console-script entry point with the process-wide error boundary.

    Maps the failure modes the spec cares about onto distinct exit codes so a
    supervising script can tell a budget abort from a replay miss from a bug.
    """
    try:
        # With standalone_mode=False click *returns* the exit code for a
        # typer.Exit rather than raising it, so the return value is
        # load-bearing. Ignoring it silently turns every `raise typer.Exit(3)`
        # into a zero exit -- a stub that claims success.
        result = app(standalone_mode=False)
        if isinstance(result, int):
            return result
    except typer.Exit as exc:
        return int(exc.exit_code)
    except click_exceptions() as exc:  # pragma: no cover - argument errors
        err_console.print(f"[red]{exc}[/red]")
        return EXIT_ERROR
    except BudgetExceeded as exc:
        err_console.print(f"[bold red]budget ceiling breached[/bold red] {exc}")
        return EXIT_BUDGET_BREACH
    except CacheMiss as exc:
        err_console.print(f"[bold red]cache miss in replay mode[/bold red] {exc}")
        return EXIT_CACHE_MISS
    except PromptTooShortToCache as exc:
        err_console.print(f"[bold red]prompt cache misconfigured[/bold red] {exc}")
        return EXIT_PRECONDITION
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("[yellow]interrupted[/yellow]")
        return EXIT_ERROR
    return EXIT_OK


def click_exceptions() -> type[BaseException] | tuple[type[BaseException], ...]:
    """Return click's user-error types without importing click at module scope."""
    import click

    return (click.ClickException, click.UsageError)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
