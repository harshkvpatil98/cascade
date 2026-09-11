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
corpus_app = typer.Typer(
    help="Build, inspect and verify the evidence corpus (M2).", no_args_is_help=True
)
app.add_typer(corpus_app, name="corpus")
ledger_app = typer.Typer(
    help="Build, seal and verify the scenario registry (M1).", no_args_is_help=True
)
app.add_typer(ledger_app, name="ledger")
retrieval_app = typer.Typer(
    help="Chronofence: index, benchmark and verify time-locked retrieval (M3).",
    no_args_is_help=True,
)
app.add_typer(retrieval_app, name="retrieval")
compile_app = typer.Typer(
    help="Lathe: compile scenarios into typed causal graphs (M4).", no_args_is_help=True
)
app.add_typer(compile_app, name="compile")

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
# corpus: the evidence corpus (M2)
# ---------------------------------------------------------------------------


def _print_corpus_stats(stats: Any, target: int) -> None:
    """Print measured corpus figures. Measured, never targeted."""
    table = Table(title="Evidence corpus")
    table.add_column("Quantity", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_column("Required", justify="right")
    table.add_column("", width=3)

    def row(name: str, measured: str, required: str, ok: bool) -> None:
        table.add_row(name, measured, required, "[green]OK[/green]" if ok else "[red]NO[/red]")

    row("chunks", f"{stats.n_chunks:,}", f">= {target:,}", stats.n_chunks >= target)
    row("documents", f"{stats.n_documents:,}", "-", True)
    row("NULL published_at", str(stats.null_dates), "0", stats.null_dates == 0)
    row("future published_at", str(stats.future_dates), "0", stats.future_dates == 0)
    row(
        "embedding coverage",
        f"{stats.embedding_coverage:.4f}",
        "1.0000",
        stats.missing_embeddings == 0 and stats.n_chunks > 0,
    )
    console.print(table)

    per_source = Table(title="Per-source breakdown")
    per_source.add_column("Source", style="cyan")
    per_source.add_column("documents", justify="right")
    per_source.add_column("chunks", justify="right")
    for name, documents, chunks in stats.per_source:
        per_source.add_row(name, f"{documents:,}", f"{chunks:,}")
    console.print(per_source)
    console.print(f"date range: {stats.earliest} .. {stats.latest}")


@corpus_app.command("build")
def corpus_build(
    config: OverlayOpt = None,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Restrict to these sources. Repeatable."),
    ] = None,
    max_units: Annotated[
        int | None,
        typer.Option("--max-units", help="Cap units per source this run. Resumable."),
    ] = None,
) -> None:
    """Ingest, dedupe, chunk and embed evidence (M2).

    Resumable: units already recorded done are skipped, so an interrupted
    ingest continues rather than restarting.
    """
    from datetime import UTC, datetime

    from cascade.corpus.pipeline import run_ingest
    from cascade.corpus.store import corpus_stats

    settings = _settings(config)
    selected = tuple(source) if source else None
    report = run_ingest(
        settings,
        now=datetime.now(UTC),
        sources=selected,
        max_units_per_source=max_units,
    )

    table = Table(title="Ingest run")
    table.add_column("Source", style="cyan")
    table.add_column("units", justify="right")
    table.add_column("skipped", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("docs", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("collapsed", justify="right")
    table.add_column("dropped", overflow="fold")
    for source_report in report.sources:
        dropped = ", ".join(
            f"{reason}={count}" for reason, count in sorted(source_report.dropped.items())
        )
        table.add_row(
            source_report.source,
            str(source_report.units_done),
            str(source_report.units_skipped),
            str(source_report.units_failed),
            f"{source_report.documents_written:,}",
            f"{source_report.chunks_written:,}",
            str(source_report.collapsed),
            dropped or source_report.detail,
        )
    console.print(table)
    console.print(
        f"dedupe collapse ratio: [bold]{report.collapse_ratio:.4f}[/bold] "
        f"({report.collapsed:,} collapsed of {report.collapsed + report.documents_written:,} kept+collapsed)"
    )

    stats = corpus_stats(settings)
    _print_corpus_stats(stats, settings.corpus.target_chunks)


@corpus_app.command("status")
def corpus_status(config: OverlayOpt = None) -> None:
    """Report measured corpus size, coverage and date integrity."""
    from cascade.corpus.store import corpus_stats

    settings = _settings(config)
    _print_corpus_stats(corpus_stats(settings), settings.corpus.target_chunks)


@corpus_app.command("verify")
def corpus_verify(config: OverlayOpt = None) -> None:
    """Assert the corpus invariants over the full table (spec §3.2).

    Exits 3 on any violation. These are leakage invariants, so they are
    asserted over every row rather than a sample: a sampled check on a
    leakage rule is a check that can miss the row that matters.
    """
    from cascade.corpus.store import corpus_stats

    settings = _settings(config)
    stats = corpus_stats(settings)
    _print_corpus_stats(stats, settings.corpus.target_chunks)

    failures: list[str] = []
    if stats.n_chunks == 0:
        failures.append("corpus is empty")
    if stats.null_dates:
        failures.append(f"{stats.null_dates} documents have a NULL published_at")
    if stats.future_dates:
        failures.append(f"{stats.future_dates} documents are dated in the future")
    if stats.missing_embeddings:
        failures.append(f"{stats.missing_embeddings} chunks have no embedding")
    if stats.n_chunks < settings.corpus.target_chunks:
        failures.append(
            f"{stats.n_chunks:,} chunks is short of the {settings.corpus.target_chunks:,} target"
        )

    if failures:
        _fail("; ".join(failures), EXIT_PRECONDITION)
    console.print("[bold green]corpus: all invariants hold[/bold green]")


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
# retrieval: Chronofence (M3)
# ---------------------------------------------------------------------------


def _print_index_report(report: Any) -> None:
    """Print what the index pass did, per partition."""
    table = Table(title="IVFFlat index pass")
    table.add_column("Partition", style="cyan")
    table.add_column("rows", justify="right")
    table.add_column("lists", justify="right")
    table.add_column("target", justify="right")
    table.add_column("action")
    table.add_column("reason", overflow="fold")
    colour = {
        "create": "green",
        "rebuild": "yellow",
        "keep": "dim",
        "skip-empty": "dim",
    }
    for plan in report.plans:
        style = colour.get(plan.action, "")
        table.add_row(
            plan.partition,
            f"{plan.rows:,}",
            "-" if plan.current_lists is None else str(plan.current_lists),
            "-" if plan.target_lists == 0 else str(plan.target_lists),
            f"[{style}]{plan.action}[/{style}]" if style else plan.action,
            plan.reason,
        )
    console.print(table)
    console.print(
        f"created [bold]{report.created}[/bold] · rebuilt [bold]{report.rebuilt}[/bold] · "
        f"kept [bold]{report.kept}[/bold] · skipped empty [bold]{report.skipped_empty}[/bold] "
        f"in {report.elapsed_s:.1f}s"
    )


@retrieval_app.command("index")
def retrieval_index(
    config: OverlayOpt = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without touching any index."),
    ] = False,
) -> None:
    """Create or rebuild one IVFFlat index per non-empty chunks partition.

    Sized `lists = clamp(round(sqrt(rows)), 1, index_max_lists)` from exact
    row counts (ADR-0012). Idempotent: a second run keeps everything already
    within tolerance.
    """
    from cascade.retrieval.index import apply_plans, measure, plan_all
    from cascade.retrieval.schema import IndexReport

    settings = _settings(config)
    retrieval = settings.retrieval

    partitions = measure(settings)
    plans = plan_all(
        partitions,
        max_lists=retrieval.index_max_lists,
        tolerance=retrieval.index_rebuild_tolerance,
    )

    if dry_run:
        _print_index_report(
            IndexReport(
                plans=plans,
                created=sum(1 for plan in plans if plan.action == "create"),
                rebuilt=sum(1 for plan in plans if plan.action == "rebuild"),
                kept=sum(1 for plan in plans if plan.action == "keep"),
                skipped_empty=sum(1 for plan in plans if plan.action == "skip-empty"),
                elapsed_s=0.0,
            )
        )
        console.print("[yellow]dry run: nothing was changed[/yellow]")
        return

    _print_index_report(apply_plans(settings, plans))


def _print_bench_result(result: Any, settings: Settings) -> None:
    """Print measured latency and recall. Measured, never targeted."""
    target = settings.retrieval

    table = Table(title="Chronofence bench")
    table.add_column("Quantity", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_column("Required", justify="right")
    table.add_column("", width=3)

    def row(name: str, measured: str, required: str, ok: bool) -> None:
        table.add_row(name, measured, required, "[green]OK[/green]" if ok else "[red]NO[/red]")

    row("queries", f"{result.queries:,}", "-", True)
    row("ivfflat probes", str(result.probes), str(target.ivfflat_probes), True)
    row("p50", f"{result.latency.p50:.2f} ms", "-", True)
    row(
        "p95",
        f"{result.latency.p95:.2f} ms",
        f"< {target.target_p95_ms:.0f} ms",
        result.latency.p95 < target.target_p95_ms,
    )
    row("p99", f"{result.latency.p99:.2f} ms", "-", True)
    row("min / max", f"{result.latency.minimum:.2f} / {result.latency.maximum:.2f} ms", "-", True)
    row(
        f"recall@{result.recall.k}",
        f"{result.recall.mean:.4f}" if result.recall.measured else "not measured",
        f"> {target.target_recall_at_k:.2f}",
        result.recall.measured and result.recall.mean > target.target_recall_at_k,
    )
    row("recall sample", f"{result.recall.scored:,} queries", "-", True)
    row("empty results", f"{result.empty_results:,}", "-", True)
    console.print(table)

    histogram = Table(title="Latency histogram (ms)")
    histogram.add_column("Bucket", style="cyan", justify="right")
    histogram.add_column("Count", justify="right")
    histogram.add_column("Share", justify="right")
    histogram.add_column("", overflow="crop")
    total = max(1, result.latency.n)
    for lower, upper, count in result.latency.histogram:
        if count == 0:
            continue
        label = f"{lower:g}-{upper:g}" if upper != float("inf") else f">{lower:g}"
        share = count / total
        histogram.add_row(label, f"{count:,}", f"{share:6.2%}", "█" * round(share * 40))
    console.print(histogram)

    console.print(
        f"cutoffs: [bold]{result.distinct_cutoffs}[/bold] distinct, "
        f"{result.earliest_cutoff} .. {result.latest_cutoff}"
    )
    if result.recall.measured:
        console.print(
            f"recall detail: min {result.recall.minimum:.4f}, "
            f"perfect on {result.recall.perfect:,}/{result.recall.scored:,}, "
            f"{result.recall.empty_ground_truth:,} sampled queries had no admissible evidence"
        )
    console.print(f"bench wall time: {result.elapsed_s:.1f}s")


@retrieval_app.command("bench")
def retrieval_bench(
    config: OverlayOpt = None,
    queries: Annotated[
        int | None,
        typer.Option("--queries", help="Override the configured query count."),
    ] = None,
) -> None:
    """Benchmark time-locked retrieval: p50/p95/p99 and recall@k (M3).

    Exits 3 when a measured value misses its acceptance criterion, so a
    regression cannot pass as success in CI.
    """
    from cascade.retrieval.bench import run_bench

    settings = _settings(config)
    result = run_bench(settings, count=queries)
    _print_bench_result(result, settings)

    passed, failures = result.meets(settings)
    if not passed:
        _fail("; ".join(failures), EXIT_PRECONDITION)
    console.print("[bold green]chronofence: latency and recall criteria met[/bold green]")


@retrieval_app.command("verify")
def retrieval_verify(config: OverlayOpt = None) -> None:
    """Assert the Chronofence preconditions hold (spec §4.2).

    Checks the structural guarantees, not a sample of results: the deployed
    function pins the configured probes, every non-empty partition carries a
    correctly sized index, and `cascade_sim` cannot reach the corpus except
    through `chronofence_search`. Exits 3 on any violation.
    """
    from cascade.retrieval.index import measure, plan_all
    from cascade.retrieval.search import Chronofence

    settings = _settings(config)
    retrieval = settings.retrieval
    failures: list[str] = []

    with Chronofence(settings, role="admin") as fence:
        deployed = fence.probes()
    if deployed != retrieval.ivfflat_probes:
        failures.append(
            f"chronofence_search pins ivfflat.probes={deployed} but config says "
            f"{retrieval.ivfflat_probes}; migrations are forward-only, so add a new "
            "migration rather than editing 004"
        )

    partitions = measure(settings)
    plans = plan_all(
        partitions,
        max_lists=retrieval.index_max_lists,
        tolerance=retrieval.index_rebuild_tolerance,
    )
    stale = [plan for plan in plans if plan.action in {"create", "rebuild"}]

    table = Table(title="Chronofence preconditions")
    table.add_column("Check", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_column("", width=3)

    def row(name: str, measured: str, ok: bool) -> None:
        table.add_row(name, measured, "[green]OK[/green]" if ok else "[red]NO[/red]")

    indexed = sum(1 for plan in plans if plan.action == "keep")
    non_empty = sum(1 for plan in plans if plan.rows > 0)
    row("ivfflat probes pinned", str(deployed), deployed == retrieval.ivfflat_probes)
    row("non-empty partitions", str(non_empty), True)
    row("correctly sized indexes", f"{indexed}/{non_empty}", not stale)

    leak_ok, leak_detail = _sim_role_is_fenced(settings)
    row("cascade_sim fenced from chunks", leak_detail, leak_ok)
    console.print(table)

    if stale:
        failures.append(
            f"{len(stale)} partition(s) need an index pass: "
            + ", ".join(f"{plan.partition} ({plan.action})" for plan in stale[:6])
            + ("..." if len(stale) > 6 else "")
            + " -- run `cascade retrieval index`"
        )
    if not leak_ok:
        failures.append(f"cascade_sim is not fenced from the corpus: {leak_detail}")

    if failures:
        _fail("; ".join(failures), EXIT_PRECONDITION)
    console.print("[bold green]chronofence: all preconditions hold[/bold green]")


@retrieval_app.command("memorization")
def retrieval_memorization(
    config: OverlayOpt = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Probe only the first N scenarios by id."),
    ] = None,
) -> None:
    """Measure what the agent model already knows with no context (M3, spec §4.2).

    The one leakage control that cannot be engineered away: Chronofence keeps
    post-cutoff *documents* out of the context window, not post-cutoff *facts*
    out of the weights. Reports the distribution; it never fails a threshold,
    because there is no threshold to enforce -- the number is disclosed
    alongside the headline metric.

    Costs money in `record` mode. The `bench` phase ceiling applies.
    """
    from cascade.ledger.store import load_records
    from cascade.retrieval.memorization import run_probe

    settings = _settings(config)

    # Checked here rather than left to the SDK: a probe that dies twenty
    # scenarios in has already spent money, and the operator needs to know
    # which variable to set before it starts, not after.
    key = settings.anthropic_api_key
    if settings.llm.mode != "replay" and (key is None or not key.get_secret_value().strip()):
        _fail(
            f"llm.mode={settings.llm.mode!r} needs CASCADE_ANTHROPIC_API_KEY, which is "
            "unset or empty. Set it in .env, or run against a recorded cache with "
            "CASCADE_LLM__MODE=replay",
            EXIT_PRECONDITION,
        )

    records = load_records(settings, role="eval")
    if not records:
        _fail("scenario registry is empty; run `cascade ledger build` first", EXIT_PRECONDITION)
    if limit is not None:
        records = tuple(sorted(records, key=lambda item: item.scenario.scenario_id))[:limit]

    report = run_probe(settings, records)

    table = Table(title="Parametric probe (memorization)")
    table.add_column("Quantity", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_row("scenarios probed", f"{report.total:,}")
    table.add_row("answers parsed", f"{report.scored:,}")
    table.add_row("unparseable", f"{report.unparseable:,}")
    table.add_row("mean confidence", f"{report.mean_confidence:.4f}")
    table.add_row("median confidence", f"{report.median_confidence:.4f}")
    table.add_row("probe Brier", "n/a" if report.brier is None else f"{report.brier:.6f}")
    table.add_row("directionally correct", f"{report.directionally_correct:,}/{report.scored:,}")
    table.add_row(
        "confident (>=0.5) and correct", f"{report.confident_and_correct:,}/{report.scored:,}"
    )
    console.print(table)

    deciles = Table(title="Confidence distribution")
    deciles.add_column("Bucket", style="cyan", justify="right")
    deciles.add_column("Scenarios", justify="right")
    deciles.add_column("", overflow="crop")
    for low, count in report.confidence_deciles:
        share = count / max(1, report.scored)
        deciles.add_row(f"{low:.1f}-{low + 0.1:.1f}", f"{count:,}", "█" * round(share * 40))
    console.print(deciles)

    if report.unparseable:
        console.print(
            f"[yellow]{report.unparseable} answers could not be parsed[/yellow] and are "
            "excluded from every statistic above rather than scored as 0.5"
        )
    console.print(
        "[dim]Measured and disclosed, never enforced: a high score is a fact about "
        "the model, not a defect in the harness (spec §4.2).[/dim]"
    )


def _sim_role_is_fenced(settings: Settings) -> tuple[bool, str]:
    """True when `cascade_sim` cannot SELECT the corpus tables directly.

    The grant is the mechanism (invariant 2, ADR-0005), so this asserts the
    mechanism rather than trusting that no code happens to issue the query.
    """
    import psycopg

    try:
        with psycopg.connect(settings.database_url("sim"), connect_timeout=10) as conn:
            for table in ("chunks", "documents"):
                with conn.cursor() as cur:
                    try:
                        cur.execute(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608
                    except psycopg.errors.InsufficientPrivilege:
                        conn.rollback()
                        continue
                    return False, f"cascade_sim can SELECT {table}"
    except psycopg.OperationalError as exc:
        return False, f"cannot connect as cascade_sim: {exc}"
    return True, "denied on chunks and documents"


# ---------------------------------------------------------------------------
# compile: Lathe, the causal decomposition compiler (M4)
# ---------------------------------------------------------------------------


def _lathe(settings: Settings) -> Any:
    """Wire the compiler to the real model, embedder and corpus.

    Retrieval goes through Chronofence as `eval` rather than `sim`: compilation
    is an offline preparation step, not part of a run, and the role separation
    that matters here is the time lock -- which `chronofence_search` enforces
    identically for both roles.
    """
    from cascade.corpus.embed import Embedder
    from cascade.decompose.compiler import Lathe
    from cascade.llm.client import LLMClient
    from cascade.retrieval.search import Chronofence

    embedder = Embedder(
        model_name=settings.models.embedding, batch_size=settings.corpus.embed_batch_size
    )
    embedder.load()

    fence = Chronofence(settings, role="eval")
    fence.__enter__()

    def retrieve(question: str, as_of: Any, k: int) -> list[tuple[str, str, str]]:
        vector = embedder.encode([question])[0]
        result = fence.search(vector, as_of=as_of, k=k)
        return [
            (chunk.published_at.isoformat(), chunk.source, chunk.body) for chunk in result.chunks
        ]

    compiler = Lathe(
        settings=settings,
        client=LLMClient(settings, phase="compile"),
        embed=embedder.encode,
        retrieve=retrieve,
    )
    return compiler, fence


def _require_api_key(settings: Settings) -> None:
    """Fail before spending anything if the credential is absent.

    Checked here rather than left to the SDK: a 180-scenario compile that dies
    on scenario 20 has already spent real money, and the operator needs to know
    which variable to set before it starts.
    """
    key = settings.anthropic_api_key
    if settings.llm.mode != "replay" and (key is None or not key.get_secret_value().strip()):
        _fail(
            f"llm.mode={settings.llm.mode!r} needs CASCADE_ANTHROPIC_API_KEY, which is unset "
            "or empty. Set it in .env, or replay a recorded compile with "
            "CASCADE_LLM__MODE=replay",
            EXIT_PRECONDITION,
        )


def _print_compile_stats(stats: Any, settings: Settings) -> None:
    """Print the M4 acceptance figures. Measured, never targeted."""
    table = Table(title="Causal graphs (M4)")
    table.add_column("Quantity", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_column("Required", justify="right")
    table.add_column("", width=3)

    def row(name: str, measured: str, required: str, ok: bool) -> None:
        table.add_row(name, measured, required, "[green]OK[/green]" if ok else "[red]NO[/red]")

    row(
        "graphs compiled",
        f"{stats.compiled:,}/{stats.scenarios:,}",
        f"{stats.scenarios:,}",
        stats.compiled == stats.scenarios and stats.scenarios > 0,
    )
    row("hard failures", f"{stats.failed:,}", "0", stats.failed == 0)
    row(
        "mean actors",
        f"{stats.mean_actors:.2f}" if stats.compiled else "-",
        "14 +/- 2",
        bool(stats.compiled) and 12.0 <= stats.mean_actors <= 16.0,
    )
    row(
        "actor range",
        f"{stats.min_actors}-{stats.max_actors}" if stats.compiled else "-",
        "8-20",
        bool(stats.compiled) and stats.min_actors >= 8 and stats.max_actors <= 20,
    )
    row(
        "mean factors",
        f"{stats.mean_factors:.2f}" if stats.compiled else "-",
        "-",
        True,
    )
    row(
        "factor range",
        f"{stats.min_factors}-{stats.max_factors}" if stats.compiled else "-",
        "4-12",
        bool(stats.compiled) and stats.min_factors >= 4 and stats.max_factors <= 12,
    )
    row(
        "distinct graph hashes",
        f"{stats.distinct_hashes:,}",
        f"{stats.compiled:,}",
        stats.distinct_hashes == stats.compiled,
    )
    row("mean edges", f"{stats.mean_edges:.1f}" if stats.compiled else "-", "-", True)
    row(
        "mean evidence chunks",
        f"{stats.mean_evidence_chunks:.1f}" if stats.compiled else "-",
        f"{settings.retrieval.k_compiler}",
        True,
    )
    row("total LLM calls", f"{stats.total_llm_calls:,}", "-", True)
    console.print(table)

    if stats.retry_histogram:
        histogram = Table(title="Repair retries")
        histogram.add_column("Retries", style="cyan", justify="right")
        histogram.add_column("Graphs", justify="right")
        histogram.add_column("Share", justify="right")
        histogram.add_column("", overflow="crop")
        for retries, count in stats.retry_histogram:
            share = count / max(1, stats.compiled)
            histogram.add_row(str(retries), f"{count:,}", f"{share:6.2%}", "#" * round(share * 40))
        console.print(histogram)


@compile_app.command("build")
def compile_build(
    config: OverlayOpt = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Compile at most N pending scenarios.")
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Recompile scenarios that already have a graph."),
    ] = False,
) -> None:
    """Compile scenarios into typed causal graphs (M4, spec §5).

    Resumable: scenarios already compiled are skipped unless --rebuild is
    given, so an interrupted run continues rather than paying for 180 scenarios
    again. Costs money in `record` mode; the `compile` phase ceiling applies.
    """
    from cascade.decompose.store import (
        compile_stats,
        completed_scenarios,
        record_failure,
        write_graph,
    )
    from cascade.ledger.store import load_scenarios

    settings = _settings(config)
    _require_api_key(settings)

    scenarios = load_scenarios(settings, role="admin")
    if not scenarios:
        _fail("scenario registry is empty; run `cascade ledger build` first", EXIT_PRECONDITION)

    done = set() if rebuild else completed_scenarios(settings)
    pending = [item for item in scenarios if item.scenario_id not in done]
    if limit is not None:
        pending = pending[:limit]

    console.print(
        f"compiling [bold]{len(pending)}[/bold] scenario(s); "
        f"{len(scenarios) - len(pending)} already done"
    )

    compiler, fence = _lathe(settings)
    compiled = failed = 0
    try:
        for index, scenario in enumerate(pending, start=1):
            outcome = compiler.compile_scenario(scenario)
            if outcome.ok:
                write_graph(settings, outcome)
                compiled += 1
                marker = f"[green]ok[/green] retries={outcome.repair_retries}"
            else:
                record_failure(settings, outcome)
                failed += 1
                marker = f"[red]failed[/red] {'; '.join(outcome.violations)[:90]}"
            console.print(f"  [{index}/{len(pending)}] {scenario.scenario_id}: {marker}")
    finally:
        fence.__exit__(None, None, None)

    console.print(f"compiled [bold]{compiled}[/bold], failed [bold]{failed}[/bold]")
    _print_compile_stats(compile_stats(settings, expected_scenarios=len(scenarios)), settings)
    if failed:
        _fail(
            f"{failed} scenario(s) could not be compiled within the retry budget", EXIT_PRECONDITION
        )


@compile_app.command("status")
def compile_status(config: OverlayOpt = None) -> None:
    """Report measured graph statistics and the repair-retry histogram."""
    from cascade.decompose.store import compile_stats

    settings = _settings(config)
    _print_compile_stats(compile_stats(settings), settings)


@compile_app.command("verify")
def compile_verify(config: OverlayOpt = None) -> None:
    """Assert every stored graph is valid and hashes to its recorded digest.

    Re-runs the full §5.3 validator against what is actually in the database,
    not against what the compiler believed at write time. Exits 3 on any
    violation, hash mismatch, or missing scenario.
    """
    from cascade.corpus.embed import Embedder
    from cascade.decompose.store import compile_stats, verify_hashes
    from cascade.decompose.validator import validate
    from cascade.ledger.store import load_scenarios

    settings = _settings(config)
    failures: list[str] = []

    mismatches = verify_hashes(settings)
    if mismatches:
        failures.append(
            f"{len(mismatches)} stored graph(s) do not match their recorded hash: "
            + ", ".join(item.scenario_id for item in mismatches[:5])
        )

    scenarios = {item.scenario_id: item for item in load_scenarios(settings, role="admin")}
    stats = compile_stats(settings)

    embedder = Embedder(
        model_name=settings.models.embedding, batch_size=settings.corpus.embed_batch_size
    )
    invalid: list[str] = []
    if stats.compiled:
        embedder.load()
        from cascade.decompose.store import load_graphs

        for graph in load_graphs(settings, role="admin"):
            scenario = scenarios.get(graph.scenario_id)
            if scenario is None:
                invalid.append(f"{graph.scenario_id}: no such scenario in the registry")
                continue
            outcome_text = f"{scenario.question} {scenario.resolution_criterion}".strip()
            report = validate(graph, outcome_text=outcome_text, embed=embedder.encode)
            if not report.ok:
                invalid.append(f"{graph.scenario_id}: {report.render() or 'checks skipped'}")

    table = Table(title="Causal graph preconditions")
    table.add_column("Check", style="cyan")
    table.add_column("Measured", justify="right")
    table.add_column("", width=3)

    def row(name: str, measured: str, ok: bool) -> None:
        table.add_row(name, measured, "[green]OK[/green]" if ok else "[red]NO[/red]")

    row(
        "graphs stored",
        f"{stats.compiled:,}/{stats.scenarios:,}",
        stats.compiled == stats.scenarios,
    )
    row(
        "hash matches content",
        f"{stats.compiled - len(mismatches):,}/{stats.compiled:,}",
        not mismatches,
    )
    row(
        "passes the §5.3 validator",
        f"{stats.compiled - len(invalid):,}/{stats.compiled:,}",
        not invalid,
    )
    row("hard failures logged", f"{stats.failed:,}", stats.failed == 0)
    console.print(table)

    if invalid:
        for line in invalid[:10]:
            err_console.print(f"  [red]{line}[/red]")
        failures.append(f"{len(invalid)} stored graph(s) fail the validator")
    if stats.compiled == 0:
        failures.append("no graphs are stored; run `cascade compile build`")
    if stats.failed:
        failures.append(f"{stats.failed} scenario(s) are logged as hard failures")

    if failures:
        _fail("; ".join(failures), EXIT_PRECONDITION)
    console.print("[bold green]causal graphs: all preconditions hold[/bold green]")


@compile_app.command("audit")
def compile_audit(
    config: OverlayOpt = None,
    score: Annotated[
        bool,
        typer.Option("--score", help="Score a completed worksheet instead of writing one."),
    ] = False,
) -> None:
    """Write or score the seeded 20-graph human audit (spec §5.4).

    Without --score, samples 20 graphs and writes `rubric.md` and
    `worksheet.json` under `reports/audit/` for a human to fill in. With
    --score, reads the completed worksheet and publishes the mean; exits 3 when
    the mean is below the §5.4 threshold, because a compiler below it cannot be
    fixed by tuning anything downstream.
    """
    from pathlib import Path

    from cascade.decompose.audit import (
        PASS_THRESHOLD,
        sample_scenarios,
        score_worksheet,
        summarise,
        write_worksheet,
    )
    from cascade.decompose.store import load_graph
    from cascade.ledger.store import load_scenarios

    settings = _settings(config)
    directory = Path(settings.paths.reports) / "audit"

    if score:
        worksheet = directory / "worksheet.json"
        if not worksheet.is_file():
            _fail(
                f"no worksheet at {worksheet}; run `cascade compile audit` first", EXIT_PRECONDITION
            )
        scores = score_worksheet(worksheet)
        summary = summarise(scores)

        table = Table(title="Graph audit (spec §5.4)")
        table.add_column("Axis", style="cyan")
        table.add_column("Mean", justify="right")
        for axis, mean in summary.per_axis:
            table.add_row(axis, f"{mean:.3f}")
        table.add_row("[bold]overall[/bold]", f"[bold]{summary.mean:.3f}[/bold]")
        console.print(table)
        console.print(f"scored [bold]{summary.scored}[/bold] graph(s); threshold {PASS_THRESHOLD}")
        if summary.worst:
            console.print(
                "lowest scoring: " + ", ".join(f"{sid} ({m:.2f})" for sid, m in summary.worst)
            )

        if summary.scored == 0:
            _fail("worksheet has no completed scores", EXIT_PRECONDITION)
        if not summary.passed:
            _fail(
                f"audit mean {summary.mean:.3f} is below {PASS_THRESHOLD}; per §5.4 the "
                "compiler prompt is the problem and simulation tuning will not fix it",
                EXIT_PRECONDITION,
            )
        console.print("[bold green]audit: mean meets the §5.4 threshold[/bold green]")
        return

    scenarios = {item.scenario_id: item for item in load_scenarios(settings, role="admin")}
    if not scenarios:
        _fail("scenario registry is empty", EXIT_PRECONDITION)

    chosen = sample_scenarios(sorted(scenarios), salt=settings.study.salt)
    graphs = []
    missing = []
    for scenario_id in chosen:
        graph = load_graph(settings, scenario_id, role="admin")
        if graph is None:
            missing.append(scenario_id)
            continue
        graphs.append((scenario_id, scenarios[scenario_id].question, graph))

    if missing:
        _fail(
            f"{len(missing)} sampled scenario(s) have no compiled graph: "
            + ", ".join(missing[:5])
            + " -- run `cascade compile build` first",
            EXIT_PRECONDITION,
        )

    rubric_path, worksheet_path = write_worksheet(
        directory, graphs=graphs, salt=settings.study.salt
    )
    console.print(f"sampled [bold]{len(graphs)}[/bold] graph(s), seeded from the study salt")
    console.print(f"  rubric:    {rubric_path}")
    console.print(f"  worksheet: {worksheet_path}")
    console.print(
        "[dim]Fill in every score, then run `cascade compile audit --score`. "
        "The graphs are shown without their outcomes on purpose (§5.4).[/dim]"
    )


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
