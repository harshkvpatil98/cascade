"""Assemble, screen, select and seal the scenario registry (spec §3.1).

The thin shell around the pure core: this module does the I/O -- reading
sources, writing the manifest -- and delegates every decision to
``rules.py``, ``select.py``, ``manifest.py`` and ``climatology.py``, none of
which can reach a network or a clock.

A source that cannot be reached is **reported, never absorbed**. If Metaculus
contributes zero scenarios because it now demands a token, the build says so
and the report carries it; the remaining loaders do not quietly make up the
difference, because a set assembled from two market sources is a different
object from the one §3.1 describes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cascade.canonical import canonical_timestamp
from cascade.config import Settings
from cascade.ledger.climatology import Climatology, climatology_of
from cascade.ledger.http import SourceCache, SourceOffline
from cascade.ledger.manifest import compute_manifest
from cascade.ledger.rules import Rejected, screen
from cascade.ledger.schema import RawQuestion, ScenarioRecord
from cascade.ledger.select import SelectionResult, select
from cascade.ledger.sources.curated import load_curated
from cascade.ledger.sources.manifold import load_manifold
from cascade.ledger.sources.metaculus import MetaculusUnavailable, load_metaculus
from cascade.ledger.sources.polymarket import load_polymarket

__all__ = ["BuildResult", "SourceStatus", "build_registry", "scenario_id_for"]


@dataclass(frozen=True)
class SourceStatus:
    """What one loader contributed, or why it contributed nothing."""

    name: str
    available: bool
    n_raw: int
    detail: str = ""


@dataclass(frozen=True)
class BuildResult:
    """Everything a build produced, including what it could not."""

    selection: SelectionResult
    sources: tuple[SourceStatus, ...]
    climatology: Climatology | None
    manifest_sha256: str
    cache_hits: int = 0
    cache_fetches: int = 0
    unavailable: tuple[str, ...] = field(default_factory=tuple)

    @property
    def records(self) -> tuple[ScenarioRecord, ...]:
        return self.selection.records


def scenario_id_for(raw: RawQuestion) -> str:
    """Derive a stable scenario id from the source reference.

    Stable across rebuilds because it is a function of the upstream identity
    alone -- not of position, not of selection order. The manifest hashes this
    id, so a derivation that moved would invalidate the seal for no reason.
    """
    return raw.source_ref


def _load_sources(
    settings: Settings, cache: SourceCache
) -> tuple[list[RawQuestion], list[SourceStatus]]:
    """Run every loader, recording what each produced."""
    salt = settings.study.salt
    ledger = settings.ledger
    raw: list[RawQuestion] = []
    statuses: list[SourceStatus] = []

    token = settings.metaculus_token.get_secret_value() if settings.metaculus_token else None
    try:
        metaculus = load_metaculus(
            cache,
            token=token,
            pages=ledger.metaculus_pages,
            page_size=ledger.metaculus_page_size,
        )
    except (MetaculusUnavailable, SourceOffline) as exc:
        statuses.append(SourceStatus("metaculus", False, 0, str(exc).split(".")[0]))
    else:
        raw.extend(metaculus)
        statuses.append(SourceStatus("metaculus", True, len(metaculus)))

    market_loaders: tuple[tuple[str, Callable[[], list[RawQuestion]]], ...] = (
        (
            "polymarket",
            lambda: load_polymarket(
                cache,
                salt=salt,
                pages=ledger.polymarket_pages,
                page_size=ledger.polymarket_page_size,
            ),
        ),
        (
            "manifold",
            lambda: load_manifold(
                cache, pages=ledger.manifold_pages, page_size=ledger.manifold_page_size
            ),
        ),
    )
    for name, loader in market_loaders:
        try:
            loaded = loader()
        except SourceOffline as exc:
            statuses.append(SourceStatus(name, False, 0, str(exc)[:120]))
            continue
        raw.extend(loaded)
        statuses.append(SourceStatus(name, True, len(loaded)))

    curated_dir: Path = settings.curated_path()
    curated = load_curated(curated_dir)
    raw.extend(curated)
    statuses.append(
        SourceStatus(
            "curated",
            True,
            len(curated),
            "" if curated else f"no entries in {curated_dir}",
        )
    )

    return raw, statuses


def build_registry(settings: Settings, *, refresh: bool = False) -> BuildResult:
    """Load every source, apply the inclusion rules, and select the study set.

    Preserves the frozen-split guarantee's precondition: the result is a pure
    function of the recorded source responses plus the configured constraints,
    so the same cache directory reproduces the same manifest anywhere.
    """
    cache = SourceCache(settings.source_cache_path(), refresh=refresh)
    try:
        raw, statuses = _load_sources(settings, cache)
    finally:
        cache.close()

    ledger = settings.ledger
    eligible: list[ScenarioRecord] = []
    rejections: list[Rejected] = []
    seen_ids: set[str] = set()

    for candidate in sorted(raw, key=lambda item: item.source_ref):
        scenario_id = scenario_id_for(candidate)
        if scenario_id in seen_ids:
            rejections.append(Rejected(scenario_id, "duplicate_event_group", "duplicate id"))
            continue
        seen_ids.add(scenario_id)
        outcome = screen(
            candidate,
            scenario_id=scenario_id,
            cutoff_fraction=ledger.cutoff_fraction,
            min_volume=ledger.min_volume,
        )
        if isinstance(outcome, Rejected):
            rejections.append(outcome)
        else:
            eligible.append(outcome)

    selection = select(
        tuple(eligible),
        rejections=tuple(rejections),
        n_candidates=len(raw),
        target_n=ledger.target_scenarios,
        yes_rate_min=ledger.yes_rate_min,
        yes_rate_max=ledger.yes_rate_max,
        max_domain_share=ledger.max_domain_share,
        salt=settings.study.salt,
    )

    climatology = climatology_of(selection.records) if selection.records else None
    manifest = compute_manifest(selection.records) if selection.records else ""

    return BuildResult(
        selection=selection,
        sources=tuple(statuses),
        climatology=climatology,
        manifest_sha256=manifest,
        cache_hits=cache.hits,
        cache_fetches=cache.fetched,
        unavailable=tuple(sorted(status.name for status in statuses if not status.available)),
    )


def summarise(result: BuildResult) -> dict[str, object]:
    """A JSON-safe summary for the build report and the run manifest."""
    report = result.selection.report
    return {
        "n_candidates": report.n_candidates,
        "n_eligible": report.n_eligible,
        "n_selected": report.n_selected,
        "target": report.target_n,
        "met_target": report.met_target,
        "n_yes": report.n_yes,
        "yes_rate": report.yes_rate,
        "max_domain_share": report.max_domain_share,
        "domains": dict(report.domain_counts),
        "party_rules": dict(report.party_rule_counts),
        "sources": dict(report.source_counts),
        "rejections": dict(report.rejection_counts),
        "dropped_duplicate_groups": report.dropped_duplicate_groups,
        "shortfall_reason": report.shortfall_reason,
        "unavailable_sources": list(result.unavailable),
        "manifest_sha256": result.manifest_sha256,
        "climatology_brier": result.climatology.brier if result.climatology else None,
        "base_rate": result.climatology.base_rate if result.climatology else None,
        "earliest_cutoff": (
            canonical_timestamp(min(r.scenario.cutoff_ts for r in result.records))
            if result.records
            else None
        ),
        "latest_resolve": (
            canonical_timestamp(max(r.scenario.resolve_ts for r in result.records))
            if result.records
            else None
        ),
    }
