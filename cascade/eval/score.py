"""Turning scored forecasts into §10's metrics. Pure: no I/O, no clock.

Takes :class:`ScoredForecast` tuples -- the only labelled type in Assay -- and
returns the metric objects the report prints. The split between this module and
``store.py`` is the same one that makes ``metrics.py`` property-testable: the
database read is a thin shell, and everything that decides what a number *is*
happens here, where a test can hand it a list.

The recalibration split is derived from the study salt, so the "held-out half"
of §10.5 is the same half on every run of the report. A half chosen by a fresh
shuffle would make the post-calibration Brier move between two runs of the same
report, and the number most likely to be quoted out of context is the one that
must not drift.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cascade.eval.metrics import (
    CALIBRATION_BINS,
    auc,
    brier,
    brier_skill_score,
    calibration,
    isotonic_fit,
    isotonic_predict,
    log_loss,
    murphy_decomposition,
    per_domain,
)
from cascade.eval.schema import (
    CalibrationReport,
    DispersionFinding,
    DomainMetrics,
    MetricSet,
    ScoredForecast,
)
from cascade.eval.stats import pearson, permutation_p, spearman

__all__ = [
    "calibration_of",
    "dispersion_finding",
    "domains_of",
    "metrics_for",
    "recalibrated_brier",
    "split_halves",
]


def _columns(scored: Sequence[ScoredForecast]) -> tuple[list[float], list[int]]:
    return [item.p_hat for item in scored], [item.outcome for item in scored]


def split_halves(
    scored: Sequence[ScoredForecast], *, salt: str
) -> tuple[tuple[ScoredForecast, ...], tuple[ScoredForecast, ...]]:
    """Deterministic two-way split of the scenario set, keyed by the study salt.

    Assignment is a keyed hash of the scenario id, so it is stable across
    runs, independent of the order forecasts were loaded in, and -- critically
    -- **independent of the outcome**. Splitting on anything correlated with
    the label would make the held-out half a different population from the
    fitting half, and the recalibrated Brier would measure that instead of
    calibration.

    The halves are not guaranteed to be exactly equal in size; a hash is not a
    partition of fixed cardinality. Both counts travel with the result.
    """
    fit: list[ScoredForecast] = []
    held: list[ScoredForecast] = []
    for item in sorted(scored, key=lambda value: value.scenario_id):
        digest = hashlib.blake2b(
            item.scenario_id.encode("utf-8"), digest_size=8, key=salt.encode("utf-8")
        ).digest()
        (fit if digest[0] & 1 else held).append(item)
    return tuple(fit), tuple(held)


def recalibrated_brier(scored: Sequence[ScoredForecast], *, salt: str) -> float | None:
    """§10.5's isotonic recalibration, fitted on one half, scored on the other.

    "Never report the recalibrated figure as the headline -- the headline is
    the raw system." It is returned as a separate, optional field for exactly
    that reason, and it is ``None`` when either half is too small to fit or
    score, rather than a number computed from four scenarios.
    """
    fit, held = split_halves(scored, salt=salt)
    if len(fit) < 2 or len(held) < 2:
        return None
    fit_p, fit_y = _columns(fit)
    held_p, held_y = _columns(held)
    thresholds, values = isotonic_fit(fit_p, fit_y)
    mapped = isotonic_predict(thresholds, values, held_p)
    return brier(mapped, held_y)


def calibration_of(
    scored: Sequence[ScoredForecast], *, bins: int = CALIBRATION_BINS
) -> CalibrationReport:
    """§10.5's reliability table for one configuration."""
    forecasts, outcomes = _columns(scored)
    return calibration(forecasts, outcomes, bins=bins)


def domains_of(scored: Sequence[ScoredForecast]) -> tuple[DomainMetrics, ...]:
    """§10.4's per-domain breakdown for one configuration."""
    forecasts, outcomes = _columns(scored)
    return per_domain(forecasts, outcomes, [item.domain for item in scored])


def metrics_for(
    scored: Sequence[ScoredForecast],
    *,
    config_id: str,
    climatology_brier: float | None = None,
    direct_brier: float | None = None,
    salt: str | None = None,
    bins: int = CALIBRATION_BINS,
) -> MetricSet:
    """Every §10.1 metric for one configuration, measured on ``scored``.

    ``climatology_brier`` comes from the sealed manifest rather than being
    recomputed here: §10.2 calls climatology the floor, and a floor recomputed
    against whatever set happens to be loaded moves in the direction that
    flatters the result.

    The skill scores are ``None`` when their reference is absent. A missing
    baseline is a missing number; substituting a plausible one is precisely
    what §1 forbids.
    """
    if not scored:
        raise ValueError(
            f"no scored forecasts for {config_id!r}; a metric set over an empty "
            "set would be a row of numbers with nothing behind it"
        )
    forecasts, outcomes = _columns(scored)
    score = brier(forecasts, outcomes)
    report = calibration(forecasts, outcomes, bins=bins)
    return MetricSet(
        config_id=config_id,
        n=len(scored),
        base_rate=sum(outcomes) / len(outcomes),
        mean_p_hat=sum(forecasts) / len(forecasts),
        brier=score,
        log_loss=log_loss(forecasts, outcomes),
        auc=auc(forecasts, outcomes),
        ece=report.ece,
        mce=report.mce,
        murphy=murphy_decomposition(forecasts, outcomes, bins=bins),
        policies=tuple(sorted({item.policy for item in scored})),
        bss_vs_climatology=(
            brier_skill_score(score, climatology_brier)
            if climatology_brier not in (None, 0.0)
            else None
        ),
        bss_vs_direct=(
            brier_skill_score(score, direct_brier) if direct_brier not in (None, 0.0) else None
        ),
        brier_recalibrated=(recalibrated_brier(scored, salt=salt) if salt else None),
    )


def dispersion_finding(
    scored: Sequence[ScoredForecast],
    *,
    sigma_threshold: float,
    seed: int,
    permutations: int = 10_000,
) -> DispersionFinding:
    """§9.2's claim that sigma is informative, tested rather than assumed.

    Two things are reported and they answer different questions. The
    correlation between sigma and absolute error asks whether dispersion
    *ranks* the forecasts by how wrong they are. The Brier of the flagged and
    unflagged subsets asks whether the flag the system actually raises picks
    out the forecasts it should distrust -- which is the operational claim, and
    the one that can be true while the correlation is weak.

    The expected and defensible finding is that flagged scenarios score
    materially worse. If they do not, that is the result, and it is reported as
    measured.
    """
    sigmas = [item.sigma for item in scored]
    errors = [item.abs_error for item in scored]
    flagged = [item for item in scored if item.modality == "multi"]
    unflagged = [item for item in scored if item.modality != "multi"]

    def subset_brier(subset: Sequence[ScoredForecast]) -> float | None:
        if not subset:
            return None
        forecasts, outcomes = _columns(subset)
        return brier(forecasts, outcomes)

    rho = spearman(sigmas, errors)
    r = pearson(sigmas, errors)
    return DispersionFinding(
        n=len(scored),
        pearson_r=r,
        pearson_p=(
            permutation_p(sigmas, errors, seed=seed, permutations=permutations, rank_based=False)
            if r is not None
            else None
        ),
        spearman_rho=rho,
        spearman_p=(
            permutation_p(sigmas, errors, seed=seed, permutations=permutations, rank_based=True)
            if rho is not None
            else None
        ),
        flagged=len(flagged),
        brier_flagged=subset_brier(flagged),
        brier_unflagged=subset_brier(unflagged),
        sigma_threshold=sigma_threshold,
    )
