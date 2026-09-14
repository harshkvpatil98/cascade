"""Assay's inference (spec §10.4). Pure: no I/O, no clock, no global RNG.

180 scenarios is a small sample and the differences under test are small --
the headline ablation effect is 0.035 Brier. Three things follow, and §10.4
requires all three:

**Pairing.** Every configuration scores the same scenarios, so the variance of
a *difference* is far smaller than the variance of either Brier. Resampling
scenarios and recomputing the difference inside each resample keeps that
pairing; resampling the two sets independently would throw it away and widen
every interval for no reason.

**Multiple-comparison control.** Twelve cells and five baselines is a family of
tests. Holm-Bonferroni is applied across it and the adjusted p-values are what
the report quotes.

**A seeded bootstrap.** The resampling RNG is derived from the study salt and
the pair being compared, so a confidence interval is reproducible from the
salt alone. An unseeded bootstrap puts a slightly different interval in the
report on every run, which is the same class of defect as an unseeded
simulation and considerably harder to notice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from cascade.eval.schema import BootstrapInterval, Comparison

__all__ = [
    "adjust_family",
    "bootstrap_seed",
    "holm_bonferroni",
    "paired_bootstrap",
    "pearson",
    "permutation_p",
    "spearman",
]


def bootstrap_seed(salt: str, *args: str) -> int:
    """Derive a bootstrap seed from the study salt and what is being compared.

    Same construction as §8.2's run seed -- keyed blake2b over a joined
    identifier -- so the report's intervals are as reproducible as its runs,
    and for the same stated reason.
    """
    digest = hashlib.blake2b(
        "|".join(args).encode("utf-8"), digest_size=8, key=salt.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def _paired_arrays(
    a: Sequence[float], b: Sequence[float], outcomes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(a) == len(b) == len(outcomes)):
        raise ValueError(
            f"paired lengths disagree: {len(a)} vs {len(b)} vs {len(outcomes)} -- "
            "a paired test over unpaired scenarios is not a paired test"
        )
    if not outcomes:
        raise ValueError("a paired bootstrap over an empty set is undefined")
    return (
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(outcomes, dtype=float),
    )


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    outcomes: Sequence[int],
    *,
    seed: int,
    b_resamples: int = 10_000,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Paired percentile bootstrap on ``Brier(a) - Brier(b)`` (spec §10.4).

    Resamples **scenarios**, not runs: the unit the study treats as independent
    is the resolved question. Resampling runs would treat 200 replicates of one
    scenario as 200 observations and shrink every interval by a factor of
    roughly sqrt(200) on a quantity that has one degree of freedom.

    The p-value is the two-sided bootstrap proportion -- twice the smaller tail
    mass on either side of zero, clipped at 1 -- with a floor of ``1/b`` rather
    than 0, because a bootstrap cannot resolve a p-value finer than its own
    resolution and printing ``p = 0.0`` claims that it can.
    """
    pa, pb, y = _paired_arrays(a, b, outcomes)
    if b_resamples <= 0:
        raise ValueError(f"b_resamples must be positive, got {b_resamples}")
    n = y.size
    point = float(np.mean((pa - y) ** 2) - np.mean((pb - y) ** 2))
    if n == 1:
        return BootstrapInterval(point=point, lo=point, hi=point, b=b_resamples, p_value=1.0)

    rng = np.random.Generator(np.random.PCG64(seed))
    # Per-scenario squared-error difference. The Brier difference is its mean,
    # so one draw of scenario indices resamples both configurations together --
    # which is the pairing, expressed as arithmetic rather than as a promise.
    per_scenario = (pa - y) ** 2 - (pb - y) ** 2
    draws = rng.integers(0, n, size=(b_resamples, n))
    differences = per_scenario[draws].mean(axis=1)

    lo = float(np.quantile(differences, alpha / 2.0))
    hi = float(np.quantile(differences, 1.0 - alpha / 2.0))
    below = float(np.mean(differences <= 0.0))
    above = float(np.mean(differences >= 0.0))
    p_value = min(1.0, 2.0 * min(below, above))
    return BootstrapInterval(
        point=point,
        lo=lo,
        hi=hi,
        b=b_resamples,
        p_value=max(p_value, 1.0 / b_resamples),
    )


def holm_bonferroni(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-Bonferroni step-down adjustment (spec §10.4).

    Returns adjusted p-values in the caller's order. The running maximum is
    what makes the adjustment monotone: without it a test can come out with a
    smaller adjusted p-value than a test with a smaller raw p-value, which is
    not a step-down procedure and is not what "adjusted" means.

    Uniformly more powerful than Bonferroni at the same family-wise error
    rate, and it makes no independence assumption -- which matters here,
    because the twelve cells share scenarios and are strongly dependent.
    """
    m = len(p_values)
    if m == 0:
        return ()
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


def adjust_family(comparisons: Sequence[Comparison]) -> tuple[Comparison, ...]:
    """Attach Holm-adjusted p-values to a family of comparisons.

    The family is whatever is passed in, which is deliberate: what counts as
    one family is a reporting decision, and burying it in this function would
    hide the choice that determines every adjusted p-value in the report.
    """
    adjusted = holm_bonferroni([item.interval.p_value for item in comparisons])
    return tuple(
        item.model_copy(update={"p_adjusted": value})
        for item, value in zip(comparisons, adjusted, strict=True)
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when either side is constant.

    ``None`` rather than 0.0 or NaN: a constant sigma column means the
    correlation is undefined, not absent, and reporting 0.0 would read as
    "tested and found nothing".
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    sx = float(x.std())
    sy = float(y.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Mid-ranks, so tied sigmas do not depend on sort order."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    index = 0
    while index < sorted_values.size:
        stop = index
        while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation via Pearson on mid-ranks.

    Reported alongside Pearson because §9.2's claim is about *order* -- that
    high-sigma forecasts are the worse ones -- and absolute error is bounded
    and skewed, so a rank statistic is the honest primary and Pearson the
    corroboration.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    return pearson(
        _ranks(np.asarray(xs, dtype=float)).tolist(),
        _ranks(np.asarray(ys, dtype=float)).tolist(),
    )


def permutation_p(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    seed: int,
    permutations: int = 10_000,
    rank_based: bool = True,
) -> float | None:
    """Two-sided permutation p-value for a correlation.

    A permutation test rather than the t-approximation, for the same reason
    the bootstrap is percentile rather than normal: absolute forecast error is
    bounded, skewed and heaped at zero, and the null distribution of a
    correlation on 180 such points is not the one the closed form assumes.

    Returns ``None`` when the statistic itself is undefined. Floored at
    ``1/permutations``, which is the finest a permutation test can resolve.
    """
    statistic = spearman(xs, ys) if rank_based else pearson(xs, ys)
    if statistic is None:
        return None
    if permutations <= 0:
        raise ValueError(f"permutations must be positive, got {permutations}")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if rank_based:
        x = _ranks(x)
        y = _ranks(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt((x**2).sum() * (y**2).sum()))
    if denominator == 0.0:
        return None

    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        if abs(float((x * shuffled).sum()) / denominator) >= abs(statistic) - 1e-12:
            extreme += 1
    return max((extreme + 1) / (permutations + 1), 1.0 / permutations)
