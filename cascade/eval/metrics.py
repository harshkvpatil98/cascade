"""Assay's scoring rules (spec §10.1, §10.5). Pure: no I/O, no clock, no RNG.

Every metric here is written from its definition. The stack pins numpy and
nothing else numerical (spec §2.3); scipy and scikit-learn are present in this
environment only as transitive dependencies of the embedding model, so calling
into them would make the headline metric depend on a package the project never
declared. The same reasoning produced ADR-0021 for the dip test, and it is
stronger here: these are the numbers the study *is*.

Purity is what makes this testable against closed forms. Every function takes
its data and its parameters and returns a value -- there is no settings object,
no database handle and no global bin count, so a property test can quantify
over inputs rather than over fixtures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cascade.eval.schema import (
    CalibrationBin,
    CalibrationReport,
    DomainMetrics,
    MurphyTerms,
)

__all__ = [
    "CALIBRATION_BINS",
    "LOG_LOSS_CLIP",
    "auc",
    "brier",
    "brier_skill_score",
    "calibration",
    "isotonic_fit",
    "isotonic_predict",
    "log_loss",
    "murphy_decomposition",
    "per_domain",
    "wilson_interval",
]

# §10.1 and §10.5 both say ten equal-width bins. It is a reporting choice, not
# a tuning knob: changing it changes ECE, so it is a module constant with the
# section that fixes it named, and any caller that wants a different count
# passes it explicitly and owns the deviation.
CALIBRATION_BINS = 10

# §10.1: "Clip p to [0.01, 0.99] and say so -- an unclipped log loss is
# dominated by one confident miss." Saying so is what this constant is for.
LOG_LOSS_CLIP = 0.01


def _as_arrays(
    forecasts: Sequence[float], outcomes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the pairing once, so no metric below has to re-check it."""
    if len(forecasts) != len(outcomes):
        raise ValueError(f"forecast/outcome length mismatch: {len(forecasts)} vs {len(outcomes)}")
    if not outcomes:
        raise ValueError("a scoring rule over an empty set is undefined")
    p = np.asarray(forecasts, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("forecasts must lie in [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def _mean_sorted(values: np.ndarray) -> float:
    """Mean of ``values``, accumulated in sorted order.

    Float addition is not associative, so summing the same set in two orders
    can differ in the last ULP. That is invisible at six decimal places and
    very visible in a committed artifact: a re-run whose forecasts loaded in a
    different order would produce a diff in `metrics.json` with no change
    behind it, and a paired bootstrap subtracting two such sums would carry the
    difference into an effect size.

    Sorting first makes the accumulation a function of the *set*, which is what
    permutation invariance means. The arbiter accumulates in sorted actor order
    for the same reason (§8.1).
    """
    if values.size == 0:
        raise ValueError("mean of an empty set is undefined")
    return float(np.sort(values).sum() / values.size)


def brier(forecasts: Sequence[float], outcomes: Sequence[int]) -> float:
    """``mean((p - y)^2)`` -- the headline (spec §10.1).

    Deliberately identical to :func:`cascade.ledger.climatology.brier_score`,
    which computes it in pure Python at M1 for the sealed set. Two independent
    implementations of the study's headline metric that must agree is a
    property test, not duplication, and one of them is asserted against the
    other.

    Exactly permutation-invariant, not approximately -- see :func:`_mean_sorted`.
    """
    p, y = _as_arrays(forecasts, outcomes)
    return _mean_sorted((p - y) ** 2)


def brier_skill_score(model: float, reference: float) -> float:
    """``1 - BS_model / BS_reference`` (spec §10.1).

    Makes a claim like "-30.5%" explicit rather than implied by two raw
    numbers. Raises on a zero reference: a skill score against a perfect
    reference is not a large number, it is undefined, and returning infinity
    would put it in a report.
    """
    if reference == 0.0:
        raise ValueError("Brier skill score against a reference of 0 is undefined")
    return 1.0 - model / reference


def log_loss(
    forecasts: Sequence[float], outcomes: Sequence[int], *, clip: float = LOG_LOSS_CLIP
) -> float:
    """``-mean(y ln p + (1-y) ln(1-p))`` with p clipped to ``[clip, 1-clip]``.

    Secondary by design (§10.1). The clip is an argument with a stated default
    rather than a buried constant, because the value materially changes the
    number and a reader has to be able to see which one produced it.
    """
    p, y = _as_arrays(forecasts, outcomes)
    if not 0.0 < clip < 0.5:
        raise ValueError(f"clip must lie in (0, 0.5), got {clip}")
    q = np.clip(p, clip, 1.0 - clip)
    return -_mean_sorted(y * np.log(q) + (1.0 - y) * np.log1p(-q))


def auc(forecasts: Sequence[float], outcomes: Sequence[int]) -> float | None:
    """Rank discrimination, by the Mann-Whitney identity (spec §10.1).

    Ties are handled with mid-ranks, which is what makes AUC = 0.5 for a
    constant forecast rather than 0 or 1 depending on sort order -- and a
    constant forecast is exactly what the climatology baseline is.

    Returns ``None`` when one class is absent: AUC is undefined there, and a
    0.5 in its place would read as "no discrimination" rather than "not
    measurable".
    """
    p, y = _as_arrays(forecasts, outcomes)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(p, kind="stable")
    ranks = np.empty(p.size, dtype=float)
    sorted_p = p[order]
    index = 0
    while index < sorted_p.size:
        stop = index
        while stop + 1 < sorted_p.size and sorted_p[stop + 1] == sorted_p[index]:
            stop += 1
        # Mid-rank over the tie group, 1-based.
        ranks[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    rank_sum = float(ranks[y == 1.0].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (spec §10.5).

    Wilson rather than the normal approximation because §10.5's bins routinely
    hold single-digit counts and observed frequencies of 0 or 1, where the
    normal interval is degenerate or leaves [0, 1] entirely. The default ``z``
    is the exact two-sided 95% normal quantile, spelled out rather than
    rounded to 1.96 so the interval does not silently differ from a reader's.
    """
    if trials <= 0:
        raise ValueError("a Wilson interval over zero trials is undefined")
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes out of {trials} trials is impossible")
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    half = (z / denominator) * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = CALIBRATION_BINS,
) -> CalibrationReport:
    """The reliability table, ECE and MCE (spec §10.1, §10.5).

    Bins are equal-width over [0, 1] and **every** bin appears in the output,
    including empty ones. §10.5's whole point is that a reliability curve
    without bin counts hides how thin the interesting bins are; dropping the
    empty ones would hide it differently.

    ECE is the count-weighted mean absolute gap and MCE the maximum over
    non-empty bins, both over the same bins the table prints -- a reader can
    recompute them from the table, which is the property that makes the table
    worth printing.
    """
    p, y = _as_arrays(forecasts, outcomes)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=False` everywhere except the last bin, which is closed so p = 1.0
    # lands in it rather than in a bin that does not exist.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    rows: list[CalibrationBin] = []
    total = float(p.size)
    ece = 0.0
    mce = 0.0
    for k in range(bins):
        mask = index == k
        count = int(mask.sum())
        if count == 0:
            rows.append(
                CalibrationBin(
                    index=k,
                    lo=float(edges[k]),
                    hi=float(edges[k + 1]),
                    count=0,
                    mean_pred=None,
                    obs_freq=None,
                    wilson_lo=None,
                    wilson_hi=None,
                )
            )
            continue
        mean_pred = float(p[mask].mean())
        successes = int(y[mask].sum())
        obs_freq = successes / count
        lo, hi = wilson_interval(successes, count)
        gap = abs(mean_pred - obs_freq)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append(
            CalibrationBin(
                index=k,
                lo=float(edges[k]),
                hi=float(edges[k + 1]),
                count=count,
                mean_pred=mean_pred,
                obs_freq=obs_freq,
                wilson_lo=lo,
                wilson_hi=hi,
            )
        )
    return CalibrationReport(bins=tuple(rows), ece=ece, mce=mce, n=int(p.size))


def murphy_decomposition(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = CALIBRATION_BINS,
) -> MurphyTerms:
    """``BS = REL - RES + UNC``, with the binning residual reported (spec §10.1).

    * ``REL`` -- count-weighted mean squared gap between a bin's mean forecast
      and its observed frequency. Low means well calibrated.
    * ``RES`` -- count-weighted mean squared distance of a bin's observed
      frequency from the base rate. High means discriminative.
    * ``UNC`` -- ``o(1-o)``: the irreducible variance of the outcome, and
      exactly the climatology Brier for this set.

    The identity is exact only when forecasts are grouped by equal value. Over
    equal-width bins the cross term does not vanish, so ``residual = BS - (REL
    - RES + UNC)`` is computed and carried. Reporting it is the difference
    between a decomposition a reader can check and three numbers that happen
    to be printed near a fourth.
    """
    p, y = _as_arrays(forecasts, outcomes)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    base = float(y.mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    total = float(p.size)
    reliability = 0.0
    resolution = 0.0
    for k in range(bins):
        mask = index == k
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(p[mask].mean())
        obs_freq = float(y[mask].mean())
        reliability += count / total * (mean_pred - obs_freq) ** 2
        resolution += count / total * (obs_freq - base) ** 2
    uncertainty = base * (1.0 - base)
    # The same accumulation `brier` uses: the two appear side by side in the
    # report and a reader comparing them must not find them differing in the
    # last digit.
    score = _mean_sorted((p - y) ** 2)
    return MurphyTerms(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        brier=score,
        residual=score - (reliability - resolution + uncertainty),
        bins=bins,
    )


def per_domain(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    domains: Sequence[str],
) -> tuple[DomainMetrics, ...]:
    """Brier by domain tag, with counts (spec §10.4).

    "A headline win driven entirely by one over-represented domain is not a
    win, and the breakdown is what reveals it." Sorted by domain so two runs
    of the report produce the same table (invariant 7).
    """
    p, y = _as_arrays(forecasts, outcomes)
    if len(domains) != len(forecasts):
        raise ValueError(f"domain/forecast length mismatch: {len(domains)} vs {len(forecasts)}")
    labels = np.asarray(domains, dtype=object)
    out: list[DomainMetrics] = []
    for domain in sorted(set(domains)):
        mask = labels == domain
        count = int(mask.sum())
        out.append(
            DomainMetrics(
                domain=domain,
                n=count,
                base_rate=float(y[mask].mean()),
                brier=_mean_sorted((p[mask] - y[mask]) ** 2),
            )
        )
    return tuple(out)


def isotonic_fit(
    forecasts: Sequence[float], outcomes: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pool-adjacent-violators isotonic regression (spec §10.5).

    Returns ``(thresholds, values)``: a non-decreasing step function mapping a
    raw forecast to a recalibrated one. Written from the algorithm rather than
    taken from scikit-learn, which this project does not declare -- and the
    algorithm is fifteen lines, so the dependency would cost more than it saves.

    **Equal forecasts are pooled into one point before the sweep.** Without
    that, two scenarios forecast at the same probability can land in different
    blocks with different fitted values, and the "step function" is then
    two-valued at that x -- so which value a later prediction gets depends on a
    ``searchsorted`` tie-break rather than on the data. Measured on
    ``p = [0.0, 0.5, 0.5]``, ``y = [0, 0, 1]``: the unpooled fit mapped all
    three to 0.0 and *doubled* the Brier on its own fitting set, which is the
    one thing isotonic regression cannot do. Found by a property test.

    Pooling first also makes ``thresholds`` strictly increasing, which is what
    makes :func:`isotonic_predict` well defined at a knot.
    """
    p, y = _as_arrays(forecasts, outcomes)
    order = np.argsort(p, kind="stable")
    xs = p[order]
    ys = y[order]

    # (x at the right edge of the block, weighted sum, weight). Distinct x
    # values only: equal forecasts are one observation with weight n.
    blocks: list[list[float]] = []
    index = 0
    while index < xs.size:
        stop = index
        while stop + 1 < xs.size and xs[stop + 1] == xs[index]:
            stop += 1
        weight = float(stop - index + 1)
        blocks.append([float(xs[index]), float(ys[index : stop + 1].sum()), weight])
        index = stop + 1
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks[-1]
            left[0] = right[0]
            left[1] += right[1]
            left[2] += right[2]
    thresholds = tuple(block[0] for block in blocks)
    values = tuple(block[1] / block[2] for block in blocks)
    return thresholds, values


def isotonic_predict(
    thresholds: Sequence[float], values: Sequence[float], forecasts: Sequence[float]
) -> tuple[float, ...]:
    """Apply a fitted isotonic map to new forecasts.

    A forecast above every threshold takes the top block's value, and one
    below every threshold takes the bottom block's -- the fit is a step
    function on the training support and extrapolating it linearly would
    invent calibration data outside the range anything was observed at.
    """
    if len(thresholds) != len(values):
        raise ValueError(f"threshold/value length mismatch: {len(thresholds)} vs {len(values)}")
    if not thresholds:
        raise ValueError("cannot apply an isotonic fit with no blocks")
    edges = np.asarray(thresholds, dtype=float)
    table = np.asarray(values, dtype=float)
    index = np.searchsorted(edges, np.asarray(forecasts, dtype=float), side="left")
    index = np.clip(index, 0, edges.size - 1)
    return tuple(float(value) for value in table[index])
