"""Hartigan's dip test and the bimodality coefficient (spec §9.2). Pure.

§9.2 asks for **two independent statistics** corroborating the sigma > 0.30 flag,
and the reason is stated there: the flag is the system's own claim about which
of its forecasts to distrust, so it has to be more than one threshold on one
moment of one distribution.

The dip statistic is the sup-norm distance from the empirical CDF to the
nearest unimodal CDF, computed by Hartigan's greatest-convex-minorant /
least-concave-majorant iteration. Zero for any sample whose ECDF is already
unimodal; 0.25 for a perfect two-point split at the bounds, which is the
maximum attainable.

The p-value is Monte Carlo against the uniform null -- the standard choice,
because the uniform is the least favourable unimodal distribution, so a dip
that is surprising under it is surprising under every unimodal alternative.
The null depends only on the sample size, so it is computed once per *n* and
memoised: 36,000 runs collapse to 2,160 forecasts, and recomputing a 2,000-draw
null for each would be 4.3M dip evaluations to answer 2,160 questions.

Determinism: the null is drawn from a PCG64 seeded from the sample size and the
study salt, never from global state. Two processes computing the same p-value
get the same number, which the M8 replay criterion requires of everything that
reaches a report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "BIMODALITY_THRESHOLD",
    "DIP_NULL_SAMPLES",
    "DipResult",
    "bimodality_coefficient",
    "dip_p_value",
    "dip_statistic",
    "dip_test",
]

# §9.2: "BC > 0.555 indicates bimodality under the uniform null." The value is
# the BC of the uniform distribution itself, which is why it is the threshold.
BIMODALITY_THRESHOLD = 0.555

# Monte Carlo draws for the null. 2,000 resolves a p-value to ~0.01, which is
# the resolution the 0.05 decision needs; the cost is paid once per sample size.
DIP_NULL_SAMPLES = 2000


class DipResult(BaseModel):
    """The dip statistic and its Monte Carlo p-value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dip: float
    p: float
    n: int


def dip_statistic(values: Sequence[float]) -> float:
    """Hartigan's dip: the sup distance from the ECDF to the nearest unimodal CDF.

    Computed from the definition rather than by a transcribed iteration. A
    unimodal CDF is convex up to some modal point and concave after it, so for
    each candidate mode the distance to the best such fit is

        max( departure from convexity on the left,
             departure from concavity on the right ) / 2

    and the dip is the minimum of that over modal positions. The halving is
    where the classical factor comes from: the error can be split evenly above
    and below the ECDF's own jump, so the attainable distance is half the gap.

    The search is a binary one, which the shape of the problem permits: adding
    points to the left prefix can only lower its convex minorant, so the left
    departure is non-decreasing in the mode, and the right departure is
    non-increasing by the same argument. The minimum of a non-decreasing and a
    non-increasing sequence sits at their crossover.

    Returns 0.0 for fewer than four points or a degenerate sample -- with that
    little data every ECDF is within rounding of unimodal, and a non-zero dip
    would manufacture a modality claim out of nothing.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = int(x.size)
    if n < 4 or x[0] == x[-1]:
        return 0.0

    # The ECDF brackets each observation: it is i/n just below x[i] and
    # (i+1)/n just above. The minorant is fitted to the lower edge and measured
    # against the upper, so the envelope is charged for the jump exactly once.
    lower = np.arange(n, dtype=float) / n
    upper = np.arange(1, n + 1, dtype=float) / n

    def left(mode: int) -> float:
        return _departure(x[: mode + 1], lower[: mode + 1], upper[: mode + 1], upper_hull=False)

    def right(mode: int) -> float:
        return _departure(x[mode:], lower[mode:], upper[mode:], upper_hull=True)

    low, high = 0, n - 1
    while low < high:
        mid = (low + high) // 2
        if left(mid) >= right(mid):
            high = mid
        else:
            low = mid + 1

    # The crossover and its neighbours: the sequences are monotone, so the
    # minimum is one of these three, and checking all three costs nothing.
    best = min(
        max(left(mode), right(mode)) for mode in sorted({max(0, low - 1), low, min(n - 1, low + 1)})
    )
    return max(0.0, best / 2.0)


def _departure(xs: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, upper_hull: bool) -> float:
    """How far this span of the ECDF departs from convexity (or concavity).

    ``upper_hull=False`` fits the greatest convex minorant to the ECDF's lower
    edge and returns the largest amount by which the upper edge rises above it;
    ``True`` fits the least concave majorant to the upper edge and returns the
    largest amount by which it sits above the lower edge. Zero exactly when the
    span is already convex (or concave), which is what makes the dip zero for a
    sample that needs no unimodal correction.
    """
    if xs.size < 2:
        return 0.0
    if upper_hull:
        knots = _hull(xs, upper, upper=True)
        envelope = np.interp(xs, xs[knots], upper[knots])
        return float(np.max(envelope - lower))
    knots = _hull(xs, lower, upper=False)
    envelope = np.interp(xs, xs[knots], lower[knots])
    return float(np.max(upper - envelope))


def bimodality_coefficient(values: Sequence[float]) -> float:
    """``BC = (skew^2 + 1) / kurtosis`` on the sample moments (spec §9.2).

    Kurtosis here is the *non-excess* fourth standardised moment, which is why
    a normal sample scores 1/3 rather than 0. Returns 0.0 for a degenerate
    sample: with no spread there is no shape to describe, and dividing by a
    zero variance would report bimodality for a constant.
    """
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    if n < 4:
        return 0.0
    # Identical values first, and by range rather than by variance: the mean of
    # fifty copies of 0.4 is not exactly 0.4 in binary, so the variance of a
    # constant sample lands at ~1e-33 rather than 0 and the moments below come
    # out as noise divided by noise -- measured, it reported BC = 2.0 for a
    # constant, which would flag a degenerate ensemble as strongly bimodal.
    if float(np.max(x) - np.min(x)) == 0.0:
        return 0.0
    centred = x - x.mean()
    variance = float(np.mean(centred**2))
    if variance <= 0.0:
        return 0.0
    skew = float(np.mean(centred**3)) / variance**1.5
    kurt = float(np.mean(centred**4)) / variance**2
    if kurt <= 0.0:
        return 0.0
    return float((skew**2 + 1.0) / kurt)


def dip_p_value(dip: float, n: int, *, salt: str, samples: int = DIP_NULL_SAMPLES) -> float:
    """Probability of a dip at least this large under the uniform null.

    Reported as ``(1 + #{null >= dip}) / (1 + samples)`` -- the standard
    add-one Monte Carlo estimate. It never returns exactly 0, because a p-value
    of 0 from 2,000 draws claims more certainty than 2,000 draws contain.
    """
    if n < 4:
        return 1.0
    null = _null_distribution(n, salt, samples)
    exceed = int(np.count_nonzero(null >= dip))
    return (1.0 + exceed) / (1.0 + float(samples))


def dip_test(values: Sequence[float], *, salt: str, samples: int = DIP_NULL_SAMPLES) -> DipResult:
    """The dip statistic with its p-value, as §9.1's ``dip_test(scores).p``."""
    dip = dip_statistic(values)
    n = len(values)
    return DipResult(dip=dip, p=dip_p_value(dip, n, salt=salt, samples=samples), n=n)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _hull(xs: np.ndarray, ys: np.ndarray, *, upper: bool) -> np.ndarray:
    """Indices of the lower (or upper) convex hull of the points, in order.

    A monotone chain over points already sorted by x. One function with a flag
    rather than two that could drift apart: the minorant and the majorant are
    the same construction with the turn direction reversed.

    Ties in x are collapsed to the extreme point -- the lowest y for a minorant,
    the highest for a majorant. Sample values repeat often here (an outcome
    score rounds to the same number in many replicates), and a hull carrying
    two points at one x is not a function of x at all.
    """
    sign = -1.0 if upper else 1.0
    stack: list[int] = []
    for index in range(xs.size):
        if stack and xs[index] == xs[stack[-1]]:
            better = ys[index] > ys[stack[-1]] if upper else ys[index] < ys[stack[-1]]
            if not better:
                continue
            stack.pop()
        while len(stack) >= 2 and sign * _cross(xs, ys, stack[-2], stack[-1], index) <= 0.0:
            stack.pop()
        stack.append(index)
    return np.asarray(stack, dtype=int)


def _cross(xs: np.ndarray, ys: np.ndarray, a: int, b: int, c: int) -> float:
    """Signed area of the turn a->b->c. Positive is a left turn."""
    return float((xs[b] - xs[a]) * (ys[c] - ys[a]) - (ys[b] - ys[a]) * (xs[c] - xs[a]))


@lru_cache(maxsize=64)
def _null_distribution(n: int, salt: str, samples: int) -> np.ndarray:
    """Dip statistics of ``samples`` uniform samples of size ``n``.

    Memoised on (n, salt, samples): the null is a property of the sample size,
    not of the data, so 2,160 forecasts share a handful of these.
    """
    seed = int.from_bytes(
        hashlib.blake2b(f"dip|{n}|{samples}".encode(), digest_size=8, key=salt.encode()).digest(),
        "big",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.random((samples, n))
    return np.asarray([dip_statistic(row) for row in draws], dtype=float)
