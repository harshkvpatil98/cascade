"""Chorus: collapsing replicates into a forecast (spec §9.1-§9.3). Pure.

Everything here is a function of a list of terminal outcome scores. No I/O, no
clock, no labels -- the collapse happens under the simulation role precisely
because a forecast must be computable without seeing an outcome (invariant 2).

The bootstrap is seeded from the (scenario, config) it describes, so a
confidence interval is reproducible from the study salt alone. An unseeded
bootstrap would put a different CI in the report on every run, which is the
same class of defect as an unseeded simulation and harder to notice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from cascade.ensemble.dip import DIP_NULL_SAMPLES, bimodality_coefficient, dip_test
from cascade.ensemble.schema import ConvergencePoint, Forecast

__all__ = [
    "CONVERGENCE_LADDER",
    "bootstrap_percentile",
    "collapse",
    "convergence_curve",
    "standard_error",
]

# §9.3: "the convergence curve ... as n goes from 25 to 400". Doubling is the
# ladder that makes a plateau visible on a log axis; the study's own 200 sits
# inside it rather than at an endpoint, so the plot can show what is gained by
# going past it and what would have been lost by stopping earlier.
CONVERGENCE_LADDER: tuple[int, ...] = (25, 50, 100, 200, 400)


def _seed(scenario_id: str, config_id: str, salt: str, tag: str) -> int:
    digest = hashlib.blake2b(
        f"{tag}|{scenario_id}|{config_id}".encode(), digest_size=8, key=salt.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def bootstrap_percentile(
    scores: Sequence[float],
    *,
    seed: int,
    b: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the ensemble mean (spec §9.1).

    Resamples the scores with replacement ``b`` times and reads the alpha/2 and
    1-alpha/2 quantiles of the resampled means. Percentile rather than normal
    approximation because the score distribution is bounded on [0, 1] and
    frequently bimodal -- a symmetric interval around the mean would routinely
    leave the unit interval, and a CI that reports p in [-0.04, 0.31] is not a
    CI anyone can use.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty ensemble")
    if values.size == 1:
        single = float(values[0])
        return single, single
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, values.size, size=(b, values.size))
    means = values[draws].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def standard_error(scores: Sequence[float]) -> float:
    """``sigma / sqrt(n)`` -- the quantity §9.3's replicate count is chosen against."""
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def collapse(
    scores: Sequence[float],
    *,
    scenario_id: str,
    config_id: str,
    salt: str,
    sigma_threshold: float,
    bootstrap_b: int = 10_000,
    dip_samples: int = DIP_NULL_SAMPLES,
    mean_events: float = 0.0,
    mean_steps: float = 0.0,
    absorbed_runs: int = 0,
) -> Forecast:
    """Collapse one (scenario, config)'s replicates into a forecast (spec §9.1).

    ``modality`` is "multi" when sigma exceeds the threshold **or** the dip
    test rejects unimodality, exactly as §9.1 writes it. The bimodality
    coefficient is computed and stored alongside without entering the decision:
    §9.2 asks for it as corroboration, and a third statistic silently voting
    would make the flag a different thing from the one the spec defines.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError(
            f"no replicates to collapse for {scenario_id!r}/{config_id!r}; a forecast "
            "over an empty ensemble would be a number with nothing behind it"
        )
    as_list = [float(value) for value in values]
    p_hat = float(values.mean())
    sigma = float(values.std(ddof=1)) if values.size > 1 else 0.0
    lo, hi = bootstrap_percentile(
        as_list,
        seed=_seed(scenario_id, config_id, salt, "bootstrap"),
        b=bootstrap_b,
    )
    dip = dip_test(as_list, salt=salt, samples=dip_samples)
    bc = bimodality_coefficient(as_list)
    multi = sigma > sigma_threshold or dip.p < 0.05
    return Forecast(
        scenario_id=scenario_id,
        config_id=config_id,
        p_hat=p_hat,
        sigma=sigma,
        ci_lo=min(max(lo, 0.0), 1.0),
        ci_hi=min(max(hi, 0.0), 1.0),
        modality="multi" if multi else "single",
        dip_p=dip.p,
        bimodality=bc,
        n_replicates=int(values.size),
        mean_events=mean_events,
        mean_steps=mean_steps,
        absorbed_runs=absorbed_runs,
    )


def convergence_curve(
    scores_by_scenario: Mapping[str, Sequence[float]],
    *,
    ladder: Sequence[int] = CONVERGENCE_LADDER,
) -> tuple[ConvergencePoint, ...]:
    """§9.3's convergence curve: how much p_hat still moves as n grows.

    For each rung, the mean absolute change in p_hat from the previous rung,
    averaged over the scenarios that have enough replicates to reach it. This
    is what turns "we ran it 200 times" into "we ran it 200 times because 200
    is where it converges" -- or, if the curve has not flattened by 200, into
    an honest statement that it has not.

    Prefixes rather than random subsamples: replicate *k* is a fixed seeded
    world (§8.2), so the first 25 replicates are a reproducible ensemble and
    a resampled 25 would differ between two runs of the report.
    """
    rungs = sorted({int(n) for n in ladder if n > 0})
    out: list[ConvergencePoint] = []
    previous: dict[str, float] = {}
    for n in rungs:
        changes: list[float] = []
        sigmas: list[float] = []
        count = 0
        for scenario_id in sorted(scores_by_scenario):
            values = np.asarray(scores_by_scenario[scenario_id], dtype=float)
            if values.size < n:
                continue
            head = values[:n]
            mean = float(head.mean())
            sigmas.append(float(head.std(ddof=1)) if head.size > 1 else 0.0)
            count += 1
            if scenario_id in previous:
                changes.append(abs(mean - previous[scenario_id]))
            previous[scenario_id] = mean
        if count == 0:
            continue
        out.append(
            ConvergencePoint(
                n=n,
                mean_abs_change=float(np.mean(changes)) if changes else 0.0,
                mean_sigma=float(np.mean(sigmas)) if sigmas else 0.0,
                scenarios=count,
            )
        )
    return tuple(out)
