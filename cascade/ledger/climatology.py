"""The climatology baseline (spec §3.1, §10.2).

Pure: no I/O, no clock.

Climatology always predicts the set's own base rate. It is the floor every
other baseline and the system itself must clear -- "a system that does not
beat climatology has produced nothing" -- so it is computed here, at M1, from
the sealed set, and stored alongside the manifest rather than recomputed at
M7 from whatever set happens to be loaded then.

Storing it with the split is the point. A climatology number recomputed later
against a different set would silently move the bar the study is judged
against, in the direction that flatters the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from cascade.ledger.schema import ScenarioRecord

__all__ = ["Climatology", "brier_score", "climatology_of"]


def brier_score(forecasts: tuple[float, ...], outcomes: tuple[int, ...]) -> float:
    """Mean squared error of probabilistic forecasts against binary outcomes.

    The definition, not a target. This is the metric the entire study is
    reported in, so it is written once, here, from first principles: no
    weighting, no clipping, no smoothing.
    """
    if len(forecasts) != len(outcomes):
        raise ValueError(f"forecast/outcome length mismatch: {len(forecasts)} vs {len(outcomes)}")
    if not outcomes:
        raise ValueError("Brier score of an empty set is undefined")
    return sum((p - y) ** 2 for p, y in zip(forecasts, outcomes, strict=True)) / len(outcomes)


@dataclass(frozen=True)
class Climatology:
    """The base rate of a sealed set, and the Brier score of predicting it."""

    n: int
    n_yes: int
    base_rate: float
    brier: float

    @property
    def n_no(self) -> int:
        return self.n - self.n_yes


def climatology_of(records: tuple[ScenarioRecord, ...]) -> Climatology:
    """Compute the base rate and the climatology Brier for a sealed set.

    For a constant forecast ``p`` the Brier score reduces to ``p(1-p)`` when
    ``p`` is the base rate, which is the variance of the outcome -- the
    irreducible uncertainty term ``UNC`` of the Murphy decomposition used at
    M7. It is computed here by the direct definition rather than by that
    identity, so that the identity remains something M7 can *check* rather
    than something it assumes.
    """
    if not records:
        raise ValueError("climatology of an empty registry is undefined")
    outcomes = tuple(record.label.outcome for record in records)
    base_rate = sum(outcomes) / len(outcomes)
    forecasts = tuple(base_rate for _ in outcomes)
    return Climatology(
        n=len(outcomes),
        n_yes=sum(outcomes),
        base_rate=base_rate,
        brier=brier_score(forecasts, outcomes),
    )
