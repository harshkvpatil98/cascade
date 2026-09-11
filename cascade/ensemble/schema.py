"""Boundary types for Chorus, the ensembling layer (spec §9, §3.3)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ConvergencePoint", "Forecast", "Modality"]

Modality = Literal["single", "multi"]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class Forecast(BaseModel):
    """One (scenario, config) collapsed from its replicates (spec §9.1).

    The forecast is the ensemble **mean** of the terminal outcome scores, not
    the median and not the mode: the mean is the natural probability estimate
    under the outcome rule and the quantity the Brier score is defined against.

    The dispersion fields travel with it rather than being recomputed at
    reporting time, because §9.2's claim is that sigma is *informative* -- the
    correlation between sigma and absolute error is a result, and a result
    cannot be recomputed from a forecast that dropped its own sigma.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    config_id: str
    p_hat: Unit
    sigma: Annotated[float, Field(ge=0.0)]
    ci_lo: Unit
    ci_hi: Unit
    modality: Modality
    dip_p: float | None = None
    bimodality: float | None = None
    n_replicates: Annotated[int, Field(gt=0)]
    mean_events: float = 0.0
    mean_steps: float = 0.0
    absorbed_runs: int = 0

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo


class ConvergencePoint(BaseModel):
    """One point on §9.3's replicate-count convergence curve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: Annotated[int, Field(gt=0)]
    mean_abs_change: float
    """Mean |p_hat(n) - p_hat(n_previous)| across the sampled scenarios."""
    mean_sigma: float
    scenarios: int
