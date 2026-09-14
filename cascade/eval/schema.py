"""Boundary types for Assay, the evaluation harness (spec §10, Appendix D).

Every type here is frozen and every field is a measurement. There is no field
for a target and no default that could stand in for a number nobody computed:
§1 requires that the report print what was measured, so a metric that could
not be produced is an absent row, never a plausible one.

The separation that matters is that a :class:`ScoredForecast` is the *only*
type in the package that carries an outcome. It is constructed in one place --
``cascade.eval.score.score_config`` -- from a label-blind forecast joined to a
label read under the eval role. Nothing upstream of that join can see a label,
which is invariant 2 expressed in the type system as well as in the grant.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AblationCell",
    "BootstrapInterval",
    "CalibrationBin",
    "CalibrationReport",
    "Comparison",
    "DispersionFinding",
    "DomainMetrics",
    "Grounding",
    "MetricSet",
    "MurphyTerms",
    "ScoredForecast",
]

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Grounding = Literal["chronofence", "parametric_only"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoredForecast(_Frozen):
    """One forecast joined to its outcome. The only labelled type in Assay."""

    scenario_id: str
    config_id: str
    p_hat: Unit
    outcome: Literal[0, 1]
    domain: str
    sigma: Annotated[float, Field(ge=0.0)] = 0.0
    n_replicates: Annotated[int, Field(gt=0)] = 1
    modality: Literal["single", "multi"] = "single"
    policy: Literal["agent", "heuristic", "mixed", "none"] = "agent"

    @property
    def abs_error(self) -> float:
        return abs(self.p_hat - self.outcome)


class MurphyTerms(_Frozen):
    """``BS = REL - RES + UNC``, plus the residual the binning leaves behind.

    The identity is exact only when forecasts are grouped by *equal value*.
    Over 10 equal-width bins a bin holds a spread of forecasts and the cross
    term does not vanish, so the residual is reported rather than assumed
    away. A decomposition that silently absorbs its own error into REL is how
    "badly calibrated" and "not discriminative" get confused, which is the one
    thing §10.1 asks this decomposition to separate.
    """

    reliability: float
    resolution: float
    uncertainty: float
    brier: float
    residual: float
    bins: int


class CalibrationBin(_Frozen):
    """One row of §10.5's reliability table."""

    index: int
    lo: float
    hi: float
    count: int
    mean_pred: float | None
    obs_freq: float | None
    wilson_lo: float | None
    wilson_hi: float | None

    @property
    def gap(self) -> float | None:
        if self.mean_pred is None or self.obs_freq is None:
            return None
        return self.mean_pred - self.obs_freq


class CalibrationReport(_Frozen):
    """The 10-bin table, ECE and MCE (spec §10.1, §10.5)."""

    bins: tuple[CalibrationBin, ...]
    ece: float
    mce: float
    n: int


class MetricSet(_Frozen):
    """Every §10.1 metric for one configuration, measured on one scored set."""

    config_id: str
    n: int
    base_rate: float
    mean_p_hat: float
    brier: float
    log_loss: float
    auc: float | None
    ece: float
    mce: float
    murphy: MurphyTerms
    policies: tuple[str, ...] = ("agent",)
    """Which deciders produced the runs behind this configuration's forecasts.

    ``"heuristic"`` or ``"mixed"`` means the metrics are a mechanism check
    rather than a study result, and the report says so rather than printing a
    Brier that looks like one. ``"none"`` is the climatology baseline, which is
    arithmetic over the sealed base rate and is a legitimate study figure with
    no decider behind it."""

    @property
    def provisional(self) -> bool:
        """True when a decider that is not study data produced these numbers."""
        return bool({"heuristic", "mixed"} & set(self.policies))

    bss_vs_climatology: float | None = None
    bss_vs_direct: float | None = None
    brier_recalibrated: float | None = None
    """Isotonic recalibration fitted on a held-out half (§10.5). Secondary,
    never the headline."""


class DomainMetrics(_Frozen):
    """§10.4's per-domain breakdown, with counts. A table without counts hides
    that the interesting cell holds nine scenarios."""

    domain: str
    n: int
    base_rate: float
    brier: float


class BootstrapInterval(_Frozen):
    """A paired percentile-bootstrap interval on a Brier difference."""

    point: float
    lo: float
    hi: float
    b: int
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


class Comparison(_Frozen):
    """One hypothesis in the ablation family, with its adjusted p-value."""

    name: str
    config_a: str
    config_b: str
    brier_a: float
    brier_b: float
    n_paired: int
    interval: BootstrapInterval
    p_adjusted: float | None = None

    @property
    def delta(self) -> float:
        return self.brier_a - self.brier_b


class AblationCell(_Frozen):
    """One of Appendix C's 12 cells, as designed and as executed.

    ``replicates_design`` is Appendix C's D factor; ``replicates_executed`` is
    what the grid driver actually ran. CLAUDE.md Q1 turns on these being
    different numbers for the 11 non-headline cells, so the report carries
    both and never reconciles them silently.
    """

    cell_id: str
    config_id: str
    decomposition: bool
    information_asymmetry: bool
    grounding: Grounding
    replicates_design: int
    replicates_executed: int | None
    scenarios_executed: int
    role: str
    metrics: MetricSet | None = None


class DispersionFinding(_Frozen):
    """§9.2's claim that sigma is informative, tested rather than asserted."""

    n: int
    pearson_r: float | None
    pearson_p: float | None
    spearman_rho: float | None
    spearman_p: float | None
    flagged: int
    brier_flagged: float | None
    brier_unflagged: float | None
    sigma_threshold: float
