"""The 12-cell ablation grid (spec §10.3, Appendix C). Pure definitions.

Twelve cells, not sixteen. Four binary factors would give sixteen, but the
visibility policy is *derived* from the compiled causal graph, so information
asymmetry cannot be enabled when decomposition is disabled -- B nests in A and
the ``(A=off, B=on)`` quadrant is undefined rather than merely untested.

The grid is defined here rather than read from the overlay files, and then
**checked against them**. Two independent statements of the same design that
must agree is a test; one statement read twice is a transcription.

Reading the deltas correctly is the other thing this module is for. Because B
nests in A the leave-one-out deltas do not sum, and §10.3 requires the report
to say so. :func:`headline_comparisons` names the comparisons explicitly,
including the ``+0.008`` net figure §10.3 asks to be published *first* -- "the
number a careful reader will compute anyway".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from cascade.eval.schema import Grounding

__all__ = [
    "CELLS",
    "HEADLINE_CELL",
    "CellSpec",
    "ComparisonSpec",
    "cell_by_id",
    "comparison_family",
    "grid_replicates",
    "grid_scenarios",
    "headline_comparisons",
    "missing_cells",
]

ReplicatePolicy = Literal["design", "budget_capped"]


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One cell of Appendix C's matrix, as designed."""

    cell_id: str
    decomposition: bool
    information_asymmetry: bool
    grounding: Grounding
    replicates_design: int
    role: str

    @property
    def is_headline(self) -> bool:
        return self.cell_id == HEADLINE_CELL


HEADLINE_CELL = "C01"

# Transcribed from Appendix C. `tests/unit/test_ablation_cells.py` asserts that
# every one of these agrees with its overlay file in configs/ablations/, so a
# drift between the grid driver and the configuration the runs are made under
# fails CI rather than surfacing as a mislabelled row in the report.
CELLS: tuple[CellSpec, ...] = (
    CellSpec("C01", True, True, "chronofence", 200, "Full system -- headline configuration"),
    CellSpec("C02", True, True, "chronofence", 1, "Isolates the value of ensembling"),
    CellSpec("C03", True, True, "parametric_only", 200, "Isolates the value of grounded retrieval"),
    CellSpec("C04", True, True, "parametric_only", 1, "Structure only, no evidence, no ensemble"),
    CellSpec("C05", True, False, "chronofence", 200, "LOO information asymmetry"),
    CellSpec("C06", True, False, "chronofence", 1, "Asymmetry x ensemble interaction"),
    CellSpec("C07", True, False, "parametric_only", 200, "Asymmetry x grounding interaction"),
    CellSpec("C08", True, False, "parametric_only", 1, "Decomposition alone"),
    CellSpec("C09", False, False, "chronofence", 200, "LOO decomposition; MAS baseline"),
    CellSpec("C10", False, False, "chronofence", 1, "Single persona-panel pass"),
    CellSpec("C11", False, False, "parametric_only", 200, "Unstructured, ungrounded, ensembled"),
    CellSpec(
        "C12", False, False, "parametric_only", 1, "Floor -- nearest the single-model baseline"
    ),
)


def cell_by_id(cell_id: str) -> CellSpec:
    """Look up a cell, or raise naming the twelve that exist."""
    for cell in CELLS:
        if cell.cell_id == cell_id:
            return cell
    raise KeyError(f"{cell_id!r} is not an ablation cell; the grid is {[c.cell_id for c in CELLS]}")


def grid_replicates(
    cell: CellSpec, *, policy: ReplicatePolicy, ablation_cap: int
) -> tuple[int, str]:
    """How many replicates this cell runs at, and the sentence that says why.

    **This is CLAUDE.md's open question Q1, resolved explicitly rather than
    silently.** Appendix C gives the D factor as ``n in {200, 1}`` per cell,
    while §10.3 and §12.3 describe the 11 non-headline cells as "90 scenarios x
    30 replicates". Both cannot be literally true of a D=200 ablation cell.

    ``policy="budget_capped"`` reads D as the *design* factor and 30 as a
    *budget cap* applied to the non-headline cells -- so a D=200 ablation cell
    executes at 30. ``policy="design"`` runs every cell at its D factor, which
    is what Appendix C says and what §12.3's 29,700-run line does not fund.

    The returned sentence is carried into the report verbatim. §10.3's warning
    is that the ensemble contribution estimate depends on which reading is
    applied, so the report states which one it used rather than leaving a
    reader to infer it from a run count.

    The sentence is written in the present tense because it describes what the
    policy **prescribes**, not what happened: it is generated for all twelve
    cells including ones that were never executed, and "ran at 30" would be a
    claim about a cell with no runs. What a cell actually executed is
    ``AblationCell.replicates_executed``, measured from its stored forecasts.
    """
    if policy == "design":
        return cell.replicates_design, (
            f"{cell.cell_id} runs at its Appendix C design factor D="
            f"{cell.replicates_design}; no budget cap is applied."
        )
    if cell.is_headline:
        return cell.replicates_design, (
            f"{cell.cell_id} is the headline cell and runs at the full "
            f"D={cell.replicates_design}; the cap applies to the 11 others."
        )
    capped = min(cell.replicates_design, ablation_cap)
    if capped == cell.replicates_design:
        return capped, (
            f"{cell.cell_id} has D={cell.replicates_design}, which is already at or "
            f"below the {ablation_cap}-replicate budget cap; it runs at D."
        )
    return capped, (
        f"{cell.cell_id} has Appendix C design factor D={cell.replicates_design} and "
        f"runs at the {ablation_cap}-replicate budget cap of §10.3/§12.3. The "
        "ensemble contribution estimated from this cell is an estimate at "
        f"{capped} replicates, not at {cell.replicates_design}."
    )


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """One named comparison in §10.3's reading of the grid."""

    name: str
    config_a: str
    config_b: str
    reading: str
    """What the delta means. Printed beside the number, not in a footnote."""


def headline_comparisons() -> tuple[ComparisonSpec, ...]:
    """§10.3's table, as comparisons rather than as expected values.

    No target appears here. §10.3 pairs each row with an expected Brier; those
    expectations are what the study is testing, and writing them into the code
    that computes the deltas is the exact failure §1 names. The *reading* of
    each delta is recorded, because that is a property of the design; the
    value is measured.

    The order matters: the net figure is listed immediately after the two
    leave-one-out deltas it reconciles, so the report cannot print one without
    the other.
    """
    return (
        ComparisonSpec(
            name="LOO information asymmetry",
            config_a="C05",
            config_b="C01",
            reading=(
                "Brier(A=on, B=off) - Brier(full). Positive means turning "
                "information asymmetry off made the system worse."
            ),
        ),
        ComparisonSpec(
            name="LOO causal decomposition",
            config_a="C09",
            config_b="C01",
            reading=(
                "Brier(A=off, B=off) - Brier(full). Positive means removing the "
                "causal decomposition made the system worse. Because B nests in "
                "A, this delta contains the asymmetry delta and the two do not sum."
            ),
        ),
        ComparisonSpec(
            name="Decomposition net of asymmetry",
            config_a="C09",
            config_b="C05",
            reading=(
                "Brier(A=off, B=off) - Brier(A=on, B=off): what decomposition "
                "contributes once asymmetry is already off. This is the number a "
                "careful reader computes from the other two, and §10.3 requires "
                "publishing it first."
            ),
        ),
        ComparisonSpec(
            name="Grounding contribution",
            config_a="C03",
            config_b="C01",
            reading=(
                "Brier(parametric_only) - Brier(chronofence), both with structure "
                "and asymmetry on. Reported, not predicted."
            ),
        ),
        ComparisonSpec(
            name="Ensemble contribution",
            config_a="C02",
            config_b="C01",
            reading=(
                "Brier(n=1) - Brier(n=200). Reported, not predicted. Its magnitude "
                "depends on the replicate policy applied to the grid -- see the "
                "replicate-policy note."
            ),
        ),
    )


def missing_cells(available: Sequence[str]) -> tuple[str, ...]:
    """Which of the twelve cells have no stored forecasts. Sorted."""
    have = set(available)
    return tuple(cell.cell_id for cell in CELLS if cell.cell_id not in have)


def grid_scenarios(
    scenario_ids: Sequence[str], *, cell: CellSpec, salt: str, cap: int | None
) -> tuple[str, ...]:
    """The scenarios one cell runs, subsampled if §10.3's cap applies. Pure.

    §10.3 and §12.3 size the 11 non-headline cells at "90 scenarios x 30
    replicates". The headline cell always runs the whole sealed set: it is the
    configuration the study reports, and reporting it on a subsample would make
    the headline Brier a different quantity from the one every baseline is
    scored against.

    The subsample is ranked by a keyed hash of the scenario id. Three
    properties follow and each of them matters:

    * **Outcome-independent.** The hash sees the id and the study salt. A
      subsample correlated with the label would make every ablation delta a
      measurement of the subsample.
    * **Identical across cells.** The ranking does not depend on the cell, so
      the eleven capped cells run the *same* 90 scenarios and are paired with
      each other exactly, not merely with the headline.
    * **Reproducible.** Same salt, same 90, on any machine and in any process.
    """
    ordered = sorted(scenario_ids)
    if cell.is_headline or cap is None or cap <= 0 or cap >= len(ordered):
        return tuple(ordered)
    ranked = sorted(
        ordered,
        key=lambda scenario_id: hashlib.blake2b(
            scenario_id.encode("utf-8"), digest_size=8, key=salt.encode("utf-8")
        ).digest(),
    )
    return tuple(sorted(ranked[:cap]))


def comparison_family(
    *, available: Sequence[str], headline: str = HEADLINE_CELL
) -> tuple[ComparisonSpec, ...]:
    """The family Holm-Bonferroni is applied across (spec §10.4).

    §10.4 sizes the family as "twelve cells plus five baselines", so it is not
    only §10.3's five named readings: every other configuration that has
    forecasts is also compared against the headline, and all of them are
    adjusted together. Restricting the adjustment to the five headline readings
    while still reporting the rest would under-correct exactly the comparisons
    a sceptical reader goes looking for.

    Only comparisons with forecasts on **both** sides are returned. A family
    padded with untestable hypotheses would inflate the Holm multiplier and
    make every surviving result look weaker than the evidence says -- which is
    conservative in the wrong way: it is still a wrong number.

    ``headline`` is a parameter rather than the constant so the family can be
    built against whichever configuration is actually being reported. A partial
    grid has a headline too.
    """
    have = set(available)
    out: list[ComparisonSpec] = []
    named = {
        (spec.config_a, spec.config_b)
        for spec in headline_comparisons()
        if spec.config_a in have and spec.config_b in have
    }
    out.extend(spec for spec in headline_comparisons() if (spec.config_a, spec.config_b) in named)
    for config_id in sorted(have):
        if config_id == headline or (config_id, headline) in named:
            continue
        if headline not in have:
            continue
        out.append(
            ComparisonSpec(
                name=f"{config_id} vs {headline}",
                config_a=config_id,
                config_b=headline,
                reading=(
                    f"Brier({config_id}) - Brier({headline}). Positive means "
                    f"{config_id} is worse than the reported configuration. Included "
                    "in the Holm family so the adjustment covers every comparison the "
                    "report prints (§10.4)."
                ),
            )
        )
    return tuple(out)
