"""The grid, its replicate policy and its scenario subsample (spec §10.3, App. C).

Two of these tests exist because of ADR-0025: `causal_decomposition` and
`grounding` were configuration fields nothing read, so six of the twelve cells
would have executed as duplicates of six others and the headline deltas would
have been measurements of nothing. A switch that is configured, documented and
inert is the failure they guard against.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from cascade.config import load_settings
from cascade.eval.ablation import (
    CELLS,
    HEADLINE_CELL,
    cell_by_id,
    comparison_family,
    grid_replicates,
    grid_scenarios,
    headline_comparisons,
    missing_cells,
)


class TestGridDefinition:
    def test_there_are_twelve_cells(self) -> None:
        assert len(CELLS) == 12

    def test_no_cell_enables_asymmetry_without_decomposition(self) -> None:
        """B nests in A. The four (A=off, B=on) cells are undefined, not untested."""
        assert not any(cell.information_asymmetry and not cell.decomposition for cell in CELLS)

    def test_the_four_factors_cover_the_defined_space_exactly(self) -> None:
        combinations = {
            (cell.decomposition, cell.information_asymmetry, cell.grounding, cell.replicates_design)
            for cell in CELLS
        }
        assert len(combinations) == 12

    @pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.cell_id)
    def test_every_cell_agrees_with_its_overlay_file(self, cell: object) -> None:
        """The grid driver and the configuration the runs are made under are two
        statements of one design; a drift between them fails here rather than
        surfacing as a mislabelled row in the report."""
        settings = load_settings(cell.cell_id)  # type: ignore[attr-defined]
        assert settings.flags.causal_decomposition is cell.decomposition  # type: ignore[attr-defined]
        assert settings.flags.information_asymmetry is cell.information_asymmetry  # type: ignore[attr-defined]
        assert settings.flags.grounding == cell.grounding  # type: ignore[attr-defined]
        assert settings.ensemble.replicates == cell.replicates_design  # type: ignore[attr-defined]

    def test_an_unknown_cell_raises_and_names_the_twelve(self) -> None:
        with pytest.raises(KeyError, match="C13"):
            cell_by_id("C13")

    def test_missing_cells_are_reported_sorted(self) -> None:
        assert missing_cells(["C01", "C05"])[:2] == ("C02", "C03")


class TestReplicatePolicy:
    """CLAUDE.md Q1, resolved explicitly rather than silently."""

    def test_budget_capped_caps_the_non_headline_cells(self) -> None:
        cell = cell_by_id("C05")
        count, note = grid_replicates(cell, policy="budget_capped", ablation_cap=30)
        assert count == 30
        assert "design factor D=200" in note
        assert "30" in note

    def test_the_headline_cell_is_never_capped(self) -> None:
        cell = cell_by_id(HEADLINE_CELL)
        count, note = grid_replicates(cell, policy="budget_capped", ablation_cap=30)
        assert count == 200
        assert "headline" in note

    def test_design_policy_runs_every_cell_at_its_d_factor(self) -> None:
        count, note = grid_replicates(cell_by_id("C05"), policy="design", ablation_cap=30)
        assert count == 200
        assert "no budget cap" in note.lower()

    def test_a_d_one_cell_is_already_under_the_cap(self) -> None:
        count, note = grid_replicates(cell_by_id("C02"), policy="budget_capped", ablation_cap=30)
        assert count == 1
        assert "already at or below" in note

    def test_the_note_is_prescriptive_not_a_claim_about_what_ran(self) -> None:
        """It is generated for all twelve cells, including ones never executed.

        "ran at 30" would be a claim about a cell with no runs. What a cell
        actually executed is measured from its stored forecasts.
        """
        for cell in CELLS:
            _, note = grid_replicates(cell, policy="budget_capped", ablation_cap=30)
            assert " ran at " not in note

    def test_the_note_states_which_reading_was_applied(self) -> None:
        """§10.3 warns that the ensemble contribution estimate depends on it,
        so the sentence is carried into the report verbatim."""
        _, note = grid_replicates(cell_by_id("C09"), policy="budget_capped", ablation_cap=30)
        assert "estimate at 30 replicates, not at 200" in note


class TestGridScenarios:
    IDS: ClassVar[list[str]] = [f"scenario-{index:03d}" for index in range(180)]

    def test_the_headline_cell_always_runs_the_whole_set(self) -> None:
        chosen = grid_scenarios(self.IDS, cell=cell_by_id("C01"), salt="s", cap=90)
        assert len(chosen) == 180

    def test_a_capped_cell_runs_the_cap(self) -> None:
        chosen = grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=90)
        assert len(chosen) == 90

    def test_every_capped_cell_runs_the_same_subsample(self) -> None:
        """So the eleven capped cells are paired with each other exactly, not
        merely with the headline."""
        first = grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=90)
        second = grid_scenarios(self.IDS, cell=cell_by_id("C11"), salt="s", cap=90)
        assert first == second

    def test_the_subsample_is_reproducible_from_the_salt(self) -> None:
        assert grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=90) == grid_scenarios(
            list(reversed(self.IDS)), cell=cell_by_id("C05"), salt="s", cap=90
        )

    def test_a_different_salt_chooses_a_different_ninety(self) -> None:
        assert grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="a", cap=90) != grid_scenarios(
            self.IDS, cell=cell_by_id("C05"), salt="b", cap=90
        )

    def test_a_cap_at_or_above_the_set_size_is_no_cap(self) -> None:
        assert len(grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=180)) == 180
        assert len(grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=None)) == 180

    def test_the_subsample_is_a_subset(self) -> None:
        chosen = grid_scenarios(self.IDS, cell=cell_by_id("C05"), salt="s", cap=90)
        assert set(chosen) <= set(self.IDS)
        assert len(set(chosen)) == len(chosen)


class TestComparisons:
    def test_the_net_figure_is_published_beside_the_two_it_reconciles(self) -> None:
        """§10.3 requires publishing +0.008 first, because it is the number a
        careful reader computes anyway."""
        names = [spec.name for spec in headline_comparisons()]
        assert names.index("Decomposition net of asymmetry") == (
            max(names.index("LOO information asymmetry"), names.index("LOO causal decomposition"))
            + 1
        )

    def test_no_expected_value_appears_in_any_comparison(self) -> None:
        """§1: a target must never be written into a report code path.

        The readings describe what a delta *means*; the values are measured.
        """
        for spec in headline_comparisons():
            assert "0.035" not in spec.reading
            assert "0.027" not in spec.reading
            assert "0.141" not in spec.reading

    def test_the_nesting_is_stated_in_the_reading(self) -> None:
        loo = next(
            spec for spec in headline_comparisons() if spec.name == "LOO causal decomposition"
        )
        assert "do not sum" in loo.reading

    def test_the_family_only_contains_testable_comparisons(self) -> None:
        """A family padded with untestable hypotheses inflates the Holm
        multiplier and weakens every surviving result."""
        family = comparison_family(available=["C01", "C05"], headline="C01")
        assert [spec.name for spec in family] == ["LOO information asymmetry"]

    def test_the_family_covers_every_available_cell(self) -> None:
        family = comparison_family(available=["C01", "C05", "C09", "C11"], headline="C01")
        pairs = {(spec.config_a, spec.config_b) for spec in family}
        assert ("C11", "C01") in pairs
        assert ("C05", "C01") in pairs
        assert ("C09", "C01") in pairs

    def test_a_missing_headline_drops_only_the_comparisons_against_it(self) -> None:
        """A named §10.3 reading that does not involve the headline survives.

        "Decomposition net of asymmetry" is C09 - C05 and is testable whenever
        both are present, headline or not. Dropping it because C01 is missing
        would withhold the one figure §10.3 says to publish first.
        """
        family = comparison_family(available=["C05", "C09"], headline="C01")
        assert [spec.name for spec in family] == ["Decomposition net of asymmetry"]
        assert all("C01" not in (spec.config_a, spec.config_b) for spec in family)

    def test_the_headline_is_not_compared_with_itself(self) -> None:
        family = comparison_family(available=["C01", "C05"], headline="C01")
        assert all(spec.config_a != spec.config_b for spec in family)
