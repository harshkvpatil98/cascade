"""What the report *says*, as opposed to which files it writes.

Split from `test_eval_report.py` because these assert on the prose a reader
actually reads. §1 is a claim about the report's text as much as about its
arithmetic: a number that could not be produced must read as not produced, and
a number produced by a stand-in must not read like a result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cascade.eval.ablation import CELLS, grid_replicates, headline_comparisons
from cascade.eval.report import StudyArtifact, write_report
from cascade.eval.schema import (
    AblationCell,
    BootstrapInterval,
    Comparison,
    DispersionFinding,
    MetricSet,
    MurphyTerms,
)


def _artifact(**overrides: object) -> StudyArtifact:
    base: dict[str, object] = {
        "report_id": "study_20260811T1440Z",
        "written_at": datetime(2026, 8, 11, 14, 40, tzinfo=UTC),
        "manifest_sha256": "a" * 64,
        "git_sha": "abc123",
        "n_scenarios_sealed": 180,
        "base_rate": 0.5,
        "climatology_brier": 0.25,
        "study_salt": "salt",
        "models": {"agent": "haiku", "compiler": "sonnet"},
        "config_snapshot": {},
    }
    base.update(overrides)
    return StudyArtifact(**base)  # type: ignore[arg-type]


def _metrics(config_id: str, brier: float) -> MetricSet:
    return MetricSet(
        config_id=config_id,
        n=180,
        base_rate=0.5,
        mean_p_hat=0.5,
        brier=brier,
        log_loss=0.6,
        auc=0.6,
        ece=0.05,
        mce=0.1,
        murphy=MurphyTerms(
            reliability=0.01,
            resolution=0.02,
            uncertainty=0.25,
            brier=brier,
            residual=0.0,
            bins=10,
        ),
        bss_vs_climatology=1 - brier / 0.25,
    )


def _comparison(name: str, point: float) -> Comparison:
    return Comparison(
        name=name,
        config_a="C05",
        config_b="C01",
        brier_a=0.2,
        brier_b=0.2 - point,
        n_paired=90,
        interval=BootstrapInterval(
            point=point, lo=point - 0.01, hi=point + 0.01, b=10000, p_value=0.002
        ),
        p_adjusted=0.01,
    )


class TestTheDeltasAreReadCorrectly:
    def test_the_nesting_warning_is_in_the_report_not_only_in_the_code(
        self, tmp_path: Path
    ) -> None:
        """§10.3: "they do not sum, and the report must say so"."""
        directory = write_report(_artifact(), root=tmp_path)
        headline = (directory / "headline.md").read_text()
        assert "do not sum" in headline

    def test_every_comparison_carries_its_reading(self, tmp_path: Path) -> None:
        readings = {spec.name: spec.reading for spec in headline_comparisons()}
        artifact = _artifact(
            comparisons=(_comparison("LOO information asymmetry", 0.02),),
            comparison_readings=readings,
        )
        directory = write_report(artifact, root=tmp_path)
        headline = (directory / "headline.md").read_text()
        assert readings["LOO information asymmetry"][:40] in headline

    def test_the_reading_is_machine_readable_beside_the_number(self, tmp_path: Path) -> None:
        readings = {spec.name: spec.reading for spec in headline_comparisons()}
        artifact = _artifact(
            comparisons=(_comparison("LOO causal decomposition", 0.03),),
            comparison_readings=readings,
        )
        directory = write_report(artifact, root=tmp_path)
        payload = json.loads((directory / "significance.json").read_text())
        assert payload["comparisons"][0]["reading"]
        assert payload["method"]["resample_unit"] == "scenario"
        assert "Holm" in payload["method"]["multiple_comparison"]

    def test_no_comparisons_says_so_rather_than_printing_an_empty_table(
        self, tmp_path: Path
    ) -> None:
        directory = write_report(_artifact(), root=tmp_path)
        assert "no delta was computed" in (directory / "headline.md").read_text()


class TestTheReplicatePolicyIsStated:
    def test_the_report_names_which_reading_of_q1_it_applied(self, tmp_path: Path) -> None:
        """§10.3 warns the ensemble-contribution estimate depends on it."""
        notes = tuple(
            grid_replicates(cell, policy="budget_capped", ablation_cap=30)[1] for cell in CELLS
        )
        artifact = _artifact(
            replicate_policy="D is the design factor; 30 is a budget cap.",
            replicate_notes=notes,
        )
        headline = (write_report(artifact, root=tmp_path) / "headline.md").read_text()
        assert "## Replicate policy" in headline
        assert "budget cap" in headline
        assert "C09 has Appendix C design factor D=200" in headline

    def test_a_grid_that_never_ran_still_says_so(self, tmp_path: Path) -> None:
        headline = (write_report(_artifact(), root=tmp_path) / "headline.md").read_text()
        assert "no ablation cell was executed" in headline


class TestDispersionIsReportedHoweverItFell:
    def test_the_finding_is_printed_with_both_statistics(self, tmp_path: Path) -> None:
        finding = DispersionFinding(
            n=180,
            pearson_r=0.31,
            pearson_p=0.001,
            spearman_rho=0.28,
            spearman_p=0.002,
            flagged=56,
            brier_flagged=0.31,
            brier_unflagged=0.12,
            sigma_threshold=0.3,
        )
        headline = (
            write_report(_artifact(dispersion=finding), root=tmp_path) / "headline.md"
        ).read_text()
        assert "0.2800" in headline
        assert "56 of 180" in headline
        assert "whichever way they fell" in headline

    def test_an_unmeasurable_correlation_reads_as_not_measured(self, tmp_path: Path) -> None:
        finding = DispersionFinding(
            n=10,
            pearson_r=None,
            pearson_p=None,
            spearman_rho=None,
            spearman_p=None,
            flagged=0,
            brier_flagged=None,
            brier_unflagged=0.2,
            sigma_threshold=0.3,
        )
        headline = (
            write_report(_artifact(dispersion=finding), root=tmp_path) / "headline.md"
        ).read_text()
        assert "not measured" in headline


class TestCellsAreMachineReadable:
    def test_design_and_executed_replicate_counts_are_both_recorded(self, tmp_path: Path) -> None:
        """CLAUDE.md Q1 turns on these being different numbers; the report
        carries both and never reconciles them silently."""
        cells = (
            AblationCell(
                cell_id="C05",
                config_id="C05",
                decomposition=True,
                information_asymmetry=False,
                grounding="chronofence",
                replicates_design=200,
                replicates_executed=30,
                scenarios_executed=90,
                role="LOO information asymmetry",
                metrics=_metrics("C05", 0.168),
            ),
        )
        directory = write_report(_artifact(cells=cells), root=tmp_path)
        payload = json.loads((directory / "metrics.json").read_text())
        cell = payload["cells"][0]
        assert cell["replicates_design"] == 200
        assert cell["replicates_executed"] == 30
        assert cell["scenarios_executed"] == 90

    def test_a_cell_with_no_metrics_records_null_not_zero(self, tmp_path: Path) -> None:
        cells = (
            AblationCell(
                cell_id="C07",
                config_id="C07",
                decomposition=True,
                information_asymmetry=False,
                grounding="parametric_only",
                replicates_design=200,
                replicates_executed=None,
                scenarios_executed=0,
                role="Asymmetry x grounding interaction",
                metrics=None,
            ),
        )
        directory = write_report(_artifact(cells=cells), root=tmp_path)
        payload = json.loads((directory / "metrics.json").read_text())
        assert payload["cells"][0]["metrics"] is None

    def test_the_baseline_table_lists_all_five_even_when_absent(self, tmp_path: Path) -> None:
        from cascade.eval.baselines import BASELINES

        baselines = tuple((spec.baseline_id, spec.name, spec.config_id, None) for spec in BASELINES)
        directory = write_report(_artifact(baselines=baselines), root=tmp_path)
        payload = json.loads((directory / "metrics.json").read_text())
        assert len(payload["baselines"]) == 5
        assert all(row["metrics"] is None for row in payload["baselines"])
