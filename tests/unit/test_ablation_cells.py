"""The 12-cell ablation grid (spec Appendix C).

Twelve cells, not sixteen. Four binary factors would give sixteen, but
``(decomposition=off, asymmetry=on)`` is undefined -- the visibility policy is
*derived* from the compiled causal graph, so with decomposition off there is
no structure from which to project partial observation. Those four cells are
absent by construction, and this module asserts that the absence is exactly
where it should be rather than an off-by-one in the overlay files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.config import load_settings, repo_root

ABLATIONS_DIR = repo_root() / "configs" / "ablations"

# Transcribed from spec Appendix C: (cell, decomp, asym, grounding, D).
APPENDIX_C: list[tuple[str, bool, bool, str, int]] = [
    ("C01", True, True, "chronofence", 200),
    ("C02", True, True, "chronofence", 1),
    ("C03", True, True, "parametric_only", 200),
    ("C04", True, True, "parametric_only", 1),
    ("C05", True, False, "chronofence", 200),
    ("C06", True, False, "chronofence", 1),
    ("C07", True, False, "parametric_only", 200),
    ("C08", True, False, "parametric_only", 1),
    ("C09", False, False, "chronofence", 200),
    ("C10", False, False, "chronofence", 1),
    ("C11", False, False, "parametric_only", 200),
    ("C12", False, False, "parametric_only", 1),
]


def overlay_files() -> list[Path]:
    return sorted(ABLATIONS_DIR.glob("*.yaml"))


def test_there_are_exactly_twelve_cells() -> None:
    assert [path.stem for path in overlay_files()] == [row[0] for row in APPENDIX_C]


@pytest.mark.parametrize(("cell", "decomp", "asym", "grounding", "replicates"), APPENDIX_C)
def test_cell_matches_appendix_c(
    cell: str, decomp: bool, asym: bool, grounding: str, replicates: int
) -> None:
    """Each overlay resolves to exactly the row the spec specifies."""
    settings = load_settings(cell)
    assert settings.flags.causal_decomposition is decomp
    assert settings.flags.information_asymmetry is asym
    assert settings.flags.grounding == grounding
    assert settings.ensemble.replicates == replicates


def test_no_cell_enables_asymmetry_without_decomposition() -> None:
    """The four undefined cells must be absent, not silently defaulted.

    With decomposition off there is no graph, so there is no topology from
    which to derive a visibility policy. A cell in that quadrant would be
    scored against a policy that cannot exist.
    """
    undefined = [
        path.stem
        for path in overlay_files()
        if not load_settings(path.stem).flags.causal_decomposition
        and load_settings(path.stem).flags.information_asymmetry
    ]
    assert undefined == [], f"cells {undefined} enable asymmetry with decomposition off"


def test_the_headline_cell_matches_the_base_configuration() -> None:
    """C01 is the headline configuration; base.yaml must already be it.

    If they diverged, the main study and the C01 ablation cell would be
    different systems while being reported as the same one.
    """
    base = load_settings()
    headline = load_settings("C01")
    assert headline.flags == base.flags
    assert headline.ensemble.replicates == base.ensemble.replicates


def test_grounding_values_are_the_two_the_config_allows() -> None:
    """Appendix C writes 'parametric'; the config Literal is 'parametric_only'."""
    values = {load_settings(path.stem).flags.grounding for path in overlay_files()}
    assert values == {"chronofence", "parametric_only"}


def test_every_cell_carries_the_appendix_c_design_factor() -> None:
    """Overlays encode D, not the budget cap. See CLAUDE.md Q1.

    ``ensemble.replicates`` here is the Appendix C *design* factor D
    (200 or 1). ``ensemble.ablation_replicates`` (30) is the budget cap that
    §10.3/§12.3 apply to the 11 non-headline cells. These are different
    numbers and the M7 grid driver must state in the report which it applied;
    reconciling them here would hide the choice.
    """
    replicates = {load_settings(path.stem).ensemble.replicates for path in overlay_files()}
    assert replicates == {1, 200}
    assert load_settings().ensemble.ablation_replicates == 30


def test_overlays_change_only_flags_and_replicate_count() -> None:
    """A cell that quietly retuned the kernel would not be an ablation.

    The ablation claim is that one factor changed and nothing else did. That
    is only true if the overlays touch nothing but the factors.
    """
    base = load_settings()
    for path in overlay_files():
        cell = load_settings(path.stem)
        assert cell.kernel == base.kernel, path.stem
        assert cell.aperture == base.aperture, path.stem
        assert cell.models == base.models, path.stem
        assert cell.pricing == base.pricing, path.stem
        assert cell.retrieval == base.retrieval, path.stem
        assert cell.study == base.study, path.stem


def test_every_overlay_documents_the_q1_ambiguity() -> None:
    """The D-vs-cap ambiguity must stay visible at the point of use."""
    for path in overlay_files():
        text = path.read_text(encoding="utf-8")
        assert "ablation_replicates" in text, path.stem
        assert "Q1" in text, path.stem
