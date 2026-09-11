"""The §5.4 audit protocol: seeded sampling, rubric, scoring.

§5.4 exists because automated validation catches structural defects, not wrong
ones. These tests cover the parts a machine owns -- that the sample cannot be
re-rolled until it flatters the compiler, that a half-finished audit reports as
half-finished rather than as a failing one, and that the published mean is the
one the threshold is applied to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade.decompose.audit import (
    AUDIT_AXES,
    PASS_THRESHOLD,
    SAMPLE_SIZE,
    AuditScore,
    sample_scenarios,
    score_worksheet,
    summarise,
    write_worksheet,
)
from tests.conftest import make_graph

IDS = tuple(f"scenario-{index:03d}" for index in range(180))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_the_sample_is_the_specified_size() -> None:
    assert len(sample_scenarios(IDS, salt="s")) == SAMPLE_SIZE


def test_the_sample_is_reproducible() -> None:
    """Re-running the audit must review the same twenty graphs."""
    assert sample_scenarios(IDS, salt="s") == sample_scenarios(IDS, salt="s")


def test_the_sample_does_not_depend_on_input_order() -> None:
    """Registry rows arrive in whatever order the database returns them."""
    assert sample_scenarios(IDS, salt="s") == sample_scenarios(tuple(reversed(IDS)), salt="s")


def test_a_different_salt_gives_a_different_sample() -> None:
    """The seed is fixed by the study, so a reviewer cannot re-roll it."""
    assert sample_scenarios(IDS, salt="a") != sample_scenarios(IDS, salt="b")


def test_the_sample_is_drawn_without_replacement() -> None:
    chosen = sample_scenarios(IDS, salt="s")
    assert len(set(chosen)) == len(chosen)
    assert set(chosen) <= set(IDS)


def test_a_smaller_registry_samples_everything_it_has() -> None:
    small = IDS[:5]
    assert set(sample_scenarios(small, salt="s")) == set(small)


def test_sampling_an_empty_registry_raises() -> None:
    with pytest.raises(ValueError, match="empty scenario set"):
        sample_scenarios((), salt="s")


# ---------------------------------------------------------------------------
# Worksheet
# ---------------------------------------------------------------------------


def test_the_worksheet_carries_the_question_but_not_the_outcome(tmp_path: Path) -> None:
    """A reviewer who knows the answer scores the graph against the answer.

    §5.4 asks whether a domain-literate person at the cutoff would recognise
    the decomposition -- which is a different question from whether it points at
    what happened.
    """
    graphs = [("scenario-001", "Will the merger be blocked?", make_graph())]
    _, worksheet_path = write_worksheet(tmp_path, graphs=graphs, salt="s")
    payload = json.loads(worksheet_path.read_text())

    entry = payload["graphs"][0]
    assert entry["question"] == "Will the merger be blocked?"
    # Structural: the entry carries exactly these three things. `outcome_rule`
    # lives inside `graph` and is part of the decomposition under review, which
    # is why this asserts on keys rather than grepping for the word "outcome".
    assert set(entry) == {"scenario_id", "question", "graph"}
    assert set(entry["graph"]) == {"actors", "factors", "edges", "outcome_rule"}

    rendered = json.dumps(payload).lower()
    for marker in ("resolved yes", "resolved no", "resolved_at", '"outcome":', '"label"'):
        assert marker not in rendered, f"worksheet leaks {marker!r}"


def test_the_worksheet_shows_edges_readably(tmp_path: Path) -> None:
    """A reviewer judging edge signs should not have to parse raw JSON."""
    graphs = [("scenario-001", "Q?", make_graph())]
    _, worksheet_path = write_worksheet(tmp_path, graphs=graphs, salt="s")
    edges = json.loads(worksheet_path.read_text())["graphs"][0]["graph"]["edges"]
    assert any("--(+" in edge and "lag" in edge for edge in edges)


def test_the_rubric_is_written_alongside(tmp_path: Path) -> None:
    """§5.4: "publish the mean, and publish the rubric"."""
    rubric_path, _ = write_worksheet(tmp_path, graphs=[("s1", "Q?", make_graph())], salt="s")
    text = rubric_path.read_text()
    for axis in AUDIT_AXES:
        assert axis in text
    assert str(PASS_THRESHOLD) in text


def test_the_worksheet_has_one_blank_score_row_per_graph(tmp_path: Path) -> None:
    graphs = [(f"s{index}", "Q?", make_graph()) for index in range(3)]
    _, worksheet_path = write_worksheet(tmp_path, graphs=graphs, salt="s")
    scores = json.loads(worksheet_path.read_text())["scores"]
    assert len(scores) == 3
    assert all(row[axis] is None for row in scores for axis in AUDIT_AXES)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def complete(tmp_path: Path, *rows: dict[str, object]) -> Path:
    graphs = [(str(row["scenario_id"]), "Q?", make_graph()) for row in rows]
    _, path = write_worksheet(tmp_path, graphs=graphs, salt="s")
    payload = json.loads(path.read_text())
    payload["scores"] = list(rows)
    path.write_text(json.dumps(payload))
    return path


def test_a_completed_worksheet_scores(tmp_path: Path) -> None:
    path = complete(
        tmp_path,
        {
            "scenario_id": "s1",
            "actor_completeness": 2,
            "edge_sign_correctness": 2,
            "missing_leverage": 1,
            "note": "",
        },
    )
    summary = summarise(score_worksheet(path))
    assert summary.scored == 1
    assert summary.mean == pytest.approx(5 / 3)
    assert summary.passed


def test_an_unscored_row_is_skipped_not_counted_as_zero(tmp_path: Path) -> None:
    """A half-finished audit and a failing one are different facts."""
    path = complete(
        tmp_path,
        {
            "scenario_id": "s1",
            "actor_completeness": 2,
            "edge_sign_correctness": 2,
            "missing_leverage": 2,
            "note": "",
        },
        {
            "scenario_id": "s2",
            "actor_completeness": None,
            "edge_sign_correctness": None,
            "missing_leverage": None,
            "note": "",
        },
    )
    summary = summarise(score_worksheet(path))
    assert summary.scored == 1
    assert summary.mean == pytest.approx(2.0)


def test_a_mean_below_the_threshold_does_not_pass(tmp_path: Path) -> None:
    path = complete(
        tmp_path,
        {
            "scenario_id": "s1",
            "actor_completeness": 1,
            "edge_sign_correctness": 1,
            "missing_leverage": 1,
            "note": "thin",
        },
    )
    summary = summarise(score_worksheet(path))
    assert summary.mean == pytest.approx(1.0)
    assert not summary.passed


def test_exactly_the_threshold_passes(tmp_path: Path) -> None:
    """§5.4 says "at least 1.5", so the boundary is a pass."""
    path = complete(
        tmp_path,
        {
            "scenario_id": "s1",
            "actor_completeness": 2,
            "edge_sign_correctness": 2,
            "missing_leverage": 2,
            "note": "",
        },
        {
            "scenario_id": "s2",
            "actor_completeness": 1,
            "edge_sign_correctness": 1,
            "missing_leverage": 1,
            "note": "",
        },
    )
    summary = summarise(score_worksheet(path))
    assert summary.mean == pytest.approx(1.5)
    assert summary.passed


def test_an_empty_audit_does_not_pass(tmp_path: Path) -> None:
    """Zero scored graphs must not read as a passing mean of 0.0."""
    summary = summarise([])
    assert summary.scored == 0
    assert not summary.passed


def test_the_per_axis_breakdown_is_reported(tmp_path: Path) -> None:
    """A 1.5 made of (2, 2, 0.5) and one made of (1.5, 1.5, 1.5) differ.

    They call for entirely different prompt changes, so the headline alone is
    not actionable.
    """
    path = complete(
        tmp_path,
        {
            "scenario_id": "s1",
            "actor_completeness": 2,
            "edge_sign_correctness": 2,
            "missing_leverage": 0,
            "note": "leverage missing",
        },
    )
    summary = summarise(score_worksheet(path))
    per_axis = dict(summary.per_axis)
    assert per_axis["actor_completeness"] == pytest.approx(2.0)
    assert per_axis["missing_leverage"] == pytest.approx(0.0)


def test_the_worst_graphs_are_surfaced(tmp_path: Path) -> None:
    path = complete(
        tmp_path,
        *[
            {
                "scenario_id": f"s{index}",
                "actor_completeness": index % 3,
                "edge_sign_correctness": 2,
                "missing_leverage": 2,
                "note": "",
            }
            for index in range(6)
        ],
    )
    summary = summarise(score_worksheet(path))
    assert summary.worst
    assert summary.worst[0][1] <= summary.worst[-1][1]


@pytest.mark.parametrize("bad", [-1, 3, 5])
def test_an_out_of_range_score_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        AuditScore(
            scenario_id="s1",
            actor_completeness=bad,
            edge_sign_correctness=2,
            missing_leverage=2,
        )
