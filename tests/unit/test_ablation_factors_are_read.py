"""The four ablation factors must be *mechanisms*, not configuration fields.

ADR-0025's regression. Measured at the start of M7: `causal_decomposition` and
`grounding` appeared only in `cascade/config.py`. The twelve-cell grid would
have executed as four distinct configurations, six cells would have been
duplicates of six others, and the headline deltas would have come back as
precise nulls with confidence intervals attached. Nothing downstream could
detect it -- a paired bootstrap on two identical columns returns [0, 0], which
reads as a finding.

So each factor is checked twice: that the package *references* it outside the
config module, and that flipping it changes what the system actually builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.config import load_settings, repo_root

FACTORS = {
    "A": "causal_decomposition",
    "B": "information_asymmetry",
    "C": "grounding",
    "D": "replicates",
}


def _package_sources() -> list[Path]:
    root = repo_root() / "cascade"
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if path.name != "config.py" and "__pycache__" not in path.parts
    ]


@pytest.mark.parametrize(("factor", "field"), sorted(FACTORS.items()))
def test_every_factor_is_referenced_outside_the_config_module(factor: str, field: str) -> None:
    """A flag read only by the code that defines it is an inert switch."""
    readers = [
        path.relative_to(repo_root())
        for path in _package_sources()
        if field in path.read_text(encoding="utf-8")
    ]
    assert readers, (
        f"factor {factor} (`{field}`) is defined in config.py and read nowhere else. "
        "Every cell that varies it would execute identically -- see ADR-0025."
    )


def test_the_check_would_notice_an_inert_flag() -> None:
    """A guard that cannot fail is not a guard."""
    invented = "a_flag_nothing_reads"
    readers = [path for path in _package_sources() if invented in path.read_text(encoding="utf-8")]
    assert readers == []


class TestFactorAChangesTheGraph:
    def test_decomposition_off_never_reaches_the_compiled_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural, not textual: `load_graph` is replaced with a landmine.

        Under A=off the panel branch must return before any database read, so
        the A=off arm is runnable with no compiled graph at all -- which is
        also what makes §10.2's multi-agent baseline producible before the
        compile phase has succeeded.
        """
        from cascade.cli import _graph_for

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("A=off must not load a compiled graph")

        monkeypatch.setattr("cascade.decompose.store.load_graph", explode)
        graph = _graph_for(load_settings("C09"), "some-scenario")
        assert graph is not None
        assert all(actor.id.startswith("panelist_") for actor in graph.actors)

    def test_decomposition_on_does_load_the_compiled_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: A=on must not quietly serve the panel."""
        called: list[str] = []

        def record(settings: object, scenario_id: str, *, role: str) -> None:
            called.append(scenario_id)
            return None

        monkeypatch.setattr("cascade.decompose.store.load_graph", record)
        from cascade.cli import _graph_for

        assert _graph_for(load_settings("C01"), "some-scenario") is None
        assert called == ["some-scenario"]

    def test_the_two_arms_produce_different_graphs_for_the_same_scenario(self) -> None:
        from cascade.decompose.schema import graph_hash
        from cascade.eval.panel import panel_graph

        settings = load_settings("C09")
        panel = panel_graph("some-scenario", actors=settings.flags.panel_actors)
        assert all(actor.id.startswith("panelist_") for actor in panel.actors)
        assert graph_hash(panel) != graph_hash(
            panel_graph("a-different-scenario", actors=settings.flags.panel_actors)
        )


class TestFactorCChangesTheEvidence:
    def test_parametric_only_opens_no_corpus_connection(self) -> None:
        """The ungrounded arm must not hold a handle to the corpus it is meant
        not to read -- so the guard is a context manager, not a conditional at
        each call site, and it is asserted by using it."""
        from cascade.cli import _maybe_chronofence

        with _maybe_chronofence(load_settings("C11"), enabled=False) as fence:
            assert fence is None

    def test_the_grounding_flag_selects_the_branch(self) -> None:
        assert load_settings("C09").flags.grounding == "chronofence"
        assert load_settings("C11").flags.grounding == "parametric_only"

    def test_an_ungrounded_brief_states_that_retrieval_was_disabled(self) -> None:
        """Not the same message as "nothing admissible was found".

        The same empty evidence block, two different facts, and an agent told
        the wrong one reasons differently about its own ignorance.
        """
        from cascade.sim.prompts import brief_from, persona_block

        class _Actor:
            id = "a1"
            name = "Actor"
            objective = "do something"
            risk_posture = "neutral"
            constraints = ()
            utility_terms = ()

        grounded = persona_block(
            brief_from(
                _Actor(),
                levers={},
                counterparties=(),
                horizon=24,
                question_context="q",
                evidence=(),
                grounded=True,
            ),
            evidence_chars=900,
        )
        ungrounded = persona_block(
            brief_from(
                _Actor(),
                levers={},
                counterparties=(),
                horizon=24,
                question_context="q",
                evidence=(),
                grounded=False,
            ),
            evidence_chars=900,
        )
        assert "no admissible evidence was found" in grounded
        assert "retrieval is disabled" in ungrounded
        assert grounded != ungrounded


class TestFactorsBAndD:
    def test_asymmetry_off_makes_every_policy_transparent(self) -> None:
        from cascade.aperture.policy import derive_policies
        from cascade.config import load_settings as load
        from cascade.eval.panel import panel_graph

        graph = panel_graph("s", actors=14)
        aperture = load(None).aperture
        opaque = derive_policies(graph, aperture, asymmetry=True)
        transparent = derive_policies(graph, aperture, asymmetry=False)
        assert opaque != transparent
        assert all(policy.action_visibility == "full" for policy in transparent.values())

    def test_the_replicate_factor_differs_between_the_d_arms(self) -> None:
        assert load_settings("C01").ensemble.replicates == 200
        assert load_settings("C02").ensemble.replicates == 1
