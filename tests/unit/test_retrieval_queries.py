"""The bench query distribution (spec §4.2).

Two properties carry the whole module's credibility and are asserted here: the
query set never touches an outcome, and it is reproducible. A benchmark built
from resolution text would measure how well the index finds the answer key,
and a benchmark whose queries drift between runs cannot detect a regression.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cascade.ledger.schema import Scenario
from cascade.retrieval.queries import BenchQuery, build_queries, query_templates

SALT = "test-salt"


def scenario(
    scenario_id: str,
    question: str = "Will the merger be blocked by the regulator?",
    parties: tuple[str, ...] = ("European Commission", "Acme Corp"),
    cutoff: datetime | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        question=question,
        resolution_criterion="Resolves YES if the regulator blocks the merger.",
        cutoff_ts=cutoff or datetime(2019, 6, 1, tzinfo=UTC),
        resolve_ts=datetime(2020, 1, 1, tzinfo=UTC),
        domain="corporate",
        source="polymarket",
        source_ref="ref",
        party_rule="named_parties",
        party_names=parties,
        event_group=None,
    )


# ---------------------------------------------------------------------------
# Leakage: the query set must not be built from anything outcome-bearing
# ---------------------------------------------------------------------------


def test_no_template_references_an_outcome() -> None:
    """A retrieval benchmark keyed on resolution text grades its own answer key."""
    forbidden = ("outcome", "resolved", "resolution", "yes", "no", "label", "answer")
    for template in query_templates():
        lowered = template.lower()
        for word in forbidden:
            assert word not in lowered, f"{template!r} references {word!r}"


def test_queries_are_built_only_from_question_and_parties() -> None:
    """The resolution criterion must not appear in generated query text."""
    subject = scenario("s1")
    queries = build_queries([subject], count=40, salt=SALT)
    for query in queries:
        assert "Resolves YES" not in query.text
        assert "resolution" not in query.text.lower()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_registry_yields_the_same_queries() -> None:
    scenarios = [scenario(f"s{i}") for i in range(5)]
    first = build_queries(scenarios, count=50, salt=SALT)
    second = build_queries(scenarios, count=50, salt=SALT)
    assert [query.text for query in first] == [query.text for query in second]


def test_scenario_order_does_not_change_the_query_set() -> None:
    """Registry rows arrive in whatever order the database returns them."""
    scenarios = [scenario(f"s{i}") for i in range(5)]
    forward = build_queries(scenarios, count=50, salt=SALT)
    reversed_ = build_queries(list(reversed(scenarios)), count=50, salt=SALT)
    assert [query.text for query in forward] == [query.text for query in reversed_]


def test_a_different_salt_yields_a_different_query_set() -> None:
    scenarios = [scenario(f"s{i}") for i in range(5)]
    a = build_queries(scenarios, count=50, salt="salt-a")
    b = build_queries(scenarios, count=50, salt="salt-b")
    assert [q.text for q in a] != [q.text for q in b]


def test_adding_a_scenario_changes_the_query_set() -> None:
    """A different population is a different benchmark, and must look like one."""
    base = [scenario(f"s{i}") for i in range(5)]
    a = build_queries(base, count=50, salt=SALT)
    b = build_queries([*base, scenario("s99")], count=50, salt=SALT)
    assert [q.text for q in a] != [q.text for q in b]


# ---------------------------------------------------------------------------
# Coverage of the cutoff range
# ---------------------------------------------------------------------------


def test_every_scenario_is_represented_roughly_equally() -> None:
    """p95 is over the *full* cutoff range; weighting would hide the tail."""
    scenarios = [scenario(f"s{i}") for i in range(10)]
    queries = build_queries(scenarios, count=100, salt=SALT)
    per_scenario = {s.scenario_id: 0 for s in scenarios}
    for query in queries:
        per_scenario[query.scenario_id] += 1
    assert set(per_scenario.values()) == {10}


def test_each_query_carries_its_own_scenarios_cutoff() -> None:
    early = scenario("early", cutoff=datetime(2018, 1, 1, tzinfo=UTC))
    late = scenario("late", cutoff=datetime(2025, 1, 1, tzinfo=UTC))
    by_id = {q.scenario_id: q.as_of for q in build_queries([early, late], count=20, salt=SALT)}
    assert by_id["early"] == early.cutoff_ts
    assert by_id["late"] == late.cutoff_ts


def test_exactly_count_queries_are_produced() -> None:
    assert len(build_queries([scenario("s1")], count=137, salt=SALT)) == 137


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_a_scenario_with_no_named_parties_still_produces_queries() -> None:
    """The `event_siblings` party rule admits scenarios with no named parties.

    Substituting an empty placeholder would inject a token appearing in no
    document, so those templates fall back to topic-only.
    """
    queries = build_queries([scenario("s1", parties=())], count=30, salt=SALT)
    assert len(queries) == 30
    assert all(query.text.strip() for query in queries)


def test_interrogative_framing_is_stripped_from_the_topic() -> None:
    """ "Will ..." carries no retrievable signal and biases every query alike."""
    queries = build_queries(
        [scenario("s1", question="Will the coalition hold through Q3?", parties=())],
        count=1,
        salt=SALT,
    )
    assert not queries[0].text.lower().startswith("will ")
    assert "?" not in queries[0].text


def test_building_from_an_empty_registry_raises() -> None:
    with pytest.raises(ValueError, match="empty scenario registry"):
        build_queries([], count=10, salt=SALT)


def test_a_non_positive_count_raises() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        build_queries([scenario("s1")], count=0, salt=SALT)


def test_a_naive_cutoff_is_rejected() -> None:
    """Invariant 1's neighbour: a naive cutoff silently shifts the time lock."""
    with pytest.raises(ValueError, match="timezone-aware"):
        BenchQuery(scenario_id="s1", text="x", as_of=datetime(2019, 1, 1))
