"""The parametric probe's scoring (spec §4.2).

§4.2 calls this the leakage control that "cannot be fixed by engineering; it is
measured and disclosed". Which puts the burden on the measurement being right:
a probe that reported low memorisation because it failed to parse the answer
would be worse than no probe at all, because it would look like reassurance.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.retrieval.memorization import (
    MemorizationScore,
    parse_probability,
    probe_prompt,
    summarise,
)
from tests.unit.test_retrieval_leakage import record

# ---------------------------------------------------------------------------
# parse_probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"p": 0.73}', 0.73),
        ('  {"p":0}  ', 0.0),
        ('{"p": 1}', 1.0),
        ('Here you go: {"p": 0.42}', 0.42),
        ('{"p": 0.5, "note": "unsure"}', 0.5),
    ],
)
def test_json_answers_are_parsed(text: str, expected: float) -> None:
    assert parse_probability(text) == pytest.approx(expected)


def test_a_bare_number_in_range_is_accepted() -> None:
    """Models drop the JSON wrapper often enough that refusing it wastes a call."""
    assert parse_probability("0.65") == pytest.approx(0.65)


@pytest.mark.parametrize("text", ["", "I cannot answer that.", "{}", '{"q": 0.5}', "abc"])
def test_an_unparseable_answer_is_none_not_a_default(text: str) -> None:
    """Defaulting to 0.5 would report *absent* memorisation when measurement failed.

    That is the most misleading possible direction for this particular number
    to fail in: it reads as reassurance.
    """
    assert parse_probability(text) is None


@pytest.mark.parametrize("text", ['{"p": 1.5}', '{"p": -0.2}', "42"])
def test_out_of_range_values_are_rejected(text: str) -> None:
    assert parse_probability(text) is None


def test_a_json_object_without_p_is_not_scavenged_for_numbers() -> None:
    """Regression: the bare-number fallback must not read structured output.

    `{"scenario": 1, "p_est": 0.7}` would otherwise return a confident 1.0 --
    the scenario index, read as a probability.
    """
    assert parse_probability('{"q": 0.5}') is None
    assert parse_probability('{"scenario": 1, "p_est": 0.7}') is None


def test_an_out_of_range_p_is_not_rescued_by_surrounding_prose() -> None:
    """A model that answers p=7 produced garbage; the prose number is a guess."""
    assert parse_probability('{"p": 7} but really 0.3') is None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0), (0.75, 0.5), (0.25, 0.5)],
)
def test_confidence_is_distance_from_ignorance(probability: float, expected: float) -> None:
    score = MemorizationScore(scenario_id="s1", probability=probability, outcome=1)
    assert score.confidence == pytest.approx(expected)


def test_confidence_is_direction_free() -> None:
    """A confidently wrong prior steers the simulation as much as a right one."""
    right = MemorizationScore(scenario_id="s1", probability=0.95, outcome=1)
    wrong = MemorizationScore(scenario_id="s2", probability=0.05, outcome=1)
    assert right.confidence == pytest.approx(wrong.confidence)
    assert right.correct_direction is True
    assert wrong.correct_direction is False


def test_an_exactly_uninformative_answer_has_no_direction() -> None:
    score = MemorizationScore(scenario_id="s1", probability=0.5, outcome=1)
    assert score.correct_direction is None


def test_an_unparseable_score_contributes_no_confidence() -> None:
    score = MemorizationScore(scenario_id="s1", probability=None, outcome=1)
    assert score.measured is False
    assert score.confidence == 0.0
    assert score.brier is None


def test_brier_is_the_squared_error_against_the_outcome() -> None:
    assert MemorizationScore(scenario_id="s1", probability=0.75, outcome=1).brier == pytest.approx(
        0.0625
    )


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


def test_unparseable_answers_are_reported_separately_not_scored() -> None:
    report = summarise(
        [
            MemorizationScore(scenario_id="a", probability=None, outcome=1),
            MemorizationScore(scenario_id="b", probability=0.9, outcome=1),
        ]
    )
    assert report.total == 2
    assert report.scored == 1
    assert report.unparseable == 1
    assert report.mean_confidence == pytest.approx(0.8)


def test_a_model_that_knows_nothing_scores_zero_confidence() -> None:
    report = summarise(
        [MemorizationScore(scenario_id=f"s{i}", probability=0.5, outcome=i % 2) for i in range(10)]
    )
    assert report.mean_confidence == pytest.approx(0.0)
    assert report.confident_and_correct == 0


def test_a_model_that_has_memorised_everything_scores_high() -> None:
    report = summarise(
        [
            MemorizationScore(scenario_id=f"s{i}", probability=float(i % 2), outcome=i % 2)
            for i in range(10)
        ]
    )
    assert report.mean_confidence == pytest.approx(1.0)
    assert report.confident_and_correct == 10
    assert report.brier == pytest.approx(0.0)


def test_deciles_partition_every_scored_answer() -> None:
    scores = [
        MemorizationScore(scenario_id=f"s{i}", probability=i / 20.0, outcome=1) for i in range(21)
    ]
    report = summarise(scores)
    assert sum(count for _, count in report.confidence_deciles) == report.scored


def test_a_confidence_of_exactly_one_lands_in_the_top_decile() -> None:
    """Half-open buckets would drop the most important observations."""
    report = summarise([MemorizationScore(scenario_id="s1", probability=1.0, outcome=1)])
    assert report.confidence_deciles[-1] == (0.9, 1)


def test_summarising_nothing_does_not_divide_by_zero() -> None:
    report = summarise([])
    assert report.scored == 0
    assert report.brier is None


@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50))
def test_mean_confidence_is_always_a_fraction(probabilities: list[float]) -> None:
    report = summarise(
        [
            MemorizationScore(scenario_id=f"s{i}", probability=p, outcome=i % 2)
            for i, p in enumerate(probabilities)
        ]
    )
    assert 0.0 <= report.mean_confidence <= 1.0


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_probe_prompt_carries_no_evidence() -> None:
    """Zero context is the whole measurement -- anything else contaminates it."""
    prompt = probe_prompt(record())
    assert "Will the merger be blocked?" in prompt
    assert "Resolves YES if blocked." in prompt
    # No retrieved chunk, no date hint, no outcome.
    assert "resolved" not in prompt.lower().replace("resolves yes if blocked.", "")


# ---------------------------------------------------------------------------
# End-to-end wiring
#
# The probe cannot be executed against the real API in this environment --
# CASCADE_ANTHROPIC_API_KEY is empty -- so the path is exercised against a mock
# transport instead. That proves everything except the credential: the request
# shape, the single call site (invariant 5), the metering, and the scoring.
# ---------------------------------------------------------------------------


def test_run_probe_scores_every_scenario_through_the_llm_client(
    settings, monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """One call per scenario, scored, with no retrieval anywhere in the path."""
    import json

    import httpx

    from cascade.llm.client import LLMClient
    from cascade.retrieval.memorization import run_probe
    from tests.conftest import message_payload

    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json=message_payload(model=body["model"], text='{"p": 0.8}'),
            headers={"content-type": "application/json"},
        )

    recorded = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"mode": "record"})}
    )
    client = LLMClient(
        recorded,
        phase="bench",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    records = [
        record(f"s{i}", outcome=i % 2, question=f"Will outcome {i} occur?") for i in range(4)
    ]
    report = run_probe(recorded, records, client=client)

    assert len(seen) == 4, "one call per scenario, no retries, no extra calls"
    assert report.total == 4
    assert report.scored == 4
    assert report.unparseable == 0
    assert report.mean_confidence == pytest.approx(0.6)  # 2 * |0.8 - 0.5|

    # Invariant: the probe is zero-context. No chunk text, no evidence, no
    # date hint may appear in any request that leaves this module.
    for body in seen:
        rendered = json.dumps(body)
        assert "chunk" not in rendered.lower()
        assert recorded.models.agent == body["model"]
        assert body["temperature"] == 0.0


def test_run_probe_reports_an_unparseable_answer_rather_than_defaulting(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """A model that refuses must not be scored as an uninformative 0.5."""
    import httpx

    from cascade.llm.client import LLMClient
    from cascade.retrieval.memorization import run_probe
    from tests.conftest import message_payload

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        return httpx.Response(
            200,
            json=message_payload(model=body["model"], text="I cannot answer that."),
            headers={"content-type": "application/json"},
        )

    recorded = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"mode": "record"})}
    )
    client = LLMClient(
        recorded,
        phase="bench",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    report = run_probe(recorded, [record("s1")], client=client)

    assert report.scored == 0
    assert report.unparseable == 1
    assert report.brier is None


def test_identical_questions_are_answered_once_from_the_cache(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Two scenarios with the same question text cost one call, not two.

    The registry admits distinct scenarios whose question text coincides. The
    prompt, model and temperature are then identical, so the content-addressed
    cache serves the second from disk. Correct *and* cheaper -- and asserted
    here because the opposite (a cache key accidentally including the scenario
    id) would silently double the probe's cost with no visible symptom.
    """
    import json

    import httpx

    from cascade.llm.client import LLMClient
    from cascade.retrieval.memorization import run_probe
    from tests.conftest import message_payload

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=message_payload(model=body["model"], text='{"p": 0.9}'),
            headers={"content-type": "application/json"},
        )

    recorded = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"mode": "record"})}
    )
    client = LLMClient(
        recorded,
        phase="bench",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    twins = [record("a", question="Same question?"), record("b", question="Same question?")]
    report = run_probe(recorded, twins, client=client)

    assert calls["n"] == 1, "the second identical prompt must be served from the cache"
    assert report.scored == 2, "both scenarios must still be scored"
