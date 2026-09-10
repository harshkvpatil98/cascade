"""The Chronofence client's database-free surface (spec §4.2).

The grant and time-lock behaviour needs a live database and is asserted in
``tests/leakage/``. What is checked here is everything that must fail *before*
a connection is opened -- the cheapest possible place to catch a caller who
would otherwise have run a query with the wrong cutoff.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cascade.config import Settings
from cascade.corpus.embed import EMBEDDING_DIM
from cascade.retrieval.search import Chronofence, vector_literal

CUTOFF = datetime(2019, 6, 1, tzinfo=UTC)


def test_vector_literal_is_pgvector_text_format() -> None:
    assert vector_literal([1.0, -0.5, 0.25]) == "[1.0,-0.5,0.25]"


def test_vector_literal_round_trips_a_float_exactly() -> None:
    """A truncated query vector perturbs every distance in the study.

    The damage would be invisible: recall would be measurably worse for a
    reason nothing in the output points at.
    """
    value = 0.1 + 0.2  # 0.30000000000000004
    assert float(vector_literal([value]).strip("[]")) == value


def test_vector_literal_accepts_an_empty_vector() -> None:
    """Rejecting it here would pre-empt the dimension check's better message."""
    assert vector_literal([]) == "[]"


def test_using_the_client_outside_a_context_manager_raises(settings: Settings) -> None:
    """The connection is the resource; forgetting `with` must not silently open one."""
    fence = Chronofence(settings, role="sim")
    with pytest.raises(RuntimeError, match="context manager"):
        fence.search([0.0] * EMBEDDING_DIM, as_of=CUTOFF, k=5)


def test_the_default_role_is_sim(settings: Settings) -> None:
    """The role the simulation runs as, so the narrowest grant is the default.

    Defaulting to `eval` would give every caller the exhaustive oracle and the
    ability to read `scenario_labels`.
    """
    assert Chronofence(settings).role == "sim"


def test_as_of_is_keyword_only(settings: Settings) -> None:
    """Invariant 1: a positional cutoff is easy to pass in the wrong slot."""
    fence = Chronofence(settings, role="sim")
    with pytest.raises(TypeError):
        fence.search([0.0] * EMBEDDING_DIM, CUTOFF, 5)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Credential preconditions
#
# Regression: `.env` carrying `CASCADE_ANTHROPIC_API_KEY=` (present, empty)
# parses to SecretStr("") rather than None, so the client's `is None` guard let
# it through and the SDK raised `TypeError: Could not resolve authentication
# method` -- an error naming neither the variable nor this project.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_an_empty_api_key_is_reported_as_absent(settings: Settings, value: str) -> None:
    from pydantic import SecretStr

    from cascade.llm.client import LLMClient
    from cascade.llm.types import LLMError

    blank = settings.model_copy(
        update={
            "anthropic_api_key": SecretStr(value),
            "llm": settings.llm.model_copy(update={"mode": "record"}),
        }
    )
    client = LLMClient(blank, phase="bench")
    with pytest.raises(LLMError, match="CASCADE_ANTHROPIC_API_KEY"):
        client._client()


def test_a_real_api_key_is_accepted(settings: Settings) -> None:
    """The guard must not reject a usable key."""
    from cascade.llm.client import LLMClient

    recorded = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"mode": "record"})}
    )
    client = LLMClient(recorded, phase="bench")
    assert client._client() is not None
