"""Shared fixtures.

Tests build ``Settings`` from the real ``configs/base.yaml`` and then override
only what they need, so a drift between the shipped config and what the tests
assume shows up as a failure rather than passing against a parallel fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cascade.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Drop the memoised Settings between tests and detach the repo's ``.env``.

    ``load_settings`` is lru_cached so config is immutable within a run. Across
    tests that would leak one test's monkeypatched environment into the next,
    making failures depend on ordering.

    It also detaches the process from every ambient ``CASCADE_*`` variable and
    points ``CASCADE_ENV_FILE`` at a path that does not exist. Both are needed:
    ``make test`` includes and **exports** ``.env``, so without this the suite
    passes under bare ``pytest`` and fails under ``make`` -- or worse, passes
    under both for reasons that differ per machine. A test that needs a value
    set now sets it explicitly with ``monkeypatch.setenv``.
    """
    import os

    from cascade.config import _cached_settings

    for name in sorted(os.environ):
        if name.startswith("CASCADE_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CASCADE_ENV_FILE", str(tmp_path / "absent.env"))
    _cached_settings.cache_clear()
    yield
    _cached_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real configuration, with all writable paths redirected into ``tmp_path``."""
    base = Settings()
    return base.model_copy(
        update={
            "llm": base.llm.model_copy(update={"cache_dir": str(tmp_path / "llm-cache")}),
            "paths": base.paths.model_copy(
                update={
                    "checkpoints": str(tmp_path / "checkpoints"),
                    "reports": str(tmp_path / "reports"),
                    "graphs": str(tmp_path / "graphs"),
                }
            ),
            "anthropic_api_key": SecretStr("sk-ant-test-key-not-real"),
        }
    )


def message_payload(
    *,
    model: str,
    text: str = "ok",
    input_tokens: int = 1900,
    output_tokens: int = 45,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    """A wire-shaped ``/v1/messages`` response body.

    Returned through a mock transport so the *real* SDK does the parsing; a
    hand-built fake response object would not catch a deserialisation change.
    """
    return {
        "id": "msg_01TEST",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
        },
    }


class CallCounter:
    """Counts HTTP requests so a test can assert *zero* network activity."""

    def __init__(self) -> None:
        self.count = 0
        self.urls: list[str] = []


@pytest.fixture
def recording_transport() -> tuple[httpx.Client, CallCounter]:
    """An httpx client that serves a canned Messages response and counts calls."""
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.count += 1
        counter.urls.append(str(request.url))
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=message_payload(model=body["model"]),
            headers={"content-type": "application/json"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler)), counter


class NetworkForbidden(AssertionError):
    """Raised by the replay transport if anything attempts to reach the network."""


@pytest.fixture
def forbidding_transport() -> httpx.Client:
    """An httpx client whose transport raises on *any* request.

    This is the M0 acceptance instrument: the replay pass must make zero
    network calls, and the only way to prove that is to make a call impossible.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise NetworkForbidden(
            f"replay mode attempted a network call to {request.url}; "
            "replay must never reach the network"
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Live-database fixtures, shared by tests/leakage/ and tests/property/
#
# Session-scoped: loading the pinned embedding model dominates these suites and
# there is no per-test state to isolate, because every probe that writes rolls
# its transaction back.
#
# These deliberately set `CASCADE_ENV_FILE` through os.environ rather than
# monkeypatch. The autouse fixture above detaches every test from the repo's
# .env, which is right for unit tests and wrong for a probe whose entire
# purpose is to run against the real database; a session-scoped fixture cannot
# use a function-scoped monkeypatch to undo it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    """Real settings against the running database, or skip.

    Session-scoped: the embedding model load dominates these tests and there
    is no per-test state to isolate, since every probe rolls back.
    """
    import os

    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        pytest.skip("no .env; run `make env` first")
    os.environ["CASCADE_ENV_FILE"] = str(env_file)
    settings = Settings()
    try:
        import psycopg

        with (
            psycopg.connect(settings.database_url("admin"), connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT to_regprocedure('chronofence_search(halfvec,timestamptz,int)')")
            row = cur.fetchone()
            if row is None or row[0] is None:
                pytest.skip("chronofence_search is absent; run `cascade db migrate`")
            cur.execute("SELECT count(*) FROM chunks")
            count_row = cur.fetchone()
            if count_row is None or count_row[0] == 0:
                pytest.skip("corpus is empty; run `cascade corpus build`")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 -- a skip needs its reason
        pytest.skip(f"postgres not reachable: {type(exc).__name__}: {exc}")
    return settings


@pytest.fixture(scope="session")
def records(live_settings: Settings) -> tuple[Any, ...]:
    """All 180 scenarios with their labels. Read as ``eval`` (invariant 2)."""
    from cascade.ledger.store import load_records

    loaded = load_records(live_settings, role="eval")
    if not loaded:
        pytest.skip("scenario registry is empty; run `cascade ledger build`")
    return loaded


@pytest.fixture(scope="session")
def embedder(live_settings: Settings) -> Any:
    """The pinned embedding model, loaded once for the whole suite."""
    from cascade.corpus.embed import Embedder, EmbeddingUnavailable

    model = Embedder(
        model_name=live_settings.models.embedding,
        batch_size=live_settings.corpus.embed_batch_size,
    )
    try:
        model.load()
    except EmbeddingUnavailable as exc:
        pytest.skip(str(exc))
    return model


# ---------------------------------------------------------------------------
# CausalGraph builders (M4)
#
# A valid graph has eight actors, four factors, full reachability and a
# two-factor outcome rule. Tests mutate one thing at a time from this baseline,
# so a failure names the rule that broke rather than the fixture.
# ---------------------------------------------------------------------------


# Semantically unrelated factor names. Enumerated rather than generated:
# "Distinct driver number 1/2/3" reads as distinct to a hashing stub and as
# near-identical to the pinned 384-dimension model, which made the fixture pass
# the orthogonality rule under a fake embedder and fail it under the real one.
FACTOR_NAMES: tuple[str, ...] = (
    "Regulatory approval probability",
    "Credit market liquidity",
    "Public opposition intensity",
    "Labour union strike readiness",
    "Currency exchange volatility",
    "Parliamentary coalition stability",
    "Semiconductor supply backlog",
    "Arctic shipping lane access",
    "Vaccine distribution coverage",
    "Agricultural commodity yield",
    "Undersea cable redundancy",
    "Judicial appointment backlog",
)

# Objectives phrased as standing interests, deliberately unrelated to any
# merger/regulator question the tests use as outcome text.
OBJECTIVES: tuple[str, ...] = (
    "Keep borrowing costs below the level at which its refinancing plan fails.",
    "Preserve harbour access rights it has held since the last treaty revision.",
    "Maintain enough spare manufacturing capacity to absorb a demand spike.",
    "Protect the pension scheme's funding ratio from a rates shock.",
    "Expand rural broadband coverage ahead of the next licensing round.",
    "Avoid any precedent that would widen its own liability in future disputes.",
    "Secure a second supplier before the existing contract lapses.",
    "Hold its membership together through the next internal election.",
    "Keep grain export tariffs from rising above its margin tolerance.",
    "Retain technical staff that a competitor has been recruiting.",
    "Limit exposure to a single shipping corridor it cannot insure.",
    "Win a budget allocation large enough to finish the pilot programme.",
    "Defend its credit rating against a downgrade it expects next quarter.",
    "Keep a regional office open that headquarters has marked for closure.",
    "Establish a standard its existing equipment already complies with.",
    "Recover arrears owed by a counterparty now in administration.",
    "Prevent a rival from acquiring the only remaining deepwater berth.",
    "Keep its insurance premium inside the band its board approved.",
    "Finish a certification process before the grandfather clause expires.",
    "Protect water rights that an upstream project would reduce.",
)


def make_factor(index: int, **overrides: Any) -> Any:
    from cascade.decompose.schema import Factor

    base: dict[str, Any] = {
        "id": f"factor_{index}",
        "name": FACTOR_NAMES[index % len(FACTOR_NAMES)],
        "state": 0.5,
        "volatility": 0.1,
        "inertia": 0.5,
        "observable_by_default": index % 2 == 0,
    }
    base.update(overrides)
    return Factor(**base)


def make_actor(index: int, *, factor_id: str = "factor_0", **overrides: Any) -> Any:
    from cascade.decompose.schema import Actor, UtilityTerm

    base: dict[str, Any] = {
        "id": f"actor_{index}",
        "name": f"Party {index}",
        "objective": OBJECTIVES[index % len(OBJECTIVES)],
        "utility_terms": (UtilityTerm(factor_id=factor_id, weight=0.5),),
        "resources": {"capital": 0.5},
        "constraints": ("may not act unilaterally",),
        "risk_posture": "neutral",
        "initial_beliefs": {factor_id: 0.5},
    }
    base.update(overrides)
    return Actor(**base)


def make_graph(
    *,
    n_actors: int = 8,
    n_factors: int = 4,
    scenario_id: str = "scenario-1",
    **overrides: Any,
) -> Any:
    """A minimal graph that satisfies every structural §5.3 rule."""
    from cascade.decompose.schema import CausalGraph, Edge, OutcomeRule, OutcomeTerm

    factors = tuple(make_factor(i) for i in range(n_factors))
    actors = tuple(make_actor(i, factor_id=f"factor_{i % n_factors}") for i in range(n_actors))

    edges: list[Any] = [
        Edge(src=actor.id, dst=f"factor_{i % n_factors}", sign=1, weight=0.3, lag=1)
        for i, actor in enumerate(actors)
    ]
    # Chain every factor into factor_0 so each actor reaches the outcome rule.
    edges += [
        Edge(src=f"factor_{i}", dst="factor_0", sign=1, weight=0.2, lag=1)
        for i in range(1, n_factors)
    ]

    base: dict[str, Any] = {
        "scenario_id": scenario_id,
        "actors": actors,
        "factors": factors,
        "edges": tuple(edges),
        "outcome_rule": OutcomeRule(
            terms=(
                OutcomeTerm(factor_id="factor_0", weight=0.8),
                OutcomeTerm(factor_id="factor_1", weight=-0.4),
            ),
            threshold=0.5,
            steepness=6.0,
        ),
    }
    base.update(overrides)
    return CausalGraph(**base)


def fake_embed(distinct: bool = True) -> Any:
    """A deterministic stand-in for the sentence embedder.

    Maps each distinct string to a near-orthogonal unit vector by hashing, so
    unrelated texts score ~0 similarity and identical texts score exactly 1.0.
    That is the property the semantic rules turn on, and it lets them be tested
    without loading a 384-dimension model.
    """
    import hashlib
    import math

    def embed(texts: Any) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.blake2b(text.strip().lower().encode("utf-8"), digest_size=32).digest()
            vector = [((byte / 255.0) - 0.5) for byte in digest]
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out

    return embed
