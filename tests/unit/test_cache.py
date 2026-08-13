"""Content-addressed cache: key derivation and the on-disk store (spec §8.3).

The key is a correctness surface, not an optimisation. Two properties are
being asserted here and they pull in opposite directions:

* *stability* -- an identical request must yield an identical key, across
  processes, hash seeds and dict insertion orders, or the hit rate collapses
  and the study stops being affordable;
* *sensitivity* -- any field that can change the response must change the key,
  or replay serves the wrong recording and nothing ever notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cascade.llm.cache import CallCache, cache_key, canonical_json
from cascade.llm.types import CachedCall, LLMRequest, Usage

MODEL = "claude-haiku-4-5-20251001"


def make_request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "model": MODEL,
        "system": "You are actor A.",
        "messages": [{"role": "user", "content": "step 1"}],
        "tools": None,
        "temperature": 0.7,
        "max_tokens": 512,
        "prompt_rev": "r1",
    }
    base.update(overrides)
    return LLMRequest(**base)


def make_call(key: str) -> CachedCall:
    return CachedCall(
        key=key,
        request_digest={"model": MODEL},
        raw_response={
            "id": "msg_01TEST",
            "model": MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        },
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=12.5,
        recorded_at="2026-08-13T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------


def test_canonical_json_is_insertion_order_independent() -> None:
    """Dict ordering must not reach the hash (invariant 7's cousin)."""
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_escapes_non_ascii() -> None:
    """The byte sequence must not depend on the filesystem or locale encoding."""
    encoded = canonical_json({"text": "café"})
    assert encoded.isascii()
    assert json.loads(encoded)["text"] == "café"


def test_canonical_json_rejects_non_finite_floats() -> None:
    """``NaN`` is not JSON and does not compare equal to itself.

    Allowing it would produce a key that never matches its own recording --
    a permanent, silent cache miss.
    """
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json({"x": float("nan")})


def test_canonical_json_has_no_incidental_whitespace() -> None:
    assert canonical_json({"a": [1, 2]}) == '{"a":[1,2]}'


# ---------------------------------------------------------------------------
# Key stability
# ---------------------------------------------------------------------------


def test_identical_requests_share_a_key() -> None:
    assert cache_key(make_request()) == cache_key(make_request())


def test_key_is_a_sha256_hex_digest() -> None:
    key = cache_key(make_request())
    assert len(key) == 64
    assert all(char in "0123456789abcdef" for char in key)


def test_cache_control_markers_do_not_change_the_key() -> None:
    """ADR-0007: prompt-cache tuning is a pure cost change.

    If the marker were in the key, every cache-boundary experiment would
    invalidate the recorded corpus and demand a paid re-record to measure a
    change that cannot alter a single output token.
    """
    plain = make_request(
        system=[{"type": "text", "text": "You are actor A."}],
    )
    marked = make_request(
        system=[
            {
                "type": "text",
                "text": "You are actor A.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    )
    assert cache_key(plain) == cache_key(marked)


def test_cache_control_is_stripped_at_every_depth() -> None:
    """Markers appear on message content blocks and tool definitions too."""
    plain = make_request(
        messages=[{"role": "user", "content": [{"type": "text", "text": "step 1"}]}],
    )
    marked = make_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "step 1",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )
    assert cache_key(plain) == cache_key(marked)


# ---------------------------------------------------------------------------
# Key sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "claude-sonnet-4-6"),
        ("system", "You are actor B."),
        ("messages", [{"role": "user", "content": "step 2"}]),
        ("tools", [{"name": "act", "input_schema": {"type": "object"}}]),
        ("temperature", 0.2),
        ("prompt_rev", "r2"),
        ("max_tokens", 1024),
    ],
)
def test_every_field_in_the_key_domain_changes_the_key(field: str, value: Any) -> None:
    """Sensitivity: serving a recording made under different inputs is silent.

    ``max_tokens`` is included beyond the spec's listing because it bounds the
    response -- sharing a key across two limits would serve a truncated
    completion as if it were whole.
    """
    assert cache_key(make_request()) != cache_key(make_request(**{field: value}))


def test_the_key_domain_is_exactly_the_documented_fields() -> None:
    """A field added to the request must be a deliberate corpus invalidation."""
    assert sorted(make_request().cache_domain()) == [
        "max_tokens",
        "messages",
        "model",
        "prompt_rev",
        "system",
        "temperature",
        "tools",
    ]


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    cache.put(make_call(key))

    fetched = cache.get(key)
    assert fetched is not None
    assert fetched.key == key
    assert fetched.usage.input_tokens == 10
    assert fetched.raw_response["content"][0]["text"] == "ok"


def test_miss_returns_none_and_is_counted(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)
    assert cache.get("0" * 64) is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
    assert cache.stats.hit_rate == 0.0


def test_hit_rate_is_measured(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    cache.put(make_call(key))
    for _ in range(9):
        cache.get(key)
    cache.get("f" * 64)
    assert cache.stats.hit_rate == pytest.approx(0.9)


def test_entries_are_sharded_on_the_key_prefix(tmp_path: Path) -> None:
    """A flat directory of ~400k recordings is slow to enumerate everywhere."""
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    cache.put(make_call(key))
    assert cache.path_for(key) == tmp_path / key[:2] / key[2:4] / f"{key}.json"
    assert cache.path_for(key).is_file()


def test_writing_the_same_key_twice_is_idempotent(tmp_path: Path) -> None:
    """Re-recording an already-recorded call must not fork the store."""
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    cache.put(make_call(key))
    cache.put(make_call(key))
    assert cache.count() == 1


def test_no_temporary_files_survive_a_write(tmp_path: Path) -> None:
    """Atomic writes must not leave debris a later rglob would pick up."""
    cache = CallCache(tmp_path)
    cache.put(make_call(cache_key(make_request())))
    assert [p.name for p in tmp_path.rglob("*.tmp")] == []


def test_misfiled_entry_is_reported_not_served(tmp_path: Path) -> None:
    """A hand-edited store must fail loudly rather than serve the wrong call."""
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_call("a" * 64).model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted or hand-edited"):
        cache.get(key)


def test_corrupt_entry_is_reported_not_treated_as_a_miss(tmp_path: Path) -> None:
    """Silently downgrading corruption to a miss would let replay 'succeed'."""
    cache = CallCache(tmp_path)
    key = cache_key(make_request())
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        cache.get(key)


def test_count_of_an_absent_store_is_zero(tmp_path: Path) -> None:
    assert CallCache(tmp_path / "never-created").count() == 0
