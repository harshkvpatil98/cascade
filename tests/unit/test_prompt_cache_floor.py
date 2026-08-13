"""The cacheable-prefix gate (ADR-0001).

Anthropic silently declines to cache a prefix below the model's floor: the
request succeeds, ``cache_creation_input_tokens`` comes back 0, and every
prefix token is billed at list price. The spec's §12.1 model assumes a
1,900-token prefix billed at the 10% cached-read rate, which puts it below
Haiku 4.5's 4,096-token floor and makes the input-cost model a ~4.8x
underestimate.

There is no prompt yet -- prompts are authored at M4/M5 -- so what is tested
here is the gate itself, before the code it will gate exists.
"""

from __future__ import annotations

import pytest

from cascade.config import Settings
from cascade.llm.client import assert_cacheable_prefix, estimate_tokens
from cascade.llm.types import PromptTooShortToCache


def test_the_configured_floor_is_the_model_floor(settings: Settings) -> None:
    assert settings.prompt_cache.min_prefix_tokens == 4096
    assert settings.prompt_cache.enforce_min_prefix is True
    assert settings.prompt_cache.enabled is True


def test_ttl_is_one_hour(settings: Settings) -> None:
    """Batches routinely lag past the 5m default; an expired cache never wrote."""
    assert settings.prompt_cache.ttl == "1h"


def test_a_short_prefix_is_rejected(settings: Settings) -> None:
    """The spec's own 1,900-token prefix must fail this gate.

    This is the defect ADR-0001 records, expressed as a test: if this ever
    passes, the cost model has been quietly restored to its broken form.
    """
    spec_prefix = "x" * int(1900 * 3.6)
    assert estimate_tokens(spec_prefix) == pytest.approx(1900, abs=2)
    with pytest.raises(PromptTooShortToCache) as caught:
        assert_cacheable_prefix(spec_prefix, settings)
    assert caught.value.minimum_tokens == 4096
    assert "silently not cache" in str(caught.value)


def test_a_long_enough_prefix_passes(settings: Settings) -> None:
    tokens = assert_cacheable_prefix("y" * int(5000 * 3.6), settings)
    assert tokens >= settings.prompt_cache.min_prefix_tokens


def test_an_exact_count_overrides_the_estimate(settings: Settings) -> None:
    """The estimate is a fallback; a real tokenizer count wins when available."""
    assert assert_cacheable_prefix("short", settings, exact_tokens=8192) == 8192
    with pytest.raises(PromptTooShortToCache):
        assert_cacheable_prefix("x" * 100_000, settings, exact_tokens=10)


def test_the_gate_can_be_disabled_deliberately(settings: Settings) -> None:
    """Opting out is a config change, so it appears in the run manifest."""
    relaxed = settings.model_copy(
        update={
            "prompt_cache": settings.prompt_cache.model_copy(update={"enforce_min_prefix": False})
        }
    )
    assert assert_cacheable_prefix("short", relaxed) == estimate_tokens("short")


def test_the_gate_is_inert_when_caching_is_off(settings: Settings) -> None:
    disabled = settings.model_copy(
        update={"prompt_cache": settings.prompt_cache.model_copy(update={"enabled": False})}
    )
    assert assert_cacheable_prefix("short", disabled) == estimate_tokens("short")


def test_the_estimator_is_never_used_for_billing() -> None:
    """Guard the boundary the docstring claims.

    ``estimate_tokens`` gates the prefix and nothing else; every dollar comes
    from the provider's reported usage. A grep is the enforcement because the
    alternative -- an estimated cost that looks plausible -- is undetectable.
    """
    import ast
    from pathlib import Path

    meter_source = (Path(__file__).resolve().parents[2] / "cascade" / "llm" / "meter.py").read_text(
        encoding="utf-8"
    )
    names = {node.id for node in ast.walk(ast.parse(meter_source)) if isinstance(node, ast.Name)}
    assert "estimate_tokens" not in names
    assert "estimate_tokens" not in meter_source
