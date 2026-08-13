"""Langfuse wiring: trace per run, span per step, generation per LLM call.

This module holds the **one sanctioned broad exception guard** in the codebase
(CLAUDE.md §4). Observability must never fail a run: a 36,000-run phase that
dies because a telemetry sidecar restarted has cost real money for nothing.
Every guard here is annotated, narrow in scope, and covered by a test that
asserts the tracer degrades to a no-op rather than propagating.

The rule that makes this safe is that nothing downstream reads back from
Langfuse during a run. The cost ledger is authoritative and lives in Postgres;
Langfuse is reconciled *against* it at M8, so a dropped span costs a
reconciliation warning, never a wrong number.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from cascade.config import Settings
from cascade.llm.types import Usage

__all__ = ["LangfuseTracer", "NullTracer", "Tracer", "build_tracer", "null_tracer"]


class Tracer:
    """No-op tracer. Also the base class, so the null path is the tested path.

    Subclasses override the three hooks. Anything that fails inside an
    override must be swallowed there, not here.
    """

    enabled: bool = False
    degraded_reason: str | None = None

    @contextmanager
    def run(self, *, run_id: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        """Open a trace for one simulation run."""
        del run_id, metadata
        yield

    @contextmanager
    def step(self, *, index: int, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        """Open a span for one step of the 24-step loop."""
        del index, metadata
        yield

    def generation(
        self,
        *,
        name: str,
        model: str,
        usage: Usage,
        cost_usd: Decimal | None,
        cached: bool,
        latency_ms: float,
    ) -> None:
        """Log one model call.

        Cache hits arrive here with ``cost_usd=None`` and ``cached=True`` and
        are logged as zero-cost generations, so hit rate and spend appear on
        the same dashboard (spec §12.2).
        """

    def flush(self) -> None:
        """Push buffered events. Safe to call when nothing is buffered."""


class NullTracer(Tracer):
    """Explicit name for the disabled tracer, for readable diagnostics."""

    def __init__(self, reason: str | None = None) -> None:
        self.degraded_reason = reason


def null_tracer() -> Tracer:
    """Return a tracer that records nothing."""
    return NullTracer()


class LangfuseTracer(Tracer):
    """Reports to a self-hosted Langfuse v2 instance (ADR-0006)."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client
        self._trace: Any | None = None
        self._span: Any | None = None

    @contextmanager
    def run(self, *, run_id: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        try:
            self._trace = self._client.trace(id=run_id, name="cascade.run", metadata=metadata)
        except Exception as exc:  # noqa: BLE001 -- sanctioned: telemetry never fails a run
            self._trace = None
            self.degraded_reason = f"trace open failed: {type(exc).__name__}"
        try:
            yield
        finally:
            self._trace = None

    @contextmanager
    def step(self, *, index: int, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        parent = self._trace
        if parent is not None:
            try:
                self._span = parent.span(name=f"step.{index:02d}", metadata=metadata)
            except Exception as exc:  # noqa: BLE001 -- sanctioned: see module docstring
                self._span = None
                self.degraded_reason = f"span open failed: {type(exc).__name__}"
        try:
            yield
        finally:
            span, self._span = self._span, None
            if span is not None:
                try:
                    span.end()
                except Exception as exc:  # noqa: BLE001 -- sanctioned: see module docstring
                    self.degraded_reason = f"span end failed: {type(exc).__name__}"

    def generation(
        self,
        *,
        name: str,
        model: str,
        usage: Usage,
        cost_usd: Decimal | None,
        cached: bool,
        latency_ms: float,
    ) -> None:
        parent = self._span or self._trace or self._client
        try:
            parent.generation(
                name=name,
                model=model,
                usage={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "unit": "TOKENS",
                    # A cache hit is a real generation that cost nothing. Logging
                    # it as 0.0 rather than omitting it keeps the dashboard's
                    # call count equal to the study's call count.
                    "totalCost": float(cost_usd) if cost_usd is not None else 0.0,
                },
                metadata={
                    "cached": cached,
                    "latency_ms": latency_ms,
                    "cache_read_input_tokens": usage.cache_read_input_tokens,
                    "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001 -- sanctioned: see module docstring
            self.degraded_reason = f"generation failed: {type(exc).__name__}"

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001 -- sanctioned: see module docstring
            self.degraded_reason = f"flush failed: {type(exc).__name__}"


def build_tracer(settings: Settings) -> Tracer:
    """Return a live tracer when Langfuse is configured, a no-op otherwise.

    Preserves the invariant that observability is optional at runtime and
    mandatory in configuration: a missing key degrades to a no-op with a
    stated reason, so ``doctor`` can report it, rather than silently
    pretending tracing is on.
    """
    if not settings.langfuse.enabled:
        return NullTracer("disabled in config")
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return NullTracer("langfuse keys not set in the environment")
    try:
        from langfuse import Langfuse  # optional dependency, probed rather than required

        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse.host,
        )
    except Exception as exc:  # noqa: BLE001 -- sanctioned: telemetry never fails a run
        return NullTracer(f"langfuse unavailable: {type(exc).__name__}: {exc}"[:160])
    return LangfuseTracer(client)
