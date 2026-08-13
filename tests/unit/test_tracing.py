"""Langfuse tracing, and an audit of the one sanctioned broad exception guard.

Observability must never fail a run: a 36,000-run phase that dies because a
telemetry sidecar restarted has spent real money for nothing. So
``cascade/llm/tracing.py`` swallows everything Langfuse throws.

That licence is dangerous exactly where it is convenient, so the second half
of this module is a static audit: broad guards are forbidden in the LLM
client, the cost meter and the cache, and every one that does exist must be
annotated and must not be a silent ``pass``.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cascade.config import Settings
from cascade.llm.tracing import LangfuseTracer, NullTracer, build_tracer, null_tracer
from cascade.llm.types import Usage

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "cascade"
SANCTIONED_GUARD_MODULE = PACKAGE_ROOT / "llm" / "tracing.py"

# Broad guards are forbidden outright in the modules that handle money,
# recordings and (from M6) the event log.
NO_BROAD_GUARDS = [
    PACKAGE_ROOT / "llm" / "client.py",
    PACKAGE_ROOT / "llm" / "meter.py",
    PACKAGE_ROOT / "llm" / "cache.py",
]

USAGE = Usage(input_tokens=260, output_tokens=45, cache_read_input_tokens=1900)


class ExplodingClient:
    """A Langfuse client where every call fails. Nothing may escape."""

    def __init__(self) -> None:
        self.calls = 0

    def trace(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise RuntimeError("langfuse is down")

    def span(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise RuntimeError("langfuse is down")

    def generation(self, **kwargs: Any) -> Any:
        self.calls += 1
        raise RuntimeError("langfuse is down")

    def flush(self) -> None:
        self.calls += 1
        raise RuntimeError("langfuse is down")


class RecordingClient:
    """Captures what would have been sent, so the payload can be asserted on."""

    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []

    def trace(self, **kwargs: Any) -> RecordingClient:
        self.traces.append(kwargs)
        return self

    def span(self, **kwargs: Any) -> RecordingClient:
        return self

    def generation(self, **kwargs: Any) -> None:
        self.generations.append(kwargs)

    def end(self) -> None:
        return None

    def flush(self) -> None:
        return None


# ---------------------------------------------------------------------------
# The no-op path
# ---------------------------------------------------------------------------


def test_null_tracer_accepts_the_full_protocol() -> None:
    """The null path is the default path, so it must be exercised."""
    tracer = null_tracer()
    assert not tracer.enabled
    with tracer.run(run_id="r1"), tracer.step(index=3):
        tracer.generation(
            name="llm.complete",
            model="claude-haiku-4-5-20251001",
            usage=USAGE,
            cost_usd=Decimal("0.000338"),
            cached=False,
            latency_ms=12.0,
        )
    tracer.flush()


def test_build_tracer_is_a_no_op_when_disabled(settings: Settings) -> None:
    disabled = settings.model_copy(
        update={"langfuse": settings.langfuse.model_copy(update={"enabled": False})}
    )
    tracer = build_tracer(disabled)
    assert isinstance(tracer, NullTracer)
    assert tracer.degraded_reason == "disabled in config"


def test_build_tracer_states_why_it_degraded(settings: Settings) -> None:
    """A missing key must be reported, not silently pretend tracing is on."""
    tracer = build_tracer(settings)
    assert isinstance(tracer, NullTracer)
    assert tracer.degraded_reason is not None
    assert "keys not set" in tracer.degraded_reason


# ---------------------------------------------------------------------------
# The live path degrades rather than propagating
# ---------------------------------------------------------------------------


def test_a_broken_backend_never_fails_a_run() -> None:
    """Every Langfuse call raises; the caller must not see any of it."""
    client = ExplodingClient()
    tracer = LangfuseTracer(client)

    with tracer.run(run_id="r1"), tracer.step(index=1):
        tracer.generation(
            name="llm.complete",
            model="claude-haiku-4-5-20251001",
            usage=USAGE,
            cost_usd=Decimal("0.000338"),
            cached=False,
            latency_ms=12.0,
        )
    tracer.flush()

    assert client.calls > 0, "the failures must have actually been triggered"
    assert tracer.degraded_reason is not None


def test_degradation_is_reported_not_hidden() -> None:
    """Swallowed is not the same as unnoticed; doctor reads this field."""
    tracer = LangfuseTracer(ExplodingClient())
    tracer.flush()
    assert tracer.degraded_reason == "flush failed: RuntimeError"


def test_cache_hits_are_logged_as_zero_cost_generations() -> None:
    """Hit rate and spend must appear on the same dashboard (spec §12.2)."""
    client = RecordingClient()
    tracer = LangfuseTracer(client)

    with tracer.run(run_id="r1"):
        tracer.generation(
            name="llm.complete",
            model="claude-haiku-4-5-20251001",
            usage=USAGE,
            cost_usd=None,
            cached=True,
            latency_ms=0.1,
        )

    assert len(client.generations) == 1
    logged = client.generations[0]
    assert logged["usage"]["totalCost"] == 0.0
    assert logged["metadata"]["cached"] is True
    # The tokens are still reported, so a cached call is visible as a call.
    assert logged["usage"]["input"] == 260


def test_billed_generations_carry_their_cost() -> None:
    client = RecordingClient()
    tracer = LangfuseTracer(client)
    with tracer.run(run_id="r1"):
        tracer.generation(
            name="llm.complete",
            model="claude-haiku-4-5-20251001",
            usage=USAGE,
            cost_usd=Decimal("0.000338"),
            cached=False,
            latency_ms=12.0,
        )
    assert client.generations[0]["usage"]["totalCost"] == pytest.approx(0.000338)


# ---------------------------------------------------------------------------
# Static audit of broad exception guards
# ---------------------------------------------------------------------------


def _broad_handlers(path: Path) -> list[tuple[int, bool, bool]]:
    """Return ``(lineno, is_annotated, is_silent_pass)`` for each *swallowing*
    broad handler.

    A handler that re-raises is not a guard and is not counted. The standard
    forbids *swallowing* an error near money or recordings, not catching one:
    the ``except BaseException: cleanup; raise`` around an atomic temp-file
    write deletes the partial file and lets the failure propagate untouched,
    which is the behaviour the standard wants rather than an exception to it.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found: list[tuple[int, bool, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        is_broad = caught is None or (
            isinstance(caught, ast.Name) and caught.id in {"Exception", "BaseException"}
        )
        if not is_broad:
            continue
        if any(isinstance(statement, ast.Raise) for statement in node.body):
            continue
        annotated = "noqa: BLE001" in lines[node.lineno - 1]
        silent = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
        found.append((node.lineno, annotated, silent))
    return found


def test_the_broad_handler_detector_distinguishes_swallowing_from_cleanup() -> None:
    """Guard the guard: the detector must not be blind, nor cry wolf."""
    import tempfile

    cases = [
        ("try:\n    x()\nexcept Exception:\n    pass\n", 1),
        ("try:\n    x()\nexcept Exception:\n    log()\n", 1),
        ("try:\n    x()\nexcept Exception:\n    cleanup()\n    raise\n", 0),
        ("try:\n    x()\nexcept BaseException:\n    cleanup()\n    raise\n", 0),
        ("try:\n    x()\nexcept ValueError:\n    pass\n", 0),
        ("try:\n    x()\nexcept:\n    pass\n", 1),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for index, (source, expected) in enumerate(cases):
            probe = Path(directory) / f"probe{index}.py"
            probe.write_text(source, encoding="utf-8")
            assert len(_broad_handlers(probe)) == expected, source


@pytest.mark.parametrize("module", NO_BROAD_GUARDS, ids=lambda p: p.name)
def test_no_broad_guard_near_money_or_recordings(module: Path) -> None:
    """A swallowed error here would corrupt the ledger or the replay corpus.

    Both failures are silent by construction: the meter would under-report and
    the cache would serve a partial recording, and the first check that would
    notice either is the M8 reconciliation.
    """
    assert _broad_handlers(module) == [], (
        f"{module.name} must not contain a broad exception guard; "
        "the only sanctioned one is in llm/tracing.py"
    )


def test_the_sanctioned_guards_are_annotated() -> None:
    """Every broad guard in the package must be marked and justified."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for lineno, annotated, _ in _broad_handlers(path)
        if not annotated
    ]
    assert offenders == [], f"unannotated broad exception guards: {offenders}"


def test_no_broad_guard_is_a_silent_pass() -> None:
    """``except Exception: pass`` is the anti-pattern the spec names by name."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for lineno, _, silent in _broad_handlers(path)
        if silent
    ]
    assert offenders == [], f"broad guards that swallow silently: {offenders}"


def test_tracing_actually_contains_the_sanctioned_guards() -> None:
    """Guard the guard: a tracer that stopped guarding would pass every test above.

    It would also propagate a Langfuse outage into a 36,000-run phase, which
    is the failure the licence exists to prevent.
    """
    handlers = _broad_handlers(SANCTIONED_GUARD_MODULE)
    assert len(handlers) >= 4, "tracing.py must guard trace, span, generation and flush"
    assert all(annotated for _, annotated, _ in handlers)
