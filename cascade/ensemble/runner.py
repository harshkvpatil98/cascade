"""The fan-out runner (spec §2.2 PHASE 2, §12.2). Thin shell over a pure kernel.

36,000 runs is 180 scenarios x 200 replicates, and the only reason that is
affordable is that most of the decisions inside it have been made before. Two
mechanisms do the work, and the runner exists to let them:

**The wavefront.** Runs advance in lockstep, a step at a time. Everything up to
and including OBSERVE is deterministic and model-free, so a whole wave's turns
can be assembled before any of them is decided -- and then resolved in one
batch at the 50% rate §12.1 assumes. Deciding run-by-run instead would submit
one request at a time to an API whose SLA is measured in hours.

**The cache.** Replicates of one scenario share their early steps exactly:
until the seeded noise pushes them apart, the same actor sees the same rounded
observation and produces the same request bytes. Those are served from disk and
never reach the batch. The M6 criterion measures that as the action-cache hit
rate; here it simply means the batch is small.

Wave size is the operator's dial and the docstring on :meth:`execute` states
the arithmetic. A larger wave means fewer, larger batch submissions -- the
difference between 24 submissions for the whole study and 4,320 -- and more
live runs held in memory. Neither end is free and the right value depends on
the machine, so it is a parameter with a stated trade-off rather than a
constant someone would have to rediscover.

Resumability (invariant 8) is a set difference, not a checkpoint format: a run
row exists only for a run that finished, so the work left to do is every
(scenario, replicate) without one. An interrupted wave loses only the runs that
had not reached their horizon.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cascade.config import Settings
from cascade.sim.kernel import Loom, PendingTurn, RunHandle, RunResult, RunSpec

__all__ = [
    "EnsembleRunner",
    "FanoutReport",
    "RunTask",
    "ScenarioRunner",
    "run_id_for",
]


def run_id_for(scenario_id: str, config_id: str, replicate: int) -> str:
    """The deterministic run id for one (scenario, config, replicate).

    Derived rather than random so a re-run of an interrupted phase produces the
    same id for the same unit of work. A random uuid4 here would make the
    append-only event log accumulate a second copy of every re-simulated run,
    and the M6 count criterion would drift upward with every restart.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cascade/{scenario_id}/{config_id}/{replicate}"))


class RunTask(BaseModel):
    """One unit of work in the fan-out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    config_id: str
    replicate: int = Field(ge=0)

    def spec(self, *, policy: str) -> RunSpec:
        return RunSpec(
            run_id=run_id_for(self.scenario_id, self.config_id, self.replicate),
            scenario_id=self.scenario_id,
            config_id=self.config_id,
            replicate=self.replicate,
            policy=policy,
        )


class FanoutReport(BaseModel):
    """What a fan-out actually did. Every figure measured, none configured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: int
    completed: int
    skipped: int
    decisions: int
    llm_calls: int
    cache_hits: int
    batches: int
    batched_turns: int
    waves: int
    steps: int
    elapsed_s: float

    @property
    def cache_hit_rate(self) -> float:
        """The M6 criterion's >= 88%. Measured over decisions, not over runs."""
        return self.cache_hits / self.llm_calls if self.llm_calls else 0.0

    @property
    def decisions_per_run(self) -> float:
        return self.decisions / self.completed if self.completed else 0.0


class ScenarioRunner(Protocol):
    """Builds the kernel for one scenario. Injected so the runner does no I/O."""

    def __call__(self, scenario_id: str) -> Loom: ...


@dataclass
class EnsembleRunner:
    """Drives a fan-out of runs, batching each step's decisions across the wave.

    ``loom_for`` and ``on_complete`` are injected: loading a graph and writing
    an event log are I/O, and keeping them out of this class is what lets the
    wavefront be tested end to end without a database.
    """

    settings: Settings
    loom_for: ScenarioRunner
    on_complete: Callable[[RunResult], None]
    policy: str = "agent"
    completed: Callable[[str, str], set[int]] | None = None
    """(scenario_id, config_id) -> replicates already stored. Absent means none."""

    _looms: dict[str, Loom] = field(default_factory=dict)

    # -- planning -----------------------------------------------------------

    def plan(
        self, *, scenario_ids: Sequence[str], config_id: str, replicates: int
    ) -> tuple[list[RunTask], int]:
        """Return the work left to do, and how much was already stored.

        Invariant 8 in one line: the plan is the full cross product minus what
        the runs table already holds.
        """
        tasks: list[RunTask] = []
        skipped = 0
        for scenario_id in sorted(scenario_ids):
            done = self.completed(scenario_id, config_id) if self.completed else set()
            for replicate in range(replicates):
                if replicate in done:
                    skipped += 1
                    continue
                tasks.append(
                    RunTask(scenario_id=scenario_id, config_id=config_id, replicate=replicate)
                )
        return tasks, skipped

    # -- execution ----------------------------------------------------------

    def execute(self, tasks: Sequence[RunTask], *, wave: int = 200) -> FanoutReport:
        """Run every task, advancing ``wave`` runs in lockstep.

        On wave size. A wave of *w* runs submits one batch per step carrying
        roughly ``w x 4.86 x (1 - hit rate)`` requests, and the study needs
        ``24 x ceil(total / w)`` submissions in total. At w = 200 that is 4,320
        submissions for 36,000 runs; at w = 36,000 it is 24. Larger waves mean
        far fewer round trips against an API whose SLA is hours, and more live
        runs in memory -- a run holds its world, its agents' memories and its
        events until it finishes, on the order of 70 KB. Pick the largest wave
        the machine can hold.
        """
        started = time.perf_counter()
        report = _Tally()
        for chunk in _chunks(tasks, max(1, wave)):
            self._run_wave(chunk, report)
        return FanoutReport(
            tasks=len(tasks),
            completed=report.completed,
            skipped=0,
            decisions=report.decisions,
            llm_calls=report.llm_calls,
            cache_hits=report.cache_hits,
            batches=report.batches,
            batched_turns=report.batched_turns,
            waves=report.waves,
            steps=report.steps,
            elapsed_s=time.perf_counter() - started,
        )

    def _loom(self, scenario_id: str) -> Loom:
        loom = self._looms.get(scenario_id)
        if loom is None:
            loom = self.loom_for(scenario_id)
            self._looms[scenario_id] = loom
        return loom

    def _run_wave(self, tasks: Sequence[RunTask], report: _Tally) -> None:
        """Advance one wave of runs to completion, a step at a time."""
        live: list[tuple[Loom, RunHandle]] = []
        for task in sorted(tasks, key=lambda t: (t.scenario_id, t.replicate)):
            loom = self._loom(task.scenario_id)
            live.append((loom, loom.start(task.spec(policy=self.policy))))
        report.waves += 1

        while live:
            turns: list[PendingTurn] = []
            for loom, handle in live:
                turns.extend(loom.observe_step(handle))
            report.steps += 1

            if turns:
                by_decider: dict[int, tuple[Any, list[PendingTurn]]] = {}
                for loom, handle in live:
                    key = id(loom.decider)
                    bucket = by_decider.setdefault(key, (loom.decider, []))
                    bucket[1].extend(turn for turn in turns if turn.run_id == handle.spec.run_id)
                for key in sorted(by_decider):
                    decider, owned = by_decider[key]
                    prepared = _prepare(decider, owned)
                    if prepared:
                        report.batches += 1
                        report.batched_turns += prepared

            for loom, handle in live:
                loom.complete_step(handle)

            finished = [(loom, handle) for loom, handle in live if handle.done]
            live = [(loom, handle) for loom, handle in live if not handle.done]
            for loom, handle in finished:
                result = loom.result(handle)
                report.completed += 1
                report.decisions += result.decisions
                report.llm_calls += result.llm_calls
                report.cache_hits += result.cache_hits
                self.on_complete(result)


def _prepare(decider: Any, turns: Sequence[PendingTurn]) -> int:
    """Resolve a group of turns up front where the decider supports it.

    Grouped by decider rather than assumed shared: one model-backed decider
    serving every scenario batches a whole wave in one submission, while the
    per-scenario stand-in has no ``prepare`` and simply decides turn by turn.
    Both are correct; only the first is cheap.
    """
    prepare = getattr(decider, "prepare", None)
    if prepare is None or not turns:
        return 0
    count = prepare(turns)
    return int(count) if count else 0


@dataclass
class _Tally:
    completed: int = 0
    decisions: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    batches: int = 0
    batched_turns: int = 0
    waves: int = 0
    steps: int = 0


def _chunks(items: Sequence[RunTask], size: int) -> Iterable[Sequence[RunTask]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
