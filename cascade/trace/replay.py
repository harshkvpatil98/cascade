"""Re-running a stored run and proving it came back identical (spec §8.4; M8).

M8's first criterion is a **byte-identical event-log hash across processes**
over 25 runs. Two halves of that matter separately.

*Byte-identical* is asserted against the hash the original run stored in
``runs.event_log_hash``, not against a second in-process run. A hash compared
only with itself proves the hash function is a function.

*Across processes* is the half that finds real defects. Within one process a
dependency on dict insertion order, on ``id()``, or on a module-level cache is
perfectly stable; ``PYTHONHASHSEED`` differs between processes, so a set
iterated for a tie-break diverges there and nowhere else. Each replay
therefore runs in its own interpreter, under a hash seed chosen to differ from
the parent's.

When a run does diverge, the answer "the hashes differ" is useless. §8.4's
rule is that the bisect names the first divergent field, so a mismatch is
localised to a step by comparing the stored per-step ``state_hash`` sequence
against the replayed one, and the first index that differs is reported with
both values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from cascade.config import Settings

__all__ = [
    "ReplayOutcome",
    "ReplayReport",
    "load_replay_targets",
    "replay_in_process",
    "verify_replays",
]

Role = Literal["admin", "sim", "eval"]

# The child's PYTHONHASHSEED. Any fixed value differs from the parent's random
# one, which is the whole point; fixing it keeps the *child* reproducible so a
# divergence is a property of the code rather than of the run that found it.
CHILD_HASH_SEED = "12345"


@dataclass(frozen=True, slots=True)
class ReplayTarget:
    """One stored run, and the hash it must reproduce."""

    run_id: str
    scenario_id: str
    config_id: str
    replicate: int
    policy: str
    event_log_hash: str
    state_hashes: tuple[str, ...]
    outcome_score: float
    decisions: int


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """What one replay produced, and where it first differed if it did."""

    run_id: str
    expected_hash: str
    actual_hash: str | None
    first_divergent_step: int | None = None
    expected_state_hash: str | None = None
    actual_state_hash: str | None = None
    error: str | None = None

    @property
    def matched(self) -> bool:
        return self.error is None and self.actual_hash == self.expected_hash


@dataclass
class ReplayReport:
    """The M8 criterion, measured."""

    outcomes: list[ReplayOutcome] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def attempted(self) -> int:
        return len(self.outcomes)

    @property
    def matched(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.matched)

    @property
    def diverged(self) -> tuple[ReplayOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.matched)

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and not self.diverged


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


def load_replay_targets(
    settings: Settings, *, limit: int, config_id: str | None = None, role: Role = "eval"
) -> tuple[ReplayTarget, ...]:
    """The runs to replay, chosen deterministically.

    Ordered by ``run_id`` and taken from the front rather than sampled: the
    criterion is "25 runs replay identically", and a random 25 would make a
    failure depend on which 25 the command happened to draw. The same 25 every
    time is what makes a green result mean the same thing twice.
    """
    clause = "WHERE config_id = %(config_id)s" if config_id else ""
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT run_id, scenario_id, config_id, replicate, policy,
                   event_log_hash, outcome_score, decisions
            FROM runs {clause}
            ORDER BY run_id
            LIMIT %(limit)s
            """,  # noqa: S608 -- `clause` is a literal chosen above, not input
            {"config_id": config_id, "limit": limit},
        )
        rows = cur.fetchall()

        targets: list[ReplayTarget] = []
        for run_id, scenario_id, cfg, replicate, policy, log_hash, outcome, decisions in rows:
            cur.execute(
                "SELECT state_hash FROM run_steps WHERE run_id = %s ORDER BY step",
                (run_id,),
            )
            targets.append(
                ReplayTarget(
                    run_id=str(run_id),
                    scenario_id=str(scenario_id),
                    config_id=str(cfg),
                    replicate=int(replicate),
                    policy=str(policy),
                    event_log_hash=str(log_hash),
                    state_hashes=tuple(str(row[0]) for row in cur.fetchall()),
                    outcome_score=float(outcome),
                    decisions=int(decisions),
                )
            )
    return tuple(targets)


def replay_in_process(settings: Settings, target: ReplayTarget) -> dict[str, Any]:
    """Re-simulate one run here and return its hashes. Writes nothing.

    The replay must not touch the event log: invariant 6 makes it append-only,
    and a verification pass that inserted its own copy would make the M6 event
    count drift every time someone checked the M8 criterion.
    """
    from cascade.aperture.policy import derive_policies
    from cascade.cli import _graph_for
    from cascade.ledger.store import load_scenarios
    from cascade.sim.kernel import Loom, RunSpec, mechanics_from
    from cascade.sim.policies import HeuristicPolicy

    cell = load_settings_for(target.config_id)
    graph = _graph_for(cell, target.scenario_id)
    if graph is None:
        raise LookupError(
            f"scenario {target.scenario_id!r} has no compiled graph, so the run that "
            "produced this hash cannot be reconstructed; run `cascade compile build`"
        )
    scenarios = {item.scenario_id: item for item in load_scenarios(cell, role="admin")}
    if target.scenario_id not in scenarios:
        raise LookupError(f"scenario {target.scenario_id!r} is not in the registry")

    policies = derive_policies(graph, cell.aperture, asymmetry=cell.flags.information_asymmetry)
    if target.policy == "heuristic":
        mechanics = mechanics_from(graph)
        decider: Any = HeuristicPolicy(
            utility={actor: dict(mechanics[actor].utility) for actor in sorted(mechanics)}
        )
    else:
        from cascade.cli import _agent_policy

        decider = _agent_policy(cell, [(scenarios[target.scenario_id], graph)])

    loom = Loom(settings=cell, graph=graph, policies=policies, decider=decider)
    result = loom.run(
        RunSpec(
            run_id=target.run_id,
            scenario_id=target.scenario_id,
            config_id=target.config_id,
            replicate=target.replicate,
            policy=target.policy,
        )
    )
    return {
        "run_id": target.run_id,
        "event_log_hash": result.event_log_hash,
        "state_hashes": [record.state_hash for record in result.step_records],
        "outcome_score": result.outcome_score,
        "decisions": result.decisions,
    }


def load_settings_for(config_id: str) -> Settings:
    """Load the settings a run was made under.

    A run records the ablation cell it belongs to, and replaying it under the
    base configuration would re-run a *different experiment* -- with a
    different visibility policy, a different graph arm, or different grounding
    -- and report the resulting mismatch as non-determinism.
    """
    from cascade.config import load_settings

    overlay = None if config_id in {"base", ""} else config_id
    try:
        return load_settings(overlay)
    except FileNotFoundError:
        return load_settings(None)


def _child_command(run_id: str) -> list[str]:
    return [sys.executable, "-m", "cascade.trace.replay_child", run_id]


def _replay_in_subprocess(target: ReplayTarget, env_file: str | None = None) -> ReplayOutcome:
    """Run one replay in its own interpreter and compare what comes back.

    The child's environment comes from :func:`cascade.config.child_environment`
    rather than from ``os.environ`` here: only ``config.py`` reads the process
    environment (a test enforces it), and a child that re-derived its settings
    location from ambient environment would work from a shell and fail wherever
    that environment is deliberately controlled -- which is exactly where
    determinism is checked.
    """
    from cascade.config import child_environment

    env = child_environment(PYTHONHASHSEED=CHILD_HASH_SEED)
    if env_file is not None:
        env["CASCADE_ENV_FILE"] = env_file
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            _child_command(target.run_id),
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ReplayOutcome(
            run_id=target.run_id,
            expected_hash=target.event_log_hash,
            actual_hash=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    if completed.returncode != 0:
        return ReplayOutcome(
            run_id=target.run_id,
            expected_hash=target.event_log_hash,
            actual_hash=None,
            error=(completed.stderr or completed.stdout).strip()[-400:] or "child exited non-zero",
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return ReplayOutcome(
            run_id=target.run_id,
            expected_hash=target.event_log_hash,
            actual_hash=None,
            error=f"child produced no parseable result: {type(exc).__name__}: {exc}",
        )

    actual_hash = str(payload["event_log_hash"])
    if actual_hash == target.event_log_hash:
        return ReplayOutcome(
            run_id=target.run_id,
            expected_hash=target.event_log_hash,
            actual_hash=actual_hash,
        )

    # §8.4: the bisect names the first divergent field. Comparing the per-step
    # state hashes turns "these runs differ" into "they differ from step 11",
    # which is the difference between a diagnosis and a symptom.
    replayed = [str(value) for value in payload.get("state_hashes", [])]
    first: int | None = None
    expected_state = actual_state = None
    for index in range(max(len(target.state_hashes), len(replayed))):
        stored = target.state_hashes[index] if index < len(target.state_hashes) else None
        fresh = replayed[index] if index < len(replayed) else None
        if stored != fresh:
            first, expected_state, actual_state = index, stored, fresh
            break

    return ReplayOutcome(
        run_id=target.run_id,
        expected_hash=target.event_log_hash,
        actual_hash=actual_hash,
        first_divergent_step=first,
        expected_state_hash=expected_state,
        actual_state_hash=actual_state,
    )


def verify_replays(
    targets: Sequence[ReplayTarget], *, workers: int = 4, env_file: str | None = None
) -> ReplayReport:
    """Replay every target in its own process and compare the hashes.

    Concurrency is over *processes*, which is safe because a replay writes
    nothing: it re-simulates and returns hashes. Results are collected in
    target order so a report reads the same way twice (invariant 7).
    """
    import time

    started = time.perf_counter()
    if not targets:
        return ReplayReport(outcomes=[], elapsed_s=0.0)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(lambda target: _replay_in_subprocess(target, env_file), targets))
    return ReplayReport(outcomes=outcomes, elapsed_s=time.perf_counter() - started)
