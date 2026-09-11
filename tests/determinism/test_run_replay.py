"""A run replays byte-identically, in this process and in another (spec §8).

M8's criterion is a byte-identical event-log hash across processes over 25
runs. That gate is three milestones away, but the property it tests is
established or lost *here*, at the kernel -- so the cheap version runs now,
while the code that would break it is being written rather than after.

The cross-process half matters more than it looks. Within one process a hidden
dependency on dict insertion order, on ``id()``, or on a module-level cache can
be perfectly stable; ``PYTHONHASHSEED`` differs between processes, and a set
iterated for a tie-break would diverge there and nowhere else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cascade.aperture.policy import derive_policies
from cascade.config import Settings
from cascade.sim.kernel import Loom, RunSpec, mechanics_from
from cascade.sim.policies import HeuristicPolicy
from cascade.sim.state import state_hash
from tests.conftest import make_graph

REPO_ROOT = Path(__file__).resolve().parents[2]

# Built in a subprocess so the run is driven by a different interpreter with a
# different hash seed. Printed as JSON so a divergence names the first step
# that differs rather than only the final hash.
SCRIPT = """
import json, sys
sys.path.insert(0, {repo!r})
sys.path.insert(0, {tests!r})
from conftest import make_graph
from cascade.aperture.policy import derive_policies
from cascade.config import Settings
from cascade.sim.kernel import Loom, RunSpec, mechanics_from
from cascade.sim.policies import HeuristicPolicy

settings = Settings()
graph = make_graph(n_actors=14, n_factors=8)
policies = derive_policies(graph, settings.aperture, asymmetry=True)
mechanics = mechanics_from(graph)
decider = HeuristicPolicy(
    utility={{a: dict(mechanics[a].utility) for a in sorted(mechanics)}}
)
loom = Loom(settings=settings, graph=graph, policies=policies, decider=decider)
result = loom.run(
    RunSpec(
        run_id="00000000-0000-0000-0000-000000000000",
        scenario_id="scenario-1",
        config_id="base",
        replicate={replicate},
        policy="heuristic",
    )
)
print(json.dumps({{
    "event_log_hash": result.event_log_hash,
    "state_hashes": [record.state_hash for record in result.step_records],
    "outcome": result.outcome_score,
    "decisions": result.decisions,
}}))
"""


def run_here(replicate: int = 0) -> dict[str, object]:
    settings = Settings()
    graph = make_graph(n_actors=14, n_factors=8)
    policies = derive_policies(graph, settings.aperture, asymmetry=True)
    mechanics = mechanics_from(graph)
    loom = Loom(
        settings=settings,
        graph=graph,
        policies=policies,
        decider=HeuristicPolicy(
            utility={actor: dict(mechanics[actor].utility) for actor in sorted(mechanics)}
        ),
    )
    result = loom.run(
        RunSpec(
            run_id="00000000-0000-0000-0000-000000000000",
            scenario_id="scenario-1",
            config_id="base",
            replicate=replicate,
            policy="heuristic",
        )
    )
    return {
        "event_log_hash": result.event_log_hash,
        "state_hashes": [record.state_hash for record in result.step_records],
        "outcome": result.outcome_score,
        "decisions": result.decisions,
    }


def run_there(replicate: int, *, hash_seed: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["CASCADE_ENV_FILE"] = str(REPO_ROOT / "does-not-exist.env")
    for name in sorted(environment):
        if name.startswith("CASCADE_") and name != "CASCADE_ENV_FILE":
            environment.pop(name)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            SCRIPT.format(repo=str(REPO_ROOT), tests=str(REPO_ROOT / "tests"), replicate=replicate),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_a_run_replays_identically_in_this_process() -> None:
    assert run_here() == run_here()


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_a_run_replays_identically_under_a_different_hash_seed(hash_seed: str) -> None:
    """Dict and set iteration order is the divergence this catches (§8.1)."""
    assert run_there(0, hash_seed=hash_seed) == run_here()


def test_replicates_diverge_from_each_other() -> None:
    """Guard the guard: identical hashes would also 'pass' if every run were equal."""
    assert run_here(0)["event_log_hash"] != run_here(1)["event_log_hash"]


def test_the_state_hash_localises_a_divergence_to_a_step() -> None:
    """Per-step hashes are what make an M8 bisect name a step instead of a run."""
    left = run_here(0)["state_hashes"]
    right = run_here(1)["state_hashes"]
    assert isinstance(left, list) and isinstance(right, list)
    assert len(left) == len(right) == 24
    first_divergence = next(
        index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b
    )
    assert first_divergence == 0


def test_state_hash_is_insensitive_to_construction_order() -> None:
    """Two states built in different key orders are one state, and must hash as one."""
    from cascade.sim.state import WorldState

    left = WorldState(
        step=1,
        factors={"b": 0.25, "a": 0.5},
        factor_history={"b": (0.2, 0.25), "a": (0.4, 0.5)},
        resources={"actor_1": {"x": 0.2, "a": 0.1}},
        relations={},
    )
    right = WorldState(
        step=1,
        factors={"a": 0.5, "b": 0.25},
        factor_history={"a": (0.4, 0.5), "b": (0.2, 0.25)},
        resources={"actor_1": {"a": 0.1, "x": 0.2}},
        relations={},
    )
    assert state_hash(left) == state_hash(right)
