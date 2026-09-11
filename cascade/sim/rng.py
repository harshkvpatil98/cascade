"""One RNG per run, seeded once, drawn in a fixed order (invariant 4, §8.2).

The spec's derivation is reproduced exactly::

    run_seed = blake2b(f"{scenario_id}|{config_id}|{replicate}",
                       digest_size=8, key=STUDY_SALT).digest()
    rng      = np.random.Generator(np.random.PCG64(int.from_bytes(run_seed)))

Two things are stronger here than §8.2 requires, and both exist to make an
ablation comparison mean what it claims (ADR-0015):

**The whole step is drawn at once.** Stages do not hold the generator. Each
step calls :meth:`RunRng.step_noise` once, which draws the step's values in the
documented order -- exogenous walk, observation noise, tie-breaks, arbiter
jitter -- and hands back a frozen record the stages read from. An out-of-order
draw is then not a bug someone has to notice: there is no generator to draw
from anywhere else.

**Draws are unit normals, and the count is a function of the graph alone.**
Every value is N(0, 1), scaled by its volatility or sigma at the point of use,
and the number drawn per step is ``2*|factors| + |actors|*|factors|`` whatever
the run does. So the stream position at the start of step *t* depends only on
the graph and *t* -- never on the visibility policy, the active set, or the
actions taken. §6.4 requires that switching information asymmetry off changes
"nothing else"; if a transparent policy skipped its noise draws, every
subsequent exogenous shock would shift too and the +0.027 would be measuring
the shift as much as the mechanism.

``rng_counter`` is therefore exactly ``step * draws_per_step``, which
:func:`assert_counter` checks. A stray draw anywhere in the system moves it and
fails a run rather than quietly producing a different world.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DRAW_ORDER",
    "RngOrderViolation",
    "RngState",
    "RunRng",
    "StepNoise",
    "assert_counter",
    "draws_per_step",
    "run_seed",
]

# The documented order, quoted by invariant 4. `tiebreak` draws nothing: §7.3
# breaks ties "by utility-weighted salience, deterministically", so the
# scheduler needs no randomness. The stage is kept in the list because it is
# part of the contract and a future tie-break that does need a draw must take
# it here, between observation noise and arbiter jitter, rather than wherever
# is convenient.
DRAW_ORDER: tuple[str, ...] = ("exogenous", "observation", "tiebreak", "arbiter")


class RngOrderViolation(RuntimeError):
    """A step's noise was requested out of order, or the generator was re-seeded.

    Invariant 4 says the RNG is seeded once and never re-seeded mid-run. This
    is what that looks like when it is violated: a step asking for noise it
    already consumed, or skipping one.
    """


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StepNoise(_Frozen):
    """Every random value one step will use, drawn in the documented order.

    Unit normals. Scaling happens where the value is used -- volatility in the
    exogenous walk, ``noise_sigma`` in the projection -- so changing either is
    a change to the world model and never to the stream.
    """

    step: int = Field(ge=0)
    exogenous: dict[str, float]
    """factor id -> N(0, 1)."""
    observation: dict[str, dict[str, float]]
    """actor id -> factor id -> N(0, 1). Drawn for every pair, used where visible."""
    arbiter: dict[str, float]
    """factor id -> N(0, 1). The ESCALATE variance penalty of §7.4."""

    def observation_for(self, actor_id: str) -> dict[str, float]:
        draws = self.observation.get(actor_id)
        if draws is None:
            raise KeyError(
                f"no observation noise drawn for actor {actor_id!r}; the draw plan is "
                "built from the graph, so an unknown actor means the plan and the run disagree"
            )
        return draws


class RngState(_Frozen):
    """A snapshot of the generator, sufficient to resume bit-for-bit.

    §7.1 stores ``rng_counter`` in the world state, and §8.2 says a checkpoint
    resume "reproduces the identical stream". A counter alone cannot do that
    for PCG64 -- it is not a counter-based generator, and the number of
    internal advances per normal draw varies with the ziggurat's rejections --
    so the full bit-generator state travels with it. The counter stays as the
    audit number: it is what :func:`assert_counter` checks.
    """

    bit_generator: str
    state: dict[str, Any]
    draws: int = Field(ge=0)
    next_step: int = Field(ge=0)
    numpy_version: str
    """Recorded because ``np.random.Generator`` does not promise stream
    stability across numpy releases (NEP 19 covers ``RandomState``, not
    ``Generator``). A replay under a different numpy is a hard error at M8
    rather than a mysterious divergence, so the version has to be in the
    manifest and in the snapshot."""


def run_seed(*, scenario_id: str, config_id: str, replicate: int, salt: str) -> int:
    """Derive the seed for one run exactly as spec §8.2 specifies.

    The byte order is stated explicitly rather than left to the default: the
    default changed meaning once already in Python's history, and a seed that
    silently byte-swaps produces a study that is internally consistent and
    irreproducible from the spec.
    """
    if replicate < 0:
        raise ValueError(f"replicate must be non-negative, got {replicate}")
    digest = hashlib.blake2b(
        f"{scenario_id}|{config_id}|{replicate}".encode(),
        digest_size=8,
        key=salt.encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


def draws_per_step(*, n_actors: int, n_factors: int) -> int:
    """Number of values drawn per step. A function of the graph, nothing else."""
    return 2 * n_factors + n_actors * n_factors


def assert_counter(*, rng_counter: int, step: int, n_actors: int, n_factors: int) -> None:
    """Assert the stream position is exactly where the step count says it should be.

    Preserves invariant 4 at runtime. A component that reached for the
    generator directly -- or a stage that drew conditionally -- shows up here
    as an arithmetic mismatch at the end of the step that did it, instead of as
    a divergent event-log hash 36,000 runs later.
    """
    expected = step * draws_per_step(n_actors=n_actors, n_factors=n_factors)
    if rng_counter != expected:
        raise RngOrderViolation(
            f"rng_counter is {rng_counter} at step {step}; expected {expected} "
            f"({draws_per_step(n_actors=n_actors, n_factors=n_factors)} draws/step for "
            f"{n_actors} actors and {n_factors} factors). Something drew from the run "
            "generator outside the step plan."
        )


class RunRng:
    """The one generator for one run. Seeded once; never re-seeded (invariant 4)."""

    def __init__(self, generator: np.random.Generator, *, draws: int = 0, next_step: int = 0):
        self._generator = generator
        self._draws = draws
        self._next_step = next_step

    @classmethod
    def for_run(cls, *, scenario_id: str, config_id: str, replicate: int, salt: str) -> RunRng:
        """Seed from ``(scenario_id, config_id, replicate)`` under the study salt."""
        seed = run_seed(
            scenario_id=scenario_id, config_id=config_id, replicate=replicate, salt=salt
        )
        return cls(np.random.Generator(np.random.PCG64(seed)))

    @property
    def draws(self) -> int:
        """Values drawn so far. Mirrors ``WorldState.rng_counter``."""
        return self._draws

    @property
    def next_step(self) -> int:
        return self._next_step

    def step_noise(
        self, *, step: int, actor_ids: Sequence[str], factor_ids: Sequence[str]
    ) -> StepNoise:
        """Draw every value step ``step`` will use, in the documented order.

        Refuses a step out of sequence. Re-drawing a step would hand two
        different worlds the same step number, and skipping one would silently
        change every later step's stream position -- both are the failure
        invariant 4 exists to prevent, and neither is visible in the output.
        """
        if step != self._next_step:
            raise RngOrderViolation(
                f"step {step} requested noise but the generator is at step {self._next_step}; "
                "the stream is drawn once per step, in order, and never re-seeded"
            )
        actors = sorted(actor_ids)
        factors = sorted(factor_ids)
        if len(set(actors)) != len(actors) or len(set(factors)) != len(factors):
            raise ValueError("duplicate ids in the draw plan; the plan must mirror the graph")

        # 1. exogenous walk
        exogenous = {factor: self._normal() for factor in factors}
        # 2. observation noise -- every (actor, factor) pair, whether visible or not
        observation = {actor: {factor: self._normal() for factor in factors} for actor in actors}
        # 3. tie-breaks -- deterministic, zero draws (see DRAW_ORDER)
        # 4. arbiter jitter
        arbiter = {factor: self._normal() for factor in factors}

        self._next_step = step + 1
        return StepNoise(step=step, exogenous=exogenous, observation=observation, arbiter=arbiter)

    def _normal(self) -> float:
        self._draws += 1
        return float(self._generator.normal())

    def snapshot(self) -> RngState:
        """Capture enough state to resume the identical stream."""
        raw: dict[str, Any] = dict(self._generator.bit_generator.state)
        return RngState(
            bit_generator=str(raw.get("bit_generator", "")),
            state=raw,
            draws=self._draws,
            next_step=self._next_step,
            numpy_version=np.__version__,
        )

    @classmethod
    def restore(cls, snapshot: RngState) -> RunRng:
        """Rebuild a generator from a snapshot, refusing an incompatible one.

        The numpy version is compared rather than ignored: ``Generator``
        distribution methods are not covered by a stream-compatibility promise,
        so resuming a checkpoint under a different numpy can produce a
        different world while every hash in the checkpoint still looks valid.
        """
        if snapshot.bit_generator != "PCG64":
            raise RngOrderViolation(
                f"snapshot was taken from {snapshot.bit_generator!r}; spec §8.2 pins PCG64"
            )
        if snapshot.numpy_version != np.__version__:
            raise RngOrderViolation(
                f"snapshot was taken under numpy {snapshot.numpy_version} and this process "
                f"runs numpy {np.__version__}. Generator streams are not guaranteed stable "
                "across numpy releases, so resuming would silently simulate a different world."
            )
        bit_generator = np.random.PCG64()
        # numpy types the property as its own TypedDict; the snapshot round-trips
        # through JSON, so what comes back is a plain dict with the same keys.
        bit_generator.state = cast(Any, snapshot.state)
        return cls(
            np.random.Generator(bit_generator),
            draws=snapshot.draws,
            next_step=snapshot.next_step,
        )
