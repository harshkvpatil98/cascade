"""One RNG per run, seeded once, drawn in a fixed order (invariant 4, §8.2).

These tests are the runtime half of invariant 4 -- the static half cannot see a
draw. What they pin down is that the stream position after a step is a function
of the graph and the step number and nothing else, because that is what makes
the §6.4 ablation a comparison rather than two different worlds.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cascade.sim.rng import (
    DRAW_ORDER,
    RngOrderViolation,
    RunRng,
    assert_counter,
    draws_per_step,
    run_seed,
)

SALT = "cascade-2026-study-01"
ACTORS = ("actor_0", "actor_1", "actor_2")
FACTORS = ("factor_0", "factor_1")


def make() -> RunRng:
    return RunRng.for_run(scenario_id="s-1", config_id="base", replicate=0, salt=SALT)


def test_seed_matches_the_spec_derivation_byte_for_byte() -> None:
    """§8.2 states the derivation; this is it, independently recomputed."""
    expected = int.from_bytes(
        hashlib.blake2b(b"s-1|base|7", digest_size=8, key=SALT.encode("utf-8")).digest(),
        "big",
    )
    assert run_seed(scenario_id="s-1", config_id="base", replicate=7, salt=SALT) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("s-1", "base", 0), ("s-2", "base", 0)),
        (("s-1", "base", 0), ("s-1", "C04", 0)),
        (("s-1", "base", 0), ("s-1", "base", 1)),
    ],
)
def test_every_seed_field_changes_the_seed(
    left: tuple[str, str, int], right: tuple[str, str, int]
) -> None:
    """Scenario, config and replicate each have to separate the streams.

    If any of the three did not, two runs the study treats as independent would
    share a world, and the ensemble's dispersion would be an artefact.
    """
    seed_left = run_seed(scenario_id=left[0], config_id=left[1], replicate=left[2], salt=SALT)
    seed_right = run_seed(scenario_id=right[0], config_id=right[1], replicate=right[2], salt=SALT)
    assert seed_left != seed_right


def test_the_salt_changes_every_stream() -> None:
    assert run_seed(scenario_id="s", config_id="c", replicate=0, salt="a") != run_seed(
        scenario_id="s", config_id="c", replicate=0, salt="b"
    )


def test_the_same_seed_reproduces_the_same_draws() -> None:
    first = make().step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    second = make().step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    assert first == second


def test_the_draw_count_is_a_function_of_the_graph_alone() -> None:
    """ADR-0015: noise is drawn for every (actor, factor) pair, visible or not.

    This is what makes the asymmetry ablation honest. If a transparent policy
    skipped the noise draws it does not use, every later exogenous shock would
    shift, and the measured delta would include the shift.
    """
    rng = make()
    rng.step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    assert rng.draws == draws_per_step(n_actors=len(ACTORS), n_factors=len(FACTORS))
    assert rng.draws == 2 * len(FACTORS) + len(ACTORS) * len(FACTORS)

    rng.step_noise(step=1, actor_ids=ACTORS, factor_ids=FACTORS)
    assert rng.draws == 2 * draws_per_step(n_actors=len(ACTORS), n_factors=len(FACTORS))


def test_steps_must_be_drawn_in_order_and_only_once() -> None:
    rng = make()
    rng.step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    with pytest.raises(RngOrderViolation, match="generator is at step 1"):
        rng.step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    with pytest.raises(RngOrderViolation, match="generator is at step 1"):
        rng.step_noise(step=5, actor_ids=ACTORS, factor_ids=FACTORS)


def test_the_documented_draw_order_is_the_order_drawn() -> None:
    """The stages are consumed in §8.2's order: exogenous, observation, jitter.

    Checked by reproducing the sequence from a bare generator with the same
    seed: if the plan drew observation noise before the exogenous walk, the
    values would land in different slots.
    """
    seed = run_seed(scenario_id="s-1", config_id="base", replicate=0, salt=SALT)
    bare = np.random.Generator(np.random.PCG64(seed))
    expected = [float(bare.normal()) for _ in range(draws_per_step(n_actors=3, n_factors=2))]

    noise = make().step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    drawn = [noise.exogenous[factor] for factor in sorted(FACTORS)]
    for actor in sorted(ACTORS):
        drawn.extend(noise.observation[actor][factor] for factor in sorted(FACTORS))
    drawn.extend(noise.arbiter[factor] for factor in sorted(FACTORS))

    assert drawn == expected
    assert DRAW_ORDER == ("exogenous", "observation", "tiebreak", "arbiter")


def test_a_snapshot_resumes_the_identical_stream() -> None:
    """§8.2: "a checkpoint resume reproduces the identical stream"."""
    rng = make()
    rng.step_noise(step=0, actor_ids=ACTORS, factor_ids=FACTORS)
    rng.step_noise(step=1, actor_ids=ACTORS, factor_ids=FACTORS)

    resumed = RunRng.restore(rng.snapshot())
    assert resumed.draws == rng.draws
    assert resumed.next_step == rng.next_step
    assert resumed.step_noise(step=2, actor_ids=ACTORS, factor_ids=FACTORS) == rng.step_noise(
        step=2, actor_ids=ACTORS, factor_ids=FACTORS
    )


def test_a_snapshot_from_a_different_numpy_is_refused() -> None:
    """ADR-0015: `Generator` makes no cross-version stream promise.

    Resuming anyway would simulate a different world while every hash in the
    checkpoint still looked valid, which is the worst available failure mode.
    """
    snapshot = make().snapshot()
    with pytest.raises(RngOrderViolation, match="numpy"):
        RunRng.restore(snapshot.model_copy(update={"numpy_version": "0.0.0-not-real"}))


def test_a_snapshot_from_a_different_bit_generator_is_refused() -> None:
    snapshot = make().snapshot()
    with pytest.raises(RngOrderViolation, match="PCG64"):
        RunRng.restore(snapshot.model_copy(update={"bit_generator": "Philox"}))


def test_the_counter_check_catches_an_unplanned_draw() -> None:
    """A stray draw moves the counter; the check fails the step that did it."""
    per_step = draws_per_step(n_actors=3, n_factors=2)
    assert_counter(rng_counter=2 * per_step, step=2, n_actors=3, n_factors=2)
    with pytest.raises(RngOrderViolation, match=f"expected {2 * per_step}"):
        assert_counter(rng_counter=2 * per_step + 1, step=2, n_actors=3, n_factors=2)


def test_a_negative_replicate_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        run_seed(scenario_id="s", config_id="c", replicate=-1, salt=SALT)


def test_duplicate_ids_in_the_plan_are_refused() -> None:
    """A duplicated id would draw twice for one pair and shift the whole stream."""
    with pytest.raises(ValueError, match="duplicate ids"):
        make().step_noise(step=0, actor_ids=("a", "a"), factor_ids=FACTORS)
