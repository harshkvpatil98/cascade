# ADR-0015 — The run RNG draws a whole step at once, in unit normals

- **Status:** accepted
- **Milestone:** M5
- **Corrects / completes:** spec §8.1 (calls the PRNG "counter-based" where
  §8.2 pins PCG64, which is not), §8.2 (`rng_counter` alone cannot resume a
  PCG64 stream), and §6.4 (the isolation it requires is not achievable if draw
  counts depend on the policy)

## Context

§8.2 is explicit about the seed and the generator:

```python
run_seed = blake2b(f"{scenario_id}|{config_id}|{replicate}",
                   digest_size=8, key=STUDY_SALT).digest()
rng      = np.random.Generator(np.random.PCG64(int.from_bytes(run_seed)))
# every consumer draws from the SAME generator in a fixed order:
#   exogenous walk -> observation noise -> tie-breaks -> arbiter jitter
```

Three problems surface when this meets §6.4, which requires that switching
information asymmetry off changes "nothing else — same graph, same agents,
same seeds".

1. **Draw counts would depend on the visibility policy.** The natural
   implementation draws observation noise only for channels that have any
   (`sigma > 0`). A transparent policy has none, so it draws nothing, so the
   generator sits at a different position when the *next* step's exogenous walk
   runs — and every factor trajectory after step 0 diverges. The measured
   ablation delta would then contain the divergence as well as the mechanism.

2. **`rng_counter` cannot resume PCG64.** §7.1 stores "position in the seeded
   stream" in `WorldState` and §8.2 says a checkpoint resume "reproduces the
   identical stream". PCG64 is not counter-based (§8.1's phrase fits Philox,
   which §8.2 does not pick), and the number of internal advances per
   `normal()` varies with the ziggurat's rejections — so a draw count does not
   locate a position.

3. **`Generator` has no cross-version stream guarantee.** NEP 19's stability
   policy covers `RandomState`; `Generator`'s distribution methods are
   explicitly allowed to change. A checkpoint resumed under a different numpy
   can therefore simulate a different world while every hash in the checkpoint
   still validates.

## Decision

**The whole step is drawn at once, at the top of the step, in the documented
order.** `RunRng.step_noise(step, actor_ids, factor_ids)` returns a frozen
`StepNoise` and is the only thing that touches the generator. No stage holds
it. "Drawn in a fixed order" is then a property of the code's shape rather
than a convention six modules have to observe.

**Every draw is a unit normal, and the count is a function of the graph
alone:**

```
draws_per_step = 2·|factors| + |actors|·|factors|
```

— one exogenous draw per factor, one observation draw for *every* (actor,
factor) pair whether or not that channel uses it, zero for tie-breaks (§7.3
breaks ties "by utility-weighted salience, deterministically", so no randomness
is needed), and one arbiter-jitter draw per factor. Scaling by `volatility` or
`noise_sigma` happens at the point of use, so changing either is a change to
the world model and never to the stream.

Consequences, all of them the point:

- The stream position at the start of step *t* depends only on the graph and
  *t*. `rng_counter == step × draws_per_step` exactly, and
  `assert_counter()` checks it at the end of every step — a stray draw
  anywhere in the system fails the step that made it rather than surfacing as
  an M8 hash mismatch.
- The §6.4 ablation runs the identical stream through a different projection.
  A 14×8 graph draws 128 values per step, ~3,072 per run; the waste is
  microseconds and the alternative is a confounded measurement.
- Tie-breaks keep their slot in `DRAW_ORDER` while consuming nothing, so a
  future tie-break that does need a draw has one place to take it.

**The checkpoint carries the full bit-generator state**, not just the counter.
`RngState` holds PCG64's own state dict plus `draws` (the audit number) and the
numpy version.

**Replay under a different numpy is refused**, by comparing the recorded
version on restore. The alternative is a run that looks valid and is not.

## Alternatives rejected

**Switch to Philox** to make §8.1's "counter-based" literally true. It would
make position addressable and parallel fan-out trivially independent, but §8.2
pins PCG64 by name and in code; substituting the generator changes every seeded
value in the study for a benefit the draw plan above already provides.

**Independent child streams per consumer** (`SeedSequence.spawn`). Cleaner
statistically, and it contradicts §8.2's "every consumer draws from the SAME
generator" directly rather than elaborating it.

**Draw lazily and accept the ablation confound**, reporting it as a caveat. The
+0.027 attributed to information asymmetry is one of the study's two headline
mechanisms; a confound in it is not a caveat, it is the result.

## Verified by

`tests/unit/test_sim_rng.py` — the seed derivation recomputed independently
from §8.2, the draw order reproduced from a bare `PCG64`, the count as a
function of the graph, snapshot/restore equality, and both refusals.
`tests/unit/test_sim_kernel.py::test_turning_asymmetry_off_changes_beliefs_and_not_the_world`
runs the ablation with inert agents and asserts the factor trajectories are
identical.
