# ADR-0017 — `WorldState` carries the bookkeeping §7.1 omits

- **Status:** accepted
- **Milestone:** M5
- **Completes:** spec §7.1 (the listed fields cannot support §7.3, §7.4 or §7.6)

## Context

§7.1 says of `WorldState`: "Everything an agent could ever influence lives
here; nothing else does", and lists `step`, `factors`, `factor_history`,
`resources`, `relations`, `pending` and `rng_counter`. Three rules stated
elsewhere need state that is not in that list.

**§7.4, ALLY and DEFECT.** ALLY "pools resources for contests next step";
DEFECT "releases pooled resources back". Neither is expressible without a
ledger of who pledged what to whom. Folding a pledge into the recipient's
`resources` loses the backer, so DEFECT cannot return it; leaving it in the
backer's `resources` means the ally cannot spend it, which is the entire
effect.

**§7.3, activation.** An actor is triggered when an observed factor "moved by
more than 0.06 since that actor's last observation", and forced to act "at
least once every 5 steps". Both are per-actor history; neither is recoverable
from `factors`.

**§7.6, termination.** "A factor pinned at a bound for 3 consecutive steps" is
a streak, not a predicate on the current state.

## Decision

Four fields are added, and each is the minimum that makes its rule executable.

```python
pools: dict[str, dict[str, float]]      # pools[ally][backer] -> pledged amount
last_acted: dict[str, int]              # actor -> step it last acted
last_observed_at: dict[str, int]        # actor -> step it last observed
pinned_streak: dict[str, int]           # factor -> consecutive steps at a bound
```

`pools` is keyed by holder then backer, so a pledge is spendable by the ally
(the arbiter drains own resources first, then pledges) and returnable to its
owner on defection. A release that would push the backer above its endowment
leaves the remainder in the pool rather than evaporating: a conservation check
that tolerated quiet destruction would tolerate quiet creation.

`last_observed_at` stores the **step**, not the values seen. §7.3's trigger is
then evaluated against the factor's own trajectory between the two steps rather
than against the noisy reading the actor took. Comparing against the noisy
reading looks more faithful and is not: a two-hop channel carries
`noise_sigma = 0.15`, which trips a 0.06 threshold on sensor noise alone, so
the activation rate — and with it the per-run cost — would become a function of
the visibility policy. §6.4 needs the asymmetry switch to change what agents
*believe*, not how often they are woken.

**`last_acted` opens staggered.** The scheduled floor says every actor acts at
least once every 5 steps. Initialising the whole cast as equally overdue
satisfies that and makes all 14 fall due on the same step, where the cap of 8
picks by salience and the remainder carry into a synchronised pulse every 5
steps. `staggered_schedule()` seeds actor *i* (in sorted order) as due at step
`interval − 1 − (i mod interval)`, so the floor is spread across the interval.
It is a deterministic function of the graph, so it replays identically.

## What is deliberately *not* added

A per-actor `reputation` scalar. §7.4 gives DEFECT a "reputation penalty", and
the arbiter implements it as a decrement to the defector's trust with **every**
other actor — which is what reputation is, expressed in the state that already
exists. Who *learns* of the defection is Aperture's business; what it does to
standing is the world's.

## Cost

A 14-actor, 8-factor state serialises to ~4 KB as §7.1 estimates; the four
additions are ~40 numbers and do not change that order of magnitude. Every one
of them is in `canonical()` and therefore in the state hash, so a checkpoint
that restored three of the four would fail the round-trip rather than resume a
subtly different world.

## Verified by

`tests/unit/test_sim_state.py` (round-trip including every added field, the
staggered opening), `tests/unit/test_sim_scheduler.py` (the floor, the trigger
window), `tests/unit/test_sim_kernel.py::test_a_pinned_outcome_factor_ends_the_run_early`,
and `tests/property/test_arbiter.py::test_resources_are_conserved_and_never_negative`,
which is what caught an ALLY answered by a same-step DEFECT booking the
returned pledge as new resource.
