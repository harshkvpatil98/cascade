# ADR-0020 — The fan-out advances runs in lockstep so a step can be batched

- **Status:** accepted
- **Milestone:** M6
- **Completes:** spec §12.2 ("Batch API. The entire backtest is offline. There
  is no reason to pay interactive rates.") against §7.2's sequential step loop

## Context

§12.1 prices the study at `× 0.5 batch`. That multiplier is not a rounding
detail: at 36,000 runs the simulate phase costs $126 with it and $252 without,
against a configured ceiling of $240. **Batching is a functional requirement,
not an optimisation.**

But the Batch API is asynchronous with an SLA measured in hours, and a
simulation step depends on the step before it. Three shapes are available and
two of them do not work:

| Shape | Submissions for the study | Why it fails |
|---|---|---|
| One request per decision | 4.2M | Each waits hours. Not a study, a geological process. |
| One batch per run per step | 864,000 | Same problem, 5 actors at a time. |
| One batch per *step* across all runs | 24 | Works — needs runs to advance in lockstep. |

The third shape requires stopping between OBSERVE and DECIDE, which a single
LangGraph invocation cannot do: it runs all six stages of §7.2 and returns.

## Decision

**The kernel exposes the step as two halves, and the runner drives thousands of
runs through them in lockstep.**

```python
handle = loom.start(spec)                  # no work
turns  = loom.observe_step(handle)         # stages 1-4: deterministic, model-free
decider.prepare(all_turns_across_the_wave) # one batch for every miss in the wave
loom.complete_step(handle)                 # stages 5-6: every call is a cache hit
```

This works because **nothing in a step's first four stages depends on any
actor's decision in that step**. The exogenous walk, the arrivals, the
activation set and every observation are fixed before the first actor decides —
so a whole wave's turns can be assembled, resolved together, and then applied.

Three consequences, each load-bearing:

**Two drivers, one ordering.** `run()` still drives a single replicate through
the compiled LangGraph graph (§2.3 pins the orchestration, and the single-run
path is what `cascade simulate run` uses). The wavefront walks the same stage
methods. Both read one `STAGE_SEQUENCE`, and a test asserts the two produce
byte-identical event-log hashes for the same spec. Duplicating the *driver* is
acceptable; duplicating the *ordering* would not be.

**The prompt is built once.** `LLMAgents.build_request` is shared by `prepare`
and `decide`. If the batch path built its own request, a one-character
difference would make every batched answer a cache miss at decide time — the
study would pay twice for every decision and still look like it worked.

**One decider serves every scenario.** Prefixes are keyed by
`(scenario_id, actor_id)`, not by actor: actor ids are per-graph slugs and two
scenarios both containing an `incumbent_party` would otherwise share a persona.
A decider per scenario would also turn one submission per step into 180.

## Wave size is the operator's dial, with the arithmetic stated

A wave of *w* runs submits one batch per step of roughly
`w × 4.86 × (1 − hit rate)` requests, and the phase needs
`24 × ceil(total / w)` submissions. At w = 200 that is 4,320 submissions; at
w = 36,000 it is 24. Larger waves mean fewer round trips against an hours-SLA
API and more live runs in memory — a run holds its world, its agents' memories
and its events until it finishes, on the order of 70 KB. There is no value that
is right on every machine, so it is a parameter with a documented trade-off
rather than a constant someone would have to rediscover by profiling.

## Alternatives rejected

**LangGraph `interrupt_before`.** The framework can pause before a node, but
resuming requires a checkpointer and one thread per run; 36,000 threads through
a checkpoint store to avoid exposing two methods is a worse trade.

**Sequential runs with a live (non-batch) API.** Simple, and it breaches the
phase ceiling by design. §15 says to cut ablation replicates before scenarios
when the budget is tight; paying double for the headline configuration is not
on that list.

## Verified by

`tests/unit/test_ensemble_runner.py` — the wavefront reproduces the single-run
driver's event-log hash, wave size does not change any run, a wave spanning
scenarios prepares once per step, and the handle refuses to be observed twice,
completed before it is observed, or collected before it terminates.
`tests/unit/test_sim_kernel.py::test_both_drivers_produce_the_same_run` pins
the equivalence from the kernel's side.
