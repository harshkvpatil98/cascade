# ADR-0019 — Evidence is retrieved once per (scenario, actor), in the cached prefix

- **Status:** accepted
- **Milestone:** M5
- **Corrects:** spec §7.2 (stage OBSERVE retrieves per step) against §12.1
  (260 dynamic tokens per call) — the two cannot both hold

## Context

§7.2's OBSERVE stage reads:

```
4. OBSERVE     for each active actor:
                 obs = aperture.project(world, policy[actor], rng)
                 ctx = memory[actor].recent(12) + rolling_summary
                 evidence = chronofence.search(q, as_of=cutoff, k=6)
```

and §12.1 budgets the per-call cost as

| Static prefix (persona + rules) | 1,900 | prompt-cached at 10% of list |
| Dynamic (observation + memory + evidence) | 260 | bounded by the 12-observation cap |

Six chunks of up to 512 tokens each is up to 3,072 tokens. The dynamic budget
is 260. The evidence alone is an order of magnitude over the line it is listed
on, and the line is what the whole $290 study cost is derived from.

The reconciling observation is in the retrieval call itself: **`as_of` is the
scenario cutoff, which does not change during a run.** The corpus is frozen
before the cutoff and the query is a function of the actor's standing
objective, so the same six chunks come back at step 0 and at step 23. Per-step
retrieval buys nothing and costs 4.2M vector searches at ~13 ms.

## Decision

Evidence is retrieved **once per (scenario, actor)**, at run preparation, and
lives in the actor's static prefix alongside the rules and the persona. The
prompt is built in three layers:

| Layer | Varies with | Cached |
|---|---|---|
| `RULES` — world model, action semantics, arbiter mechanics | nothing | yes |
| persona — objective, utility, levers, counterparties, **evidence** | scenario × actor | yes |
| turn — observation, memory digest | step | no |

Retrieval per scenario drops from `24 steps × ~4.86 actors × 200 replicates`
to `|actors|`, i.e. from ~23,000 to 14 searches — 2,520 for the whole study
instead of 4.2M.

**This is also what makes prompt caching work at all.** ADR-0001 records that
Anthropic silently declines to cache a prefix below 4,096 tokens while still
charging the write premium. Measured here: `RULES` is ~1,389 tokens and the
generated tool schema ~1,527, so rules + schema + persona is ~2,900 — under the
floor. The evidence block is what carries it over. `kernel.evidence_chars`
(900) bounds each excerpt so the prefix has a stable length as well as a
sufficient one; `prepare_actor()` measures the result and marks the block
cacheable **only if it clears the floor**, because marking a short prefix is
strictly worse than not marking it.

## What is given up

An agent cannot retrieve in response to what it sees at step 17. That capability
would be worth having if the corpus could answer a question that arose during
the run — but every document in it predates the cutoff, so the only thing a
late query can surface is a different slice of the same fixed evidence, chosen
by a world state that the evidence cannot reflect. The exchange is a capability
with no information behind it for a cost model that closes and a cache that
works.

If a later milestone wants per-step retrieval, the honest version is a second
`k` budget and a restated §12.1, not a quiet extra call.

## Verified by

`tests/unit/test_sim_agent.py::test_the_prefix_is_marked_cacheable_only_above_the_floor`
and `::test_evidence_is_what_carries_the_prefix_over_the_floor` (both measured,
not asserted from the spec's numbers), and
`::test_an_identical_turn_is_served_from_the_cache`, which is the action cache
of §12.1 doing its job with no separate mechanism.
