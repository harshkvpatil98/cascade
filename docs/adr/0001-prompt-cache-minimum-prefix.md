# ADR-0001 — The cacheable prefix must clear the model's cache floor

- **Status:** accepted
- **Milestone:** M0
- **Corrects:** spec §12.1

## Context

The §12.1 cost model bills the static persona + rules + action-schema prefix
at the prompt-cache read rate:

| Line | Value |
|---|---|
| Static prefix tokens | 1,900 |
| Dynamic tokens | 260 |
| Effective billed input per call | **450** = 1,900 × 0.1 + 260 |

That 0.1 multiplier only applies if the provider actually caches the prefix.
Anthropic will not cache a prefix shorter than a per-model minimum — 4,096
tokens for Haiku 4.5. Below the floor there is **no error**: the request
succeeds, `cache_creation_input_tokens` comes back `0`, and every prefix token
is billed at list price.

The spec's prefix is 1,900 tokens. It sits below the floor.

## Consequence if unaddressed

Effective billed input per call becomes `1,900 + 260 = 2,160` rather than
`450` — **4.8×** the modelled input cost. Input is ⅔ of the marginal run cost,
so the $290 study becomes roughly $1,000, and the first visible symptom is the
invoice.

## Decision

1. `prompt_cache.min_prefix_tokens: 4096` is config, not a constant, because
   the floor is a property of the model and models are pinned per milestone.
2. `prompt_cache.enforce_min_prefix: true` — `assert_cacheable_prefix()` in
   `cascade/llm/client.py` raises `PromptTooShortToCache` (exit 3) rather than
   letting a silent non-cache through.
3. The M4/M5 prompt authoring must pad the static prefix past 4,096 tokens
   with content that is genuinely invariant across calls for a given actor —
   world rules, the full action schema with examples, refusal boundaries. Not
   filler: padding that varies per call defeats the cache it was added to
   satisfy.
4. `prompt_cache.ttl: "1h"`. Batch submissions routinely lag past the 5-minute
   default TTL, and a cache that has expired by the time the batch runs is a
   cache that was never written.

The check is offline (a character-based token estimate) so it runs in CI. The
estimate is used **only** for this gate — every dollar in the ledger comes from
the provider's own reported `usage`.

## Status at M0

The gate exists and is tested. No prompt exists yet, so nothing is gated. The
prefix must be measured against this floor when it is authored at M4/M5.
