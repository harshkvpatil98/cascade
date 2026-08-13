# ADR-0007 — The cache key excludes `cache_control` markers

- **Status:** accepted
- **Milestone:** M0
- **Records a choice the spec left open**

## Context

Spec §8.3 defines the record/replay key:

```python
key = sha256(canonical_json({
  "model": ..., "system": ..., "messages": ...,
  "tools": ..., "temperature": ..., "prompt_rev": ...,
}))
```

Anthropic prompt caching is expressed *inside* those fields, as a
`cache_control` marker on a content block:

```json
{"type": "text", "text": "...persona...", "cache_control": {"type": "ephemeral"}}
```

So the question the spec does not answer: does moving a cache boundary change
the key?

## Decision

**No.** `cache_control` markers are stripped recursively from `system`,
`messages` and `tools` before hashing (`_strip_cache_control` in
`cascade/llm/types.py`).

## Rationale

The key's contract is *"everything that determines the response, and nothing
that does not."* `cache_control` is a **billing directive**. It does not
change the token sequence the model sees and it does not change the
completion. Two requests differing only in where the cache boundary sits are
the same request.

The practical consequence is the reason this is an ADR and not a comment.
ADR-0001 requires the static prefix to be re-measured and probably re-padded
when prompts are authored at M4/M5, and §12.2 calls the hit rate the single
biggest cost lever — so cache-boundary tuning is *expected*, iterative work.
If the marker were in the key, every boundary experiment would invalidate the
entire recorded corpus and demand a **paid** re-record to measure a change
that cannot alter a single output token. That is a strong incentive not to
tune the largest cost lever in the study.

## Cost of being wrong

Two requests sharing a key can be billed differently. This is correct and
intended: the ledger prices each call from the provider's **reported**
`usage`, never from the cache entry. A replayed call contributes zero cost by
construction, so no billing figure is ever derived from a shared key.

The residual risk is that recorded `usage` on a cache entry reflects whichever
boundary was in force when it was recorded, so `cache_read_input_tokens` in an
old recording may not match current markers. This matters only for cost
*estimation* from recordings — which is why `--estimate` (spec §12.4) runs 20
**live** sample units rather than extrapolating from the cache.

## Verified by

`tests/unit/test_cache.py` — two requests identical but for `cache_control`
placement produce the same key; changing any field in the documented key
domain (including `max_tokens`, see `LLMRequest.cache_domain`) produces a
different one.
