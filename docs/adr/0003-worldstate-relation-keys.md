# ADR-0003 — `WorldState.relations` uses a canonical string key

- **Status:** accepted
- **Milestone:** M5
- **Corrects:** spec §7.1

## Context

Spec §7.1 types the relation matrix as `dict[tuple[str, str], float]`.

JSON object keys are strings. A tuple key does not round-trip:

```python
>>> json.dumps({("a", "b"): 0.5})
'{"[\'a\', \'b\']": 0.5}'          # str() of the tuple -- lossy and Python-specific
>>> json.loads(_)
{"['a', 'b']": 0.5}                # never becomes a tuple again
```

`WorldState` is serialized on every one of the 24 steps for checkpointing, and
M5 requires that serialization to round-trip **bit-exactly**. It cannot, as
typed. Pydantic v2 will coerce on the way out and fail on the way back in.

## Decision

The wire type is `dict[str, float]` with a canonical key:

```
relation_key(a: str, b: str) -> f"{a}\x1f{b}"
```

- **`\x1f`** (ASCII unit separator) is the delimiter. Actor ids are opaque
  strings; any printable delimiter is a substring an id could legitimately
  contain, and a delimiter collision here silently merges two relations.
- **Directed, so not sorted.** `relations[(a, b)]` is *a*'s posture toward *b*
  and is not symmetric — `DEFECT` is unilateral. Sorting the pair would
  average away the asymmetry the simulation exists to model.
- Accessors `get_relation(state, a, b)` / `set_relation(...)` are the only
  supported path. Nothing indexes the dict directly.

## Ordering

Iteration over `relations` is sorted by key at every read site (invariant 7).
The insertion order of a relation dict depends on the order actors were
activated, which depends on the scheduler, which is exactly the kind of
coupling that produces a replay divergence three milestones later.

## Verified by

- `tests/property/` — hypothesis round-trip: `parse(serialize(s)) == s` for
  arbitrary `WorldState`, compared on the canonical JSON bytes, not on `==`.
- `tests/determinism/` — checkpoint-resume mid-run reproduces the identical
  terminal score.
