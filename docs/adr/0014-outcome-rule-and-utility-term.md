# ADR-0014 — `OutcomeRule` and `UtilityTerm` are declarative and monotone by construction

- **Status:** accepted
- **Milestone:** M4
- **Completes:** spec §5.1 (both types are referenced but never defined)

## Context

Spec §5.1 gives the `CausalGraph` contract as Pydantic models and references
two types it never defines:

```python
utility_terms: list[UtilityTerm]   # weighted refs to factor ids
outcome_rule: OutcomeRule          # maps terminal factor state -> [0,1]
```

The trailing comments are the whole specification. Two further constraints
appear elsewhere and are binding:

- §5.3 requires the outcome rule to depend on **≥ 2 factors** and be
  **monotone in each**.
- §7.2 evaluates it as a callable at the end of a run:
  `outcome_score = graph.outcome_rule(world.factors) -> [0,1]`.

So the type must be *data* the compiler can emit under a JSON schema, and
*executable* at terminal state. Those two requirements are what constrain the
design; everything else is open.

## Decision

### `UtilityTerm`

```python
class UtilityTerm(_Frozen):
    factor_id: str
    weight: float   # non-zero, [-1, 1]
```

The weight is **signed**, and that is the load-bearing part. An actor that
wants a factor *low* is expressed as a negative weight on that factor, not as
a positive weight on some inverted mirror factor. The alternative — unsigned
weights plus per-actor "direction" prose — puts the direction in a string that
nothing downstream can read, and M5's agents have to act on it mechanically.

### `OutcomeRule`

```python
class OutcomeTerm(_Frozen):
    factor_id: str
    weight: float          # non-zero, [-1, 1]

class OutcomeRule(_Frozen):
    terms: list[OutcomeTerm]   # >= 2, distinct factor_ids
    threshold: float           # [0, 1], the pivot in factor space
    steepness: float           # > 0
```

evaluated as

```
z = Σ wᵢ · (sᵢ − threshold)
p = 1 / (1 + exp(−steepness · z))
```

**Monotonicity is structural, not checked-and-hoped-for.** The partial
derivative is

```
∂p/∂sᵢ = steepness · wᵢ · p · (1 − p)
```

`steepness > 0` and `p ∈ (0, 1)` strictly, so the sign of the derivative is
exactly the sign of `wᵢ` everywhere in the domain. A rule that parses is
monotone in each of its factors; there is no value of the inputs for which it
is not. Requiring `wᵢ ≠ 0` is what makes it *strictly* monotone rather than
flat in a factor it nominally depends on — a zero weight would satisfy "depends
on ≥ 2 factors" by arity while depending on one in substance.

The output is open on (0, 1) rather than closed on [0, 1]. That is deliberate:
a terminal score of exactly 0 or 1 is a claim of certainty, and it makes the
Brier decomposition at M7 degenerate for that scenario.

### Why not the obvious alternatives

**A threshold on a single aggregate** (`p = 1 if Σwᵢsᵢ > t else 0`) is monotone
but not *strictly* — it is flat almost everywhere and the gradient carries no
information, so M5's 24-step dynamics would show nothing until a discontinuity.

**An expression string evaluated at runtime** would be maximally flexible and
is rejected outright: it makes the model's output executable code from an
untrusted source, it cannot be checked for monotonicity without solving the
general problem, and it cannot be canonicalised into a stable hash.

**A learned or fitted rule** would be trained on outcomes, which are behind the
`cascade_eval` grant precisely so the compiler cannot see them (invariant 2).

## Consequence for the validator

§5.3's "depends on ≥ 2 factors; is monotone in each" reduces to structural
checks the validator can make without sampling: ≥ 2 terms, distinct
`factor_id`s that exist in the graph, non-zero weights, `steepness > 0`.

The validator nonetheless **also** verifies monotonicity numerically, by
sweeping each referenced factor across [0, 1] with the others held at several
fixed points and asserting the sign of every successive difference matches the
term's weight. That is redundant today, by construction. It is kept because
the redundancy is what catches a *future* rule form added without re-reading
this ADR — the check costs microseconds and the failure it guards against
silently invalidates every forecast the rule produces.

## Verified by

`tests/unit/test_decompose_schema.py` (round-trip, bounds, canonical hashing)
and `tests/property/test_outcome_rule.py` — a Hypothesis property asserting
strict monotonicity in the sign of each weight over randomly generated rules
and factor states.
