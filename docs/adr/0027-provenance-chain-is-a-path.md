# ADR-0027 — §11.2's provenance walk is a path, not an ancestor set

**Status:** accepted · **Milestone:** M8

## Context

§11.2 specifies the provenance walk as a recursive CTE and gives it verbatim:

```sql
SELECT p.*, c.depth + 1
FROM chain c
CROSS JOIN LATERAL jsonb_array_elements(c.caused_by) AS cb
JOIN events p ON ...
WHERE c.depth < 24
```

It recurses over **every** element of `caused_by`. Immediately below it, the
same section shows what the command should print:

```
  d0  step 19  actor=regional_bloc      ESCALATE  +0.11
  d1  step 17  actor=incumbent_party    DEFECT    -0.09
  d2  step 14  actor=external_guarantor SIGNAL
  d3  step 11  actor=external_guarantor ALLY      +0.07
  root  step  4  exogenous shock on "external_financing" -0.14
```

That is a **single path**. The two are not the same object, and the difference
is not cosmetic. `caused_by` holds up to `CAUSED_BY_LIMIT` (6) antecedents, so
recursing over all of them expands 6^depth: at the 24-step horizon the
ancestor set is on the order of 10^18 rows. Measured on a real 24-step run
from the stored log, the ancestor-set form **did not return** — the command
was killed after ten minutes.

## Decision

The recursion follows **one** antecedent per level: `caused_by->0`.

`caused_by` is written by `CausalLedger.antecedents`, which selects the top
`CAUSED_BY_LIMIT` movements by magnitude and then emits them in `(step, seq)`
order. Element 0 is therefore the *earliest of the largest movers* — a
deterministic choice, and the one that walks fastest toward the origin.

The walk is linear in the horizon. Measured on the same run that did not
return: **0.27 s**, against §11.2's own standard of "demonstrable in thirty
seconds rather than described".

Three guards, none of which should ever fire on a well-formed log:

- `depth < MAX_CHAIN_DEPTH` (64), above the 24-step horizon.
- `jsonb_array_length(caused_by) > 0`, so the walk stops at a decision with no
  recorded antecedent rather than joining against nothing.
- `(p.step, p.seq) < (c.step, c.seq)`, so a corrupt row cannot make it loop.
  An antecedent is always strictly earlier by construction; a recursive CTE
  with no such guard is an outage waiting for one bad row.

A `statement_timeout` of 30 s bounds it regardless.

## Consequences

- The command answers "what is the principal causal path to this outcome",
  which is the question §11.2's display asks, rather than "what is the
  complete set of decisions that contributed", which is a different and much
  larger object.
- A contribution that is not on the principal path is not shown. That is a
  real limitation and it is the cost of a walk that terminates. The full
  antecedent set is still in the log — `caused_by` is written whole — so a
  future analysis that wants the DAG can read it without a schema change.
- `OutcomeExplanation.complete` is the M8 criterion made checkable: a chain
  that does not reach an exogenous shock in `run_steps`, or that hits the depth
  bound with antecedents left, exits 3. An incomplete chain is the study's
  provenance claim failing, not the command failing.
