# ADR-0008 — Constraint precedence when the pool cannot satisfy §3.1

- **Status:** accepted
- **Milestone:** M1
- **Resolves:** CLAUDE.md open question **Q2**

## Context

Spec §3.1 requires, simultaneously:

1. exactly **180** scenarios;
2. a resolved-YES rate in **[0.40, 0.60]**, "by construction";
3. **no domain above 25%** of the set;
4. **≥ 3 distinguishable parties** with non-identical objectives, a binary
   unambiguous outcome, and resolution ≥ 21 days after cutoff.

The spec does not say what gives when the available pool cannot satisfy all
four. At M1 it could not, so the question stopped being hypothetical.

## Decision

Weakest constraint sacrificed first:

| Rank | Constraint | Why it sits here |
|---|---|---|
| 1 | Eligibility (binary, ≥3 parties, >21-day horizon) | Defines what a scenario **is**. Relaxing one does not shrink the study, it changes what the study is about. |
| 2 | Base rate ∈ [0.40, 0.60] | The only constraint protecting the headline number from a degenerate always-predict-the-base-rate model. Outside it, Brier is not interpretable and neither is any comparison drawn from it. |
| 3 | Domain cap ≤ 25% | A breach degrades a result that can still be read honestly — the per-domain table is published anyway. |
| 4 | **N = 180 — sacrificed first** | Reducing N costs statistical power, which is bounded and disclosable. |

`select()` therefore returns **the largest N ≤ 180 for which 1–3 all hold**,
and reports N. It never returns a set that reaches 180 by bending a rule.

## Why the count is the right thing to give

The two alternatives are worse in a specific, asymmetric way.

*Admitting ineligible scenarios* changes the population being measured while
the report still says "180 multi-party strategic questions". Nothing
downstream can detect it: the metrics compute fine, the bootstrap runs, and
the number is wrong for a reason that never appears in the artifact.

*Dropping scenarios until the numbers work* is a selection effect applied
directly to the headline metric. If you drop the questions that spoil the base
rate, you have chosen the set as a function of the outcomes.

A smaller N is the only failure that is **visible in the output**. It widens
every confidence interval, and the report says so.

## Consequence

`cascade ledger build` exits **3** on a shortfall, so a script cannot mistake
a short set for a complete one. The shortfall reason names the binding
constraint.

## Measured at M1

57 of 180, base rate 0.4912, max domain share 0.2456. The binding constraint
is the eligible pool, not the algorithm — see ADR-0009.
