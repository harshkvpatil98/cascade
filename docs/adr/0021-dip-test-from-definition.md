# ADR-0021 — The dip test is implemented from its definition, with a Monte Carlo null

- **Status:** accepted
- **Milestone:** M6
- **Completes:** spec §9.1 (`dip_test(scores).p`) and §9.2 (the two
  corroborating statistics), neither of which says where the test comes from

## Context

§9.1 makes the dip test part of the collapse — `modality = "multi" if (sigma >
0.30 or dip_test(scores).p < 0.05)` — and §9.2 explains why it has to be there:
σ alone is one threshold on one moment, and the claim that σ is *informative*
about forecast error is a result the study reports, not an assumption.

Three ways to get one, and the pinned stack (§2.3) admits no new dependency:

1. Add `diptest` (a C extension). A substitution requiring its own ADR, for one
   statistic.
2. Transcribe Hartigan's 1985 FORTRAN. Compact and famously fiddly; a
   transcription error produces a *plausible* p-value, which is the worst
   failure mode available for a number that decides which forecasts the study
   tells you to distrust.
3. Implement the definition directly.

## Decision

**Implement the definition, and validate against cases with closed-form
answers.**

The dip is the sup-norm distance from the ECDF to the nearest unimodal CDF. A
unimodal CDF is convex up to its mode and concave after it, so for each
candidate mode the distance to the best such fit is

```
max( departure from convexity on the left, departure from concavity on the right ) / 2
```

and the dip is the minimum over modal positions. The halving is the classical
factor: the error can be split evenly above and below the ECDF's own jump.

The search is binary, which the shape of the problem permits: adding points to
the left prefix can only lower its convex minorant, so the left departure is
non-decreasing in the mode; the right departure is non-increasing by the same
argument; the minimum of a non-decreasing and a non-increasing sequence is at
their crossover. That is ~20 hull constructions per dip rather than *n*, and it
is what makes a 2,000-draw null affordable.

**The p-value is Monte Carlo against the uniform null** — the standard choice,
because the uniform is the least favourable unimodal distribution, so a dip
surprising under it is surprising under every unimodal alternative. The null
depends only on the sample size, so it is memoised per *n*: 2,160 forecasts
would otherwise trigger 4.3M dip evaluations to answer 2,160 questions.

**The null is seeded from the study salt**, never from global state. A p-value
that moved between processes would put a different number in the report on
every run — the same defect class as an unseeded simulation, and harder to
notice.

## Validation is the argument

An implementation from a definition is only as good as its checks, so the tests
are the cases where the answer is known independently:

| Case | Expected | Measured |
|---|---|---|
| 50/50 two-point at the bounds | 0.25 (the attainable maximum) | 0.25 |
| Exactly uniform grid, n = 50/100/200 | 1/(2n) | 1/(2n) |
| Constant sample, n < 4 | 0 | 0 |
| Bimodal vs unimodal, n = 200 | separated | > 5× the unimodal dip, p = 0.002 vs 0.14 |

The bimodality coefficient is checked the same way: 5/9 for a uniform sample
and 1/3 for a normal one, which is *why* §9.2's threshold is 0.555.

**BC does not vote.** §9.1 defines the flag as σ or the dip; a third statistic
quietly entering the disjunction would make the reported flag a different thing
from the specified one. It is computed, stored beside the forecast, and
reported.

## Defect this found

A constant ensemble scored BC = 2.0 — strongly "bimodal". The mean of fifty
copies of 0.4 is not exactly 0.4 in binary, so the variance lands near 1e-33
instead of zero, the `variance <= 0` guard never fires, and the third and
fourth moments come out as noise over noise. Degeneracy is now detected by
range, which is exact.

## Verified by

`tests/unit/test_ensemble_dip.py` (the table above, plus seeding and the
add-one p-value floor) and `tests/unit/test_ensemble_aggregate.py`, which
includes the case the second statistic exists for: two tight clusters at 0.40
and 0.60, where σ stays under 0.30 and the dip rejects unimodality anyway.
