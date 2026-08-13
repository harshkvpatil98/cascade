# ADR-0009 — The ≥3-party screen, and the loss of the primary source

- **Status:** accepted
- **Milestone:** M1
- **Amends:** spec §3.1 sourcing

## Part 1 — Metaculus is no longer a public API

Spec §3.1 names Metaculus the primary source, "resolved binary questions via
the public API", targeting ~90 of 180 scenarios — half the study set.

As of M1 every unauthenticated request returns:

```
Permission Error: The API is only available to authenticated users.
Please create an account and use your API token to access the API.
```

This is an upstream policy change since the spec was written.

**Decision.** The loader is written, tested against recorded payloads, and
wired into the registry. It activates when `CASCADE_METACULUS_TOKEN` is set
and contributes nothing until then. It is **not** silently skipped: the build
reports the source as unavailable and names the variable, because a primary
source contributing zero is exactly the shortfall that must be visible rather
than absorbed by the other loaders.

Polymarket and Manifold remain reachable and require no credential.

## Part 2 — Counting proper nouns is not counting parties

The ≥3-party rule is a semantic judgement, and at M1 there is no compiler to
make it — the causal graph, with its authoritative 8–20 actor count, is not
built until M4. The first implementation approximated it by counting distinct
capitalised names in the question and its resolution criterion.

**Measured against the real pool, that admitted:**

- *"Will the 'Blaze Star' (T Coronae Borealis) go nova during Blazing Swan 2025?"*
- *"Will Sean 'Diddy' Combs be alive on Jan 1st 2025?"*
- *"NYT review of Taylor Swift's 'The Life of a Showgirl' use >5 em dashes?"*

Each carries three or more capitalised tokens and exactly **zero** parties with
objectives. 95 of 163 scenarios entered under that rule.

Two narrower defects were found the same way and are fixed:

- the proper-noun pattern joined names across "and", turning
  *"Russia and Ukraine"* into a **third**, phantom party;
- the stop-word list was case-sensitive, so **"YES"** — which opens nearly
  every market resolution criterion — counted as a party, inflating every
  market question by one and letting two-party questions pass.

**Decision.** `named_parties` counts only **recognised institutional actors**
from an enumerated lexicon of states, blocs, regulators and institutions.
Aliases collapse (`US` / `United States` / `White House` → one party), nested
names collapse to the shorter form, and anything unrecognised **fails closed**.

Party evidence is ranked, strongest first, and the rule that fired is stored
per scenario so the composition of the set is auditable:

| Rule | Evidence | Selected at M1 |
|---|---|---|
| `curated` | Author named ≥3 parties, with provenance | 10 |
| `event_siblings` | ≥3 mutually exclusive outcomes in one real-world event — structural, needs no text | 41 |
| `named_parties` | ≥3 recognised institutional actors in the text | 6 |

## Cost, stated plainly

Precision was bought with recall. Enforcing the rule correctly took the set
from 163 to **57**. A genuine two-company merger dispute is excluded when
neither company is in the lexicon, and the lexicon presently covers states and
regulators but almost no corporations — which under-serves the "merger
reviews" scenarios §3.1 explicitly asks for.

That is the right direction for the error to run. A set of 163 in which 95
scenarios have no parties is not a larger version of this study; it is a
different and unreportable one. **The diagnosed path to a larger set is
extending the actor lexicon deliberately and re-running — not loosening the
rule.**

## Verified by

`tests/unit/test_ledger_rules.py` (party rules, alias collapse, the "and" and
"YES" regressions), `tests/unit/test_ledger_sources.py` (loader refusals,
Metaculus reported not absorbed).
