# Contributing to Cascade

This project has an unusual working method, and it is the reason the build log
reads the way it does. Read this before opening a pull request.

## The one rule

**If an acceptance criterion cannot be met, stop and report the measured value
with a diagnosis.** Do not relax the criterion. Do not add a tolerance. Do not
mark it "approximately passing".

Every milestone in `CLAUDE.md` has a row that says a criterion was missed, with
the number it missed by and why. That is what the build log is for. A run of
green checkmarks that was achieved by moving the line is worth less than a red
one with an explanation.

## Setting up

```bash
make install      # uv sync --extra dev
make env          # writes .env from .env.example
make up           # Postgres 16 + pgvector 0.8, Langfuse; waits for healthy
make migrate      # forward-only SQL migrations
cascade doctor    # asserts the pinned stack; exits 0 or tells you what drifted
```

For the corpus and retrieval paths you also need the embedding stack:

```bash
uv sync --extra dev --extra embed --extra kernel
```

## Before you push

```bash
make ci           # ruff + black + mypy strict + the offline suite
make test-all     # adds integration and leakage — needs `make up`
make verify       # every structural gate that needs no credential
```

All four gates are required. `mypy` runs in strict mode over `cascade/`, and
Pydantic v2 models sit at every subsystem boundary — **a raw dict crossing a
module boundary is a bug**.

## The nine invariants

Each is enforced by a test rather than by intention. If you find yourself
wanting an exemption, the exemption is the bug.

1. **`as_of` is never defaulted** — not in Python, not in SQL, not in a test
   helper. A missing `as_of` is a `TypeError` at the Python boundary and a
   signature error at the SQL boundary. Defaults are how leakage gets in.
2. **The simulation never reads `scenario_labels`** — enforced by a Postgres
   grant, not by code review. `cascade_sim` gets a permission error.
3. **The arbiter contains no LLM call and no I/O.** Asserted statically over
   its imports, along with the scheduler, the dynamics and the validators.
4. **One RNG per run**, seeded once, drawn in a fixed documented order. Never
   re-seeded mid-run.
5. **One LLM call site**: `cascade/llm/client.py`. A grep for the Anthropic SDK
   import anywhere else fails CI.
6. **The event log is append-only.** No `UPDATE`, no `DELETE`, ever.
7. **All iteration over collections is sorted.** Dict and set order is a
   nondeterminism vector, and the static check is deliberately blunt — uniform
   compliance is what keeps it meaningful.
8. **Every phase is resumable.** Checkpoint and skip completed units on restart.
9. **No target value is ever written into a report code path.** A static check
   parses every module on that path and fails on any literal from the
   measurement contract.

Only `cascade/config.py` reads the process environment. If you need an
environment value elsewhere, add a function to `config.py` rather than an
exemption to the check.

## Changing the database

Migrations are **forward-only** and numbered. Never edit one that has been
applied; add a new one. There is no ORM autogeneration — the partition strategy
is load-bearing and must be explicit in SQL.

Each migration's header explains *why*, with the measurement that motivated it
where one exists. See `migrations/014_hnsw_and_bounded_pool.sql` for the shape.

## Changing the pinned stack

Don't, without an ADR. `cascade doctor` asserts the pinned list in
`cascade/version.py`. If you believe a substitution is warranted, write a
record in `docs/adr/` and **ask** — do not swap silently.

Note the distinction: switching from IVFFlat to HNSW was *not* a substitution,
because both are features of the pinned pgvector. Adding matplotlib for four
charts *would* have been, which is why the report's figures are hand-written
SVG ([ADR-0024](docs/adr/0024-report-figures-are-svg.md)).

## Changing a prompt

Any edit to an agent, compiler or arbiter prompt after the first full backtest
must be recorded in the `prompt_revisions` table with the Brier before and
after, so tuning-on-test is visible in the record rather than hidden. Bump
`llm.prompt_rev` in the same change — the revision is part of the LLM cache
key, and an unchanged rev would serve recordings made against text the model
never saw.

## Writing tests

Five categories, and the distinction matters:

| Directory | Runs | For |
|---|---|---|
| `tests/unit/` | offline | Pure logic, one behaviour per test |
| `tests/property/` | offline | Hypothesis, for invariants that should hold over all inputs |
| `tests/integration/` | needs `make up` | Real Postgres, real grants, real schema |
| `tests/leakage/` | needs a corpus | Plants post-cutoff documents and asserts none is retrieved |
| `tests/determinism/` | offline | Re-runs in subprocesses under different hash seeds |

Two conventions worth copying:

- **Write the acceptance criterion as a failing test first**, then make it pass.
- **A test name should say what breaks if it fails**, not what it calls. Prefer
  `test_growth_alone_never_triggers_a_rebuild` over `test_plan_partition`.

And a warning learned the hard way: a test must measure the code, not the
machine. Two tests once failed only under a full-suite run competing with a
corpus ingest — one tripped a Hypothesis health check during input generation,
the other computed a pacing wait by subtracting real elapsed time.

## Commit and PR style

Commit messages describe the software change. Lead with what broke or what is
new, then the measurement that justifies it. The repository's history is part
of its documentation — `git log` should read as an engineering record.

Pull requests should state which milestone they belong to, which acceptance
criteria they move, and the measured values. If you changed a number that
appears in `README.md` or `CLAUDE.md`, update it there too.
