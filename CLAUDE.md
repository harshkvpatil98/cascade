# Cascade — build contract

Multi-agent causal simulation for strategic forecasting. Read this file before
writing code. The authoritative specification is
`Documents/Cascade_Technical_Specification.docx`; where this file and the spec
disagree, the spec wins — flag the disagreement rather than silently picking one.

---

## 1. The measurement contract

**These are targets, not results.** They do not exist until the harness
produces them. Build the harness so it *cannot* be steered toward them. If the
true Brier lands at 0.168, the report says 0.168 and every downstream claim is
restated. **Never write a target value into a report code path. Never hardcode
0.141.**

| Quantity | Target | Produced by |
|---|---|---|
| Backtested scenarios | 180 resolved binary questions | M1 |
| Evidence corpus | ≥ 1.30M pre-cutoff chunks | M2 |
| Retrieval p95 | < 15 ms, recall@20 > 0.92 | M3 |
| Mean actors / scenario | 14 (range 8–20) | M4 |
| Simulation horizon | 24 steps per run | M5 |
| Runs | 200 × 180 = 36,000 | M6 |
| Logged decision events | ≈ 4.2M (expected 4,199,040) | M6 |
| Cascade Brier | 0.141 | M7 |
| Single-model baseline | 0.203 (−30.5%) | M7 |
| Multi-agent, no decomposition | 0.176 (−19.9%) | M7 |
| Ablation: decomposition | ΔBrier +0.035 (leave-one-out) | M7 |
| Ablation: info asymmetry | ΔBrier +0.027 (leave-one-out) | M7 |
| Decomposition net of asymmetry | +0.008 — **publish this first** | M7 |
| Ablation cells | 12 | M7 |
| High-dispersion scenarios | σ > 0.3 in ≈ 31% | M7 |
| Calibration defect | ≈ 8 pts overconfident in the 0.70–0.90 bin | M7 |
| Replay | byte-identical event-log hash | M8 |
| Marginal cost | $0.0035 / run | M8 |
| Study cost | $290 total ($0.008/run fully loaded) | M8 |

Three mechanisms enforce integrity, and they are functional requirements:
**frozen splits** (scenario hash asserted at the start of every eval run),
**no outcome text in the loop** (separate table, separate grant), and a
**prompt-change audit** (`prompt_revisions`, before/after Brier).

---

## 2. The eight hard invariants

Each is enforced by a test, not by intention. `tests/unit/test_invariants.py`
enforces 1, 5 and 7 statically.

1. **`as_of` is never defaulted.** Not in Python, not in SQL, not in a test
   helper. A missing `as_of` is a `TypeError` at the Python boundary and a
   signature error at the SQL boundary. Defaults are how leakage gets in.
2. **The simulation never reads `scenario_labels`.** Enforced by a Postgres
   grant (migration 001, ADR-0005), not by code review. `cascade_sim` gets a
   permission error; only `cascade_eval` can read it.
3. **The arbiter contains no LLM call and no I/O.** Pure Python, ~400 lines.
   The single most important architectural decision in the system.
4. **One RNG per run**, seeded once from
   `blake2b(scenario_id|config_id|replicate, key=STUDY_SALT)`, drawn in a fixed
   documented order: exogenous walk → observation noise → tie-breaks → arbiter
   jitter. Never re-seeded mid-run.
5. **One LLM call site**: `cascade/llm/client.py`. A grep for the Anthropic SDK
   import anywhere else fails CI.
6. **The event log is append-only.** No `UPDATE`, no `DELETE`, ever.
7. **All iteration over collections is sorted.** Dict/set iteration order is a
   nondeterminism vector.
8. **Every phase is resumable.** Checkpoint and skip completed units on restart.

---

## 3. Pinned stack — do not substitute

Python 3.12 · LangGraph 0.2.x · PostgreSQL 16 + pgvector 0.8 ·
`BAAI/bge-small-en-v1.5` (384-d, local) · Claude Haiku 4.5 (agents) ·
Claude Sonnet 4.6 (compiler) · Langfuse self-hosted · DuckDB + Parquet ·
Typer + Rich · uv + Docker Compose · pytest + hypothesis.

`cascade doctor` asserts this list (`cascade/version.py`). If you believe a
substitution is warranted, write an ADR in `docs/adr/` and **ask**. Do not swap
silently.

---

## 4. Engineering standards

- **mypy strict** on `cascade/`. Pydantic v2 models at every subsystem
  boundary — a raw dict crossing a module boundary is a bug.
- **Pure core, thin shell.** `arbiter.py`, `validator.py`, `metrics.py`,
  `scheduler.py` take no clock, no RNG from global state, and do no I/O. That
  is what makes them property-testable.
- **Forward-only SQL migrations.** No ORM autogeneration — the partition
  strategy is load-bearing and must be explicit.
- **Tests first at each gate.** Write the acceptance criteria as failing tests,
  then make them pass.
- Ruff + black. Docstrings on every public function stating *what invariant it
  preserves*, not what it does.
- **No `except Exception: pass`** anywhere near the LLM client, the cost meter
  or the event log. The one sanctioned broad guard is in `tracing.py`, because
  observability must never fail a run; it is annotated and tested for.

### Exit codes

`0` ok · `1` unexpected error · `2` budget ceiling breached ·
`3` precondition failed · `4` cache miss in replay.

---

## 5. Session protocol

- **One milestone per session.** Start fresh context at each gate. Long agentic
  sessions across milestone boundaries are where architectural drift enters.
- At the end of each milestone: run the full suite, update the build log below,
  print the acceptance criteria with **measured** values, and **stop**.
- If an acceptance criterion cannot be met, **stop and report** with the
  measured value and a diagnosis. Do not relax the criterion, do not add a
  tolerance, do not mark it "approximately passing".

---

## 6. Commands

```bash
make install    # uv sync --extra dev
make up         # Postgres + Langfuse, waits for health
make migrate    # forward-only SQL migrations
make ci         # ruff + black + mypy strict + pytest
make test       # pytest, excluding tests needing live services
make test-all   # includes integration and leakage tests (needs `make up`)
make test-leakage  # the M3 time-lock probes alone
cascade doctor  # toolchain, pinned stack, service health

cascade retrieval index    # (re)build one IVFFlat index per chunks partition
cascade retrieval verify   # assert the Chronofence preconditions; exits 3 on drift
cascade retrieval bench    # p50/p95/p99 + recall@20; exits 3 if a criterion is missed
cascade retrieval memorization  # the parametric probe (costs money in record mode)

cascade compile build      # draft -> critique -> repair -> validate, per scenario
cascade compile status     # graph statistics and the repair-retry histogram
cascade compile verify     # re-validate and re-hash every stored graph; exits 3 on drift
cascade compile audit      # write the seeded 20-graph audit worksheet (§5.4)
cascade compile audit --score  # publish the audit mean; exits 3 below 1.5
```

---

## 7. Architecture decisions

Fourteen ADRs in `docs/adr/`. Seven correct defects found in the spec; the rest
record choices the spec left open.

| ADR | Decision | Milestone |
|---|---|---|
| 0001 | Static prompt prefix must exceed the model's 4,096-token cache floor — the spec's 1,900-token prefix silently does not cache and makes the §12.1 cost model a 4.8× underestimate | M0 |
| 0002 | `chronofence_search` needs `SECURITY DEFINER` + pinned `search_path`; as specified the app role gets a permission error on its own function | M3 |
| 0003 | `WorldState.relations` uses a canonical string key — `dict[tuple[str,str], float]` cannot round-trip through JSON, breaking the M5 bit-exact criterion | M5 |
| 0004 | `chunks` partitions start quarterly, not monthly — ~108 monthly partitions means ~100 index scans per late-cutoff query against a 15 ms budget | M3 |
| 0005 | Role grants deny by default (`REVOKE ... FROM PUBLIC`) and forbid superuser; the spec's `REVOKE ... FROM cascade_sim` is a no-op | M0 |
| 0006 | Langfuse pinned to v2 (Postgres only) rather than v3 (needs ClickHouse + Redis + MinIO) | M0 |
| 0007 | The cache key excludes `cache_control` markers, so prompt-cache tuning is a pure cost change and does not force a paid re-record | M0 |
| 0008 | Constraint precedence when the pool cannot satisfy §3.1: eligibility > base rate > domain cap > **N**. Resolves Q2 | M1 |
| 0009 | `named_parties` counts recognised institutional actors, not proper nouns; Metaculus now needs a token and is reported, never absorbed | M1 |
| 0010 | Corpus dates are read from the document, never the crawl; Wikipedia is anchored per-scenario; a bounded window replaces a DEFAULT partition | M2 |
| 0011 | The 512-token chunk cap is verified per chunk, not estimated -- sentence, then word, then character splitting | M2 |
| 0012 | IVFFlat index lifecycle is an operational command, not a migration -- `lists ≈ sqrt(rows)` is a function of measured data, and an index built on an empty partition is permanently degenerate | M3 |
| 0013 | `ivfflat.probes` is 40, not the spec's 10: quarterly partitioning fans one query across 36 index scans and probes applies per scan, so the single-index value measures recall@20 = 0.8331 against a > 0.92 criterion | M3 |
| 0014 | `OutcomeRule` and `UtilityTerm` -- referenced by §5.1 but never defined there -- are declarative, signed and monotone by construction, so §5.3's monotonicity rule holds for any rule that parses | M4 |

---

## 8. Open questions — resolve before the milestone that needs them

**Q1 (M7): ablation replicate count.** Appendix C gives the D factor as
`n ∈ {200, 1}` per cell, while §10.3 and §12.3 say the 11 non-headline cells
run at "90 scenarios × 30 replicates". These cannot both be literally true for
a D=200 cell. The likely reading is that D is the *design* factor and 30 is the
*budget* cap applied to non-headline cells, so a D=200 ablation cell is
executed at 30 replicates. `configs/ablations/*.yaml` currently encode the D
factor; `ensemble.ablation_replicates` (30) exists for the cap. **The M7 grid
driver must state which it applies, in the report,** because the ensemble
contribution estimate depends on it. Do not resolve this silently.

**Q2 (M1): base-rate control vs. the other inclusion rules. — RESOLVED at M1,
see ADR-0008.** Precedence, weakest sacrificed first: (1) eligibility is
absolute; (2) base rate ∈ [0.40, 0.60]; (3) domain cap ≤ 25%; (4) **N = 180
gives first**. `select()` returns the largest N ≤ 180 satisfying 1–3 and
reports N; `cascade ledger build` exits 3 on a shortfall. A smaller N is the
only failure mode that is visible in the output — the other two are selection
effects on the headline metric that nothing downstream could detect.

---

## 9. Build log

One line per completed milestone: what shipped, what the acceptance numbers
actually were, what was deferred.

### M0 — Foundations · *complete*

Shipped: repo scaffold per §13; `configs/base.yaml` (Appendix B) + typed
`Settings` with env override and secrets-from-env-only; `llm/client.py` as the
sole SDK call site with record/replay/live; content-addressed `llm/cache.py`;
exact-`Decimal` `llm/meter.py` with per-phase ceiling, atomic checkpointing and
`--estimate` extrapolation; Langfuse `llm/tracing.py` degrading to a no-op;
Typer CLI with all eight spec subcommands plus `db` and hidden `dev`; migration
001 (extensions + roles); `docker-compose.yml`; `Makefile`; 7 ADRs; 12 ablation
overlays.

**Measured acceptance values — 5 of 5 met.**

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| 1 | `make up` healthy; `cascade doctor` exits 0, prints every pinned dependency | `make up` brings both containers to **healthy**; `cascade doctor` (online, no `--offline`) exits **0** and prints all **20** pinned entries plus python 3.12.13, uv 0.12.3, docker 29.7.2, **PostgreSQL 16.10**, **pgvector 0.8.0**, Langfuse healthy | **PASS** |
| 2 | Record→replay round trip; replay makes **zero** network calls | Record 25/25 calls served; replay **0** network calls against an `httpx.MockTransport` that raises on any request. Replay additionally never constructs an SDK client (asserted with the API key unset). 12 tests | **PASS** |
| 3 | Cost meter vs. known-token fixture; USD matches to 6 dp | Exact `Decimal` equality at 6 dp across 5 fixtures. Independently reproduces the spec's own §12.1 derivation: $0.000338/call × 10.5 = **$0.003544/run** (spec: $0.0035). 21 tests | **PASS** |
| 4 | $0.01 ceiling aborts, checkpoints, exits non-zero | Real console script in a subprocess: exit code **2**, `simulate.checkpoint.json` written with resume state. Paired with a control asserting the same probe exits **0** under an unreachable ceiling | **PASS** |
| 5 | CI green: ruff, black, mypy strict, pytest | ruff **clean**, black **clean**, mypy strict **clean (20 files)**, pytest **185 passed** in 3.59 s; **17 integration passed** against live services (`make test-all`: **202 passed**) | **PASS** |

Scale: 2,420 lines in `cascade/` (20 modules), 2,658 lines of tests
(11 test modules), 202 tests total.

**Recovery event (2026-08-13).** The working tree and the git object store were
destroyed by the disk-full condition recorded in the previous entry: `git fsck`
reported **0 objects**, the branch was unborn, and 49 staged files were gone
from disk — all of `cascade/llm/`, all nine subsystem packages, all 12 ablation
overlays, all 7 ADRs and every test except `conftest.py`. `uv` and Python 3.12
were also gone from the host. Nothing was recoverable from git; the surviving
files were `config.py`, `cli.py`, `db.py`, `version.py`, `base.yaml`, migration
001, `docker-compose.yml`, `Makefile`, `pyproject.toml`, `uv.lock` and
`conftest.py`. M0 was rebuilt from those plus the spec. The index has been
repaired and `git fsck` is clean, but **there is still no commit** — commit
before doing anything else.

**Defects found and fixed during the rebuild** (each has a regression test):

- `base.yaml` documented `CASCADE_DB__SIM_PASSWORD` / `CASCADE_LANGFUSE__PUBLIC_KEY`.
  Secrets are top-level `Settings` fields, so the `__` nesting delimiter does
  not apply: those names bind to **nothing**, silently. Corrected to single
  underscore, and every documented secret variable is now asserted to bind.
- A `CASCADE_`-prefixed variable whose **root segment** is not a field was
  discarded silently by pydantic-settings — `CASCADE_ENSEMBLE_REPLICATES`
  (single underscore) set nothing and reported nothing. `extra="forbid"` only
  catches typos *inside* a known section. Now a validation error.
- `.env` was read only by the Makefile, so `cascade doctor` / `cascade db
  status` failed from a plain shell. `Settings` now reads it, located by
  `CASCADE_ENV_FILE`; the test suite points that at a nonexistent path **and**
  strips ambient `CASCADE_*`, so no test depends on machine state.
- `LANGFUSE_INIT_USER_EMAIL=cascade@localhost` fails Langfuse's own validator,
  crash-looping the container with no symptom but a 500 on `/health`.
- The Langfuse healthcheck probed `localhost`, which resolves to `[::1]` inside
  that image while Next.js binds only to `$HOSTNAME`. The probe could never
  pass; `make up` failed on a service that was serving correctly.
- `scripts/postgres-init/01-langfuse-db.sh` never ran — the bind mount reports
  the executable bit but refuses the exec (`bad interpreter: Permission
  denied`), and init scripts run before the server accepts connections, so the
  failure is easy to miss. Replaced with `.sql`, which has no exec bit to get
  wrong.
- `cascade doctor` ellipsised long version strings on a narrow terminal, which
  defeats the criterion that it *print* every pinned version. Columns now fold.

Deferred, with reasons:
- Batch API submission → M6. `complete()` raises `NotImplementedError` for
  `batch=True`; the meter and price table already model the 50% discount, so
  no accounting changes when it lands.
- Agent/compiler prompt authoring → M4/M5. `assert_cacheable_prefix()` is in
  place and will gate the prefix when the prompt exists (ADR-0001).
- ML and analytics extras (`torch`, `sentence-transformers`, `duckdb`,
  `pyarrow`, `langgraph`) are pinned but not installed at M0; `doctor` reports
  them as absent without failing, so an M0 checkout stays small.
- `tests/property/` and `tests/leakage/` exist as empty packages. Their subjects
  (the arbiter, Chronofence) land at M5 and M3; a property test with nothing to
  quantify over would be scaffolding, which §"anti-patterns" rules out.

### M1 — Scenario registry · *complete*

Shipped: migration 002 (`scenarios` / `scenario_labels` / `scenario_manifest`
with the invariant-2 grants); `cascade/ledger/` — four loaders, a
content-addressed source cache, pure `rules.py` / `select.py` / `manifest.py` /
`climatology.py` / `taxonomy.py`; `data/curated/historical.yaml` (16 entries
against a fixed template); `cascade ledger build|seal|verify|status`;
ADR-0008 and ADR-0009. Source cache: 2.9 GB, 52,763 raw candidates.

**Measured acceptance values — 4 of 4 met.**

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| 1 | Exactly 180 scenarios; YES rate in [0.40, 0.60]; no domain > 25%; print the domain histogram | **180** scenarios; YES rate **0.5000**; max domain share **0.2500**; histogram printed across 11 domains | **PASS** |
| 2 | `manifest.sha256` written; a test that mutates one label asserts the check fails | sha256 `30d9c61d…` sealed and re-verified. Mutation caught in both directions: a flipped label and a moved `cutoff_ts` each raise `ManifestMismatch`, on fixtures **and** on real stored rows. 14 tests | **PASS** |
| 3 | Grant test: `cascade_sim` selecting from `scenario_labels` raises `InsufficientPrivilege` | `cascade_sim` → **InsufficientPrivilege** on `scenario_labels`, **180 rows** on `scenarios`; `cascade_eval` reads both. Enforced by the absence of a grant, not a REVOKE (ADR-0005) | **PASS** |
| 4 | Climatology Brier computed and stored | base rate **0.500000**, climatology Brier **0.250000**, stored in `scenario_manifest` with the split it describes. Asserted against the p(1−p) identity | **PASS** |

Composition: party rules — `event_siblings` 153, `named_parties` 18,
`curated` 9. Sources — polymarket 164, manifold 7, curated 9. Pool: 34,613
Polymarket + 18,134 Manifold + 16 curated raw candidates; rejections
low_volume 22,262, insufficient_parties 7,448, horizon_too_short 18,564,
single_quantity 2,985. Cutoffs span 2018-01-12 → 2026-09-01.

**How the earlier 57-scenario shortfall was cleared.** Two fixes, no rule
relaxed:

1. **Polymarket keyset pagination.** The offset endpoint refuses offsets past
   ~2,000, which capped the reachable archive at ~1,900 questions — and since
   the ≥3-party rule is carried mostly by Polymarket's event structure, that
   cap *was* the binding constraint on the whole set. The cursor parameter is
   `after_cursor`, taken from the service's own `/openapi.json` rather than
   guessed. Raw Polymarket candidates went 1,908 → **34,613**.
2. **The actor lexicon was extended to corporations and competition
   regulators** (96 → 241 entries). It previously held states and regulators
   but almost no companies, which excluded exactly the merger-review and
   corporate-event episodes §3.1 asks for.

**Metaculus remains unavailable** and is reported, never absorbed: its API
rejects unauthenticated requests, and the loader activates on
`CASCADE_METACULUS_TOKEN`. The set is therefore sourced from Polymarket,
Manifold and the curated file — a documented deviation from §3.1's
"~90 from Metaculus".

**Defects found and fixed at M1** (each has a regression test):

- `named_parties` counted **proper nouns, not parties** — admitting "will the
  Blaze Star go nova", "will Diddy be alive", "will the NYT review use >5 em
  dashes". 95 of 163 scenarios had entered that way. Now restricted to
  recognised institutional actors, failing closed (ADR-0009).
- The proper-noun pattern joined names across "and", turning *"Russia and
  Ukraine"* into a phantom **third** party.
- The stop-word list was case-sensitive, so **"YES"** — which opens nearly
  every market resolution criterion — counted as a party, inflating every
  market question by one.
- Nested names were collapsed *after* aliasing, so "European Parliament" and
  "Parliament" counted as two parties for one body. Collapsing now happens on
  the raw names, before aliases.
- The Polymarket pagination boundary was never recorded, so the next cached
  build asked for a page that was never stored, read the resulting
  `SourceOffline` as "source unreachable", and silently dropped **every**
  Polymarket scenario.

Additional decisions recorded at M1:
- **One scenario per real-world event.** A Polymarket event carries one market
  per candidate; 60 markets under "World Cup Winner" are 60 views of one event.
  Admitting several would inflate the effective sample the M7 paired bootstrap
  treats as independent. The representative is chosen by a keyed hash of the
  question — **outcome-independent**, because always taking the YES leg would
  destroy the base-rate control by construction.
- **The curated set requires provenance.** A curated label nobody can check is
  indistinguishable from an invented one, and a wrong label corrupts the
  headline metric silently. The 16 shipped entries are authored from the public
  record and are **not independently verified** — verify before publication.
- `canonical_json` moved to `cascade/canonical.py`: the LLM cache key, the
  manifest hash and (at M8) the event-log hash must agree on bytes, so there is
  one definition.

### M2 — Evidence corpus · *pipeline complete, chunk target not reached*

Shipped: migration 003 (`documents` / `chunks` partitioned quarterly per
ADR-0004, `corpus_ingest_state`, no DEFAULT partition); `cascade/corpus/` —
five source adapters, pure `normalize.py` / `chunker.py` / `simhash.py`,
`embed.py` on the pinned `BAAI/bge-small-en-v1.5`, COPY-based `store.py`,
resumable `pipeline.py`; `cascade corpus build|status|verify`; ADR-0010 and
ADR-0011.

**Measured acceptance values** (3 of 4 met; 1 short on volume, not correctness):

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| 1 | ≥ 1,300,000 chunks; exact count and per-source breakdown | **409,899** chunks from **100,339** documents — ccnews 396,075 / wikipedia 6,280 / govpr 6,158 / edgar 1,386 / gdelt 0. Breakdown printed by `corpus status` | **SHORT** — see below |
| 2 | Zero NULL, future or naive `published_at`, asserted over the full table | **0** NULL, **0** future, over all 100,339 rows (unqualified aggregates, never a sample). Naive dates are rejected before insert and counted as `naive_date` drops | **PASS** |
| 3 | 100% embedding coverage (`COUNT(*) WHERE embedding IS NULL` = 0) | **0** chunks without a vector; coverage **1.0000** | **PASS** |
| 4 | Dedupe collapse ratio reported; earliest-date retention verified on a fixture | Collapse ratio reported per run (measured **0.0191** on a single WARC file, 791 collapsed on a six-file unit). Earliest-date retention asserted in both arrival orders on a hand-built fixture, plus cross-batch and seeded-index cases | **PASS** |

Chunk quality: mean **413.6** tokens, max **512** — the cap holds over every
stored row. Date range 2015-01-08 → 2026-07-19. CI: ruff clean, black clean,
mypy strict clean (**49 files**). Test totals at the close of M2 were restated
at M3 against a measured collection — see the M3 entry; the figure recorded
here originally (475) did not match what pytest collects.

**Criterion 1 — throughput, not correctness.** Every invariant holds at the
measured scale; the shortfall is wall-clock. Measured end-to-end throughput is
**~93 chunks/s** (single WARC stream: 51 chunks/s; six parallel streams:
93 chunks/s, CPU-bound at ~50%), so 1.3M chunks is roughly **4 hours** of
continuous ingest. Ingest continued after this entry was first written and the
corpus stands at **409,899 chunks / 100,339 documents** (31.5% of target); the
counts above are the current measured values, not those of the original run. The pipeline is resumable per unit — `corpus_ingest_state`
skips completed units, verified as 40 units skipped on a re-run — so reaching
the target is a matter of running `cascade corpus build` until it does.
`cascade corpus verify` exits **3** while short, so the gap cannot be mistaken
for success.

**GDELT contributed zero.** The adapter is written and was verified returning
articles earlier in the session. GDELT enforces one request per five seconds;
an early retry burst from this client — backoff started at 1 s, below GDELT's
own floor — earned an IP-level throttle that outlasted the run, and it now
answers 429 even at 20-second intervals. The backoff bug is fixed (retries now
never wait less than the source's own interval, and honour `Retry-After`); the
three units are recorded `failed`, and since only `done` units are skipped they
retry automatically on the next run.

**Defects found and fixed at M2** (each has a regression test):

- **Chunks exceeded the 512-token cap in three distinct ways**, all measured in
  real data. Overlap carried an oversized sentence into the next chunk
  (**1,022** tokens); summed per-sentence estimates understate the joined
  string; and word-splitting cannot divide text with no spaces — Japanese prose
  and a minified JSON blob reached **2,125** tokens. The cap is now *verified*
  per chunk, splitting at sentence, then word, then character boundaries
  (ADR-0011). Two documents written before the fix were purged.
- **The hard-split path was quadratic**: it re-measured the growing prefix
  after every word, and SEC filings routinely contain single "sentences" of
  thousands of words. Each word is now measured once.
- **NUL bytes aborted the COPY of an entire batch.** Scraped HTML and SGML
  filings carry them routinely; control characters are now stripped at the one
  funnel every document passes through.
- **Out-of-window dates aborted a batch too.** CC-NEWS re-crawls archive pages,
  so a 2016 crawl yielded a 2013 article, and with no DEFAULT partition
  (deliberately — it could never be pruned) it had nowhere to go. There is now
  an explicit corpus window, asserted against the DDL by an integration test.
- **Unsorted iteration in the pipeline** (invariant 7) — caught by the static
  invariant test, not by review.
- **SEC EDGAR returns 403** to any User-Agent without a contact address; a
  blocked client looks exactly like a source with no documents. The agent is
  now configurable via `corpus.contact`.
- **CC-NEWS months before 2016-08 do not exist**; generating them turned a
  known gap into a stream of fetch failures.

Improvements beyond the roadmap, implemented rather than suggested:

- **Parallel WARC streaming.** Single-stream ingest ran at 26% CPU — almost all
  wall time waiting on a ~1 GB download. Files are independent, so they stream
  concurrently: 51 → 93 chunks/s.
- **Batched tokenization.** Per-sentence tokenizer calls dominated ingest CPU;
  the fast tokenizer batches them into one call per document. 28.1 → 20.6 ms
  per document.
- **Bounded write batches.** A CC-NEWS month can hold ~100k chunks; they are
  now flushed in configurable batches so memory is bounded and progress is
  durable within a unit.
- **Per-source politeness budgets.** Each source gets its own `Fetcher`, so
  GDELT's five-second floor cannot throttle EDGAR and EDGAR's retries cannot
  spend GDELT's allowance.
- **`corpus_stats` reads the denormalised `n_chunks`** rather than joining two
  partitioned tables on a non-partition key, which at corpus scale degenerates
  into a full scan of both.

Deferred, with reasons:
- **CC-News via HuggingFace `datasets`** — the pinned stack has no such
  dependency, and adding one is a substitution requiring an ADR. The Common
  Crawl WARC path is parsed with the standard library instead and needs no new
  dependency.
- **IVFFlat indexes over the vectors** → M3. Index build is part of the
  Chronofence latency work (§4.2) and wants the final row counts; building one
  now would only have to be rebuilt.

### M3 — Chronofence · *complete, 4 of 5 acceptance criteria met*

Shipped: migrations 004–007 (`chronofence_search` and `chronofence_search_exact`
as SECURITY DEFINER with pinned `search_path` and `ivfflat.probes` per ADR-0002;
the `chronofence_partitions` view; grants); `cascade/retrieval/` — pure
`metrics.py` / `queries.py` / sizing in `index.py`, the `Chronofence` client,
the 10,000-query `bench.py`, `leakage.py` and `memorization.py`;
`cascade retrieval index|verify|bench|memorization`; the leakage suite in
`tests/leakage/` and the date-monotonicity property test in `tests/property/`;
ADR-0012 and ADR-0013.

**Measured acceptance values — 4 of 5 met; 1 blocked on a credential.**

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| 1 | p95 < 15 ms over 10,000 queries; print the full histogram | **p95 13.43 ms** (p50 10.95, p99 14.60, min 8.52, max 45.63) over **10,000** queries spanning **180** distinct cutoffs, 2018-01-12 → 2026-07-19. Histogram printed: 13.60% under 10 ms, 72.59% in 10–12.5, 13.23% in 12.5–15, 0.58% above. **0** empty results | **PASS** |
| 2 | recall@20 > 0.92 vs exact search | **0.9372** over a 500-query sample against `chronofence_search_exact`; perfect on 293/500, worst query 0.2500; 0 sampled queries had no admissible evidence | **PASS** |
| 3 | Poison-pill: 0 of 500 retrieved across all 180 scenarios | **0 of 500**. 500 synthetic post-resolution documents inserted, queried at all 180 cutoffs with the poison's own text — the most favourable query it could receive. A positive control asserts the poison *is* retrievable with the time filter removed, and a post-run check asserts the corpus is unchanged | **PASS** |
| 4 | Date-monotonicity property test over the full trace | **PASS** over 120 Hypothesis examples on the indexed path and 60 on the exact oracle, plus a monotonicity-in-`as_of` property. `published_at >= as_of` is treated as a violation — the promise is *strictly* before | **PASS** |
| 5 | `memorization_score` for all 180; distribution reported | **NOT RUN** — `CASCADE_ANTHROPIC_API_KEY` is empty in this environment. Implemented, CLI-wired and tested end to end against a mock transport (33 tests); `cascade retrieval memorization` exits **3** with the variable named | **BLOCKED** |

CI: ruff clean, black clean, mypy strict clean (**57 files**), **612 tests**
(520 unit + 12 determinism + 55 integration + 21 leakage + 4 property), all
passing. This restates the M2 figure, which recorded 475 against a collection
that measures 431 at that commit.

**Criterion 5 — a credential, not a defect.** The probe asks the agent model
each question with zero context and scores `2*|p-0.5|`, deliberately
direction-free: a confidently *wrong* prior steers the simulation as much as a
right one. An unparseable answer is reported as unparseable, never scored as
0.5 — for this number, failing toward "no memorisation detected" would read as
reassurance. Run it before publication; it is ~180 Haiku calls (well under the
$5 `bench` ceiling) and the result belongs beside the headline Brier.

**The corpus is concentrated where the scenarios are not.** 88.8% of the
409,899 chunks fall in 2016q3–2017q2, while scenario cutoffs span 2018-01 to
2026-09. Retrieval is therefore *correct* — nothing post-cutoff is ever
returned — but for late-cutoff scenarios the nearest admissible evidence is
often years stale. That is an M2 coverage problem surfaced by M3, not a
Chronofence defect, and it is why criterion 1's latency barely varies with the
cutoff: almost every query scans the same five large partitions. It needs
closing before the M7 headline means what it claims.

**Defects found and fixed at M3** (each has a regression test):

- **The provenance join cost 5x the entire latency budget.** `chronofence_search`
  joined `documents` for source/url/title. SECURITY DEFINER and a `SET` clause
  each independently disqualify a SQL function from inlining, so the planner
  had no constant for `published_at` and joined against all 52 partitions:
  15.37 ms against a 2.95 ms body. `CROSS JOIN LATERAL ... LIMIT 1` makes the
  partition key a runtime constant, so executor pruning probes exactly one
  partition per hit — **3.34 ms**. The same query *inlined* costs 4.25 ms,
  which is why this never looked like a slow query, only a slow function
  (migration 006).
- **`chronofence_partitions` returned two rows per partition.** The opclass
  predicate sat on the second of two LEFT JOINs, which nulls non-matching rows
  rather than removing them; every partition carries a primary key and a
  document index. 104 rows for 52 partitions, which would have made the index
  command plan each partition twice and `CREATE INDEX` fail on the second pass
  (migration 005).
- **`probes = 10` measured recall@20 = 0.8331.** The spec's value assumes one
  index; partitioning applies probes per scan. 40 is the smallest value on the
  measured curve clearing 0.92 (ADR-0013). `lists` was varied too and the
  spec's `sqrt(rows)` rule is the best of four tried.
- **`::halfvec(384)` was denied to every role.** A typmod cast runs pgvector's
  `halfvec(halfvec,integer,boolean)` coercion function, and migration 001
  revokes EXECUTE on every function in `public` from PUBLIC (ADR-0005), which
  catches pgvector's own. The client casts to bare `halfvec` — the type input
  function, which is not privilege-checked — and validates the width in Python
  with a better message than Postgres would give.
- **An empty `CASCADE_ANTHROPIC_API_KEY` bypassed the client's guard.**
  `KEY=` in a `.env` parses to `SecretStr("")`, not `None`, so the SDK raised
  `TypeError: Could not resolve authentication method` — an error naming
  neither the variable nor this project.
- **`percentile()` could report p99 < p95.** `low*(1-w) + high*w` lands one ULP
  low when `low == high`; the numpy-equivalent `low + (high-low)*w` is exact.
  Found by a property test, and it would have been dismissed as noise in a
  bench report.
- **GDELT's refusals were recorded as this client's parse errors.** The notice
  arrives under HTTP 200 as readily as 429, so it reached the JSON parser and
  all three units died as `returned non-JSON`. A marker whitelist fixed two of
  three; the rule that works is that a 200 which is not JSON is not an answer,
  because every request carries `format=json` (`Fetcher.classify_body`).
- **`/dev/shm` was Docker's 64 MB default.** A parallel aggregate over 409,899
  chunks failed with `DiskFull: could not resize shared memory segment` while
  the host had 801 GB free. `shm_size: 1gb` in `docker-compose.yml`.
- **The static `as_of` DEFAULT check fired on prose.** Migration 004's
  `COMMENT ON FUNCTION ... 'as_of has no default by design'` tripped the
  invariant it documents. The check now strips `--` comments and string
  literals before matching, which cannot hide a real default — a parameter
  default is executable SQL — and a regression test asserts both real forms
  are still caught.

Improvements beyond the roadmap, implemented rather than suggested:

- **`cascade retrieval verify`** asserts the preconditions structurally: the
  deployed function's pinned probes equals config, every non-empty partition
  carries a correctly sized index, and `cascade_sim` is denied on `chunks` and
  `documents`. Exits 3 on drift, so recall rot is a loud failure.
- **A recall oracle as a separate function.** `chronofence_search_exact` is
  granted to `cascade_eval` only, so recall has a ground truth while the
  simulation keeps exactly one corpus read path.
- **Deterministic tie-breaking.** `ORDER BY distance, chunk_id` over the k
  returned rows, so equal distances cannot reorder between replays — M8 has to
  hash the event log byte-identically.
- **A positive control in the poison-pill probe.** Without it, a retrieval path
  that returned nothing at all would pass every leakage assertion.

Deferred, with reasons:
- **The 1.3M-chunk corpus target and its coverage skew** → M2 work, not M3.
  `cascade corpus verify` still exits 3 while short.
- **GDELT** remains externally throttled from this network; all three units are
  recorded `failed` with an accurate `RateLimited` diagnostic and retry
  automatically.
- **`retrieval.k_agent` / `k_compiler` tuning** → M4/M5, when there is an agent
  whose answers can be scored against the k it was given.

### M4 — Lathe · *implemented and integration-tested; acceptance run blocked on a credential*

Shipped: `cascade/decompose/` — `schema.py` (the §5.1 `CausalGraph` contract,
plus `UtilityTerm` and `OutcomeRule` which §5.1 references but never defines,
see ADR-0014), pure `validator.py` (all six §5.3 rules), `prompts.py` (tool
schemas generated from the Pydantic models), three-pass `compiler.py`,
`store.py` and `audit.py`; migration 008 (`causal_graphs`,
`causal_graph_failures`, grants); `cascade compile build|status|verify|audit`;
`tool_choice` added to `LLMRequest` and its cache domain.

**Acceptance — not measurable in this environment.** All four criteria are read
off 180 compiled graphs, and compilation needs ~540 Sonnet calls.
`CASCADE_ANTHROPIC_API_KEY` is empty here, so no graph has been compiled by a
model and **no acceptance number in §M4 is claimed**.

| # | Criterion | Status |
|---|---|---|
| 1 | 180/180 pass the validator within 2 repair attempts; report the retry histogram | **NOT RUN** — histogram is implemented (`causal_graphs.repair_retries`, printed by `compile status`) and has no data |
| 2 | Mean actor count 14 ± 2; factor count within [4, 12] | **NOT RUN** — bounds are enforced at parse time and by a CHECK constraint; the mean needs real graphs |
| 3 | Recompiling an unchanged input reproduces the graph hash | **PASS, mechanism only** — asserted on canonical hashing, on emission-order invariance, at the compiler boundary, and against a tampered database row |
| 4 | Human audit: 20 seeded graphs, mean ≥ 1.5, rubric written to `reports/audit/` | **NOT RUN** — sampler, rubric and scorer are implemented and tested; there are no graphs to sample |

What *is* verified without a model: the full pipeline end to end against the
live database, with only the LLM stubbed — Chronofence retrieval at the cutoff,
the real 384-dimension embedder driving the two semantic rules, the upsert, the
hash round-trip, and the `cascade_sim` grant M5 will read graphs through.
One integration test pins the rule §5.2 calls the most common failure: the
pinned model scores a verbatim restatement of the outcome **above** the 0.85
threshold and an independent interest **below** it, so objective-independence
catches what it is meant to catch.

CI: ruff clean, black clean, mypy strict clean (**63 files**), **743 tests**
(634 unit + 12 determinism + 66 integration + 21 leakage + 10 property).

**Design decisions recorded:**

- **`OutcomeRule` is monotone by construction** (ADR-0014). §5.1 names the type
  and never defines it; §7.2 calls it as `outcome_rule(world.factors) -> [0,1]`
  and §5.3 requires monotonicity in each factor. A signed-weight logistic makes
  the partial derivative's sign exactly the sign of the weight everywhere, so
  the rule cannot parse and be non-monotone. The validator checks it numerically
  anyway, to catch a future rule form added without re-reading the ADR.
- **The model never emits `scenario_id`.** The compiler attaches it. A
  hallucinated or reformatted id would key a graph to the wrong scenario and
  surface at M7 as a scenario scored against someone else's decomposition.
- **Tool schemas are generated from the Pydantic models**, never written
  alongside them, so the schema the model is held to and the schema the
  validator enforces cannot drift.
- **A skipped validator rule is not a passed one.** Without an embedder the two
  semantic rules report as `skipped` and `ValidationReport.ok` is False.
- **Prompt caching is deliberately off at M4.** Measured: the draft prefix
  (tool schema + system prompt) is ~3,384 tokens, below the provider's
  4,096-token floor, where Anthropic silently declines to cache while still
  charging the write premium (ADR-0001). Padding to reach the floor would buy
  ~$2 across the milestone in exchange for inflating every call with filler.
  M5's agent prompts run 36,000 times and are where caching pays.

**Defects found and fixed at M4** (each has a regression test):

- **GDELT's zero contribution was misdiagnosed at M2 and M3.** The DOC 2.0 API
  serves a rolling recent window, not the archive, and answers anything older
  with HTTP 200 and the 26-byte body `Invalid query start date.` Measured at
  the boundary: 2016-01 and 2016-12 refused, **2017-01 returns articles**,
  2020/2024/2026 fine. `corpus.start_year` is 2016 because that is when
  CC-NEWS begins, so every GDELT unit this study generated asked for a window
  the API will never serve — 96 doomed requests per excluded year against a
  five-second politeness floor, recorded as throttling. `unit_keys` now clamps
  to `COVERAGE_START_YEAR = 2017`, mirroring what `ccnews.unit_keys` already
  does for its own collection start. The three stale `failed` rows were
  removed; a live single-unit run now reaches 2017-01 and fails only on this
  IP's throttle.
- **A permanent refusal was retried as a transient one.** `classify_body` had
  two verdicts, both retried. A third, `rejected`, fails immediately: retrying
  a refusal the source will never change spends the politeness budget the rest
  of the ingest needs and reports a fixed configuration error as an outage.
- **The validator iterated two mappings unsorted** (invariant 7). Both sorted
  their output, so the result was deterministic, but the static check is
  deliberately blunt and uniform compliance is what keeps it meaningful.
- **The test fixture's factor names were near-duplicates.** "Distinct driver
  number 1/2/3" reads as distinct to a hashing stub and as near-identical to
  the pinned model, so the fixture passed the orthogonality rule under a fake
  embedder and failed it under the real one — found by the integration test,
  which is the reason it exists.

**Corpus (M2 criterion 1) — narrowing, not closed.** Ingest of the remaining
CC-NEWS units is running and has taken the corpus from 409,899 to **474487**
chunks across **117839** documents. Measured throughput this session is ~1,700
chunks/minute, so the 1.30M target is roughly eight more hours of continuous
ingest; the pipeline is resumable per unit, so `cascade corpus build --source
ccnews` continues from where it stopped. `cascade corpus verify` still exits 3
while short. Note that a materially larger corpus drifts the IVFFlat `lists`
sizing — re-run `cascade retrieval index` and `cascade retrieval bench` once
ingest settles, because ADR-0013 records that recall falls as partitions grow
at fixed probes.

Deferred, with reasons:
- **Every M4 acceptance number** → needs `CASCADE_ANTHROPIC_API_KEY`. Budget is
  the `compile` phase ceiling ($40); the estimate for 540 Sonnet calls at
  k_compiler = 60 is roughly $35, which is inside it but not by much.
- **The M3 parametric probe** (`cascade retrieval memorization`) → same
  credential.
- **GDELT ingest** → the adapter is correct and now targets a servable window,
  but this IP is throttled hard enough that 768 units is impractical from here.
