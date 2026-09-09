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
make test-all   # includes integration tests (needs `make up`)
cascade doctor  # toolchain, pinned stack, service health
```

---

## 7. Architecture decisions

Seven ADRs in `docs/adr/`. Five correct defects found in the spec; two record
choices the spec left open.

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
| 1 | ≥ 1,300,000 chunks; exact count and per-source breakdown | **107,835** chunks from **22,173** documents — ccnews 94,011 / govpr 6,158 / wikipedia 6,280 / edgar 1,386 / gdelt 0. Breakdown printed by `corpus status` | **SHORT** — see below |
| 2 | Zero NULL, future or naive `published_at`, asserted over the full table | **0** NULL, **0** future, over all 22,173 rows (unqualified aggregates, never a sample). Naive dates are rejected before insert and counted as `naive_date` drops | **PASS** |
| 3 | 100% embedding coverage (`COUNT(*) WHERE embedding IS NULL` = 0) | **0** chunks without a vector; coverage **1.0000** | **PASS** |
| 4 | Dedupe collapse ratio reported; earliest-date retention verified on a fixture | Collapse ratio reported per run (measured **0.0191** on a single WARC file, 791 collapsed on a six-file unit). Earliest-date retention asserted in both arrival orders on a hand-built fixture, plus cross-batch and seeded-index cases | **PASS** |

Chunk quality: mean **413.6** tokens, max **512** — the cap holds over every
stored row. Date range 2015-01-08 → 2026-07-19. CI: ruff clean, black clean,
mypy strict clean (**49 files**), **432 unit + 43 integration = 475 tests**.

**Criterion 1 — throughput, not correctness.** Every invariant holds at the
measured scale; the shortfall is wall-clock. Measured end-to-end throughput is
**~93 chunks/s** (single WARC stream: 51 chunks/s; six parallel streams:
93 chunks/s, CPU-bound at ~50%), so 1.3M chunks is roughly **4 hours** of
continuous ingest. The pipeline is resumable per unit — `corpus_ingest_state`
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
