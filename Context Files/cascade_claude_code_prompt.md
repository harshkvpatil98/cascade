# Cascade — Claude Code Build Prompt

> **How to use this file.**
> Paste **Part 0** at the start of every session (or better: let Claude Code write it into `CLAUDE.md` on the first session, after which it is read automatically).
> Then paste **exactly one milestone block** per session. Do not run two milestones in one session — that is where architectural drift comes from.
> Attach `Cascade_Technical_Specification.docx` to the first session and tell Claude Code to read it before M0.

---

## PART 0 — Master brief (paste every session / put in CLAUDE.md)

You are building **Cascade**, a multi-agent causal simulation system for strategic forecasting, end to end. The full technical specification is in `Cascade_Technical_Specification.docx` (attached / in repo root). Read it before writing code. Where this prompt and the spec disagree, the spec wins; flag the disagreement instead of silently picking one.

### What Cascade does

Given a strategic question ("will this coalition hold through Q3?"), Cascade:

1. **Decomposes** it into a typed causal graph — actors with objectives, resources and constraints; factors with volatility and inertia; signed, lagged influence edges.
2. Assigns each actor to an **LLM agent** that observes only a projection of world state defined by a derived **information-asymmetry policy** (per-channel noise, lag, quantization).
3. **Simulates** 24 discrete steps under a **deterministic, non-LLM arbiter**.
4. Runs each scenario **200 times** under different seeds and collapses the outcome distribution to a probability plus a dispersion measure.
5. Is validated by a **180-scenario backtest** with a **12-cell ablation grid**, bit-exact replay, and a reconciled cost ledger.

### The measurement contract

The project is done when a reproducible study produces these, written to a report artifact:

| Quantity | Target |
|---|---|
| Backtested scenarios | 180 resolved binary questions |
| Evidence corpus | ≥ 1.30M pre-cutoff chunks |
| Retrieval p95 | < 15 ms, recall@20 > 0.92 |
| Mean actors / scenario | 14 (range 8–20) |
| Runs | 200 × 180 = 36,000 |
| Logged decision events | ≈ 4.2M |
| Cascade Brier | 0.141 |
| Single-model baseline | 0.203 (−30.5%) |
| Multi-agent, no decomposition | 0.176 (−19.9%) |
| Ablation: decomposition | ΔBrier +0.035 (leave-one-out) |
| Ablation: info asymmetry | ΔBrier +0.027 (leave-one-out) |
| Ablation cells | 12 |
| High-dispersion scenarios | σ > 0.3 in ≈ 31% |
| Calibration defect | ≈ 8 pts overconfident in the 0.70–0.90 bin |
| Replay | byte-identical event-log hash |
| Cost | $0.0035 marginal/run; $290 study total |

**These are targets, not results.** Build the harness so it *cannot* be steered toward them. If the true Brier is 0.168, the report says 0.168. Never write a target value into a report code path. Never tune a prompt after seeing test metrics without logging it in `prompt_revisions` with the before/after Brier.

### Hard invariants — every one has a test, none is enforced by intention

1. **`as_of` is never defaulted.** Not in Python, not in SQL, not in a test helper. A missing `as_of` is a `TypeError` at the Python boundary and a signature error at the SQL boundary.
2. **The simulation process never reads `scenario_labels`.** Enforced by a Postgres role grant, not code review. `cascade_sim` gets a permission error; only `cascade_eval` can read it.
3. **The arbiter contains no LLM call and no I/O.** Pure Python, ~400 lines. This is the single most important architectural decision in the system.
4. **One RNG per run**, seeded once from `blake2b(scenario_id|config_id|replicate, key=STUDY_SALT)`, drawn in a fixed documented order. Never re-seeded mid-run.
5. **One LLM call site**: `cascade/llm/client.py`. A grep for the Anthropic SDK import anywhere else fails CI.
6. **The event log is append-only.** No `UPDATE`, no `DELETE`, ever.
7. **All iteration over collections is sorted.** Dict/set iteration order is a nondeterminism vector; a lint rule catches it.
8. **Every phase is resumable.** Checkpoint and skip completed units on restart.

### Pinned stack — do not substitute

Python 3.12 · LangGraph 0.2.x · PostgreSQL 16 + pgvector 0.8 · `BAAI/bge-small-en-v1.5` (384-d, local) · Claude Haiku 4.5 (agents) · Claude Sonnet 4.6 (compiler) · Langfuse self-hosted · DuckDB + Parquet (analytics) · Typer + Rich (CLI) · uv + Docker Compose · pytest + hypothesis

If you believe a substitution is warranted, write an ADR in `docs/adr/` and *ask*. Do not swap silently.

### Engineering standards

- **mypy strict** on `cascade/`. Pydantic v2 models at every subsystem boundary — a raw dict crossing a module boundary is a bug.
- **Pure core, thin shell.** `arbiter.py`, `validator.py`, `metrics.py`, `scheduler.py` take no clock, no RNG from global state, and do no I/O. That is what makes them property-testable.
- **Forward-only SQL migrations.** No ORM autogeneration — the partition strategy is load-bearing and must be explicit.
- **Tests first at each gate.** Write the acceptance criteria as failing tests, then make them pass.
- Ruff + black. Docstrings on every public function stating *what invariant it preserves*, not what it does.

### Session protocol

- Maintain `CLAUDE.md` at repo root: invariants above + a build log with **one line per completed milestone** (what shipped, what the acceptance numbers actually were, what you deferred).
- At the end of each milestone: run the full test suite, update `CLAUDE.md`, print the acceptance criteria with **measured** values, and **stop**. Do not begin the next milestone.
- If an acceptance criterion cannot be met, **stop and report** with the measured value and your diagnosis. Do not relax the criterion, do not add a tolerance, do not mark it "approximately passing."

### Anti-patterns — do not do these

- Writing all nine subsystems as stubs and "wiring them later." Each milestone ends with something that *runs*.
- Making the arbiter an LLM because conflict resolution "needs judgement." It does not. It needs a contest function.
- Personas as a substitute for information asymmetry. Different tone is not different information; it is worth ~0 Brier.
- `except Exception: pass` anywhere near the LLM client, the cost meter, or the event log.
- Notebook-driven pipelines. Every phase is a CLI subcommand.
- Retrieval helpers with a default `as_of`. This is how the entire study gets silently invalidated.
- Hardcoding 0.141 anywhere.

---

## PART 1 — Milestone blocks (paste one per session)

### M0 — Foundations

Build the skeleton: repo, infra, the LLM client, the cost meter, the CLI.

**Deliver**

- `pyproject.toml` (uv), `docker-compose.yml` (Postgres 16 + pgvector 0.8, Langfuse), `Makefile` with `up / migrate / seed / test / study / report`.
- `cascade/config.py` — pydantic-settings, loads `configs/base.yaml`. **No bare `os.environ` reads anywhere else in the codebase.**
- `cascade/llm/client.py` — the *only* module that imports the Anthropic SDK. Three modes:
  - `record` — cache miss calls the API and persists `(key, response, usage, latency)`
  - `replay` — cache miss raises `CacheMiss`; **never** falls back to the network
  - `live` — bypasses the cache; used only by the latency benchmark
  Cache key = `sha256(canonical_json({model, system, messages, tools, temperature, prompt_rev}))`.
- `cascade/llm/meter.py` — token and USD accounting, per-phase budget ceiling, `abort_on_breach`. Aborting writes a resumable checkpoint and exits non-zero. Never a warning.
- `cascade/cli.py` — Typer app. Subcommands stubbed: `doctor`, `ledger`, `corpus`, `compile`, `simulate`, `evaluate`, `trace`, `report`.
- Langfuse wiring: trace per run → span per step → generation per LLM call. Cache hits logged as zero-cost generations so hit rate and spend appear on the same dashboard.
- Migration 001: extensions, roles `cascade_sim` / `cascade_eval`.

**Acceptance (report measured values)**

- [ ] `make up` brings Postgres and Langfuse to healthy; `cascade doctor` exits 0 and prints versions of every pinned dependency.
- [ ] Record→replay round-trip test passes; the replay pass makes **zero** network calls (assert with a patched transport that raises).
- [ ] Cost meter unit-tested against a fixture with known token counts; computed USD matches to 6 decimal places.
- [ ] Budget-ceiling test: a phase configured with a $0.01 ceiling aborts, checkpoints, and exits non-zero.
- [ ] CI green: ruff, black, mypy strict, pytest.

---

### M1 — Scenario registry (`ledger`)

180 resolved binary questions, sealed.

**Deliver**

- Loaders for Metaculus (resolved binary, API), Polymarket/Manifold (resolved, volume-filtered), and a hand-curated historical set against a fixed template in `data/curated/*.yaml`.
- Inclusion rules enforced in code, not prose: binary and unambiguous; **≥ 3 distinguishable parties with non-identical objectives** (single-quantity trend questions are out of scope); `resolve_ts ≥ cutoff_ts + 21 days`; no domain > 25% of the set.
- Base-rate control: resolved-YES rate in **[0.40, 0.60]** by construction. Also compute and store the climatology baseline.
- `scenarios` and `scenario_labels` as **separate tables with separate grants** (invariant 2).
- `cascade ledger seal` → writes `manifest.sha256` over `(scenario_id, cutoff_ts, resolve_ts, outcome)` for all 180. Every later phase asserts this hash on startup.

**Acceptance**

- [ ] Exactly 180 scenarios; YES rate in [0.40, 0.60]; no domain > 25%; print the domain histogram.
- [ ] `manifest.sha256` written; a test that mutates one label and asserts the manifest check fails.
- [ ] Grant test: connecting as `cascade_sim` and selecting from `scenario_labels` raises `InsufficientPrivilege`.
- [ ] Climatology Brier computed and stored.

---

### M2 — Evidence corpus (`quarry`)

≥ 1.30M date-stamped, embedded chunks.

**Deliver**

- Ingest: GDELT 2.0 (~600k), CC-News 2016–2024 (~420k), **Wikipedia revision snapshots** (~180k — the revision *as of the cutoff*, never the current article), SEC EDGAR 8-K/DEF 14A (~70k), gov/IGO releases (~40k).
- Pipeline: `fetch → normalize → date-validate → dedupe → chunk → embed → index`.
  - **date-validate**: reject if `published_at` is NULL, in the future, or timezone-naive. A document with an uncertain date is a leakage vector — drop it, never infer it.
  - **dedupe**: 64-bit SimHash, Hamming ≤ 3 collapses; keep the **earliest** `published_at`.
  - **chunk**: 512 tokens, 64 overlap, sentence-boundary aligned.
  - **embed**: bge-small-en-v1.5, batch 512, fp16, normalized, stored as `halfvec(384)`.
- Resumable: re-running skips already-ingested source IDs.

**Acceptance**

- [ ] ≥ 1,300,000 chunks; report the exact count and per-source breakdown.
- [ ] Zero NULL, future, or naive `published_at`. Assert over the full table, not a sample.
- [ ] 100% embedding coverage (`COUNT(*) WHERE embedding IS NULL` = 0).
- [ ] Dedupe collapse ratio reported; earliest-date retention verified on a hand-built fixture.

---

### M3 — Chronofence (time-locked retrieval)

This is the milestone that decides whether the whole study is valid. Take it seriously.

**Deliver**

- `chunks` **partitioned by `RANGE (published_at)`** at monthly granularity, one IVFFlat index per partition (`lists ≈ sqrt(rows_in_partition)`, `probes = 10`). The planner prunes post-cutoff partitions before touching a vector — the time filter must be *structural*, not a predicate someone can forget.
- SQL function `chronofence_search(q halfvec(384), as_of timestamptz, k int)`. The app role has `EXECUTE` on the function and **no `SELECT` on `chunks`**.
- `cascade/retrieval/bench.py` — 10,000 queries sampled from real agent query distributions across the full cutoff range. Report p50/p95/p99 **and** recall@20 vs exact search on a 5% sample. A latency number without a recall number is meaningless.
- Leakage suite in `tests/leakage/`:
  1. **Poison pill** — inject 500 synthetic docs stating each scenario's outcome, dated one day post-resolution. Run full retrieval for all 180. Assert **zero** retrieved.
  2. **Signature scan** — regex + embedding similarity of every retrieved chunk against resolution text; flag above threshold; report the reviewed count.
  3. **Date monotonicity** — property test over the entire retrieval trace: `published_at < scenario.cutoff_ts`.
  4. **Parametric probe** — ask the agent model each question with zero context. Store a `memorization_score` per scenario. This one cannot be fixed by engineering; it is measured and disclosed.

**Acceptance**

- [ ] p95 < 15 ms over 10,000 queries; print the full histogram.
- [ ] recall@20 > 0.92 vs exact search.
- [ ] Poison-pill: **0** of 500 retrieved across all 180 scenarios.
- [ ] Date-monotonicity property test passes over the full trace.
- [ ] `memorization_score` computed for all 180; distribution reported.

If p95 blows out on early cutoffs, the cause is almost always too many sparse partitions for old months — merge to quarterly and re-tune `lists`.

---

### M4 — Lathe (causal decomposition compiler)

**Deliver**

- `decompose/schema.py` — `Actor`, `Factor`, `Edge`, `CausalGraph`, `OutcomeRule` as Pydantic v2 models exactly as specified in §5.1 of the spec.
- `decompose/compiler.py` — three LLM passes, ~3 calls per scenario:
  1. **Draft** (Sonnet, temp 0.2): question + resolution criterion + top-60 Chronofence chunks at cutoff → candidate graph under strict tool-use JSON schema.
  2. **Adversarial critique**: fixed checklist — *which party with real leverage is missing; which actor's objective is a restatement of the outcome rather than an independent interest; which edge has the wrong sign; which factor is actually two factors*. Returns a structured defect list. **It does not rewrite.**
  3. **Repair**: applies the defect list. Separated from critique so the model cannot quietly paper over a defect it just found.
- `decompose/validator.py` — **pure Python, no LLM, no I/O**. Rules: actor count 8–20; every actor has an outbound path to the outcome rule; no actor objective with cosine similarity > 0.85 to the outcome text; pairwise factor-name similarity < 0.80; no zero-lag factor self-loops and inbound weight ≤ 3.0 per factor; outcome rule depends on ≥ 2 factors and is monotone in each.
- Repair loop: max 2 retries, then hard-fail the scenario and log it. Canonicalize and hash the graph; a scenario compiles **once** for the whole study.

**Acceptance**

- [ ] 180/180 graphs pass the validator within 2 repair attempts (report the retry histogram).
- [ ] Mean actor count 14 ± 2; factor count within [4, 12].
- [ ] Recompiling an unchanged input reproduces the graph hash exactly.
- [ ] Human audit: sample 20 graphs (seeded), score 0–2 on *actor completeness / edge-sign correctness / missing leverage*, mean ≥ 1.5. Write the rubric and the scores to `reports/audit/`.

The most common failure here is every actor being given the objective "make the outcome happen." The critique pass must name that failure explicitly.

---

### M5 — Aperture + Loom (kernel)

**Deliver — Aperture**

- `VisibilityPolicy` **derived** from graph topology, not authored: direct outbound edge → noise 0, lag 0. One hop → `noise_sigma 0.08`, `lag 1`. Two hops → `noise_sigma 0.15`, `lag 2`, quantized to {low, mid, high}. Three+ hops → not visible. `observable_by_default` factors → `noise_sigma 0.03`, `lag 0`, visible to all.
- Action visibility: `full` if two actors share a factor; `type_only` (class but not magnitude or target) at two hops; `none` otherwise.
- Private memory keyed `(run_id, actor_id)`: last 12 observations + a rolling summary refreshed every 8 steps. Agents read their own namespace and Chronofence — **never** another agent's.
- The `information_asymmetry: false` flag replaces every policy with fully transparent. Nothing else changes — same graph, same agents, same seeds. That isolation is what makes the ablation attributable.

**Deliver — Loom**

- `WorldState` (Pydantic, ~4 KB serialized): step, factors, factor_history, resources, relations, pending effects, `rng_counter`.
- The six-stage step loop as LangGraph: `EXOGENOUS → ARRIVE → ACTIVATE → OBSERVE → DECIDE → ARBITRATE`, 24 steps. Checkpoint every step.
- `scheduler.py` — activation on salience: triggered if an observed factor moved > 0.06 since last observation or an observed actor targeted a factor in this actor's utility terms; forced at least every 5 steps; capped at 8 actors/step with deterministic tie-breaks.
- `actions.py` — closed discriminated union: `COMMIT`, `SIGNAL` (may be false — truth is not enforced), `ALLY`, `DEFECT`, `ESCALATE`, `CONCEDE`, `WAIT`. `rationale` field capped at 30 tokens, never parsed.
- `arbiter.py` — **deterministic, no LLM, no I/O.** Tullock contest over committed resources (`gamma = 1.6`), `MAX_STEP_DELTA = 0.12`, resource debits, lagged effect scheduling, relation-matrix update.

**Acceptance**

- [ ] Four arbiter **property tests** (hypothesis): bounded (no factor moves > 0.12/step), conserving (no negative budgets), **permutation-invariant** (shuffling within-step action order yields an identical delta), monotone (more resource → weakly higher contest share).
- [ ] Mean activation rate **34.7% ± 4 pts** on a 50-run sample. This is emergent, but assert it — drift to 90% triples cost, drift to 10% stops simulating interaction.
- [ ] One scenario runs 24 steps end to end and produces a terminal `outcome_score ∈ [0,1]`.
- [ ] `WorldState` serialization round-trips bit-exactly; checkpoint-resume mid-run reproduces the same terminal score.

---

### M6 — Chorus (ensemble at scale)

**Deliver**

- Fan-out runner: 36,000 runs = 180 scenarios × 200 replicates, `seed = blake2b(scenario_id|config_id|replicate, key=STUDY_SALT)`.
- Aggregation: `p_hat = mean(scores)`, `sigma = std(scores, ddof=1)`, percentile bootstrap CI (B = 10,000), `modality = multi if sigma > 0.30 or dip_test.p < 0.05`. Also compute the bimodality coefficient.
- Event log: `events` partitioned by `HASH(run_id)` (32 partitions), PK `(run_id, step, seq)`, append-only, with `caused_by` populated by the arbiter.
- Cost control: `--estimate` runs 20 sample units and extrapolates before any full launch. Batch API on for everything except the latency bench. Prompt-cached persona/rules prefix. Trajectory-prefix action cache — **quantize observed floats to 2 decimals before hashing the observation**, this alone typically moves hit rate 15–20 points.

**Acceptance**

- [ ] 36,000 runs complete; event count **4.2M ± 5%** (expected 4,199,040).
- [ ] Action-cache hit rate ≥ 88%; report it.
- [ ] Measured spend within the configured ceiling; report marginal $/run.
- [ ] Kill the worker pool mid-phase and resume: **no duplicated and no lost runs** (assert on the `UNIQUE(scenario_id, config_id, replicate)` constraint plus a count check).
- [ ] Convergence curve: mean |Δp̂| as n goes 25 → 400 on a 20-scenario subsample, showing the plateau at 200.

---

### M7 — Assay (evaluation)

**Deliver**

- `metrics.py` — Brier, Brier skill score, **Murphy decomposition** (BS = REL − RES + UNC), log loss (clipped to [0.01, 0.99], and say so), ECE/MCE over 10 bins, AUC.
- Five baselines, **all reported**: climatology; single-model direct; **single-model self-consistency with k = 200** (matched sample budget — without this the result gets dismissed as "you just sampled more"); multi-agent without decomposition; Cascade full.
- The **12-cell ablation grid** — 4 binary factors would give 16, but `(decomposition=off, asymmetry=on)` is undefined (the visibility policy is derived from the graph), eliminating 4 cells. Run 11 non-headline cells at 90 scenarios × 30 replicates.
- Report leave-one-out deltas **and** the net figure: LOO(asymmetry) = +0.027, LOO(decomposition) = +0.035, and **decomposition net of asymmetry = +0.008**. They do not sum because asymmetry nests inside decomposition. Publish the net number first — a careful reader computes it anyway.
- `stats.py` — paired bootstrap **over scenarios, not runs** (B = 10,000), Holm–Bonferroni across the ablation family, per-domain Brier breakdown with counts.
- Calibration: 10 equal-width bins with count, mean predicted, observed frequency, **Wilson 95% CI**. Reliability diagram with a bin-count histogram underneath. Isotonic recalibration fitted on a held-out half, reported as a *secondary* number only.
- Test the claim that σ is informative: correlation between σ and |forecast error|, and Brier split by modality flag.
- `report.py` writes `reports/study_{ts}/` exactly as specified in Appendix D of the spec. The README quotes `headline.md` verbatim — nothing retyped by hand.

**Acceptance**

- [ ] All 12 cells executed; all 5 baselines scored.
- [ ] Full metric set produced; report artifact written with manifest (scenario hash, git SHA, model versions, config).
- [ ] Paired bootstrap CIs with Holm-adjusted p-values for every ablation comparison.
- [ ] Per-domain table with counts. A headline win driven by one over-represented domain is not a win.
- [ ] Print measured vs target for every row of the measurement contract, with deviations flagged. **Do not adjust anything to close a gap.**

---

### M8 — Strata (determinism, provenance, cost ledger)

**Deliver**

- Replay verification: run 25 runs in `record` mode, hash the canonical event log, re-run in `replay` mode **in a fresh process with a different `PYTHONHASHSEED` and a different worker count**, assert hashes are equal. On mismatch, bisect and report the first `(run_id, step, seq)` and the differing field.
- `cascade trace --run <id> --explain-outcome` — recursive CTE over `caused_by`, printing the chain from terminal outcome back to root cause with depth, step, actor, action type, and factor delta.
- Cost ledger: reconcile the `runs` table totals against Langfuse. Discrepancy > 2% is a bug in the meter and blocks the report.

**Acceptance**

- [ ] 25-run replay: **byte-identical** event-log hash. Not "similar." Equal.
- [ ] Divergence bisect tested by deliberately introducing an unsorted iteration, asserting the bisect names the right location, then reverting.
- [ ] `--explain-outcome` returns a complete chain to a root cause on a real run.
- [ ] Cost ledger reconciles within 2%; print both totals.

---

### M9 — Presentation

**Deliver**

- `README.md` with the **real measured numbers**, exact reproduction commands, and a **stated limitations** section covering: parametric memorization (with the measured scores), 180-scenario statistical power, domain coverage, and the fact that agent reasoning quality is bounded by the agent model.
- Optional read-only FastAPI + React dashboard: reliability diagram, ablation grid forest plot, σ-vs-error scatter, trace explorer.
- A 90-second demo path: clean clone → `make up` → seeded sample → one rendered causal trace.

**Acceptance**

- [ ] README numbers are read from `headline.md`, never retyped.
- [ ] Demo path executes from a clean clone on a second machine.
- [ ] `CLAUDE.md` build log complete: one line per milestone with measured acceptance values.

---

## PART 2 — Kickoff message for session 1

> Read `Cascade_Technical_Specification.docx` in full before writing any code.
>
> Then: (1) write `CLAUDE.md` at repo root containing the measurement contract, the eight hard invariants, the pinned stack, the engineering standards, and an empty build log; (2) scaffold the repository exactly as laid out in §13 of the spec; (3) execute **M0 only**.
>
> When M0's acceptance criteria are met, print each one with its **measured** value, update the build log, and stop. Do not start M1.
>
> If anything in the spec is ambiguous or you believe a design decision is wrong, say so before implementing it rather than picking silently. If an acceptance criterion cannot be met, stop and report the measured value and your diagnosis — do not relax the criterion.

---

## PART 3 — Recovery prompts (for when things go sideways)

**Replay is diverging (M8)**
> The replay hash mismatch is almost certainly (a) unsorted dict/set iteration, (b) a second RNG instance, (c) a float reduction whose order changes with worker count, or (d) an LLM cache key that includes a non-canonical field. Run the bisect, report the first divergent `(run_id, step, seq)` and field, then find the source. Do not add a tolerance — a tolerance means every downstream reproducibility claim is false.

**Cache hit rate is too low (M6)**
> Observation hashing is too fine-grained. Quantize observed floats to 2 decimals before hashing, exclude the rolling-summary text from the key (hash the summary version instead), and confirm the persona prefix is byte-identical across calls for the same actor. Report hit rate by step index — it should start near 1.0 and decay.

**The simulation collapses to consensus by step 8**
> Factor variance across replicates is decaying too fast. Check that `MAX_STEP_DELTA` is not swamping exogenous `volatility`, that hop-2 channels are actually quantized, and that `DEFECT`/`SIGNAL` are ever being chosen. Plot per-step cross-replicate factor variance before changing any constant.

**Cascade does not beat the self-consistency baseline (M7)**
> This is a legitimate finding — report it. Then investigate in this order: decomposition audit score, agent memory truncation, arbiter damping. Do **not** iterate prompts against the test set. If you do change a prompt, log it in `prompt_revisions` with the before/after Brier so the tuning is visible in the record.
