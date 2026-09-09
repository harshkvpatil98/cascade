# ADR-0010 — Corpus dates: stated, never inferred; and a bounded window

- **Status:** accepted
- **Milestone:** M2
- **Implements and extends:** spec §3.2

## Part 1 — The publication date comes from the document, never the crawl

Spec §3.2 says of CC-News: *"Drop records with missing or inferred dates — no
guessing."* Implementing that turned out to be the central design decision of
this milestone, because the easy date is the wrong one.

A CC-NEWS WARC record carries `WARC-Date`: when Common Crawl fetched the page.
It is always present, always well-formed, and always **later** than
publication — typically by hours, sometimes by years when the crawler revisits
an archive page. Using it would date evidence *after* the moment it became
knowable, so a cutoff that should have excluded a document might not. It errs
in the unsafe direction and it never looks wrong.

So the date is read from what the publisher states: `article:published_time`,
`datePublished` in JSON-LD, `pubdate`, or a `<time datetime=...>` element. A
page that declares none is **dropped**. Measured on a real WARC file, that is
35% of records — a large loss, and the correct one.

The same rule holds across all five sources, applied once in
`normalize.validate` rather than five times in five fetchers:

| Source | Date | Inference? |
|---|---|---|
| EDGAR | `Date Filed` from the form index | none — a legal fact |
| Federal Register | `publication_date` | midnight UTC, a **stated convention** |
| Wikipedia | the revision's own timestamp | none |
| GDELT | `seendate` | none |
| CC-News | publisher metadata, else dropped | none |

A **naive** timestamp is dropped rather than assumed to be UTC. An hour on the
wrong side of a cutoff is a leak, and after the fact there is no way to tell
which documents were guessed at.

## Part 2 — Wikipedia is anchored to each scenario's cutoff

§3.2 marks this source critical: *"use the revision as of the cutoff, never the
current article."* The current article about a 2016 referendum says who won.

Every snapshot is therefore anchored: the API is asked for the last revision at
or before an `as_of` instant, and `published_at` is that revision's timestamp.
The anchor is each scenario's own `cutoff_ts`, and the articles are that
scenario's `party_names` — so the snapshot is contemporaneous with the question
it will be used to reason about. Verified: the *Brexit* article anchored at
2016-04-15 returns revision 715020967 from 2016-04-13, two months before the
referendum.

Reading the registry here touches `scenarios` only. No outcome is read
(invariant 2).

## Part 3 — A bounded window, and no DEFAULT partition

`chunks` is partitioned by `RANGE (published_at)` so the planner prunes
post-cutoff partitions before touching a vector. A `DEFAULT` partition would
undo that: it can never be pruned, so every time-locked query would scan it.
There is none.

The consequence is that a document outside the partition range has nowhere to
go, and — measured — it aborts the `COPY` of its entire batch. Real data
produces such documents: CC-NEWS re-crawls archive pages, so a 2016 crawl
yielded an article published in 2013.

So the corpus has an explicit window, `WINDOW_START`/`WINDOW_END` in
`cascade/corpus/schema.py`, matching the partitions migration 003 creates.
Documents outside it are dropped as `out_of_window` — out of scope, not an
error. An integration test asserts the DDL and the Python constants agree,
because if they drift a valid document becomes an aborted batch.

## Verified by

`tests/unit/test_corpus_normalize.py` (every drop reason, earliest-date
retention, clock-independence), `tests/unit/test_corpus_sources.py` (date
extraction per source, including naive dates left naive for the shared rule to
reject), `tests/integration/test_corpus_schema.py` (partition strategy, window
agreement, and the invariants over the full table).
