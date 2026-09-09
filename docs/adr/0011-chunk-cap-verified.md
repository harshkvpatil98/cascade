# ADR-0011 — The 512-token cap is verified, not estimated

- **Status:** accepted
- **Milestone:** M2
- **Implements:** spec §3.2 chunking

## Context

Spec §3.2 specifies chunks of 512 tokens with 64 overlap, sentence-boundary
aligned. 512 is not a style preference: it is `bge-small-en-v1.5`'s context
window. A chunk above it is **silently truncated at embed time**, so its tail
sits in the database as text that is absent from its own vector — present to a
reader, invisible to retrieval, and wrong in a way nothing downstream reports.

The natural implementation packs sentences until their summed token counts
reach the cap. Three measured failures showed that summing is not enough.

1. **Overlap carried an oversized sentence.** The overlap helper kept "at
   least one" trailing sentence regardless of size, so a 512-token sentence
   became the next chunk's overlap and that chunk reached **1,022 tokens**.
2. **No tokenizer is additive.** Costs summed per sentence understate the
   joined string — wordpiece merges across the join, and the whitespace
   fallback counter is off by its own constant per piece.
3. **Word splitting cannot divide text without spaces.** Japanese prose and a
   minified JSON blob that survived HTML extraction are each a single "word".
   Both reached the corpus at up to **2,125 tokens**, four times the cap.

## Decision

Every emitted chunk is **measured once against the real counter** and split
further if it is over. The estimate plans; the measurement decides.

Splitting is ordered so the cheapest guarantee is preserved longest:

1. split at **sentence boundaries** while more than one sentence remains —
   preserving alignment, since a chunk ending mid-sentence is evidence of
   nothing;
2. for a single oversized sentence, split on **word** boundaries — the one
   documented case where alignment yields to the cap;
3. for a single oversized "word", split on **characters** — which always
   terminates, so the cap holds for CJK and for machine-generated text, not
   only for whitespace-delimited English.

The stored `token_count` is the measured value, so it is exact rather than
estimated — which matters because M3 reports retrieval quality by chunk length.

## Cost

One extra tokenizer call per emitted chunk. Against that, per-sentence
measurement is batched through the fast tokenizer in a single call per
document, which more than pays for it: measured 28.1 ms/document before,
20.6 ms/document after.

A related fix belongs to the same story: the first implementation re-measured
the growing prefix after every word while hard-splitting, which is quadratic in
sentence length. SEC filings routinely contain single "sentences" of thousands
of words, and that turned a whole-corpus ingest into something that would not
finish. Each word is now measured once.

## Verified by

`tests/unit/test_corpus_chunker.py` — the cap is asserted for empty input,
one-word input, prose, a 5,000-word sentence, text with no terminators,
Japanese, Chinese, a JSON blob, and against a deliberately hostile counter that
overstates by 3x. `tests/integration/test_corpus_schema.py` re-asserts
`token_count <= 512` over every stored row.
