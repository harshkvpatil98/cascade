"""Corpus source adapters (spec §3.2).

Each adapter turns one upstream into ``RawDocument`` values and nothing else.
Date validation, dedupe, chunking and embedding are shared downstream stages,
so an adapter never decides whether a document is usable -- it reports what the
source said, including a missing or malformed date, and ``normalize.validate``
applies one rule to all five sources.

Every adapter is *unit-addressable*: it exposes discrete units of work (an
EDGAR quarter, a Federal Register page, one scenario's Wikipedia snapshots)
keyed by a stable string, so ``corpus_ingest_state`` can record what is done
and a restart can skip it (invariant 8).
"""

from __future__ import annotations

__all__: list[str] = []
