"""The Python side of the Chronofence boundary (spec §4.2, ADR-0002).

``as_of`` is keyword-only and has no default in every function here, mirroring
the SQL signature. That is invariant 1 and it is enforced twice on purpose:
Python raises ``TypeError`` before a connection is opened, and Postgres raises
a function-resolution error if the Python layer is ever bypassed. A retrieval
helper with a default ``as_of`` is how the entire study gets silently
invalidated, so there is nowhere in this package to add one.

The connection is held open across queries because the bench measures query
latency against a 15 ms budget, and a fresh TCP connection and authentication
handshake per query would be most of that budget measuring the wrong thing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, Literal

from cascade.config import Settings
from cascade.corpus.embed import EMBEDDING_DIM
from cascade.retrieval.schema import RetrievedChunk, SearchResult

__all__ = ["Chronofence", "Role", "vector_literal"]

Role = Literal["sim", "eval", "admin"]

# The cast is `::halfvec`, deliberately without a `(384)` typmod.
#
# A typmod cast is executed by pgvector's `halfvec(halfvec, integer, boolean)`
# coercion function, and migration 001 revokes EXECUTE on every function in
# `public` from PUBLIC (ADR-0005) -- which catches pgvector's own functions,
# because the extension installs into `public`. Measured: `cascade_eval` gets
# "permission denied for function halfvec" on `'[...]'::halfvec(384)` while
# `'[...]'::halfvec` succeeds, since a bare cast uses the type's input
# function, which is not privilege-checked.
#
# Granting EXECUTE on the coercion function would work and would also punch a
# hole in deny-by-default for every role, to buy a dimension check that
# `_call` performs in Python with a better error message. Function-parameter
# typmods are not enforced by Postgres anyway, so the cast was never the thing
# validating the width.
_QUERY_TEMPLATE = (
    "SELECT chunk_id, document_id, ordinal, body, published_at, "
    "source, url, title, distance "
    "FROM {function}(%s::halfvec, %s, %s)"
)


def vector_literal(values: Sequence[float]) -> str:
    """Render a query vector in pgvector's text input format.

    ``repr`` rather than ``str`` so a float round-trips exactly; a truncated
    query vector would perturb distances and make a recall measurement
    irreproducible for reasons invisible in the output.
    """
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


class Chronofence:
    """A time-locked view of the corpus, bound to one database role.

    Use as a context manager. Opened as ``sim`` -- the default and the role
    the simulation actually runs as -- only ``search`` is available, because
    ``chronofence_search_exact`` is not granted to ``cascade_sim``. That is
    the point: the simulation gets exactly one corpus read path, and this
    class cannot quietly give it a second.
    """

    def __init__(self, settings: Settings, *, role: Role = "sim") -> None:
        self._settings = settings
        self._role = role
        self._conn: Any = None

    def __enter__(self) -> Chronofence:
        import psycopg

        self._conn = psycopg.connect(self._settings.database_url(self._role), connect_timeout=30)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def role(self) -> Role:
        return self._role

    def _require_connection(self) -> Any:
        if self._conn is None:
            raise RuntimeError("Chronofence must be used as a context manager")
        return self._conn

    def _call(
        self,
        function: str,
        vector: Sequence[float],
        *,
        as_of: datetime,
        k: int,
    ) -> SearchResult:
        """Invoke one of the two chronofence functions and time it.

        ``as_of`` is passed positionally into a signature with no DEFAULT, so
        omitting it upstream is a Python ``TypeError`` and could not reach here
        as a NULL. A NULL that did reach here would return zero rows rather
        than every row -- ``published_at < NULL`` is NULL, never true -- so the
        failure mode of this boundary is empty, not leaky.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(vector) != EMBEDDING_DIM:
            # Checked here because the SQL cast below carries no typmod (see
            # `_QUERY_TEMPLATE`), so Postgres would not catch a wrong-sized
            # query vector until the distance operator did -- by which point
            # the error names an operator rather than the caller's mistake.
            raise ValueError(
                f"query vector has {len(vector)} dimensions, expected {EMBEDDING_DIM}; "
                "the query and the corpus must come from the same pinned embedding model"
            )
        if as_of.tzinfo is None:
            raise ValueError(
                f"as_of must be timezone-aware, got naive {as_of!r}; a naive cutoff is "
                "interpreted in the server's timezone and silently shifts the time lock"
            )

        conn = self._require_connection()
        started = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                _QUERY_TEMPLATE.format(function=function),
                (vector_literal(vector), as_of, k),
            )
            rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        chunks = tuple(
            RetrievedChunk(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                ordinal=int(row[2]),
                body=str(row[3]),
                published_at=row[4],
                source=str(row[5]),
                url=str(row[6]),
                title=str(row[7]),
                distance=float(row[8]),
            )
            for row in rows
        )
        return SearchResult(as_of=as_of, k=k, chunks=chunks, elapsed_ms=elapsed_ms)

    def search(self, vector: Sequence[float], *, as_of: datetime, k: int) -> SearchResult:
        """The k nearest chunks published strictly before ``as_of``.

        Approximate: IVFFlat with the probes pinned into the function by
        migration 004. Recall against exact search is the M3 acceptance
        criterion measured by :mod:`cascade.retrieval.bench`.
        """
        return self._call("chronofence_search", vector, as_of=as_of, k=k)

    def search_exact(self, vector: Sequence[float], *, as_of: datetime, k: int) -> SearchResult:
        """Exhaustive ground truth for recall. ``cascade_eval`` only.

        Called as ``cascade_sim`` this raises ``InsufficientPrivilege`` -- the
        grant is the mechanism, so the error is left to propagate rather than
        pre-empted with a friendlier Python check that could drift from it.
        """
        return self._call("chronofence_search_exact", vector, as_of=as_of, k=k)

    def probes(self) -> int:
        """The ``ivfflat.probes`` actually pinned into the deployed function.

        Read from ``pg_proc.proconfig`` rather than from config, so
        `cascade retrieval verify` compares the *deployed* value against the
        configured one instead of comparing config with itself.
        """
        conn = self._require_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proconfig FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = 'public' AND p.proname = 'chronofence_search'"
            )
            row = cur.fetchone()
        if row is None or not row[0]:
            raise RuntimeError(
                "chronofence_search has no pinned configuration; migration 004 "
                "has not been applied to this database"
            )
        for entry in row[0]:
            key, _, value = str(entry).partition("=")
            if key == "ivfflat.probes":
                return int(value)
        raise RuntimeError(
            "chronofence_search does not pin ivfflat.probes; callers would silently "
            "get pgvector's default of 1 and a recall profile nothing asserts"
        )
