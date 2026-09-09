"""Local embedding with ``BAAI/bge-small-en-v1.5`` (spec §2.3, §3.2).

384 dimensions, normalised, stored as ``halfvec(384)``. The model is pinned:
the vectors in the corpus and the vectors a query is embedded with at M3 must
come from the same model, or every cosine distance in the study is meaningless.
``cascade doctor`` asserts the pin.

Normalisation is load-bearing rather than cosmetic. pgvector's ``<->`` is L2
distance; on unit-norm vectors L2 and cosine rank identically, so normalising
here is what lets the M3 index use the operator it is built for while the
study reasons in cosine terms.

This module also owns the **tokenizer**, because the chunker's 512-token cap
only means what the spec intends if it is counted with the same tokenizer the
model uses. Counting whitespace words instead would let a chunk overflow the
context window and be silently truncated at embed time -- the tail of the
chunk would be in the database as text but absent from its own vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EMBEDDING_DIM", "Embedder", "EmbeddingUnavailable"]

EMBEDDING_DIM = 384


class EmbeddingUnavailable(RuntimeError):
    """The embedding stack is pinned but not installed."""


@dataclass
class Embedder:
    """Loads the pinned model once and encodes batches with it."""

    model_name: str
    batch_size: int = 512
    device: str | None = None
    use_fp16: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch
        except ImportError:  # pragma: no cover - guarded by load()
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self) -> None:
        """Load the model. Idempotent, and the only place it is constructed.

        Raises rather than falling back to a different model: a silent
        substitution would produce a corpus whose vectors cannot be compared
        with anything a later phase embeds.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailable(
                "sentence-transformers is pinned but not installed. "
                "Run `uv sync --extra embed` -- the corpus cannot be built without it."
            ) from exc

        device = self._resolve_device()
        model = SentenceTransformer(self.model_name, device=device)
        if self.use_fp16 and device != "cpu":
            # fp16 halves memory and roughly doubles throughput on MPS/CUDA.
            # Left off on CPU, where half precision is emulated and slower.
            model = model.half()
        self._model = model
        self._tokenizer = model.tokenizer
        self.device = device

        # sentence-transformers 5.x renamed this; support both so the pin can
        # move within the major line without an import-time failure.
        dimension_of = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        dim = dimension_of()
        if dim != EMBEDDING_DIM:
            raise EmbeddingUnavailable(
                f"{self.model_name} produces {dim}-d vectors but the schema stores "
                f"halfvec({EMBEDDING_DIM}); the pin and the DDL disagree"
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens the way the model does.

        This is the counter the chunker must use. ``add_special_tokens`` is
        left on because the model prepends [CLS] and appends [SEP] at encode
        time, and those occupy the same 512-token window the chunk has to fit
        inside.
        """
        self.load()
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """Count tokens for many strings in one call.

        The chunker measures every sentence of every document. Called one at a
        time, that crosses the Python/Rust boundary once per sentence and
        dominates ingest CPU; the fast tokenizer batches the same work into a
        single call. Same counts, materially less overhead.
        """
        self.load()
        if not texts:
            return []
        encoded = self._tokenizer(
            list(texts), add_special_tokens=True, truncation=False, padding=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into unit-norm 384-d vectors, in input order."""
        self.load()
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]
