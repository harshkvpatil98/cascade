"""The single door to the model provider.

Everything about determinism, cost accounting, caching and tracing depends on
there being exactly one place that talks to the Anthropic SDK. That place is
``cascade.llm.client``; ``tests/unit/test_invariants.py`` greps for the import
anywhere else and fails CI when it finds one (invariant 5).
"""

from __future__ import annotations

__all__: list[str] = []
