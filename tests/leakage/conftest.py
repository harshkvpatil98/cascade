"""Marks everything under ``tests/leakage/`` as a leakage probe.

The live-database fixtures these suites use (`live_settings`, `records`,
`embedder`) are defined in ``tests/conftest.py`` so ``tests/property/`` can
share them -- date monotonicity is asserted over the same live corpus.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.leakage
