"""Lathe: the causal decomposition compiler (M4, spec §5).

Three passes -- draft, adversarial critique, repair -- kept separate so the
model cannot quietly paper over a defect it just found. ``validator.py`` is
pure Python with no LLM and no I/O, which is what makes it property-testable.
"""

from __future__ import annotations

__all__: list[str] = []
