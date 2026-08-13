"""Loom: the simulation kernel (M5, spec §7).

Six stages per step -- ``EXOGENOUS -> ARRIVE -> ACTIVATE -> OBSERVE ->
DECIDE -> ARBITRATE`` -- over 24 steps, checkpointed every step.

``arbiter.py`` contains no LLM call and no I/O (invariant 3). Conflict
resolution is a Tullock contest, not a judgement call; moving it out of the
model is what removes 864,000 calls from the study and what makes replay
bit-exact.
"""

from __future__ import annotations

__all__: list[str] = []
