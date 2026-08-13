"""Aperture: the information-asymmetry layer (M5, spec §6).

The visibility policy is *derived* from graph topology, never authored: hop
distance to a factor sets noise, lag and quantization. Personas are not a
substitute -- a different tone is not different information and is worth
roughly zero Brier.

The ``information_asymmetry: false`` flag replaces every policy with a fully
transparent one and changes nothing else: same graph, same agents, same seeds.
That isolation is what makes the M7 ablation attributable.
"""

from __future__ import annotations

__all__: list[str] = []
