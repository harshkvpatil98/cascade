"""A deterministic stand-in for the agent model.

**This is not part of the study.** It exists so the kernel can be exercised --
in tests, and in an environment with no API credential -- without a model in
the loop. Every run made with it is stamped ``policy='heuristic'`` in the
``runs`` table, and no acceptance number, baseline or headline metric may be
read off such a run. The honest failure mode of a stand-in policy is that its
numbers get quoted as if they came from agents; stamping the run is what makes
that visible in the data rather than only in a footnote.

It is deliberately simple and deliberately *not* tuned: push the lever where
the gap between the current value and the preferred pole is largest, wait when
there is nothing worth pushing or nothing to push with. It has no memory, no
model of other actors, and no strategy. That is the point -- it exercises the
machinery without pretending to be an agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from cascade.aperture.projection import Observation
from cascade.sim.actions import ActionSpace, Commit, Wait
from cascade.sim.agent import Decision

__all__ = ["HeuristicPolicy"]


@dataclass(frozen=True)
class HeuristicPolicy:
    """Utility-greedy, memoryless, deterministic."""

    utility: dict[str, dict[str, float]]
    """actor id -> factor id -> signed weight."""
    min_gap: float = 0.05
    """Below this, the factor is close enough to the preferred pole to leave alone."""
    min_budget: float = 0.05
    spend_fraction: float = 0.30

    def decide(
        self,
        *,
        actor_id: str,
        observation: Observation,
        memory: str,
        space: ActionSpace,
    ) -> Decision:
        del memory  # a stand-in with a memory would be a model of something
        weights = self.utility.get(actor_id, {})
        best: tuple[float, str] | None = None
        for factor_id in sorted(space.levers):
            weight = weights.get(factor_id, 0.0)
            observed = observation.factors.get(factor_id)
            if weight == 0.0 or observed is None:
                continue
            target = 1.0 if weight > 0 else 0.0
            gain = abs(weight) * abs(target - observed)
            if abs(target - observed) < self.min_gap:
                continue
            if best is None or (gain, factor_id) > (best[0], best[1]):
                best = (gain, factor_id)

        if best is None or observation.budget < self.min_budget:
            return Decision(action=Wait(rationale="nothing worth pushing"))

        _, factor_id = best
        observed = observation.factors[factor_id]
        weight = weights[factor_id]
        gap = abs((1.0 if weight > 0 else 0.0) - observed)
        return Decision(
            action=Commit(
                target_factor=factor_id,
                magnitude=min(1.0, max(0.05, gap)),
                resource_spend=self.spend_fraction,
                rationale="heuristic: largest weighted gap",
            )
        )
