"""Aperture: deriving each actor's visibility policy from the graph (spec §6.2).

"Give each agent a different persona" is style variance and is worth roughly
nothing. Real asymmetry means two agents holding incompatible beliefs about the
same world variable because they genuinely observed different things, and that
requires a projection function. The policy is the input to that function, and
it is **derived, not authored**: an actor observes what its edges imply it can
observe, so the conflicting worldviews are a mechanical consequence of the
topology the compiler produced.

Everything here is pure. Derivation takes a graph and the aperture constants
and returns policies; there is no clock, no RNG and no I/O, which is what makes
the derivation rules property-testable.

One departure from §6.2's schema, recorded in ADR-0016: the spec gives
``VisibilityPolicy`` a list ``sees_actions_of`` and a single policy-level
``action_visibility``, while its own derivation rule assigns *different*
visibility per target -- full for actors sharing a factor, type_only for actors
two hops away. Those cannot both be expressed by one enum per policy, so
``sees_actions_of`` carries a visibility per target and ``action_visibility``
remains as the default applied to actors not listed (``none`` under asymmetry,
``full`` when the ablation switch turns it off).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_VISIBLE_HOPS",
    "ActionChannel",
    "ActionVisibility",
    "Channel",
    "VisibilityPolicy",
    "actor_distances",
    "derive_policies",
    "factor_hops",
]

ActionVisibility = Literal["full", "type_only", "none"]

# §6.2: "Three or more hops: not visible." Hops count *intermediate* factors,
# so a direct actor->factor edge is zero hops -- "you see clearly what you
# directly act on".
MAX_VISIBLE_HOPS = 2


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Channel(_Frozen):
    """One actor's observation channel onto one factor (spec §6.2)."""

    factor_id: str
    visible: bool
    noise_sigma: Annotated[float, Field(ge=0.0)]
    lag: Annotated[int, Field(ge=0)]
    quantized: bool

    def informativeness(self) -> tuple[int, float, int]:
        """Order channels best-first: fresher, then cleaner, then unbanded.

        Used where two rules offer a channel onto the same factor -- a public
        factor an actor also acts on directly. Taking the better one is the
        only defensible reading: a rule that makes an actor *less* informed
        about something it controls because the value is also published would
        be a defect dressed as a policy.
        """
        return (self.lag, self.noise_sigma, int(self.quantized))


class ActionChannel(_Frozen):
    """How much of another actor's action this actor sees (spec §6.2)."""

    actor_id: str
    visibility: Literal["full", "type_only"]


class VisibilityPolicy(_Frozen):
    """What one actor can observe. Derived from topology, not written by hand."""

    actor_id: str
    channels: tuple[Channel, ...]
    sees_actions_of: tuple[ActionChannel, ...]
    action_visibility: ActionVisibility
    """Applied to actors absent from ``sees_actions_of`` (ADR-0016)."""

    def channel(self, factor_id: str) -> Channel | None:
        for channel in self.channels:
            if channel.factor_id == factor_id:
                return channel
        return None

    def visible_factors(self) -> tuple[str, ...]:
        return tuple(sorted(c.factor_id for c in self.channels if c.visible))

    def action_view(self, other_actor_id: str) -> ActionVisibility:
        for entry in self.sees_actions_of:
            if entry.actor_id == other_actor_id:
                return entry.visibility
        return self.action_visibility

    def is_transparent(self) -> bool:
        """True when this policy hides nothing -- the §6.4 ablation state."""
        return (
            all(
                channel.visible
                and channel.noise_sigma == 0.0
                and channel.lag == 0
                and not channel.quantized
                for channel in self.channels
            )
            and self.action_visibility == "full"
        )


def factor_hops(graph: Any) -> dict[str, dict[str, int]]:
    """Shortest directed distance from each actor to each factor, in hops.

    Zero hops means a direct outbound edge. The walk follows edge direction:
    influence flows from ``src`` to ``dst``, and an actor that can move a
    factor two links downstream is watching the thing it can affect. Walking
    the edges undirected would make every actor an observer of everything
    upstream of its own levers, which is the transparency the asymmetry layer
    exists to deny.

    Returned as hop counts so the caller does not have to remember the
    off-by-one between "edges traversed" and §6.2's "hops away".
    """
    adjacency: dict[str, list[str]] = {}
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag)):
        adjacency.setdefault(edge.src, []).append(edge.dst)

    out: dict[str, dict[str, int]] = {}
    for actor in sorted(graph.actors, key=lambda a: a.id):
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(actor.id, 0)])
        seen = {actor.id}
        while queue:
            node, depth = queue.popleft()
            for neighbour in sorted(adjacency.get(node, ())):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                # depth counts edges from the actor; hops count intermediates.
                distances[neighbour] = depth
                queue.append((neighbour, depth + 1))
        out[actor.id] = distances
    return out


def actor_distances(graph: Any) -> dict[str, dict[str, int]]:
    """Distance between actors in the actor projection of the graph.

    Two actors are adjacent (distance 1) when they share a factor -- both have
    an edge onto it -- which is §6.2's "share a factor". Distance 2 is its
    "two hops apart": a common neighbour, some third actor that shares a factor
    with each. Beyond that, none.
    """
    levers: dict[str, set[str]] = {actor.id: set() for actor in graph.actors}
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag)):
        if edge.src in levers:
            levers[edge.src].add(edge.dst)

    actor_ids = sorted(levers)
    neighbours: dict[str, set[str]] = {actor_id: set() for actor_id in actor_ids}
    for index, left in enumerate(actor_ids):
        for right in actor_ids[index + 1 :]:
            if levers[left] & levers[right]:
                neighbours[left].add(right)
                neighbours[right].add(left)

    out: dict[str, dict[str, int]] = {}
    for source in actor_ids:
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        seen = {source}
        while queue:
            node, depth = queue.popleft()
            for neighbour in sorted(neighbours[node]):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                distances[neighbour] = depth + 1
                queue.append((neighbour, depth + 1))
        out[source] = distances
    return out


def derive_policies(graph: Any, aperture: Any, *, asymmetry: bool) -> dict[str, VisibilityPolicy]:
    """Derive one policy per actor (spec §6.2), or the transparent policy (§6.4).

    ``asymmetry=False`` is the ablation switch: every factor visible, zero
    noise, zero lag, every action visible to everyone. Nothing else changes --
    same graph, same agents, same seeds, and (because the noise draws are made
    whether or not a channel uses them, see :mod:`cascade.sim.rng`) the same
    exogenous stream. That isolation is what makes the measured delta
    attributable to this component rather than to a confound.
    """
    hops = factor_hops(graph)
    distances = actor_distances(graph)
    public = {factor.id for factor in graph.factors if factor.observable_by_default}
    factor_ids = sorted(factor.id for factor in graph.factors)
    actor_ids = sorted(actor.id for actor in graph.actors)

    policies: dict[str, VisibilityPolicy] = {}
    for actor_id in actor_ids:
        channels = tuple(
            _channel_for(
                factor_id,
                hops=hops[actor_id].get(factor_id),
                is_public=factor_id in public,
                aperture=aperture,
                asymmetry=asymmetry,
            )
            for factor_id in factor_ids
        )
        if asymmetry:
            seen = tuple(
                ActionChannel(actor_id=other, visibility="full" if distance == 1 else "type_only")
                for other, distance in sorted(distances[actor_id].items())
                if distance <= 2
            )
            fallback: ActionVisibility = "none"
        else:
            seen = tuple(
                ActionChannel(actor_id=other, visibility="full")
                for other in actor_ids
                if other != actor_id
            )
            fallback = "full"
        policies[actor_id] = VisibilityPolicy(
            actor_id=actor_id,
            channels=channels,
            sees_actions_of=seen,
            action_visibility=fallback,
        )
    return policies


def _channel_for(
    factor_id: str,
    *,
    hops: int | None,
    is_public: bool,
    aperture: Any,
    asymmetry: bool,
) -> Channel:
    """Apply §6.2's derivation rules to one (actor, factor) pair."""
    if not asymmetry:
        return Channel(factor_id=factor_id, visible=True, noise_sigma=0.0, lag=0, quantized=False)

    candidates: list[Channel] = []
    if hops is not None and hops <= MAX_VISIBLE_HOPS:
        if hops == 0:
            candidates.append(
                Channel(factor_id=factor_id, visible=True, noise_sigma=0.0, lag=0, quantized=False)
            )
        elif hops == 1:
            candidates.append(_from_config(factor_id, aperture.hop1))
        else:
            candidates.append(_from_config(factor_id, aperture.hop2))
    if is_public:
        candidates.append(_from_config(factor_id, aperture.public))

    if not candidates:
        return Channel(factor_id=factor_id, visible=False, noise_sigma=0.0, lag=0, quantized=False)
    return min(candidates, key=Channel.informativeness)


def _from_config(factor_id: str, channel: Any) -> Channel:
    return Channel(
        factor_id=factor_id,
        visible=True,
        noise_sigma=float(channel.noise_sigma),
        lag=int(channel.lag),
        quantized=bool(channel.quantized),
    )


def levers_by_actor(graph: Any) -> dict[str, tuple[str, ...]]:
    """Factors each actor has a direct outbound edge to -- what it can act on.

    Separate from the visibility policy on purpose: what an actor can *see* and
    what it can *touch* are different questions, and conflating them is how an
    agent ends up able to move a factor it only reads about.
    """
    out: dict[str, set[str]] = {actor.id: set() for actor in graph.actors}
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag)):
        if edge.src in out:
            out[edge.src].add(edge.dst)
    return {actor_id: tuple(sorted(out[actor_id])) for actor_id in sorted(out)}


def counterparties(policies: Mapping[str, VisibilityPolicy]) -> dict[str, tuple[str, ...]]:
    """Actors each actor may address, derived from its action channels."""
    return {
        actor_id: tuple(sorted(entry.actor_id for entry in policies[actor_id].sees_actions_of))
        for actor_id in sorted(policies)
    }
