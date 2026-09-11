# ADR-0016 — Action visibility is per target, not per policy

- **Status:** accepted
- **Milestone:** M5
- **Corrects:** spec §6.2 (the `VisibilityPolicy` schema cannot express the
  derivation rule stated three lines below it)

## Context

§6.2 gives the policy as:

```python
class VisibilityPolicy(BaseModel):
    actor_id: str
    channels: list[Channel]
    sees_actions_of: list[str]                          # subset of other actor ids
    action_visibility: Literal["full","type_only","none"]
```

and then states the derivation rule:

> An actor sees another actor's actions at full if they share a factor;
> `type_only` (the action class but not its magnitude or target) if they are
> two hops apart; `none` otherwise.

The rule assigns a *different* visibility per target. The schema carries one
enum for the whole policy. In a 14-actor graph an actor typically shares a
factor with three or four parties and sits two hops from another five; there is
no single value of `action_visibility` that describes that, and no way to
recover the split from a flat `list[str]`.

## Decision

`sees_actions_of` carries the visibility with each target:

```python
class ActionChannel(_Frozen):
    actor_id: str
    visibility: Literal["full", "type_only"]

class VisibilityPolicy(_Frozen):
    actor_id: str
    channels: tuple[Channel, ...]
    sees_actions_of: tuple[ActionChannel, ...]
    action_visibility: ActionVisibility   # default for actors not listed
```

`action_visibility` is kept and reinterpreted as the **default for actors
absent from the list**: `none` under asymmetry, `full` when §6.4's ablation
switch is off (where every actor is also listed explicitly, so the default is
never the operative value). `VisibilityPolicy.action_view(other)` is the one
accessor, and it answers for any actor id whether or not it is listed.

Actor distance is measured on the **actor projection** of the graph: two actors
are adjacent when they both hold an edge onto the same factor (§6.2's "share a
factor"), and "two hops apart" is a common neighbour in that projection. The
alternative reading — two *factors* in between — makes "hops" mean one thing
for factors and another for actors in the same paragraph.

## Why it matters beyond tidiness

`type_only` is not a weaker `full`; it is the difference between knowing that a
party moved and knowing what it moved. The scheduler's second activation
trigger (§7.3: "an observed actor took an action targeting a factor in the
actor's utility terms") is only evaluable under a full view. Collapsing the
two would either wake every actor on every visible action or none, and in both
directions the activation rate — and with it the 4.2M event count and the cost
model — moves.

## Verified by

`tests/unit/test_aperture.py::test_action_visibility_is_per_target` and
`::test_type_only_visibility_hides_the_target_and_the_magnitude`;
`tests/unit/test_sim_scheduler.py::test_a_type_only_view_does_not_trigger_on_the_target`.
