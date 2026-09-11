# ADR-0018 — An inadmissible action is coerced to WAIT and recorded

- **Status:** accepted
- **Milestone:** M5
- **Completes:** spec §7.4 (the action union is given; what makes an instance
  of it *legal for this actor* is not)

## Context

§7.4 defines seven actions with typed payloads, and §5.1 defines an edge from
an actor to a factor as the claim that the actor can act on it. Nothing joins
the two: the schema will happily accept `COMMIT(target_factor="global_rates")`
from an actor with no edge to `global_rates`, and `ALLY(target_actor=...)` with
a party the actor cannot see.

Both will happen. An agent handed eight factor names and told it has levers on
two will occasionally act on a third, and at 36,000 runs "occasionally" is a
large number.

Three responses are available: re-prompt, execute anyway, or refuse.

## Decision

**Refuse, substitute WAIT, and record why.**

```python
def refusal(action, space) -> str | None   # e.g. "no_lever:global_rates"
```

The executed action becomes `WAIT`, carrying the original `rationale` so the
provenance chain still says what the actor thought it was doing, and the reason
code goes into `events.coercion` (migration 009). The codes are stable strings,
not prose, because they are aggregated across 4.2M rows.

Admissibility is checked in **two** places, and that is deliberate. The agent
adapter admits what a model emitted — it also handles a payload that does not
parse at all. The kernel re-checks whatever any decider returned, because the
arbiter sits behind the kernel's boundary and a stand-in policy
(`cascade/sim/policies.py`) never goes near the schema. The second check found
the first hole: a policy returning an action on a lever its actor does not hold
was executed with no record of it.

## Why not the alternatives

**Re-prompt.** Costs a second call on every occurrence, which is a cost line
that scales with how often the model is confused — the one variable the budget
cannot bound in advance. It also makes the action-cache hit rate depend on
model mood, since a re-prompted turn has a different prompt.

**Execute anyway.** The arbiter would compute leverage 0 and the action would
do nothing, which is the same outcome with no record. "How often did agents
reach for a lever they do not have" is then unanswerable, and it is exactly the
diagnostic that distinguishes a decomposition whose actors were given
implausible levers from a prompt that fails to state them clearly.

**Reject the run.** One bad answer would cost the remaining 23 steps of a run
that is otherwise fine.

## Consequence

`events.coercion` is a measurement, not an error log. A scenario whose coercion
rate is high is a scenario whose graph gave its actors objectives they have no
means to pursue, which is an M4 finding surfaced by M5 — and it is visible in a
`GROUP BY` rather than in a post-mortem.

## Verified by

`tests/unit/test_sim_actions.py` (each refusal code, and that a coerced action
keeps its rationale), `tests/unit/test_sim_agent.py` (the same through a real
tool-call response), and
`tests/unit/test_sim_kernel.py::test_an_inadmissible_action_is_recorded_as_a_coercion`.
