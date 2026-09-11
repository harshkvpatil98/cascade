"""Loom: the 24-step simulation kernel (spec §7.2).

Six ordered stages per step, all deterministic given the seed::

    EXOGENOUS -> ARRIVE -> ACTIVATE -> OBSERVE -> DECIDE -> ARBITRATE

The stages are LangGraph nodes over a single run context, as §2.3 pins the
orchestration, and every node is a thin shell over a pure function that lives
somewhere else: :mod:`cascade.sim.dynamics` for the world's own motion,
:mod:`cascade.sim.scheduler` for activation, :mod:`cascade.aperture.projection`
for observation, :mod:`cascade.sim.arbiter` for the fold. The only stage that
does I/O is DECIDE, and it does it through the one LLM call site.

The run's randomness is drawn once per step, at the top, in the documented
order (:mod:`cascade.sim.rng`). No stage holds the generator, so "drawn in a
fixed order" is a property of the code's shape rather than a convention to
uphold -- and :func:`cascade.sim.rng.assert_counter` checks at the end of every
step that nothing drew outside the plan.

What the kernel itself owns is bookkeeping the pure parts must not: whose turn
it is, what each agent privately remembers, which decision moved which factor
(for ``caused_by``), and when to stop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from cascade.aperture.memory import AgentMemory
from cascade.aperture.policy import VisibilityPolicy, counterparties, levers_by_actor
from cascade.aperture.projection import (
    Observation,
    SignalClaim,
    observation_hash,
    project,
    visible_actions,
)
from cascade.config import Settings
from cascade.sim.actions import Action, ActionSpace, Wait, refusal
from cascade.sim.agent import DecisionPolicy
from cascade.sim.arbiter import ActorMechanics, arbitrate
from cascade.sim.dynamics import (
    absorbed_factors,
    arrive,
    dynamics_from,
    exogenous,
    record_observable,
    update_streaks,
)
from cascade.sim.rng import RunRng, assert_counter
from cascade.sim.scheduler import ActorProfile, activate, profiles_from
from cascade.sim.state import WorldState, initial_state, state_hash
from cascade.trace.events import CausalLedger, DecisionEvent, StepRecord, action_payload
from cascade.trace.events import event_log_hash as compute_event_log_hash

__all__ = [
    "STAGE_SEQUENCE",
    "Loom",
    "PendingTurn",
    "RunHandle",
    "RunResult",
    "RunSpec",
    "mechanics_from",
    "propagation_from",
]

# The six stages of §7.2, in order, named once. Both drivers read this: the
# LangGraph graph that runs a single replicate, and the wavefront runner at M6
# that advances thousands of replicates in lockstep so one step's decisions can
# be batched. Two drivers over one ordering is the only duplication that is
# safe here -- the ordering itself is what a divergence would corrupt.
STAGE_SEQUENCE: tuple[str, ...] = (
    "stage_exogenous",
    "stage_arrive",
    "stage_activate",
    "stage_observe",
    "stage_decide",
    "stage_arbitrate",
)
DECIDE_STAGE_INDEX = STAGE_SEQUENCE.index("stage_decide")

Termination = Literal["horizon", "absorbed"]


class RunSpec(BaseModel):
    """Identifies one run. The seed is a function of exactly these fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    config_id: str
    replicate: int = Field(ge=0)
    policy: str = "agent"
    """``agent`` for the model, or the name of a stand-in (see
    :mod:`cascade.sim.policies`). Written to the ``runs`` table so a run that
    did not involve the agent model can never be mistaken for one that did."""


class PendingTurn(BaseModel):
    """One actor's turn, assembled but not yet decided.

    Produced by the OBSERVE stage and consumed by DECIDE. Splitting them is
    what lets a runner collect every pending turn across thousands of runs and
    resolve them in one batch (§12.2) -- the observation is already fixed by
    the time DECIDE runs, because no actor's decision can change another's
    observation within a step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    actor_id: str
    observation: Observation
    memory: str
    space: ActionSpace


class RunResult(BaseModel):
    """Everything one run produced. The unit M6 fans out and M7 aggregates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: RunSpec
    outcome_score: float
    steps_run: int
    termination: Termination
    absorbed: tuple[str, ...]
    world: WorldState
    events: tuple[DecisionEvent, ...]
    step_records: tuple[StepRecord, ...]
    event_log_hash: str
    llm_calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def decisions(self) -> int:
        return len(self.events)

    @property
    def activation_rate(self) -> float:
        """Mean fraction of the cast active per step -- the §7.3 figure.

        Measured over the steps that actually ran, not over the horizon: a run
        that terminated early at step 9 has 9 steps of evidence about
        activation, and dividing by 24 would report a rate the scheduler never
        produced.
        """
        if not self.step_records:
            return 0.0
        return sum(record.activation_rate for record in self.step_records) / len(self.step_records)


def mechanics_from(graph: Any) -> dict[str, ActorMechanics]:
    """Read the arbiter's view of each actor off the compiled graph."""
    leverage: dict[str, dict[str, float]] = {actor.id: {} for actor in graph.actors}
    signs: dict[str, dict[str, int]] = {actor.id: {} for actor in graph.actors}
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag)):
        if edge.src in leverage:
            # An actor with two edges onto one factor at different lags holds
            # one lever; the strongest is what it can bring to bear.
            leverage[edge.src][edge.dst] = max(
                leverage[edge.src].get(edge.dst, 0.0), float(edge.weight)
            )
            signs[edge.src][edge.dst] = int(edge.sign)
    return {
        actor.id: ActorMechanics(
            actor_id=actor.id,
            utility={term.factor_id: float(term.weight) for term in actor.utility_terms},
            leverage=leverage[actor.id],
            edge_sign=signs[actor.id],
            endowment={kind: float(actor.resources[kind]) for kind in sorted(actor.resources)},
        )
        for actor in sorted(graph.actors, key=lambda a: a.id)
    }


def propagation_from(graph: Any) -> dict[str, list[Any]]:
    """Factor -> its outgoing factor-to-factor edges.

    These are what make the decomposition a causal model rather than eight
    independent random walks: a contest won on one factor reaches the others
    through them, at the lag the compiler assigned.
    """
    factor_ids = {factor.id for factor in graph.factors}
    out: dict[str, list[Any]] = {}
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst, e.lag)):
        if edge.src in factor_ids and edge.dst in factor_ids:
            out.setdefault(edge.src, []).append(edge)
    return out


@dataclass
class _RunContext:
    """The kernel's working state for one run. Never leaves this module."""

    spec: RunSpec
    world: WorldState
    rng: RunRng
    memories: dict[str, AgentMemory]
    ledger: CausalLedger
    last_actions: dict[str, Action] = field(default_factory=dict)
    pending_signals: dict[str, list[SignalClaim]] = field(default_factory=dict)
    events: list[DecisionEvent] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    llm_calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    termination: Termination = "horizon"
    absorbed: tuple[str, ...] = ()
    # Within-step scratch, written by one stage and read by the next.
    noise: Any = None
    active: tuple[str, ...] = ()
    eligible: int = 0
    observations: dict[str, Observation] = field(default_factory=dict)
    turns: dict[str, PendingTurn] = field(default_factory=dict)
    observed_before: dict[str, int] = field(default_factory=dict)
    """Each active actor's *previous* observation step, captured before OBSERVE
    overwrites it -- the window `caused_by` is computed over."""
    decisions: dict[str, Any] = field(default_factory=dict)
    caused_by: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    exogenous_delta: dict[str, float] = field(default_factory=dict)
    arrivals: dict[str, float] = field(default_factory=dict)


@dataclass
class RunHandle:
    """A run in flight, driven a step at a time.

    Opaque on purpose: the context inside it is the kernel's working state, not
    a boundary type, and a runner that reached into it could advance a run
    without the stage ordering that makes it reproducible.
    """

    context: _RunContext
    pending: bool = False
    done: bool = False

    @property
    def spec(self) -> RunSpec:
        return self.context.spec

    @property
    def step(self) -> int:
        return self.context.world.step


class _LoopState(TypedDict):
    ctx: _RunContext


@dataclass
class Loom:
    """One compiled scenario, ready to run any number of seeded replicates.

    Construction is per scenario; :meth:`run` is per replicate. The derived
    structures -- mechanics, profiles, action spaces, propagation -- are
    functions of the graph alone, so they are built once and shared across the
    200 runs of a scenario rather than rebuilt 200 times.
    """

    settings: Settings
    graph: Any
    policies: Mapping[str, VisibilityPolicy]
    decider: DecisionPolicy

    def __post_init__(self) -> None:
        self.mechanics = mechanics_from(self.graph)
        self.dynamics = dynamics_from(self.graph)
        self.propagation = propagation_from(self.graph)
        self.profiles: dict[str, ActorProfile] = profiles_from(self.graph, self.policies)
        levers = levers_by_actor(self.graph)
        parties = counterparties(self.policies)
        self.spaces = {
            actor_id: ActionSpace(
                actor_id=actor_id,
                levers=levers.get(actor_id, ()),
                counterparties=parties.get(actor_id, ()),
            )
            for actor_id in sorted(self.profiles)
        }
        self.actor_ids = tuple(sorted(self.profiles))
        self.factor_ids = tuple(sorted(factor.id for factor in self.graph.factors))
        self._compiled = _build_graph()

    # -- the run ------------------------------------------------------------

    def _context(self, spec: RunSpec) -> _RunContext:
        """Build the working state for one replicate. Seeded, never re-seeded."""
        rng = RunRng.for_run(
            scenario_id=spec.scenario_id,
            config_id=spec.config_id,
            replicate=spec.replicate,
            salt=self.settings.study.salt,
        )
        ctx = _RunContext(
            spec=spec,
            world=initial_state(
                self.graph, forced_interval=self.settings.kernel.activation.forced_interval
            ),
            rng=rng,
            memories={actor_id: AgentMemory(actor_id=actor_id) for actor_id in self.actor_ids},
            ledger=CausalLedger(),
        )
        return ctx

    def run(self, spec: RunSpec) -> RunResult:
        """Run one replicate to the horizon or to an absorbing state (§7.6)."""
        ctx = self._context(spec)
        final: _LoopState = self._compiled.invoke(
            {"ctx": ctx},
            # Six nodes per step plus the router; the default limit of 25 would
            # stop a 24-step run four steps in, silently returning a partial
            # world that still looks like a result.
            {"recursion_limit": self.settings.kernel.steps * 8 + 16, "loom": self},
        )
        return self._result(final["ctx"])

    def _result(self, done: _RunContext) -> RunResult:
        """Project a finished context onto the boundary type."""
        return RunResult(
            spec=done.spec,
            outcome_score=float(self.graph.outcome_rule(done.world.factors)),
            steps_run=len(done.steps),
            termination=done.termination,
            absorbed=done.absorbed,
            world=done.world,
            events=tuple(done.events),
            step_records=tuple(done.steps),
            event_log_hash=compute_event_log_hash(done.events, done.steps),
            llm_calls=done.llm_calls,
            cache_hits=done.cache_hits,
            tokens_in=done.tokens_in,
            tokens_out=done.tokens_out,
        )

    # -- the stepwise driver (M6's wavefront) --------------------------------
    #
    # `run` above drives one replicate through the compiled graph. A runner
    # that wants to batch a step's decisions across many replicates needs to
    # stop between OBSERVE and DECIDE, which a single graph invocation cannot
    # do -- so the same stage methods are also exposed as two halves. Both
    # drivers walk STAGE_SEQUENCE; neither owns any simulation logic.

    def start(self, spec: RunSpec) -> RunHandle:
        """Open a run without advancing it. The wavefront's constructor."""
        return RunHandle(self._context(spec))

    def observe_step(self, handle: RunHandle) -> tuple[PendingTurn, ...]:
        """Advance one run through OBSERVE and return the turns awaiting decisions.

        Deterministic and model-free: everything up to and including the
        observation is fixed before any actor decides, which is exactly why a
        step's decisions can be resolved together.
        """
        ctx = handle.context
        if handle.pending:
            raise RuntimeError(
                f"run {ctx.spec.run_id} is already waiting on decisions for step "
                f"{ctx.world.step}; complete the step before observing the next"
            )
        for stage in STAGE_SEQUENCE[:DECIDE_STAGE_INDEX]:
            getattr(self, stage)(ctx)
        handle.pending = True
        return tuple(ctx.turns[actor_id] for actor_id in ctx.active)

    def complete_step(self, handle: RunHandle) -> None:
        """Run DECIDE and ARBITRATE for a step whose turns have been observed."""
        ctx = handle.context
        if not handle.pending:
            raise RuntimeError(
                f"run {ctx.spec.run_id} has no observed step to complete; "
                "call observe_step first"
            )
        for stage in STAGE_SEQUENCE[DECIDE_STAGE_INDEX:]:
            getattr(self, stage)(ctx)
        handle.pending = False
        handle.done = self.should_continue(ctx) == "stop"

    def result(self, handle: RunHandle) -> RunResult:
        """Collapse a finished run into its result. Refuses an unfinished one."""
        if not handle.done:
            raise RuntimeError(
                f"run {handle.context.spec.run_id} has not terminated; a partial run "
                "scored as if complete would enter the ensemble as a real forecast"
            )
        return self._result(handle.context)

    # -- stages -------------------------------------------------------------

    def stage_exogenous(self, ctx: _RunContext) -> None:
        """Stage 1: draw the step's plan, then walk every factor."""
        ctx.noise = ctx.rng.step_noise(
            step=ctx.world.step, actor_ids=self.actor_ids, factor_ids=self.factor_ids
        )
        before = dict(ctx.world.factors)
        ctx.world = exogenous(ctx.world, self.dynamics, ctx.noise.exogenous)
        ctx.exogenous_delta = {
            factor_id: round(ctx.world.factors[factor_id] - before[factor_id], 12)
            for factor_id in sorted(before)
            if ctx.world.factors[factor_id] != before[factor_id]
        }

    def stage_arrive(self, ctx: _RunContext) -> None:
        """Stage 2: land the lagged effects, then freeze what this step shows."""
        before = dict(ctx.world.factors)
        ctx.world, landed = arrive(ctx.world, max_step_delta=self.settings.kernel.max_step_delta)
        ctx.arrivals = {
            factor_id: round(ctx.world.factors[factor_id] - before[factor_id], 12)
            for factor_id in sorted(before)
            if ctx.world.factors[factor_id] != before[factor_id]
        }
        for effect in landed:
            if effect.source_actor_id is None:
                continue
            ctx.ledger.record(
                effect.factor_id,
                magnitude=effect.delta,
                ref=_ref(ctx.spec.run_id, effect.origin_step, effect.origin_seq),
                at_step=ctx.world.step,
            )
        ctx.world = record_observable(ctx.world)

    def stage_activate(self, ctx: _RunContext) -> None:
        """Stage 3: choose who acts (spec §7.3)."""
        decision = activate(
            profiles=self.profiles,
            world=ctx.world,
            last_actions=ctx.last_actions,
            salience_threshold=self.settings.kernel.activation.salience_threshold,
            forced_interval=self.settings.kernel.activation.forced_interval,
            max_active=self.settings.kernel.activation.max_active_per_step,
        )
        ctx.active = decision.active
        ctx.eligible = len(self.profiles)

    def stage_observe(self, ctx: _RunContext) -> None:
        """Stage 4: project the world, then assemble each active actor's turn.

        The memory digest is refreshed here rather than in DECIDE so that the
        turn handed to a batching runner is byte-identical to the one DECIDE
        will use. A second refresh would be idempotent but would leave two
        places that must agree on the prompt, which is the same class of
        divergence the single ``STAGE_SEQUENCE`` exists to prevent.
        """
        ctx.observations = {}
        observed_at = dict(ctx.world.last_observed_at)
        ctx.observed_before = {actor_id: observed_at.get(actor_id, 0) for actor_id in ctx.active}
        for actor_id in ctx.active:
            policy = self.policies[actor_id]
            ctx.observations[actor_id] = project(
                ctx.world,
                policy,
                noise=ctx.noise.observation_for(actor_id),
                signals=tuple(ctx.pending_signals.get(actor_id, ())),
                actions=visible_actions(policy, ctx.last_actions),
            )
            observed_at[actor_id] = ctx.world.step
        ctx.world = ctx.world.model_copy(update={"last_observed_at": observed_at})

        interval = self.settings.kernel.memory.summary_interval
        ctx.turns = {}
        for actor_id in ctx.active:
            observation = ctx.observations[actor_id]
            memory = ctx.memories[actor_id].maybe_refresh(observation, interval=interval)
            ctx.memories[actor_id] = memory
            ctx.turns[actor_id] = PendingTurn(
                run_id=ctx.spec.run_id,
                scenario_id=ctx.spec.scenario_id,
                actor_id=actor_id,
                observation=observation,
                memory=memory.render(),
                space=self.spaces[actor_id],
            )

        # A claim addressed to an actor that did not observe this step is not
        # discarded: it is waiting when that actor next looks. Dropping it
        # would make a signal's effect depend on the scheduler's choice of who
        # was salient, which is not something the sender can see.
        heard = set(ctx.active)
        ctx.pending_signals = {
            actor_id: claims
            for actor_id, claims in sorted(ctx.pending_signals.items())
            if actor_id not in heard
        }

    def stage_decide(self, ctx: _RunContext) -> None:
        """Stage 5: one turn per active actor, through the one call site."""
        ctx.decisions = {}
        ctx.caused_by = {}
        window = self.settings.kernel.memory.recent_observations
        for actor_id in ctx.active:
            turn = ctx.turns[actor_id]
            observation = turn.observation
            memory = ctx.memories[actor_id]
            decision = self.decider.decide(
                scenario_id=ctx.spec.scenario_id,
                actor_id=actor_id,
                observation=observation,
                memory=turn.memory,
                space=turn.space,
            )
            # Admissibility is re-checked here, not trusted to the decider
            # (ADR-0018). The agent adapter already admits what a model
            # emitted, but the arbiter is behind *this* boundary and any
            # decider reaches it -- a stand-in policy that returned an action
            # on a lever the actor does not hold would otherwise be executed
            # with no record that it happened.
            reason = refusal(decision.action, self.spaces[actor_id])
            if reason is not None:
                decision = decision.model_copy(
                    update={
                        "action": Wait(rationale=decision.action.rationale),
                        "coercion": decision.coercion or reason,
                    }
                )
            ctx.decisions[actor_id] = decision
            # Computed before this step's own movements enter the ledger: an
            # actor is responding to what happened before it looked.
            ctx.caused_by[actor_id] = ctx.ledger.antecedents(
                self.profiles[actor_id].observed,
                since=ctx.observed_before.get(actor_id, 0),
                before=observation.step,
            )
            ctx.memories[actor_id] = memory.remember(
                observation, action=decision.action, window=window
            )
            ctx.llm_calls += 1
            ctx.cache_hits += int(decision.cache_hit)
            ctx.tokens_in += decision.tokens_in
            ctx.tokens_out += decision.tokens_out

    def stage_arbitrate(self, ctx: _RunContext) -> None:
        """Stage 6: fold the actions into the world, then write the step's rows."""
        actions = {actor_id: ctx.decisions[actor_id].action for actor_id in ctx.active}
        seq_of = {actor_id: index for index, actor_id in enumerate(ctx.active)}
        result = arbitrate(
            world=ctx.world,
            actions=actions,
            mechanics=self.mechanics,
            jitter=ctx.noise.arbiter,
            propagation=self.propagation,
            gamma=self.settings.kernel.contest_gamma,
            max_step_delta=self.settings.kernel.max_step_delta,
            seq_of=seq_of,
        )
        step = ctx.world.step
        acted = dict(result.world.last_acted)
        for actor_id in ctx.active:
            acted[actor_id] = step
        ctx.world = result.world.model_copy(update={"last_acted": acted})

        for actor_id in ctx.active:
            decision = ctx.decisions[actor_id]
            ref = _ref(ctx.spec.run_id, step, seq_of[actor_id])
            deltas = result.factor_delta.get(actor_id, {})
            for factor_id in sorted(deltas):
                ctx.ledger.record(factor_id, magnitude=deltas[factor_id], ref=ref, at_step=step)
            ctx.events.append(
                DecisionEvent(
                    run_id=ctx.spec.run_id,
                    step=step,
                    seq=seq_of[actor_id],
                    actor_id=actor_id,
                    obs_hash=observation_hash(ctx.observations[actor_id]),
                    action=action_payload(decision.action),
                    caused_by=ctx.caused_by.get(actor_id, ()),
                    factor_delta={key: deltas[key] for key in sorted(deltas)},
                    cache_hit=decision.cache_hit,
                    tokens_in=decision.tokens_in,
                    tokens_out=decision.tokens_out,
                    latency_ms=decision.latency_ms,
                    coercion=decision.coercion,
                )
            )

        for delivery in result.signals:
            ctx.pending_signals.setdefault(delivery.to_actor, []).append(
                SignalClaim(
                    from_actor=delivery.from_actor,
                    factor_id=delivery.factor_id,
                    claimed_value=delivery.claimed_value,
                )
            )
        ctx.last_actions = dict(actions)

        ctx.world = update_streaks(ctx.world)
        ctx.absorbed = absorbed_factors(ctx.world, self.graph.outcome_rule.factor_ids)
        ctx.world = ctx.world.model_copy(update={"step": step + 1, "rng_counter": ctx.rng.draws})
        assert_counter(
            rng_counter=ctx.world.rng_counter,
            step=ctx.world.step,
            n_actors=len(self.actor_ids),
            n_factors=len(self.factor_ids),
        )
        ctx.ledger.prune(before_step=step - self.settings.kernel.activation.forced_interval * 2)
        ctx.steps.append(
            StepRecord(
                run_id=ctx.spec.run_id,
                step=step,
                exogenous_delta=ctx.exogenous_delta,
                arrivals=ctx.arrivals,
                contest_delta={
                    outcome.factor_id: outcome.delta
                    for outcome in sorted(result.outcomes, key=lambda o: o.factor_id)
                },
                active=ctx.active,
                eligible=ctx.eligible,
                rng_counter=ctx.world.rng_counter,
                state_hash=state_hash(ctx.world),
                absorbed=ctx.absorbed,
            )
        )

    def should_continue(self, ctx: _RunContext) -> str:
        """Stop at the horizon, or early on an absorbing condition (§7.6)."""
        if ctx.absorbed:
            ctx.termination = "absorbed"
            return "stop"
        if ctx.world.step >= self.settings.kernel.steps:
            ctx.termination = "horizon"
            return "stop"
        return "continue"


# ---------------------------------------------------------------------------
# LangGraph wiring
#
# The graph is built once at import and the Loom is passed per invocation in
# the config, so one compiled graph serves every scenario. The nodes are
# deliberately thin: each one calls a method and returns the same context, so
# the orchestration cannot become the place where the simulation's logic hides.
# ---------------------------------------------------------------------------


def _loom_of(config: Any) -> Loom:
    loom = config.get("loom") if isinstance(config, dict) else None
    if loom is None and hasattr(config, "get"):
        loom = config.get("configurable", {}).get("loom")
    if loom is None:
        raise RuntimeError("the compiled step graph was invoked without its Loom")
    return loom  # type: ignore[no-any-return]


def _build_graph() -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(_LoopState)

    def node(name: str) -> Any:
        def run_stage(state: _LoopState, config: Any = None) -> _LoopState:
            ctx = state["ctx"]
            getattr(_loom_of(config), name)(ctx)
            return {"ctx": ctx}

        run_stage.__name__ = name
        return run_stage

    stages = STAGE_SEQUENCE
    for stage in stages:
        builder.add_node(stage, node(stage))
    builder.add_edge(START, stages[0])
    for current, following in pairwise(stages):
        builder.add_edge(current, following)

    def router(state: _LoopState, config: Any = None) -> str:
        return _loom_of(config).should_continue(state["ctx"])

    builder.add_conditional_edges(stages[-1], router, {"continue": stages[0], "stop": END})
    return builder.compile()


def _ref(run_id: str, step: int, seq: int) -> Any:
    from cascade.trace.events import EventRef

    return EventRef(run_id=run_id, step=step, seq=seq)
