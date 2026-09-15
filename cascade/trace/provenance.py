"""Tracing an outcome back to the decision that caused it (spec §11.2; M8).

The study's strongest claim is that "every outcome traces back to the exact
agent decision that triggered it". §11.2 is explicit that this must be a
*command*, "demonstrable in thirty seconds rather than described" -- so the
walk is a recursive CTE over the append-only log, and the rendering is the
shape §11.2 prints.

Three things make the chain terminate rather than wander:

* **``caused_by`` is written by the arbiter**, which knows which observations
  an action responded to (§11.1). Nothing is inferred here; the walk follows
  edges the kernel recorded.
* **Depth is bounded by the horizon.** A run is 24 steps and an antecedent is
  always strictly earlier, so a cycle is impossible -- but the bound is stated
  anyway, because a recursive CTE with no bound is an outage waiting for a
  corrupt row.
* **The root is an exogenous shock**, read from ``run_steps``. §11.1's table is
  one row per agent decision (adding 864,000 world-movement rows would put the
  M6 count 20% over its criterion), so the chain crosses into that table at its
  end rather than stopping at the earliest decision and calling it a cause.

Pure rendering, impure reading: :func:`explain` takes a settings object and
returns typed rows; :func:`render` turns those rows into the §11.2 display and
is testable without a database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from cascade.config import Settings

__all__ = [
    "MAX_CHAIN_DEPTH",
    "ChainLink",
    "OutcomeExplanation",
    "explain",
    "render",
    "terminal_factors",
]

Role = Literal["admin", "sim", "eval"]

# One run is `kernel.steps` long and every antecedent is strictly earlier, so
# the walk cannot exceed the horizon. Stated as a constant because an unbounded
# recursive CTE over a 4.2M-row table is an outage waiting for one bad row.
MAX_CHAIN_DEPTH = 64


@dataclass(frozen=True, slots=True)
class ChainLink:
    """One decision on the path from an outcome to its cause."""

    depth: int
    step: int
    seq: int
    actor_id: str
    action_type: str
    factor_delta: dict[str, float]
    coercion: str | None = None

    def dominant(self) -> tuple[str, float] | None:
        """The factor this decision moved most, or ``None`` if it moved none.

        "Most" by magnitude with the factor id settling a tie, so the rendered
        line is a function of the row rather than of dict order (invariant 7).
        """
        if not self.factor_delta:
            return None
        factor = min(self.factor_delta, key=lambda key: (-abs(self.factor_delta[key]), key))
        return factor, self.factor_delta[factor]


@dataclass(frozen=True, slots=True)
class RootShock:
    """The exogenous movement the chain bottoms out in (§7.2 stage 1)."""

    step: int
    factor_id: str
    delta: float


@dataclass
class OutcomeExplanation:
    """Everything `cascade trace explain` prints for one run."""

    run_id: str
    scenario_id: str
    config_id: str
    replicate: int
    outcome_score: float
    steps_run: int
    termination: str
    seed_factor: str | None
    seed_value: float | None
    links: list[ChainLink] = field(default_factory=list)
    root: RootShock | None = None
    truncated: bool = False
    """True when the walk hit :data:`MAX_CHAIN_DEPTH` with antecedents left."""

    @property
    def complete(self) -> bool:
        """A chain is complete when it reaches an exogenous shock.

        The M8 criterion is "returns a complete chain to a root cause", so the
        command asserts this rather than printing whatever it found.
        """
        return self.root is not None and not self.truncated


def terminal_factors(outcome_rule_factors: Sequence[str]) -> tuple[str, ...]:
    """The factors the outcome rule reads, sorted (invariant 7).

    The chain starts from the decision that last moved one of these. Starting
    anywhere else would explain a movement the outcome does not depend on.
    """
    return tuple(sorted(set(outcome_rule_factors)))


# ---------------------------------------------------------------------------
# The walk. Spec §11.2's recursive CTE, with the root lookup it implies.
# ---------------------------------------------------------------------------

# §11.2's walk, as a *path* rather than an ancestor set.
#
# The spec's CTE recurses over every element of `caused_by`, and its own
# example output is a single path -- d0, d1, d2, d3, root. Those are not the
# same thing, and the difference is not cosmetic: `caused_by` holds up to
# CAUSED_BY_LIMIT (6) antecedents, so recursing over all of them expands
# 6^depth. Measured, the ancestor-set form did not return on a 24-step run.
#
# So the recursion follows one antecedent per level: `caused_by->0`, the
# earliest of the antecedents the ledger judged the largest movers. That is
# deterministic (the array is written in (step, seq) order from a
# magnitude-ranked selection), linear in the horizon, and it is the chain
# §11.2 prints.
_CHAIN_SQL = """
WITH RECURSIVE seed AS (
    SELECT e.run_id, e.step, e.seq
    FROM events e
    WHERE e.run_id = %(run_id)s
      AND e.factor_delta ? %(factor)s
    ORDER BY e.step DESC, e.seq DESC
    LIMIT 1
),
chain AS (
    SELECT e.run_id, e.step, e.seq, e.actor_id, e.action, e.caused_by,
           e.factor_delta, e.coercion, 0 AS depth
    FROM events e
    JOIN seed s ON s.run_id = e.run_id AND s.step = e.step AND s.seq = e.seq

    UNION ALL

    SELECT p.run_id, p.step, p.seq, p.actor_id, p.action, p.caused_by,
           p.factor_delta, p.coercion, c.depth + 1
    FROM chain c
    JOIN events p
      ON  p.run_id = (c.caused_by->0->>'run_id')::uuid
      AND p.step   = (c.caused_by->0->>'step')::smallint
      AND p.seq    = (c.caused_by->0->>'seq')::smallint
    WHERE c.depth < %(max_depth)s
      AND jsonb_array_length(c.caused_by) > 0
      -- An antecedent is always strictly earlier, so this cannot loop on a
      -- well-formed log. It is asserted anyway: a recursive CTE with no
      -- guard is an outage waiting for one corrupt row.
      AND (p.step, p.seq) < (c.step, c.seq)
)
SELECT depth, step, seq, actor_id, action->>'type' AS action_type,
       factor_delta, coercion
FROM chain
ORDER BY depth
"""

_ROOT_SQL = """
SELECT step, key, value::text::double precision AS delta
FROM run_steps, jsonb_each(exogenous_delta)
WHERE run_id = %(run_id)s AND step <= %(before_step)s
ORDER BY abs(value::text::double precision) DESC, step DESC, key
LIMIT 1
"""


def _connect(settings: Settings, role: Role) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=30)


def explain(
    settings: Settings,
    *,
    run_id: str,
    factor: str | None = None,
    role: Role = "eval",
    max_depth: int = MAX_CHAIN_DEPTH,
) -> OutcomeExplanation:
    """Walk one run's log from its outcome back to an exogenous shock.

    ``factor`` selects which of the outcome rule's factors to explain; the
    default is the one whose terminal value the run's own log shows was moved
    last, which is the movement an operator asking "why this outcome" means.

    Read under ``eval``. The walk touches ``events`` and ``run_steps`` and no
    label, but it is an analysis path and the study's analysis role is eval.
    """
    with _connect(settings, role) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT scenario_id, config_id, replicate, outcome_score, steps_run, termination "
            "FROM runs WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(
                f"no run {run_id!r} in the runs table; `cascade simulate run` writes a row "
                "only when a run reaches its horizon, so an interrupted run has none"
            )
        scenario_id, config_id, replicate, outcome_score, steps_run, termination = row

        chosen = factor
        if chosen is None:
            cur.execute(
                "SELECT key FROM run_steps, jsonb_each(contest_delta) "
                "WHERE run_id = %s ORDER BY step DESC, abs(value::text::double precision) DESC, "
                "key LIMIT 1",
                (run_id,),
            )
            picked = cur.fetchone()
            chosen = str(picked[0]) if picked else None

        links: list[ChainLink] = []
        if chosen is not None:
            cur.execute(_CHAIN_SQL, {"run_id": run_id, "factor": chosen, "max_depth": max_depth})
            for depth, step, seq, actor_id, action_type, factor_delta, coercion in cur.fetchall():
                links.append(
                    ChainLink(
                        depth=int(depth),
                        step=int(step),
                        seq=int(seq),
                        actor_id=str(actor_id),
                        action_type=str(action_type),
                        # Sorted (invariant 7). The values are already
                        # deterministic, but the static check is deliberately
                        # blunt and uniform compliance is what keeps it useful.
                        factor_delta={
                            key: float(value)
                            for key, value in sorted(dict(factor_delta or {}).items())
                        },
                        coercion=str(coercion) if coercion else None,
                    )
                )

        earliest = links[-1].step if links else int(steps_run)
        cur.execute(_ROOT_SQL, {"run_id": run_id, "before_step": earliest})
        root_row = cur.fetchone()
        root = (
            RootShock(step=int(root_row[0]), factor_id=str(root_row[1]), delta=float(root_row[2]))
            if root_row
            else None
        )

    seed_value: float | None = None
    if links:
        dominant = links[0].dominant()
        if dominant is not None:
            seed_value = dominant[1]

    return OutcomeExplanation(
        run_id=run_id,
        scenario_id=str(scenario_id),
        config_id=str(config_id),
        replicate=int(replicate),
        outcome_score=float(outcome_score),
        steps_run=int(steps_run),
        termination=str(termination),
        seed_factor=chosen,
        seed_value=seed_value,
        links=links,
        root=root,
        truncated=any(link.depth >= max_depth for link in links),
    )


def render(explanation: OutcomeExplanation) -> list[str]:
    """Render §11.2's display. Pure, so the shape is testable without a run."""
    head = (
        f"outcome_score {explanation.outcome_score:.2f}" f"  <- factor {explanation.seed_factor!r}"
        if explanation.seed_factor
        else f"outcome_score {explanation.outcome_score:.2f}"
    )
    if explanation.seed_factor:
        head += f" moved last at step {explanation.links[0].step}" if explanation.links else ""
    lines = [head]

    width = max((len(link.actor_id) for link in explanation.links), default=0)
    for link in explanation.links:
        dominant = link.dominant()
        # Four decimals, not two. A per-step factor movement is bounded by
        # `kernel.max_step_delta` (0.12) and is routinely an order of magnitude
        # under it, so `%+.2f` prints `+0.00` for a real movement -- which
        # reads as "this decision did nothing" in the one display whose job is
        # to show that it did.
        movement = f"{dominant[1]:+.4f} on {dominant[0]}" if dominant else "no factor movement"
        suffix = f"  (coerced: {link.coercion})" if link.coercion else ""
        lines.append(
            f"  d{link.depth:<2d} step {link.step:2d}  actor={link.actor_id:<{width}s} "
            f"{link.action_type:<9s} {movement}{suffix}"
        )

    if explanation.root is not None:
        lines.append(
            f"  root  step {explanation.root.step:2d}  exogenous shock on "
            f"{explanation.root.factor_id!r} {explanation.root.delta:+.2f}"
        )
    else:
        lines.append("  root  not reached: this run recorded no exogenous movement")
    return lines
