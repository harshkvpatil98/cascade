"""The bench query distribution (spec §4.2).

Spec §4.2 asks for "10,000 queries sampled from real agent query
distributions across the full cutoff range". At M3 no agent exists yet -- the
actor agents land at M5 -- so there is no recorded distribution to sample. The
options are to invent one, to defer the bench to M5, or to derive one from the
thing the agents will provably be asking about. This module does the third.

**What an agent will retrieve on.** Per spec §7, an actor agent observes a
projection of world state and reasons about its own objectives against the
other parties. Its queries are therefore about (a) the question under
forecast, (b) the parties named in it, and (c) the interaction between them.
Those three are exactly what `scenarios` already holds, per scenario, with a
cutoff attached. Templating over them produces a distribution with the right
*shape* -- topic drawn from the study's own subject matter, cutoff drawn from
the study's own cutoff range -- without pretending to a fidelity it cannot have
until M5.

**Two properties are load-bearing and are asserted by tests:**

*Nothing here reads an outcome.* Queries are built from `question`,
`resolution_criterion` and `party_names`, all of which live in `scenarios` and
none of which is a label. Building a retrieval benchmark from resolution text
would measure how well the index finds the answer key, which is the one number
that must never look good by construction.

*Generation is deterministic.* The RNG is seeded once from the study salt
(invariant 4's derivation, applied to a bench rather than a run), so two bench
runs on the same registry compare like with like. A bench whose query set
drifts between runs cannot detect a regression.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from cascade.ledger.schema import Scenario

__all__ = ["BenchQuery", "build_queries", "query_templates"]


@dataclass(frozen=True)
class BenchQuery:
    """One query and the cutoff it must be answered under.

    ``as_of`` travels with the text because the pair is the unit of work: a
    query without its cutoff is not a retrieval this study ever performs.
    """

    scenario_id: str
    text: str
    as_of: datetime

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError(f"as_of must be timezone-aware, got naive {self.as_of!r}")


# Templates approximating the three retrieval intents §7 gives an actor agent:
# orient on the question, assess a counterparty, and look for the interaction
# between two of them. `{topic}` is the scenario question with its
# interrogative framing stripped; `{party}` and `{other}` are named parties.
_TEMPLATES: tuple[str, ...] = (
    "{topic}",
    "{party} {topic}",
    "{party} position statement",
    "{party} recent developments",
    "{party} and {other} negotiations",
    "{party} response to {other}",
    "{topic} analysis and outlook",
    "background on {topic}",
)


def query_templates() -> tuple[str, ...]:
    """The templates, exposed so a test can assert none references an outcome."""
    return _TEMPLATES


def _topic(scenario: Scenario) -> str:
    """The scenario question reduced to a search phrase.

    Interrogative scaffolding ("will", "by 2026", a trailing question mark)
    carries no retrievable signal and, left in, biases every query in the set
    toward the same handful of function words.
    """
    text = scenario.question.strip().rstrip("?").strip()
    lowered = text.lower()
    for prefix in ("will ", "is ", "are ", "does ", "do ", "did ", "can ", "shall "):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    return " ".join(text.split())


def build_queries(
    scenarios: Sequence[Scenario], *, count: int, salt: str
) -> tuple[BenchQuery, ...]:
    """Build ``count`` queries spread evenly across ``scenarios``.

    Even spread is deliberate rather than proportional-to-anything: the
    acceptance criterion is a p95 over the *full cutoff range*, and sampling
    scenarios by any weight would let the dense end of the corpus dominate the
    percentile and hide the tail the criterion exists to catch.

    Determinism comes from a blake2b-keyed seed over the scenario set, so
    adding a scenario changes the query set (correctly -- it is a different
    population) while re-running on the same set does not.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if not scenarios:
        raise ValueError("cannot build a query set from an empty scenario registry")

    ordered = sorted(scenarios, key=lambda item: item.scenario_id)
    digest = hashlib.blake2b(
        "|".join(item.scenario_id for item in ordered).encode("utf-8"),
        key=salt.encode("utf-8"),
        digest_size=8,
    ).digest()
    # S311 is suppressed below: this is a bench fixture generator, not a
    # security control.
    # The requirement is reproducibility -- the same registry must yield the
    # same 10,000 queries -- which is exactly what a seeded Mersenne Twister
    # gives and what a cryptographic RNG would take away.
    rng = random.Random(int.from_bytes(digest, "big"))  # noqa: S311

    out: list[BenchQuery] = []
    index = 0
    while len(out) < count:
        scenario = ordered[index % len(ordered)]
        index += 1

        topic = _topic(scenario)
        parties = [name for name in scenario.party_names if name.strip()]
        template = _TEMPLATES[rng.randrange(len(_TEMPLATES))]

        if "{party}" in template and not parties:
            # A scenario with no named parties (the `event_siblings` rule
            # admits these) can still supply topic-only queries; substituting a
            # placeholder would inject a token that appears in no document.
            template = "{topic}"

        party = rng.choice(parties) if parties else ""
        others = [name for name in parties if name != party] or parties
        other = rng.choice(others) if others else ""

        text = " ".join(template.format(topic=topic, party=party, other=other).split()).strip()
        if not text:
            continue

        out.append(
            BenchQuery(scenario_id=scenario.scenario_id, text=text, as_of=scenario.cutoff_ts)
        )

    return tuple(out)
