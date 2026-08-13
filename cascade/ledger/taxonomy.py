"""Deterministic screening: domain tagging, party detection, scope exclusion.

Pure functions, no I/O, no clock, no RNG (spec §13 "pure core, thin shell").

**What this module is and is not.** Spec §3.1 requires "at least three
distinguishable parties with non-identical objectives". That is a semantic
judgement, and at M1 there is no compiler to make it -- the causal graph, with
its authoritative 8-20 actor count, is not built until M4. So this is a
*conservative screen*, not a measurement: it admits a question only when the
evidence is structural or explicit, and it fails closed on everything else.
Each admission records *which* rule fired (``party_rule``) so the composition
of the set can be audited afterwards rather than taken on trust.

The strongest rule needs no text at all. When a real-world event carries three
or more mutually exclusive outcomes -- a race with three candidates -- the
parties are established by the structure of the event, and no amount of
heuristic reading can be more reliable than that.
"""

from __future__ import annotations

import re

from cascade.ledger.schema import Domain

__all__ = [
    "classify_domain",
    "extract_known_actors",
    "extract_parties",
    "is_single_quantity",
]

# ---------------------------------------------------------------------------
# Scope exclusion: single-quantity questions
#
# "Will inflation exceed 3%" is out of scope (spec §3.1) -- it belongs to the
# trend-forecasting literature. These questions have no parties to model, so
# they would compile into a degenerate causal graph and quietly widen the
# study's claimed scope.
# ---------------------------------------------------------------------------

_QUANTITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:exceed|surpass|be above|be below|go above|go below|stay above|stay below)\b",
        r"\b(?:greater|higher|lower|less) than\b",
        r"\b(?:increase|decrease|cut|hike|raise)s? (?:by|to) \d",
        r"\b\d+\s*(?:\+\s*)?bps\b",
        r"\b(?:close|closing|open|trade|trading) (?:above|below|at or above|at or below)\b",
        r"\b(?:reach|hit|top|exceed)\b[^?]{0,40}\$\s?\d",
        r"\bprice of\b",
        r"\b(?:inflation|unemployment|cpi|gdp|interest rate)s?\b[^?]{0,40}\b\d",
        r"\bhow many\b",
        r"\bwhat will the\b.*\bbe\b",
    )
)


def is_single_quantity(text: str) -> bool:
    """Return whether ``text`` reads as a question about one number.

    Preserves the scope boundary in spec §3.1: a single-quantity trend question
    has no distinguishable parties, so admitting one would put a scenario into
    the set that the causal decomposition cannot represent -- and the resulting
    degenerate graph would be scored as though it were a strategic forecast.
    """
    return any(pattern.search(text) for pattern in _QUANTITY_PATTERNS)


# ---------------------------------------------------------------------------
# Party detection
# ---------------------------------------------------------------------------

# Actors that recur across strategic questions. Deliberately institutions,
# states and blocs rather than individuals: an individual is usually acting for
# one of these, and counting both would double-count one party.
_ACTOR_LEXICON: tuple[str, ...] = (
    "African Union",
    "Argentina",
    "Australia",
    "Austria",
    "Bank of England",
    "Bank of Japan",
    "Belgium",
    "BRICS",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Congress",
    "DOJ",
    "Democrats",
    "Denmark",
    "ECB",
    "EU",
    "Egypt",
    "European Central Bank",
    "European Commission",
    "European Union",
    "FDA",
    "FTC",
    "Federal Reserve",
    "Finland",
    "France",
    "GOP",
    "Germany",
    "Greece",
    "Hamas",
    "Hezbollah",
    "House",
    "Houthis",
    "IAEA",
    "ICC",
    "IMF",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Kremlin",
    "Lebanon",
    "Mexico",
    "NATO",
    "NHS",
    "Netherlands",
    "Nigeria",
    "North Korea",
    "Norway",
    "OPEC",
    "Ofcom",
    "Pakistan",
    "Palestine",
    "Parliament",
    "Pentagon",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Republicans",
    "Russia",
    "SEC",
    "Saudi Arabia",
    "Senate",
    "Serbia",
    "Singapore",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Syria",
    "Taiwan",
    "Thailand",
    "Turkey",
    "UAE",
    "UK",
    "UN",
    "US",
    "USA",
    "Ukraine",
    "United Kingdom",
    "United Nations",
    "United States",
    "Venezuela",
    "Vietnam",
    "WHO",
    "WTO",
    "White House",
    "Yemen",
)

# Names that mean the same party. Collapsed before counting so that
# "US" + "United States" is one actor, not two -- an inflated count is exactly
# the failure this screen exists to prevent.
_ACTOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("USA", "United States"),
    ("US", "United States"),
    ("White House", "United States"),
    ("Pentagon", "United States"),
    ("UK", "United Kingdom"),
    ("EU", "European Union"),
    ("European Commission", "European Union"),
    ("ECB", "European Central Bank"),
    ("UN", "United Nations"),
    ("Kremlin", "Russia"),
    ("GOP", "Republicans"),
)

_ALIAS_TARGET = dict(_ACTOR_ALIASES)

# Capitalised runs that are never parties: they are dates, question scaffolding
# or venue names that would otherwise inflate the count.
_NOT_A_PARTY: frozenset[str] = frozenset(
    {
        "April",
        "August",
        "December",
        "February",
        "Friday",
        "January",
        "July",
        "June",
        "March",
        "May",
        "Monday",
        "November",
        "October",
        "Saturday",
        "September",
        "Sunday",
        "Thursday",
        "Tuesday",
        "Wednesday",
        "A",
        "An",
        "And",
        "As",
        "At",
        "Be",
        "Before",
        "By",
        "Did",
        "Do",
        "Does",
        "For",
        "From",
        "How",
        "If",
        "In",
        "Is",
        "It",
        "Of",
        "On",
        "Or",
        "The",
        "There",
        "This",
        "To",
        "What",
        "When",
        "Whether",
        "Which",
        "Who",
        "Will",
        "With",
        "Would",
        "Yes",
        "No",
        "Not",
        "Resolves",
        "Resolved",
        "Resolution",
        "Market",
        "Question",
        "Note",
        "Otherwise",
        "Source",
    }
)

# Compared casefolded. Resolution criteria are overwhelmingly written
# "YES if ..." / "NO if ...", and a case-sensitive check let "YES" through as
# a proper noun -- adding a phantom party to nearly every market question and
# letting two-party questions pass the >= 3 rule. Actors that are genuinely
# uppercase (US, EU, UN, WHO) are recognised through the lexicon, which does
# not consult this set, so excluding them here costs nothing.
_NOT_A_PARTY_FOLDED: frozenset[str] = frozenset(word.casefold() for word in _NOT_A_PARTY)

# "of" and "the" are internal to a single name ("Bank of England", "United
# Nations"). "and" is not: it joins two *different* actors, and allowing it
# would turn "Russia and Ukraine" into a third, phantom party -- inflating the
# very count this screen exists to hold down.
# Straight and curly apostrophes. Real question text uses both, and treating
# them as different characters would split one name into two candidate parties.
_APOSTROPHES = "'’"  # noqa: RUF001 -- the ambiguity is the point; both are matched
_PROPER_NOUN = re.compile(
    rf"\b(?:[A-Z][\w{_APOSTROPHES}-]*)(?:\s+(?:of|the)?\s*[A-Z][\w{_APOSTROPHES}-]*)*"
)


def _canonical(name: str) -> str:
    return _ALIAS_TARGET.get(name, name)


def extract_known_actors(text: str) -> tuple[str, ...]:
    """Return only the *recognised institutional actors* ``text`` names.

    This is what the ``named_parties`` rule counts, and the distinction from
    :func:`extract_parties` is the whole point.

    Counting capitalised words does not count parties. Measured against the
    real pool, a proper-noun count admitted "Will the Blaze Star go nova?",
    "Will Diddy be alive on Jan 1st?" and "Will the NYT review use more than
    five em dashes?" -- each carrying three or more capitalised tokens and
    exactly zero parties with objectives. A party is an actor with interests
    that can act on them, so the set of things that count as one is enumerated
    rather than inferred, and anything unrecognised fails closed.

    The cost is real and is the right cost to pay: a strategic question
    between two companies not in the lexicon is excluded. Precision here
    protects the study's claim about what it is forecasting; recall is
    recoverable by extending the lexicon deliberately.
    """
    found: set[str] = set()
    for actor in _ACTOR_LEXICON:
        if re.search(rf"\b{re.escape(actor)}\b", text):
            found.add(_canonical(actor))
    return tuple(sorted(_collapse_nested(found)))


def extract_parties(text: str) -> tuple[str, ...]:
    """Return the distinct parties ``text`` names, canonicalised and sorted.

    Preserves the >= 3 party rule's meaning rather than its letter: aliases are
    collapsed so one actor referred to two ways counts once, and calendar and
    scaffolding words are excluded so a date cannot masquerade as a party.

    Returned sorted (invariant 7) -- this feeds a stored field and, through it,
    the manifest hash.
    """
    found: set[str] = set()

    for actor in _ACTOR_LEXICON:
        # Word-boundary match so "US" does not fire inside "Australia" and
        # "Iran" does not fire inside "Iranian"... which it should, but a
        # substring match would also fire inside "Iraq"-adjacent noise. The
        # adjectival forms are handled by the lexicon entries themselves.
        if re.search(rf"\b{re.escape(actor)}\b", text):
            found.add(_canonical(actor))

    for match in _PROPER_NOUN.finditer(text):
        candidate = match.group(0).strip()
        words = candidate.split()
        if not words or len(candidate) < 3:
            continue
        if any(word.casefold() in _NOT_A_PARTY_FOLDED for word in words):
            continue
        found.add(_canonical(candidate))

    return tuple(sorted(_collapse_nested(found)))


def _collapse_nested(names: set[str]) -> set[str]:
    """Drop a name that wholly contains another recognised name.

    "United States" and "United States Department of Justice" are one party
    for screening purposes far more often than they are two, so the longer
    form is dropped. The conservative direction is fewer parties: this screen
    fails closed, and M4's validator is the authoritative actor count.
    """
    ordered = sorted(names, key=lambda name: (len(name), name))
    kept: list[str] = []
    for name in ordered:
        if any(re.search(rf"\b{re.escape(shorter)}\b", name) for shorter in kept):
            continue
        kept.append(name)
    return set(kept)


# ---------------------------------------------------------------------------
# Domain tagging
#
# The domain drives the <= 25% stratification cap and the per-domain breakdown
# in the M7 report, where "a headline win driven by one over-represented domain
# is not a win". First match wins, so the order is the precedence.
# ---------------------------------------------------------------------------

_DOMAIN_PATTERNS: tuple[tuple[Domain, re.Pattern[str]], ...] = (
    (
        "conflict",
        re.compile(
            r"\b(?:war|ceasefire|truce|invade|invasion|missile|troops|"
            r"military|air ?strike|nuclear test|hostage|armistice|peace deal|"
            r"occupation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "elections",
        re.compile(
            r"\b(?:elect(?:ed|ion|oral)?|primary|nominee|nomination|ballot|"
            r"vote share|win the presidency|presidential|referendum|poll)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "labor",
        re.compile(
            r"\b(?:strike|union|collective bargaining|walkout|labou?r dispute)\b", re.IGNORECASE
        ),
    ),
    (
        "regulation",
        re.compile(
            r"\b(?:antitrust|regulator|regulatory|approv(?:e|es|ed|ing|al)|ban|sanction|"
            r"lawsuit|court|ruling|indict|tariff|investigation|FDA|FTC|DOJ|SEC)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "corporate",
        re.compile(
            r"\b(?:merger|acquisition|acquire|IPO|bankrupt|CEO|layoff|"
            r"shareholder|takeover|buyout|earnings|resign)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "macro_policy",
        re.compile(
            r"\b(?:federal reserve|central bank|interest rate|inflation|"
            r"recession|budget|debt ceiling|fiscal|monetary|shutdown)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "technology",
        re.compile(
            r"\b(?:AI|artificial intelligence|model|chip|semiconductor|launch|"
            r"satellite|rocket|software|crypto|bitcoin|blockchain)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "health",
        re.compile(r"\b(?:pandemic|vaccine|outbreak|epidemic|virus|WHO|disease)\b", re.IGNORECASE),
    ),
    (
        "sports",
        re.compile(
            r"\b(?:championship|finals|world cup|olympic|league|tournament|"
            r"playoff|super bowl|match|season|NBA|NFL|FIFA|UEFA)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "geopolitics",
        re.compile(
            r"\b(?:treaty|summit|alliance|diplomat|accession|withdraw|embassy|"
            r"border|NATO|United Nations|bilateral|negotiat)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_domain(text: str) -> Domain:
    """Assign the stratification domain. First pattern wins.

    Preserves the <= 25% domain cap's usefulness: a tag that varied with
    phrasing would let one real-world domain enter under several labels and
    defeat the cap it exists to enforce.
    """
    for domain, pattern in _DOMAIN_PATTERNS:
        if pattern.search(text):
            return domain
    return "other"
