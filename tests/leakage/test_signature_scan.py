"""Signature scan over what retrieval actually returns (spec §4.2, criterion 2).

§4.2 asks for "regex + embedding similarity of every retrieved chunk against
resolution text; flag above threshold; **report the reviewed count**". The
deliverable is a number and a list, not a pass/fail — which is why the
assertions here are about the scan having *run over the real corpus*, not
about the flag count being zero.

Flagging is not failing, and the distinction is load-bearing. A pre-cutoff
document that discusses the question in confident language is exactly the
evidence this study is built to retrieve. Dropping it because it pattern-matches
resolution text would be curating the corpus toward a conclusion — a worse
methodological failure than the leak it imitates, because it would be
invisible in the results.

What *would* be a leak is a flagged chunk that also postdates its cutoff. That
is asserted, and it is the only hard assertion in this file.
"""

from __future__ import annotations

from typing import Any

import pytest

from cascade.config import Settings
from cascade.retrieval.leakage import scan_signatures, violations
from cascade.retrieval.search import Chronofence

pytestmark = pytest.mark.leakage

# How many scenarios to scan. The full 180 would take minutes for a number
# that is stable well before then; the sample is deterministic (the first N by
# scenario id) so the reported count is reproducible.
SCAN_SCENARIOS = 60


def _cosine(left: Any, right: Any) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


@pytest.fixture(scope="module")
def scan(live_settings: Settings, records: tuple[Any, ...], embedder: Any) -> Any:
    """Run the scan once; several assertions read the same result."""
    selected = sorted(records, key=lambda item: item.scenario.scenario_id)[:SCAN_SCENARIOS]
    threshold = live_settings.retrieval.signature_similarity_threshold
    k = live_settings.retrieval.bench_recall_k

    # The query is the scenario's question; the *reference* it is compared
    # against is the resolution criterion. Retrieval must never be driven by
    # resolution text, so the two are kept apart deliberately.
    question_vectors = embedder.encode([item.scenario.question for item in selected])
    criterion_vectors = embedder.encode([item.scenario.resolution_criterion for item in selected])

    reviewed = 0
    findings: list[Any] = []
    post_cutoff: list[Any] = []

    with Chronofence(live_settings, role="eval") as fence:
        for record, question_vector, criterion_vector in zip(
            selected, question_vectors, criterion_vectors, strict=True
        ):
            result = fence.search(question_vector, as_of=record.scenario.cutoff_ts, k=k)
            reviewed += len(result.chunks)
            post_cutoff.extend(violations(result.chunks, as_of=record.scenario.cutoff_ts))

            if not result.chunks:
                continue
            bodies = embedder.encode([chunk.body for chunk in result.chunks])
            similarities = [_cosine(criterion_vector, body) for body in bodies]
            findings.extend(
                scan_signatures(
                    result.chunks,
                    scenario_id=record.scenario.scenario_id,
                    cutoff_ts=record.scenario.cutoff_ts,
                    similarities=similarities,
                    threshold=threshold,
                )
            )

    return {
        "scenarios": len(selected),
        "reviewed": reviewed,
        "findings": findings,
        "post_cutoff": post_cutoff,
        "threshold": threshold,
    }


def test_the_scan_reviewed_real_retrieved_chunks(scan: dict[str, Any]) -> None:
    """Guards the reported count against being zero for the wrong reason.

    A scan that reviewed nothing would report no findings and look reassuring.
    """
    assert scan["scenarios"] == SCAN_SCENARIOS
    assert scan["reviewed"] > 0, "the scan reviewed no chunks; it tested nothing"


def test_no_flagged_chunk_postdates_its_cutoff(scan: dict[str, Any]) -> None:
    """The only hard assertion: a flag plus a late date is an actual leak."""
    leaked = [finding for finding in scan["findings"] if finding.published_at >= finding.cutoff_ts]
    assert leaked == [], (
        f"{len(leaked)} chunks were both flagged as resolution-like and dated at "
        f"or after their cutoff: "
        f"{[(f.chunk_id, f.published_at.isoformat()) for f in leaked[:5]]}"
    )


def test_nothing_retrieved_in_the_scan_postdates_its_cutoff(scan: dict[str, Any]) -> None:
    """The same property over every reviewed chunk, flagged or not."""
    assert (
        scan["post_cutoff"] == []
    ), f"{len(scan['post_cutoff'])} retrieved chunks postdate their cutoff"


def test_the_reviewed_count_is_reported(scan: dict[str, Any], capsys: Any) -> None:
    """§4.2 asks for the count, so the suite emits it rather than hiding it.

    Printed with `-s` or captured in the report; either way the number exists
    and is attached to a run rather than to a claim.
    """
    findings = scan["findings"]
    with capsys.disabled():
        print(
            f"\nsignature scan: reviewed {scan['reviewed']:,} chunks across "
            f"{scan['scenarios']} scenarios at threshold {scan['threshold']:.2f}; "
            f"flagged {len(findings)} "
            f"({len(findings) / max(1, scan['reviewed']):.2%})"
        )
        by_phrase = sum(1 for finding in findings if finding.phrases)
        print(
            f"  {by_phrase} flagged on a retrospective phrase, "
            f"{len(findings) - by_phrase} on similarity alone"
        )
        for finding in findings[:5]:
            print(
                f"  - {finding.chunk_id} ({finding.published_at.date()} < "
                f"{finding.cutoff_ts.date()}) sim={finding.similarity:.3f} "
                f"phrases={finding.phrases}"
            )
    assert isinstance(findings, list)
