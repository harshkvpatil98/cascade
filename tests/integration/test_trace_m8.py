"""M8's criteria against the live database (spec §8.4, §11.2, §12.4).

The three things M8 gates on are all properties of stored data, so they are
asserted against stored data rather than against fixtures: a replay that
reproduces a hash computed in the same process proves only that the hash
function is a function.
"""

from __future__ import annotations

from typing import Any

import pytest

from cascade.config import Settings
from cascade.trace.ledger import local_spend, reconcile
from cascade.trace.provenance import explain
from cascade.trace.replay import load_replay_targets, verify_replays

pytestmark = pytest.mark.integration


def _env_file() -> str:
    """The env file the live fixture pointed `Settings` at.

    The repo's `.env` rather than `os.environ["CASCADE_ENV_FILE"]`: the suite's
    autouse fixture re-points that variable at a nonexistent path before every
    test, deliberately, so nothing depends on machine state. A replay child
    reading it would find no password, fail to connect, and the failure would
    be reported as a replay divergence -- which is the wrong diagnosis for a
    harness problem.
    """
    from cascade.config import REPO_ROOT

    return str(REPO_ROOT / ".env")


def _connect(settings: Settings) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url("admin"), connect_timeout=10)


@pytest.fixture(scope="module")
def stored_runs(live_settings: Settings) -> tuple[Any, ...]:
    targets = load_replay_targets(live_settings, limit=25)
    if not targets:
        pytest.skip("no stored runs; run `cascade simulate all` or `cascade eval grid`")
    return targets


class TestReplayDeterminism:
    """M8 criterion 1: a byte-identical event-log hash across processes."""

    def test_the_stored_runs_carry_a_hash_and_step_hashes(
        self, stored_runs: tuple[Any, ...]
    ) -> None:
        """Without both, a divergence can be detected but not localised."""
        for target in stored_runs:
            assert len(target.event_log_hash) == 64
            assert target.state_hashes, f"{target.run_id} recorded no per-step state hashes"

    def test_a_small_batch_replays_byte_identically_in_fresh_processes(
        self, stored_runs: tuple[Any, ...]
    ) -> None:
        """Three rather than 25, because the criterion is about the *property*
        and the command runs the full 25. Each child gets a different
        `PYTHONHASHSEED`, which is where a set iterated for a tie-break shows
        up and nowhere else.
        """
        report = verify_replays(stored_runs[:3], workers=3, env_file=_env_file())
        assert report.attempted == 3
        assert report.ok, [
            (item.run_id, item.first_divergent_step, item.error) for item in report.diverged
        ]

    def test_the_replay_writes_nothing_to_the_append_only_log(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        """Invariant 6. A verification pass that inserted its own copy would
        make the M6 event count drift every time someone checked M8."""
        with _connect(live_settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM events")
            before = cur.fetchone()[0]
        verify_replays(stored_runs[:2], workers=2, env_file=_env_file())
        with _connect(live_settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM events")
            after = cur.fetchone()[0]
        assert after == before

    def test_targets_are_chosen_deterministically(self, live_settings: Settings) -> None:
        """A random 25 would make a failure depend on which 25 were drawn."""
        first = [item.run_id for item in load_replay_targets(live_settings, limit=5)]
        second = [item.run_id for item in load_replay_targets(live_settings, limit=5)]
        assert first == second == sorted(first)


class TestProvenanceChain:
    """M8 criterion 2: a complete chain to a root cause."""

    def test_an_outcome_traces_back_to_an_exogenous_shock(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        explanation = explain(live_settings, run_id=stored_runs[0].run_id)
        assert explanation.links, "no decision moved the traced factor"
        assert explanation.root is not None
        assert explanation.complete

    def test_the_chain_walks_strictly_backwards(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        """An antecedent is always earlier. If it were not the walk could loop,
        and a recursive CTE that loops is an outage."""
        explanation = explain(live_settings, run_id=stored_runs[0].run_id)
        steps = [(link.step, link.seq) for link in explanation.links]
        assert steps == sorted(steps, reverse=True)

    def test_the_root_shock_precedes_the_earliest_decision(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        explanation = explain(live_settings, run_id=stored_runs[0].run_id)
        assert explanation.root is not None
        assert explanation.root.step <= explanation.links[-1].step

    def test_it_returns_quickly(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        """§11.2 asks for this to be demonstrable in thirty seconds.

        The bound also catches the shape defect it was written against: the
        spec's CTE recurses over every element of `caused_by`, which expands
        6^depth and does not return at all on a 24-step run.
        """
        import time

        started = time.perf_counter()
        explain(live_settings, run_id=stored_runs[0].run_id)
        assert time.perf_counter() - started < 30.0

    def test_an_unknown_run_raises_rather_than_returning_an_empty_chain(
        self, live_settings: Settings
    ) -> None:
        with pytest.raises(LookupError, match="no run"):
            explain(live_settings, run_id="00000000-0000-0000-0000-000000000000")

    def test_every_stored_run_traces_to_a_root(
        self, live_settings: Settings, stored_runs: tuple[Any, ...]
    ) -> None:
        """The claim is "every outcome", not "an outcome"."""
        incomplete = [
            target.run_id
            for target in stored_runs[:10]
            if not explain(live_settings, run_id=target.run_id).complete
        ]
        assert incomplete == []


class TestCostLedger:
    """M8 criterion 3: the ledger reconciles with Langfuse within 2%."""

    def test_the_local_ledger_sums_the_runs_table(self, live_settings: Settings) -> None:
        spend = local_spend(live_settings)
        assert spend.source == "runs ledger"
        assert spend.total_usd >= 0

    def test_a_heuristic_run_books_no_model_calls(self, live_settings: Settings) -> None:
        """Migration 016's correction, asserted rather than assumed.

        Counting one call per decision made a stand-in run report hundreds of
        thousands of model calls costing nothing, which let §12.4's gate
        compare zero with zero and pass.
        """
        with _connect(live_settings) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(sum(llm_calls), 0), COALESCE(sum(cost_usd), 0) "
                "FROM runs WHERE policy = 'heuristic'"
            )
            calls, cost = cur.fetchone()
        assert int(calls) == 0
        assert float(cost) == 0.0

    def test_the_verdict_distinguishes_unreachable_from_disagreeing(
        self, live_settings: Settings
    ) -> None:
        result = reconcile(live_settings)
        if result.remote is None:
            assert result.relative_difference is None
            assert not result.reconciled
        else:
            assert result.relative_difference is not None

    def test_a_ledger_with_no_spend_does_not_reconcile(self, live_settings: Settings) -> None:
        """Zero agrees with zero; the gate must still refuse."""
        result = reconcile(live_settings)
        if result.vacuous:
            assert not result.reconciled
