"""Assay against the live database (migrations 011/012, spec §10, §1.3).

The load-bearing checks here are the two that cannot be faked in a unit test:
the frozen-split assertion actually re-hashes the sealed registry, and the join
that puts a forecast next to an outcome runs under the one role permitted to
see a label. Everything else in `cascade/eval/` is pure and is tested without a
database, which is the point of the split.
"""

from __future__ import annotations

from typing import Any

import pytest

from cascade.config import Settings
from cascade.ensemble.schema import Forecast
from cascade.ensemble.store import write_forecast
from cascade.eval.store import (
    assert_frozen_split,
    available_configs,
    record_prompt_revision_brier,
    scored_forecasts,
    write_study_report,
)

pytestmark = pytest.mark.integration

CONFIG = "test-m7"


def _connect(settings: Settings, role: str) -> Any:
    import psycopg

    return psycopg.connect(settings.database_url(role), connect_timeout=10)


@pytest.fixture(scope="module")
def scenario_ids(live_settings: Settings) -> list[str]:
    with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
        cur.execute("SELECT scenario_id FROM scenarios ORDER BY scenario_id LIMIT 4")
        rows = cur.fetchall()
    if len(rows) < 4:
        pytest.skip("scenario registry is empty; run `cascade ledger build`")
    return [str(row[0]) for row in rows]


@pytest.fixture
def stored(live_settings: Settings, scenario_ids: list[str]) -> Any:
    written = []
    for index, scenario_id in enumerate(scenario_ids):
        forecast = Forecast(
            scenario_id=scenario_id,
            config_id=CONFIG,
            p_hat=0.2 + 0.2 * index,
            sigma=0.05 * index,
            ci_lo=0.1,
            ci_hi=0.9,
            modality="multi" if index else "single",
            n_replicates=30,
            policy="heuristic",
        )
        write_forecast(live_settings, forecast, role="admin")
        written.append(forecast)
    yield written
    with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM forecasts WHERE config_id = %s", (CONFIG,))
        cur.execute("DELETE FROM study_reports WHERE report_id LIKE 'test-m7-%'")
        conn.commit()


class TestFrozenSplit:
    def test_it_re_hashes_the_registry_and_matches_the_seal(self, live_settings: Settings) -> None:
        """§1.3's first integrity mechanism, asserted against the real rows.

        Not a stored-hash comparison: the registry is loaded and re-hashed, so
        a label edited in place fails here rather than at publication.
        """
        split = assert_frozen_split(live_settings)
        assert len(split.manifest_sha256) == 64
        assert split.n_scenarios > 0
        assert 0.0 <= split.base_rate <= 1.0

    def test_the_sealed_climatology_travels_with_the_split(self, live_settings: Settings) -> None:
        """§10.2's floor comes from the split the report describes, never
        recomputed from whatever happens to be loaded at report time."""
        split = assert_frozen_split(live_settings)
        assert split.climatology_brier == pytest.approx(
            split.base_rate * (1 - split.base_rate), abs=1e-9
        )

    def test_a_split_that_does_not_match_its_seal_is_caught(
        self, live_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assertion is wired into the eval path, not merely available.

        The seal is replaced rather than the labels: mutating a real label
        would need a committed write to the study registry, and a test that
        can corrupt the sealed set if it crashes halfway is a worse test than
        the one it replaces. That the *hash* detects a flipped label is M1's
        criterion and is asserted there against real stored rows; what M7 adds
        is that nothing under `cascade eval` reads a label before this runs.
        """
        from cascade.ledger.manifest import ManifestMismatch
        from cascade.ledger.store import SealedManifest, read_manifest

        real = read_manifest(live_settings, role="eval")
        assert real is not None

        def wrong_seal(*args: object, **kwargs: object) -> SealedManifest:
            return SealedManifest(
                manifest_sha256="f" * 64,
                sealed_at=real.sealed_at,
                n_scenarios=real.n_scenarios,
                n_yes=real.n_yes,
                yes_rate=real.yes_rate,
                climatology_brier=real.climatology_brier,
                study_salt=real.study_salt,
                notes=real.notes,
            )

        monkeypatch.setattr("cascade.ledger.store.read_manifest", wrong_seal)
        with pytest.raises(ManifestMismatch) as caught:
            assert_frozen_split(live_settings)
        assert "Do not reseal" in str(caught.value)

    def test_an_unsealed_registry_is_refused_rather_than_scored(
        self, live_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cascade.eval.store import SplitNotSealed

        monkeypatch.setattr("cascade.ledger.store.read_manifest", lambda *args, **kwargs: None)
        with pytest.raises(SplitNotSealed, match="never been sealed"):
            assert_frozen_split(live_settings)


class TestScoring:
    def test_a_forecast_meets_its_outcome_exactly_once(
        self, live_settings: Settings, stored: list[Forecast]
    ) -> None:
        scored = scored_forecasts(live_settings, config_id=CONFIG)
        assert len(scored) == len(stored)
        assert {item.outcome for item in scored} <= {0, 1}
        assert all(item.domain for item in scored)

    def test_the_decider_policy_survives_the_round_trip(
        self, live_settings: Settings, stored: list[Forecast]
    ) -> None:
        """M5 stamps the policy on a run because a footnote is not a mechanism.
        Migration 012 carries it onto the forecast for the same reason: a Brier
        over stand-in runs must not be indistinguishable from a study result."""
        scored = scored_forecasts(live_settings, config_id=CONFIG)
        assert {item.policy for item in scored} == {"heuristic"}

    def test_an_unknown_config_scores_nothing_rather_than_something(
        self, live_settings: Settings
    ) -> None:
        assert scored_forecasts(live_settings, config_id="no-such-config") == ()

    def test_available_configs_is_sorted(self, live_settings: Settings, stored: Any) -> None:
        names = [name for name, _ in available_configs(live_settings)]
        assert names == sorted(names)
        assert CONFIG in names

    def test_a_config_that_never_ran_is_absent_rather_than_zero(
        self, live_settings: Settings, stored: Any
    ) -> None:
        counts = dict(available_configs(live_settings))
        assert "never-run" not in counts
        assert counts[CONFIG] == len(stored)


class TestGrants:
    def test_the_simulation_role_still_cannot_read_a_label(self, live_settings: Settings) -> None:
        """Invariant 2. M7 is the milestone that finally reads labels, which is
        exactly when it is worth re-asserting that the simulation cannot."""
        import psycopg.errors

        with _connect(live_settings, "sim") as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("SELECT outcome FROM scenario_labels LIMIT 1")
            conn.rollback()

    def test_the_eval_role_can_write_a_study_report(self, live_settings: Settings) -> None:
        split = assert_frozen_split(live_settings)
        write_study_report(
            live_settings,
            report_id="test-m7-report",
            manifest_sha256=split.manifest_sha256,
            git_sha="0" * 40,
            headline_config=CONFIG,
            n_scenarios=split.n_scenarios,
            headline_brier=0.2201,
            notes="integration test",
        )
        with _connect(live_settings, "eval") as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT manifest_sha256, headline_brier FROM study_reports WHERE report_id = %s",
                ("test-m7-report",),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == split.manifest_sha256
        assert float(row[1]) == pytest.approx(0.2201)
        with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM study_reports WHERE report_id = 'test-m7-report'")
            conn.commit()

    def test_writing_the_same_report_id_twice_updates_rather_than_duplicates(
        self, live_settings: Settings
    ) -> None:
        split = assert_frozen_split(live_settings)
        for brier in (0.3, 0.25):
            write_study_report(
                live_settings,
                report_id="test-m7-idempotent",
                manifest_sha256=split.manifest_sha256,
                git_sha="0" * 40,
                headline_config=CONFIG,
                n_scenarios=split.n_scenarios,
                headline_brier=brier,
            )
        with _connect(live_settings, "eval") as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), max(headline_brier) FROM study_reports WHERE report_id = %s",
                ("test-m7-idempotent",),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == 1
        assert float(row[1]) == pytest.approx(0.25)
        with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM study_reports WHERE report_id = 'test-m7-idempotent'")
            conn.commit()

    def test_a_report_with_no_headline_stores_null_not_zero(self, live_settings: Settings) -> None:
        """A blocked study is a real artifact; 0.0 would be a measurement
        nobody made."""
        split = assert_frozen_split(live_settings)
        write_study_report(
            live_settings,
            report_id="test-m7-blocked",
            manifest_sha256=split.manifest_sha256,
            git_sha="0" * 40,
            headline_config="C01",
            n_scenarios=split.n_scenarios,
            headline_brier=None,
        )
        with _connect(live_settings, "eval") as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT headline_brier FROM study_reports WHERE report_id = %s",
                ("test-m7-blocked",),
            )
            row = cur.fetchone()
        assert row is not None and row[0] is None
        with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM study_reports WHERE report_id = 'test-m7-blocked'")
            conn.commit()


class TestPromptAudit:
    def test_the_before_after_brier_can_be_recorded(self, live_settings: Settings) -> None:
        """§1.3's third integrity mechanism. It landed at M6 with both columns
        NULL because the revision predates the first backtest; filling them in
        is an evaluation output, so PHASE 4 is what writes it."""
        updated = record_prompt_revision_brier(
            live_settings,
            prompt_rev="r2",
            subsystem="compiler",
            brier_before=None,
            brier_after=0.2201,
        )
        assert updated == 1
        with _connect(live_settings, "eval") as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT brier_before, brier_after FROM prompt_revisions "
                "WHERE prompt_rev = 'r2' AND subsystem = 'compiler'"
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] is None
        assert float(row[1]) == pytest.approx(0.2201)
        # Restore: this predates the first backtest and must read as NULL.
        with _connect(live_settings, "admin") as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE prompt_revisions SET brier_after = NULL "
                "WHERE prompt_rev = 'r2' AND subsystem = 'compiler'"
            )
            conn.commit()

    def test_an_unknown_revision_updates_nothing(self, live_settings: Settings) -> None:
        """A revision is recorded by the migration that makes it, not by a
        command that could invent one."""
        assert (
            record_prompt_revision_brier(
                live_settings,
                prompt_rev="r-does-not-exist",
                subsystem="agent",
                brier_before=0.1,
                brier_after=0.2,
            )
            == 0
        )
