"""The replay verifier's pure half (spec §8.4; M8).

The verdict logic is what decides whether the study's determinism claim passes,
so it is tested without a database or a subprocess. The parts that need both
are in the integration suite.
"""

from __future__ import annotations

from cascade.trace.replay import CHILD_HASH_SEED, ReplayOutcome, ReplayReport, load_settings_for


def outcome(
    run: str, *, actual: str | None, expected: str = "abc", **extra: object
) -> ReplayOutcome:
    return ReplayOutcome(run_id=run, expected_hash=expected, actual_hash=actual, **extra)  # type: ignore[arg-type]


class TestVerdict:
    def test_a_matching_hash_matches(self) -> None:
        assert outcome("r1", actual="abc").matched

    def test_a_differing_hash_does_not(self) -> None:
        assert not outcome("r1", actual="def").matched

    def test_a_child_that_failed_is_not_a_match(self) -> None:
        """An error is not a pass. A replay that could not run has not shown
        that the run is deterministic -- it has shown nothing."""
        assert not outcome("r1", actual=None, error="connection refused").matched

    def test_a_matching_hash_with_an_error_is_still_a_failure(self) -> None:
        """Defence against a child that prints a plausible hash and then dies."""
        assert not outcome("r1", actual="abc", error="died after printing").matched


class TestReport:
    def test_an_all_matching_report_is_ok(self) -> None:
        report = ReplayReport(outcomes=[outcome("a", actual="abc"), outcome("b", actual="abc")])
        assert report.ok
        assert report.matched == 2
        assert report.diverged == ()

    def test_one_divergence_fails_the_whole_report(self) -> None:
        """25 of 25, not 24 of 25: §8.4's rule is that the fix is the source,
        never a tolerance."""
        report = ReplayReport(outcomes=[outcome("a", actual="abc"), outcome("b", actual="zzz")])
        assert not report.ok
        assert report.matched == 1
        assert [item.run_id for item in report.diverged] == ["b"]

    def test_an_empty_report_is_not_ok(self) -> None:
        """Replaying nothing proves nothing, and must not read as a pass."""
        assert not ReplayReport(outcomes=[]).ok

    def test_divergences_preserve_target_order(self) -> None:
        report = ReplayReport(outcomes=[outcome(name, actual="zzz") for name in ("c", "a", "b")])
        assert [item.run_id for item in report.diverged] == ["c", "a", "b"]


class TestBisect:
    def test_a_divergence_carries_the_first_differing_step(self) -> None:
        """§8.4: the bisect names the first divergent field. "The hashes
        differ" is a symptom; "they differ from step 11" is a diagnosis."""
        item = outcome(
            "r1",
            actual="zzz",
            first_divergent_step=11,
            expected_state_hash="aaa",
            actual_state_hash="bbb",
        )
        assert not item.matched
        assert item.first_divergent_step == 11
        assert item.expected_state_hash != item.actual_state_hash


class TestChildIsolation:
    def test_the_child_hash_seed_is_fixed(self) -> None:
        """Fixed so the *child* is reproducible -- a divergence is then a
        property of the code rather than of the run that happened to find it.
        Any fixed value differs from the parent's random one, which is the
        point."""
        assert CHILD_HASH_SEED.isdigit()

    def test_a_run_is_replayed_under_the_configuration_it_was_made_with(self) -> None:
        """Replaying an ablation cell under the base configuration would re-run
        a different experiment -- different visibility policy, different graph
        arm, different grounding -- and report the mismatch as
        non-determinism."""
        assert load_settings_for("C09").flags.causal_decomposition is False
        assert load_settings_for("C01").flags.causal_decomposition is True

    def test_an_unknown_config_falls_back_rather_than_crashing(self) -> None:
        """A run may carry a config id whose overlay has since been renamed.
        Falling back is wrong in a way the hash comparison will catch loudly;
        crashing hides every other run in the batch."""
        assert load_settings_for("no-such-overlay") is not None

    def test_the_base_configuration_needs_no_overlay(self) -> None:
        assert load_settings_for("base").flags.causal_decomposition is True
