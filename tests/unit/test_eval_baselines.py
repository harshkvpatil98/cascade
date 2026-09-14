"""The three baselines that are not ablation cells (spec §10.2).

The one that matters most is self-consistency: §10.2 calls it critical because
it matches Cascade's sample budget, so a gain over it cannot be dismissed as
"you just sampled more". It is also the one the content-addressed cache would
silently break, which is what `sample_index` exists for.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cascade.config import load_settings
from cascade.eval.baselines import (
    BASELINES,
    baseline_prompt,
    climatology_forecasts,
    collapse_samples,
    evidence_for,
    sample_requests,
    system_prompt,
)
from cascade.ledger.schema import Scenario
from cascade.llm.cache import cache_key
from cascade.llm.types import LLMRequest


def _scenario(scenario_id: str = "s1") -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        question="Will the merger be approved?",
        resolution_criterion="Resolves YES if the regulator clears it before the deadline.",
        cutoff_ts=datetime(2024, 3, 1, tzinfo=UTC),
        resolve_ts=datetime(2024, 9, 1, tzinfo=UTC),
        domain="corporate",
        source="polymarket",
        source_ref="ref",
        party_rule="named_parties",
        party_names=("Acme", "Regulator"),
        event_group=None,
    )


class TestTheFiveAreAllPresent:
    def test_there_are_exactly_five(self) -> None:
        assert len(BASELINES) == 5

    def test_two_of_them_are_grid_cells_and_are_not_re_run(self) -> None:
        """Running them twice would put two differently-seeded copies of one
        experiment in the report."""
        from_grid = [spec.config_id for spec in BASELINES if spec.from_grid]
        assert sorted(from_grid) == ["C01", "C09"]

    def test_the_fair_compute_baseline_is_present(self) -> None:
        """Omitting it is the most common way a multi-agent result gets
        dismissed (§10.2)."""
        assert any("self-consistency" in spec.name for spec in BASELINES)

    def test_config_ids_are_unique(self) -> None:
        ids = [spec.config_id for spec in BASELINES]
        assert len(set(ids)) == len(ids)


class TestClimatology:
    def test_it_predicts_the_base_rate_for_every_scenario(self) -> None:
        rows = climatology_forecasts(["b", "a"], base_rate=0.5)
        assert rows == (("a", 0.5), ("b", 0.5))

    def test_a_base_rate_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            climatology_forecasts(["a"], base_rate=1.4)


class TestPrompt:
    def test_the_cutoff_is_stated_explicitly(self) -> None:
        """Without it a model reasons from its own sense of "now", which
        imports §4.4's parametric leakage into the baseline."""
        text = baseline_prompt(_scenario(), [], evidence_chars=900)
        assert "2024-03-01" in text
        assert "knowable at the cutoff" in text

    def test_absent_evidence_is_stated_rather_than_left_blank(self) -> None:
        text = baseline_prompt(_scenario(), [], evidence_chars=900)
        assert "no admissible evidence" in text

    def test_evidence_is_truncated_to_the_agents_own_width(self) -> None:
        long_excerpt = "x" * 5000
        text = baseline_prompt(
            _scenario(), [("2024-01-01", "ccnews", long_excerpt)], evidence_chars=100
        )
        assert "x" * 100 in text
        assert "x" * 101 not in text

    def test_the_system_prompt_is_not_the_agent_prompt(self) -> None:
        """Handing the agent's world model to a single model asked for one
        number would measure a differently-prompted single model."""
        from cascade.sim.prompts import RULES

        assert system_prompt() != RULES
        assert "arbiter" not in system_prompt().lower()

    def test_missing_evidence_for_a_scenario_is_an_empty_tuple(self) -> None:
        assert evidence_for(_scenario(), {}) == ()


class TestSampleRequests:
    def test_each_draw_gets_a_distinct_cache_key(self) -> None:
        """The defect this prevents: 200 identical requests collapse to one
        recording replayed 200 times, and the fair-compute baseline reports a
        dispersion of exactly zero for a reason that is an artefact of the
        cache."""
        settings = load_settings(None)
        items = sample_requests(
            settings, _scenario(), [], samples=5, temperature=0.7, config_id="B3"
        )
        keys = {cache_key(item.request) for item in items}
        assert len(keys) == 5

    def test_a_single_draw_keeps_the_key_it_always_had(self) -> None:
        """`sample_index` enters the cache domain only when non-zero, so every
        recording made before the field existed still resolves."""
        base = LLMRequest(
            model="m",
            system="s",
            messages=[{"role": "user", "content": "q"}],
            temperature=0.0,
            max_tokens=64,
            prompt_rev="r2",
        )
        with_zero = base.model_copy(update={"sample_index": 0})
        assert cache_key(base) == cache_key(with_zero)
        assert "sample_index" not in base.cache_domain()

    def test_custom_ids_identify_config_scenario_and_draw(self) -> None:
        settings = load_settings(None)
        items = sample_requests(
            settings, _scenario("abc"), [], samples=3, temperature=0.0, config_id="B3"
        )
        assert [item.custom_id for item in items] == ["B3|abc|0", "B3|abc|1", "B3|abc|2"]

    def test_zero_samples_raises(self) -> None:
        settings = load_settings(None)
        with pytest.raises(ValueError, match="samples must be positive"):
            sample_requests(settings, _scenario(), [], samples=0, temperature=0.0, config_id="B")

    def test_the_sample_index_never_reaches_the_wire(self) -> None:
        """It is a cache-domain field, not an API parameter."""
        from cascade.llm.client import _batch_params

        settings = load_settings(None)
        item = sample_requests(
            settings, _scenario(), [], samples=2, temperature=0.5, config_id="B3"
        )[1]
        assert item.request.sample_index == 1
        assert "sample_index" not in _batch_params(item.request)


class TestCollapse:
    def test_it_averages_the_parseable_draws(self) -> None:
        result = collapse_samples("s", [0.2, 0.4, 0.6], requested=3)
        assert result is not None
        assert result.p_hat == pytest.approx(0.4)
        assert result.n_parsed == 3

    def test_unparseable_answers_are_dropped_and_counted_never_scored_as_half(
        self,
    ) -> None:
        """Substituting 0.5 would make the baseline look well calibrated, which
        is the direction that flatters the system under test."""
        result = collapse_samples("s", [0.9, None, 0.9], requested=3)
        assert result is not None
        assert result.p_hat == pytest.approx(0.9)
        assert result.n_parsed == 2
        assert result.unparseable == 1

    def test_all_unparseable_yields_no_forecast_at_all(self) -> None:
        assert collapse_samples("s", [None, None], requested=2) is None

    def test_sigma_is_the_sample_standard_deviation(self) -> None:
        result = collapse_samples("s", [0.2, 0.8], requested=2)
        assert result is not None
        assert result.sigma == pytest.approx(0.4242640687, abs=1e-9)

    def test_a_single_draw_has_no_dispersion(self) -> None:
        result = collapse_samples("s", [0.42], requested=1)
        assert result is not None
        assert result.sigma == 0.0
