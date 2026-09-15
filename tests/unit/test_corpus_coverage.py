"""Ingest ordering and the coverage measure (ADR-0023).

The bug these guard against is not a crash. It is a corpus that passes every
M2 criterion -- 1.76M chunks, zero bad dates, full embedding coverage -- while
holding nothing a 2026 scenario could have read. Ordering is what prevents it
and the coverage report is what detects it, so both are tested as behaviour
rather than as plumbing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cascade.corpus.coverage import (
    demand_profile,
    month_index,
    order_units,
    unit_depth,
    unit_month,
)
from cascade.corpus.sources import ccnews


def _at(year: int, month: int, day: int = 15) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class TestUnitKeyParsing:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("2026/07#3", (2026, 7)),
            ("2016/08#0", (2016, 8)),
            ("2019/03", (2019, 3)),  # legacy month-level CC-NEWS key
            ("2021-11", (2021, 11)),  # govpr
            ("2023-04:2", (2023, 4)),  # gdelt, query index 2
            ("2024Q3", (2024, 7)),  # edgar quarter -> its first month
        ],
    )
    def test_every_source_key_shape_yields_its_month(
        self, key: str, expected: tuple[int, int]
    ) -> None:
        assert unit_month(key) == month_index(_at(*expected, day=1))

    def test_a_scenario_keyed_unit_has_no_month(self) -> None:
        """Wikipedia units are already cutoff-anchored and must not be reordered."""
        assert unit_month("polymarket:some-question:12345") is None
        assert unit_month("curated:may-withdrawal-agreement-2019") is None

    @pytest.mark.parametrize(
        ("key", "depth"),
        [("2026/07#3", 3), ("2026/07#0", 0), ("2019/03", 0), ("2023-04:2", 2), ("2024Q3", 0)],
    )
    def test_depth_is_the_index_within_the_month(self, key: str, depth: int) -> None:
        assert unit_depth(key) == depth


class TestDemandProfile:
    def test_a_month_before_a_cutoff_earns_more_than_an_older_one(self) -> None:
        profile = demand_profile([_at(2026, 6)], lookback_months=12)
        assert profile[month_index(_at(2026, 6))] > profile[month_index(_at(2026, 1))]

    def test_months_at_or_after_a_cutoff_earn_nothing(self) -> None:
        """A post-cutoff document is inadmissible; fetching it is pure waste."""
        profile = demand_profile([_at(2026, 6)], lookback_months=12)
        assert profile.get(month_index(_at(2026, 7)), 0.0) == 0.0
        assert profile.get(month_index(_at(2027, 1)), 0.0) == 0.0

    def test_demand_accumulates_across_scenarios(self) -> None:
        one = demand_profile([_at(2025, 3)], lookback_months=6)
        two = demand_profile([_at(2025, 3), _at(2025, 3)], lookback_months=6)
        key = month_index(_at(2025, 3))
        assert two[key] == pytest.approx(2 * one[key])

    def test_a_month_outside_every_window_is_absent(self) -> None:
        profile = demand_profile([_at(2026, 6)], lookback_months=6)
        assert month_index(_at(2017, 4)) not in profile

    def test_lookback_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="lookback_months"):
            demand_profile([_at(2026, 6)], lookback_months=0)


class TestOrdering:
    def test_the_span_is_swept_before_any_month_is_deepened(self) -> None:
        """The property the whole change exists for.

        An ingest is always interrupted. Depth-major order is what makes the
        prefix of the queue a thin layer over the whole study window instead
        of a complete copy of its first quarter.
        """
        keys = [f"2026/{month:02d}#{index}" for month in (1, 2, 3) for index in (0, 1)]
        demand = demand_profile([_at(2026, 4)], lookback_months=12)
        ordered = [plan.unit_key for plan in order_units(keys, demand=demand)]
        assert [key.split("#")[1] for key in ordered] == ["0", "0", "0", "1", "1", "1"]

    def test_within_a_sweep_the_needed_months_come_first(self) -> None:
        keys = ["2017/03#0", "2026/05#0", "2021/09#0"]
        demand = demand_profile([_at(2026, 6)], lookback_months=18)
        ordered = [plan.unit_key for plan in order_units(keys, demand=demand)]
        assert ordered[0] == "2026/05#0"

    def test_ties_break_on_the_key_so_the_order_is_total(self) -> None:
        """Invariant 7: no queue position depends on dict or set order."""
        keys = ["2017/05#0", "2017/03#0", "2017/04#0"]
        first = order_units(keys, demand={})
        second = order_units(list(reversed(keys)), demand={})
        assert [plan.unit_key for plan in first] == [plan.unit_key for plan in second]
        assert [plan.unit_key for plan in first] == ["2017/03#0", "2017/04#0", "2017/05#0"]

    def test_undated_units_sort_after_dated_ones(self) -> None:
        keys = ["polymarket:q:1", "2017/03#0"]
        ordered = [plan.unit_key for plan in order_units(keys, demand={})]
        assert ordered == ["2017/03#0", "polymarket:q:1"]

    def test_ordering_is_a_permutation(self) -> None:
        keys = [f"2026/{month:02d}#{index}" for month in range(1, 13) for index in range(3)]
        demand = demand_profile([_at(2026, 12)], lookback_months=18)
        ordered = [plan.unit_key for plan in order_units(keys, demand=demand)]
        assert sorted(ordered) == sorted(keys)


class TestCCNewsUnits:
    def test_units_are_one_per_file_and_clamped_to_the_collection_start(self) -> None:
        keys = ccnews.unit_keys(start_year=2015, end_year=2016, max_files=3)
        assert keys == [
            f"2016/{month:02d}#{index}" for month in (8, 9, 10, 11, 12) for index in range(3)
        ]

    def test_split_reads_the_file_ordinal(self) -> None:
        assert ccnews.split_unit("2026/07#5") == ("2026/07", 5)

    def test_a_legacy_month_key_reads_as_file_zero(self) -> None:
        assert ccnews.split_unit("2026/07") == ("2026/07", 0)

    def test_a_legacy_done_month_expands_to_the_files_it_covered(self) -> None:
        """Bookkeeping, not a rewrite: the stored row is read as what it meant.

        Without this, switching to file-level units would re-fetch every month
        already ingested -- 18 months of work, silently repeated.
        """
        expanded = ccnews.expand_legacy_unit("2017/06")
        assert expanded == [f"2017/06#{index}" for index in range(ccnews.LEGACY_MONTH_FILES)]

    def test_expanding_a_file_key_is_the_identity(self) -> None:
        assert ccnews.expand_legacy_unit("2017/06#4") == ["2017/06#4"]


class TestPrefetch:
    """The lookahead must be invisible in everything except wall-clock time."""

    def test_results_arrive_in_submission_order(self) -> None:
        import time

        from cascade.corpus.pipeline import _prefetch
        from cascade.corpus.schema import RawDocument

        def load(unit_key: str) -> list[RawDocument]:
            # Later units finish first; order must survive that.
            time.sleep(0.05 / (int(unit_key) + 1))
            return [
                RawDocument(
                    source="ccnews",
                    source_ref=unit_key,
                    url=f"https://example.test/{unit_key}",
                    title="",
                    body="body",
                    published_at=_at(2026, 1),
                )
            ]

        keys = [str(index) for index in range(8)]
        seen = [key for key, _ in _prefetch(keys, workers=4, load=load)]
        assert seen == keys

    def test_one_failing_unit_does_not_stop_the_queue(self) -> None:
        from cascade.corpus.pipeline import _prefetch
        from cascade.corpus.schema import RawDocument

        def load(unit_key: str) -> list[RawDocument]:
            if unit_key == "2":
                raise RuntimeError("unreachable")
            return []

        results = dict(_prefetch([str(index) for index in range(5)], workers=3, load=load))
        assert isinstance(results["2"], RuntimeError)
        assert all(results[key] == [] for key in ("0", "1", "3", "4"))
        assert RawDocument is not None  # import is exercised, not merely resolved

    def test_a_single_worker_takes_the_sequential_path(self) -> None:
        from cascade.corpus.pipeline import _prefetch

        calls: list[str] = []

        def load(unit_key: str) -> list:  # type: ignore[type-arg]
            calls.append(unit_key)
            return []

        list(_prefetch(["a", "b"], workers=1, load=load))
        assert calls == ["a", "b"]

    def test_stopping_early_cancels_the_lookahead(self) -> None:
        """A budget abort or --max-units must not leave downloads running."""
        from cascade.corpus.pipeline import _prefetch

        started: list[str] = []

        def load(unit_key: str) -> list:  # type: ignore[type-arg]
            started.append(unit_key)
            return []

        stream = _prefetch([str(index) for index in range(50)], workers=3, load=load)
        next(stream)
        stream.close()
        assert len(started) < 50


class TestEmbedderOfflineFallback:
    """A cached model must not need the network (M7 finding).

    The weights are cached after the first load, but the loader still calls
    the hub to check for a newer revision. A transient failure there took down
    1 test and errored 12 more -- the whole leakage suite and the
    date-monotonicity properties -- on a machine that already had the model.
    """

    def test_a_network_failure_retries_against_the_local_cache(self) -> None:
        from cascade.corpus.embed import Embedder

        calls: list[bool] = []

        def factory(name: str, *, device: str, local_files_only: bool = False) -> str:
            calls.append(local_files_only)
            if not local_files_only:
                raise ConnectionError("huggingface.co went away mid-stream")
            return "model"

        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        assert embedder._construct(factory, device="cpu") == "model"
        assert calls == [False, True]

    def test_a_genuinely_absent_model_still_fails(self) -> None:
        """The fallback can only succeed from a populated cache. A pinned model
        that is not there has no substitution to fall back to."""
        from cascade.corpus.embed import Embedder, EmbeddingUnavailable

        def factory(name: str, *, device: str, local_files_only: bool = False) -> str:
            raise OSError("no such model anywhere")

        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        with pytest.raises(EmbeddingUnavailable, match="local cache did not satisfy it"):
            embedder._construct(factory, device="cpu")

    def test_a_healthy_load_makes_one_call(self) -> None:
        from cascade.corpus.embed import Embedder

        calls: list[bool] = []

        def factory(name: str, *, device: str, local_files_only: bool = False) -> str:
            calls.append(local_files_only)
            return "model"

        assert Embedder(model_name="m")._construct(factory, device="cpu") == "model"
        assert calls == [False]
