r"""
Tests for graph_bench.types module.
"""

import pytest

from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig, Status, TimingStats


class TestStatus:
    def test_status_values(self):
        assert Status.SUCCESS.value == 1
        assert Status.FAILED.value == 2
        assert Status.TIMEOUT.value == 3
        assert Status.SKIPPED.value == 4

    def test_status_comparison(self):
        assert Status.SUCCESS < Status.FAILED
        assert Status.TIMEOUT > Status.FAILED


class TestTimingStats:
    def test_create_timing_stats(self):
        stats = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_500_000.0,
            median_ns=2_400_000.0,
            std_ns=500_000.0,
            p99_ns=4_800_000.0,
            iterations=10,
        )
        assert stats.min_ns == 1_000_000
        assert stats.iterations == 10

    def test_timing_stats_conversions(self):
        stats = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_000_000.0,
            median_ns=2_000_000.0,
            std_ns=500_000.0,
            p99_ns=4_000_000.0,
            iterations=10,
        )
        assert stats.mean_ms == 2.0
        assert stats.median_ms == 2.0
        assert stats.p99_ms == 4.0

    def test_ops_per_second(self):
        stats = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=1_000_000_000.0,
            median_ns=1_000_000_000.0,
            std_ns=0.0,
            p99_ns=1_000_000_000.0,
            iterations=1,
        )
        assert stats.ops_per_second == 1.0

    def test_ops_per_second_zero(self):
        stats = TimingStats(
            min_ns=0,
            max_ns=0,
            mean_ns=0.0,
            median_ns=0.0,
            std_ns=0.0,
            p99_ns=0.0,
            iterations=0,
        )
        assert stats.ops_per_second == float("inf")

    def test_timing_stats_immutable(self):
        stats = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_000_000.0,
            median_ns=2_000_000.0,
            std_ns=500_000.0,
            p99_ns=4_000_000.0,
            iterations=10,
        )
        with pytest.raises(AttributeError):
            stats.min_ns = 0  # type: ignore


class TestMetrics:
    def test_create_metrics(self):
        timing = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_000_000.0,
            median_ns=2_000_000.0,
            std_ns=500_000.0,
            p99_ns=4_000_000.0,
            iterations=10,
        )
        metrics = Metrics(
            timing=timing,
            throughput=1000.0,
            items_processed=10000,
        )
        assert metrics.throughput == 1000.0
        assert metrics.items_processed == 10000
        assert metrics.memory_bytes is None

    def test_metrics_with_custom(self):
        timing = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_000_000.0,
            median_ns=2_000_000.0,
            std_ns=500_000.0,
            p99_ns=4_000_000.0,
            iterations=10,
        )
        metrics = Metrics(
            timing=timing,
            throughput=1000.0,
            items_processed=10000,
            custom={"extra": "value"},
        )
        assert metrics.custom["extra"] == "value"


class TestScaleConfig:
    def test_create_scale_config(self):
        scale = ScaleConfig(
            name="test",
            nodes=1000,
            edges=5000,
        )
        assert scale.name == "test"
        assert scale.nodes == 1000
        assert scale.edges == 5000
        assert scale.warmup_iterations == 3
        assert scale.measurement_iterations == 10
        assert scale.timeout_seconds == 300

    def test_scale_config_custom(self):
        scale = ScaleConfig(
            name="custom",
            nodes=100,
            edges=200,
            warmup_iterations=1,
            measurement_iterations=5,
            timeout_seconds=60,
        )
        assert scale.warmup_iterations == 1
        assert scale.measurement_iterations == 5
        assert scale.timeout_seconds == 60


class TestBenchmarkResult:
    def test_create_result_success(self):
        scale = ScaleConfig(name="test", nodes=100, edges=200)
        timing = TimingStats(
            min_ns=1_000_000,
            max_ns=5_000_000,
            mean_ns=2_000_000.0,
            median_ns=2_000_000.0,
            std_ns=500_000.0,
            p99_ns=4_000_000.0,
            iterations=10,
        )
        metrics = Metrics(timing=timing, throughput=1000.0, items_processed=100)

        result = BenchmarkResult(
            benchmark_name="test_bench",
            database="test_db",
            scale=scale,
            metrics=metrics,
        )
        assert result.ok is True
        assert result.status == Status.SUCCESS
        assert result.error is None

    def test_create_result_failed(self):
        scale = ScaleConfig(name="test", nodes=100, edges=200)

        result = BenchmarkResult(
            benchmark_name="test_bench",
            database="test_db",
            scale=scale,
            metrics=None,
            status=Status.FAILED,
            error="Something went wrong",
        )
        assert result.ok is False
        assert result.status == Status.FAILED
        assert result.error == "Something went wrong"

    def test_result_generic_type(self):
        scale = ScaleConfig(name="test", nodes=100, edges=200)

        result: BenchmarkResult[str] = BenchmarkResult(
            benchmark_name="test",
            database="db",
            scale=scale,
            metrics=None,
        )
        assert result.benchmark_name == "test"
