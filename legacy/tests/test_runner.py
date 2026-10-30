r"""
Tests for graph_bench.runner module.
"""

import pytest

from graph_bench.runner import BenchmarkOrchestrator, OrchestratorConfig, Timer, measure_time
from graph_bench.runner.timing import TimerResult


class TestTimer:
    def test_timer_context_manager(self):
        with Timer() as t:
            total = sum(range(1000))

        assert t.elapsed_ns > 0
        assert t.elapsed_ms > 0
        assert t.elapsed_seconds > 0

    def test_timer_elapsed_values(self):
        with Timer() as t:
            pass

        assert t.elapsed_ms == t.elapsed_ns / 1_000_000
        assert t.elapsed_seconds == t.elapsed_ns / 1_000_000_000


class TestMeasureTime:
    def test_measure_time_returns_result(self):
        def add(a, b):
            return a + b

        result = measure_time(add, 1, 2)

        assert isinstance(result, TimerResult)
        assert result.result == 3
        assert result.elapsed_ns > 0

    def test_measure_time_with_kwargs(self):
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = measure_time(greet, "World", greeting="Hi")

        assert result.result == "Hi, World!"

    def test_timer_result_conversions(self):
        result = TimerResult(elapsed_ns=1_000_000_000)
        assert result.elapsed_ms == 1000.0
        assert result.elapsed_seconds == 1.0


class TestOrchestratorConfig:
    def test_default_config(self):
        config = OrchestratorConfig()
        assert config.scale == "medium"
        assert config.benchmarks is None
        assert config.categories is None
        assert config.continue_on_error is True
        assert config.verbose is False

    def test_custom_config(self):
        config = OrchestratorConfig(
            scale="small",
            benchmarks=["node_insertion"],
            categories=["storage"],
            verbose=True,
        )
        assert config.scale == "small"
        assert config.benchmarks == ["node_insertion"]
        assert config.categories == ["storage"]
        assert config.verbose is True


class TestBenchmarkOrchestrator:
    def test_create_orchestrator(self):
        orchestrator = BenchmarkOrchestrator()
        assert orchestrator is not None

    def test_create_with_config(self):
        config = OrchestratorConfig(scale="small", verbose=True)
        orchestrator = BenchmarkOrchestrator(config=config)
        assert orchestrator._config.scale == "small"
        assert orchestrator._config.verbose is True

    def test_set_progress_callback(self):
        calls = []

        def callback(db, bench, status):
            calls.append((db, bench, status))

        orchestrator = BenchmarkOrchestrator()
        orchestrator.set_progress_callback(callback)
        assert orchestrator._progress_callback is not None
