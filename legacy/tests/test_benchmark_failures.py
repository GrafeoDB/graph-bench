r"""
A failing query fails its benchmark; it is never timed.

For every registered benchmark, the embedded Grafeo database starts failing
every call once the benchmark's setup is done, as an engine upgrade that rejects
a query would. The orchestrator must then record the benchmark as FAILED with
the engine's error, not as a SUCCESS with a timing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from grafeo import GrafeoError

from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.benchmarks.base import BaseBenchmark
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.runner import BenchmarkOrchestrator, OrchestratorConfig
from graph_bench.types import ScaleConfig, Status

from .test_grafeo_benchmarks import TINY

FAILURE = "query rejected by the engine (simulated)"


class FailingDatabase:
    """Stands in for a GrafeoDB whose every call fails."""

    def __getattr__(self, name: str) -> Any:
        raise GrafeoError(FAILURE)


@pytest.fixture
def adapter() -> Iterator[GrafeoAdapter]:
    adapter = GrafeoAdapter()
    adapter.connect(path=":memory:")
    yield adapter
    adapter.disconnect()


class FailingAfterSetup(BaseBenchmark):
    """Runs a benchmark with the database failing from the first measured call on."""

    def __init__(self, inner: BaseBenchmark, adapter: GrafeoAdapter) -> None:
        self._inner = inner
        self._adapter = adapter
        self._database = adapter._db
        self.category = inner.category

    @property
    def name(self) -> str:
        return self._inner.name

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        self._inner.setup(adapter, scale)
        self._adapter._db = FailingDatabase()

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        return self._inner.run_iteration(adapter, scale)

    def teardown(self, adapter: GraphDatabaseAdapter) -> None:
        self._adapter._db = self._database
        self._inner.teardown(adapter)


def run_one(benchmark: BaseBenchmark, adapter: GrafeoAdapter) -> Any:
    orchestrator = BenchmarkOrchestrator(config=OrchestratorConfig(scale=TINY))
    (result,) = orchestrator.run([adapter], benchmarks=[benchmark], scale=TINY).results
    return result


@pytest.mark.parametrize("name", sorted(BenchmarkRegistry.list()))
def test_failing_query_fails_the_benchmark(adapter: GrafeoAdapter, name: str) -> None:
    bench_cls = BenchmarkRegistry.get(name)
    assert bench_cls is not None
    result = run_one(FailingAfterSetup(bench_cls(), adapter), adapter)
    assert result.status == Status.FAILED, f"{name} timed a failing query"
    assert result.metrics is None
    assert FAILURE in (result.error or "")


def test_working_benchmark_succeeds(adapter: GrafeoAdapter) -> None:
    # The same path without the failure, so the test above fails for the right reason.
    bench_cls = BenchmarkRegistry.get("filter_equality")
    assert bench_cls is not None
    result = run_one(bench_cls(), adapter)
    assert result.status == Status.SUCCESS
    assert result.metrics is not None
    assert result.metrics.items_processed == 40


def test_property_update_reports_every_dialect_error(adapter: GrafeoAdapter) -> None:
    bench_cls = BenchmarkRegistry.get("property_update")
    assert bench_cls is not None
    result = run_one(FailingAfterSetup(bench_cls(), adapter), adapter)
    assert result.status == Status.FAILED
    assert result.error is not None
    assert result.error.startswith("no update query ran on Grafeo")
    assert result.error.count(FAILURE) == 3
