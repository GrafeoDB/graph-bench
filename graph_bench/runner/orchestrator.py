r"""
Benchmark orchestrator for coordinating execution.

    from graph_bench.runner import BenchmarkOrchestrator

    orchestrator = BenchmarkOrchestrator()
    results = orchestrator.run(adapters, benchmarks, scale="medium")
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.config import get_scale
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import BenchmarkResult, ScaleConfig, Status

__all__ = ["BenchmarkOrchestrator", "OrchestratorConfig", "ProgressCallback"]

ProgressCallback = Callable[[str, str, str], None]


@dataclass
class OrchestratorConfig:
    """Configuration for benchmark orchestration.

    Attributes:
        scale: Scale configuration or name.
        benchmarks: List of benchmark names to run (None = all).
        categories: List of categories to run (None = all).
        timeout_override: Override per-benchmark timeout.
        continue_on_error: Continue running after failures.
        verbose: Enable verbose output.
    """

    scale: str | ScaleConfig = "medium"
    benchmarks: list[str] | None = None
    categories: list[str] | None = None
    timeout_override: int | None = None
    continue_on_error: bool = True
    verbose: bool = False


@dataclass
class OrchestratorResult:
    """Results from orchestrator run.

    Attributes:
        results: List of benchmark results.
        started_at: Timestamp when run started.
        completed_at: Timestamp when run completed.
        scale: Scale configuration used.
        databases: List of database names tested.
    """

    results: list[BenchmarkResult[Any]] = field(default_factory=list)
    started_at: float = 0.0
    completed_at: float = 0.0
    scale: ScaleConfig | None = None
    databases: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.completed_at - self.started_at

    @property
    def success_count(self) -> int:
        """Number of successful benchmarks."""
        return sum(1 for r in self.results if r.ok)

    @property
    def failure_count(self) -> int:
        """Number of failed benchmarks."""
        return sum(1 for r in self.results if not r.ok)


class BenchmarkOrchestrator:
    """Orchestrates benchmark execution across adapters."""

    def __init__(self, *, config: OrchestratorConfig | None = None) -> None:
        self._config = config or OrchestratorConfig()
        self._progress_callback: ProgressCallback | None = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _run_with_timeout(
        self,
        benchmark: BaseBenchmark,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
        timeout_seconds: int,
    ) -> Any:
        """Run benchmark with timeout enforcement.

        Args:
            benchmark: Benchmark to run.
            adapter: Database adapter.
            scale: Scale configuration.
            timeout_seconds: Maximum seconds before timeout.

        Returns:
            Metrics from benchmark execution.

        Raises:
            TimeoutError: If benchmark exceeds timeout.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(benchmark.run, adapter, scale)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeout:
                raise TimeoutError(f"Benchmark timed out after {timeout_seconds}s")

    def run(
        self,
        adapters: list[GraphDatabaseAdapter],
        *,
        benchmarks: list[BaseBenchmark] | None = None,
        scale: str | ScaleConfig | None = None,
    ) -> OrchestratorResult:
        """Run benchmarks across all adapters.

        Args:
            adapters: List of database adapters to benchmark.
            benchmarks: List of benchmarks to run (None = use config/all).
            scale: Scale configuration (None = use config).

        Returns:
            OrchestratorResult with all results.
        """
        result = OrchestratorResult()
        result.started_at = time.time()

        scale_config = self._resolve_scale(scale)
        result.scale = scale_config
        result.databases = [a.name for a in adapters]

        benchmark_list = self._resolve_benchmarks(benchmarks)

        for adapter in adapters:
            self._run_adapter_benchmarks(adapter, benchmark_list, scale_config, result)

        result.completed_at = time.time()
        return result

    def _resolve_scale(self, scale: str | ScaleConfig | None) -> ScaleConfig:
        """Resolve scale configuration."""
        if scale is None:
            scale = self._config.scale

        if isinstance(scale, str):
            return get_scale(scale)
        return scale

    def _resolve_benchmarks(self, benchmarks: list[BaseBenchmark] | None) -> list[BaseBenchmark]:
        """Resolve list of benchmarks to run."""
        if benchmarks is not None:
            return benchmarks

        benchmark_names = self._config.benchmarks
        if benchmark_names is None:
            benchmark_names = BenchmarkRegistry.list()

        if self._config.categories:
            benchmark_names = [
                name for name in benchmark_names
                if BenchmarkRegistry.get(name)
                and BenchmarkRegistry.get(name).category in self._config.categories  # type: ignore
            ]

        result = []
        for name in benchmark_names:
            bench_cls = BenchmarkRegistry.get(name)
            if bench_cls:
                result.append(bench_cls())
        return result

    def _run_adapter_benchmarks(
        self,
        adapter: GraphDatabaseAdapter,
        benchmarks: list[BaseBenchmark],
        scale: ScaleConfig,
        result: OrchestratorResult,
    ) -> None:
        """Run all benchmarks for a single adapter."""
        timeout = self._config.timeout_override or scale.timeout_seconds

        for benchmark in benchmarks:
            if self._progress_callback:
                self._progress_callback(adapter.name, benchmark.name, "running")

            try:
                metrics = self._run_with_timeout(benchmark, adapter, scale, timeout)
                bench_result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    database=adapter.name,
                    scale=scale,
                    metrics=metrics,
                    status=Status.SUCCESS,
                )
            except TimeoutError:
                bench_result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    database=adapter.name,
                    scale=scale,
                    metrics=None,
                    status=Status.TIMEOUT,
                    error="Benchmark timed out",
                )
            except Exception as e:
                bench_result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    database=adapter.name,
                    scale=scale,
                    metrics=None,
                    status=Status.FAILED,
                    error=str(e),
                )

                if not self._config.continue_on_error:
                    result.results.append(bench_result)
                    raise

            result.results.append(bench_result)

            if self._progress_callback:
                status = "success" if bench_result.ok else "failed"
                self._progress_callback(adapter.name, benchmark.name, status)
