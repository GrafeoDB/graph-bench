r"""
Base benchmark implementation.

Provides common functionality for all benchmarks including
timing, warmup, and metric collection.

    from graph_bench.benchmarks.base import BaseBenchmark

    class MyBenchmark(BaseBenchmark):
        ...
"""

from __future__ import annotations

import gc
import statistics
import time
from abc import ABC, abstractmethod
from typing import Any

from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import Metrics, ScaleConfig, TimingStats

__all__ = ["BaseBenchmark", "BenchmarkRegistry"]


class BenchmarkRegistry:
    """Registry for benchmark implementations."""

    _benchmarks: dict[str, type[BaseBenchmark]] = {}

    @classmethod
    def register(cls, name: str, *, category: str = "general") -> Any:
        """Decorator to register a benchmark class."""

        def decorator(benchmark_cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
            cls._benchmarks[name] = benchmark_cls
            return benchmark_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseBenchmark] | None:
        """Get benchmark class by name."""
        return cls._benchmarks.get(name)

    @classmethod
    def list(cls) -> list[str]:
        """List registered benchmark names."""
        return list(cls._benchmarks.keys())

    @classmethod
    def by_category(cls, category: str) -> list[str]:
        """List benchmarks in a category."""
        return [name for name, bench in cls._benchmarks.items() if bench.category == category]


class BaseBenchmark(ABC):
    """Base class for benchmark implementations."""

    category: str = "general"

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.__class__.__doc__ or self.name

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Prepare benchmark (load data, create indices, etc.)."""
        pass

    @abstractmethod
    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run single benchmark iteration, return items processed."""
        ...

    def teardown(self, adapter: GraphDatabaseAdapter) -> None:
        """Clean up after benchmark."""
        pass

    def run(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> Metrics:
        """Execute the benchmark and return metrics."""
        self.setup(adapter, scale)

        try:
            for _ in range(scale.warmup_iterations):
                gc.disable()
                self.run_iteration(adapter, scale)
                gc.enable()
                gc.collect()

            timings_ns: list[int] = []
            items_processed = 0

            for _ in range(scale.measurement_iterations):
                gc.disable()
                start = time.perf_counter_ns()
                items = self.run_iteration(adapter, scale)
                end = time.perf_counter_ns()
                gc.enable()

                timings_ns.append(end - start)
                items_processed += items
                gc.collect()

            timing_stats = self._compute_stats(timings_ns)
            total_items = items_processed // scale.measurement_iterations if scale.measurement_iterations else 0
            throughput = timing_stats.ops_per_second * total_items if total_items else timing_stats.ops_per_second

            return Metrics(
                timing=timing_stats,
                throughput=throughput,
                items_processed=items_processed,
            )
        finally:
            self.teardown(adapter)

    def _compute_stats(self, timings_ns: list[int]) -> TimingStats:
        """Compute timing statistics from raw nanosecond measurements."""
        if not timings_ns:
            return TimingStats(
                min_ns=0,
                max_ns=0,
                mean_ns=0.0,
                median_ns=0.0,
                std_ns=0.0,
                p99_ns=0.0,
                iterations=0,
            )

        sorted_timings = sorted(timings_ns)
        n = len(sorted_timings)

        p99_idx = int(n * 0.99)
        p99_idx = min(p99_idx, n - 1)

        return TimingStats(
            min_ns=sorted_timings[0],
            max_ns=sorted_timings[-1],
            mean_ns=statistics.mean(timings_ns),
            median_ns=statistics.median(timings_ns),
            std_ns=statistics.stdev(timings_ns) if n > 1 else 0.0,
            p99_ns=float(sorted_timings[p99_idx]),
            iterations=n,
        )
