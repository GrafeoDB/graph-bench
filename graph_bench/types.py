r"""
Core types for graph database benchmarks.

    from graph_bench.types import BenchmarkResult, Status, Metrics

    result = run_benchmark(adapter, scale="medium")
    if result.ok:
        print(f"Throughput: {result.metrics.throughput}")
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

__all__ = [
    "Status",
    "TimingStats",
    "Metrics",
    "ScaleConfig",
    "BenchmarkResult",
]


class Status(IntEnum):
    """Benchmark outcome status."""

    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    SKIPPED = auto()
    WARMUP = auto()


@dataclass(frozen=True, slots=True)
class TimingStats:
    """Timing statistics for a benchmark run.

    Attributes:
        min_ns: Minimum execution time in nanoseconds.
        max_ns: Maximum execution time in nanoseconds.
        mean_ns: Mean execution time in nanoseconds.
        median_ns: Median execution time in nanoseconds.
        std_ns: Standard deviation in nanoseconds.
        p99_ns: 99th percentile in nanoseconds.
        iterations: Number of iterations performed.
    """

    min_ns: int
    max_ns: int
    mean_ns: float
    median_ns: float
    std_ns: float
    p99_ns: float
    iterations: int

    @property
    def mean_ms(self) -> float:
        """Mean execution time in milliseconds."""
        return self.mean_ns / 1_000_000

    @property
    def median_ms(self) -> float:
        """Median execution time in milliseconds."""
        return self.median_ns / 1_000_000

    @property
    def p99_ms(self) -> float:
        """99th percentile in milliseconds."""
        return self.p99_ns / 1_000_000

    @property
    def ops_per_second(self) -> float:
        """Operations per second based on mean time."""
        if self.mean_ns == 0:
            return float("inf")
        return 1_000_000_000 / self.mean_ns


@dataclass(frozen=True, slots=True)
class Metrics:
    """Collected metrics from a benchmark.

    Attributes:
        timing: Timing statistics.
        throughput: Operations per second.
        items_processed: Number of items (nodes/edges) processed.
        memory_bytes: Memory usage in bytes (if available).
        custom: Additional custom metrics.
    """

    timing: TimingStats
    throughput: float
    items_processed: int
    memory_bytes: int | None = None
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Scale configuration for benchmarks.

    Attributes:
        name: Scale name (small, medium, large).
        nodes: Number of nodes.
        edges: Number of edges.
        warmup_iterations: Warmup iterations before measurement.
        measurement_iterations: Iterations for measurement.
        timeout_seconds: Maximum time per benchmark.
    """

    name: str
    nodes: int
    edges: int
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: int = 300


@dataclass(frozen=True, slots=True)
class BenchmarkResult[T]:
    """Result from a benchmark execution.

    Attributes:
        benchmark_name: Name of the benchmark.
        database: Database adapter name.
        scale: Scale configuration used.
        metrics: Collected metrics.
        status: Outcome status.
        error: Error message if failed.
        metadata: Additional result metadata.
    """

    benchmark_name: str
    database: str
    scale: ScaleConfig
    metrics: Metrics | None
    status: Status = Status.SUCCESS
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if benchmark completed successfully."""
        return self.status == Status.SUCCESS
