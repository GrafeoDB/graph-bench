r"""
Result collection and aggregation.

    from graph_bench.reporting.collector import ResultCollector

    collector = ResultCollector()
    collector.add_result(result)
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from graph_bench.types import BenchmarkResult

__all__ = ["ResultCollector", "SessionInfo"]


@dataclass
class SessionInfo:
    """Information about a benchmark session.

    Attributes:
        session_id: Unique session identifier.
        started_at: Session start timestamp.
        completed_at: Session end timestamp (None if ongoing).
        scale: Scale name used.
        databases: List of database names tested.
    """

    session_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    scale: str = ""
    databases: list[str] = field(default_factory=list)


@dataclass
class EnvironmentInfo:
    """Information about the benchmark environment.

    Attributes:
        platform: Operating system platform.
        python_version: Python version string.
        cpu: CPU description.
        memory_gb: Total memory in GB.
    """

    platform: str = ""
    python_version: str = ""
    cpu: str = ""
    memory_gb: float = 0.0


class ResultCollector:
    """Collects and aggregates benchmark results."""

    def __init__(self) -> None:
        self._results: list[BenchmarkResult[Any]] = []
        self._session = SessionInfo()
        self._environment = EnvironmentInfo()
        self._started_at: datetime | None = None

    def start_session(self, *, scale: str, databases: list[str]) -> None:
        """Start a new benchmark session."""
        self._started_at = datetime.now(UTC)
        self._session = SessionInfo(
            session_id=f"bench_{self._started_at.strftime('%Y%m%d_%H%M%S')}",
            started_at=self._started_at.isoformat(),
            scale=scale,
            databases=databases,
        )
        self._collect_environment()

    def _collect_environment(self) -> None:
        """Collect environment information."""
        import platform
        import sys

        self._environment = EnvironmentInfo(
            platform=platform.system().lower(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            cpu=platform.processor() or "unknown",
            memory_gb=self._get_memory_gb(),
        )

    def _get_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            import os

            if hasattr(os, "sysconf"):
                mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                return round(mem_bytes / (1024**3), 1)
        except Exception:
            pass
        return 0.0

    def end_session(self) -> None:
        """End the current benchmark session."""
        self._session.completed_at = datetime.now(UTC).isoformat()

    def add_result(self, result: BenchmarkResult[Any]) -> None:
        """Add a benchmark result."""
        self._results.append(result)

    def add_results(self, results: list[BenchmarkResult[Any]]) -> None:
        """Add multiple benchmark results."""
        self._results.extend(results)

    @property
    def results(self) -> list[BenchmarkResult[Any]]:
        """Get all collected results."""
        return self._results

    @property
    def session(self) -> SessionInfo:
        """Get session information."""
        return self._session

    @property
    def environment(self) -> EnvironmentInfo:
        """Get environment information."""
        return self._environment

    def get_results_by_database(self, database: str) -> list[BenchmarkResult[Any]]:
        """Get results for a specific database."""
        return [r for r in self._results if r.database == database]

    def get_results_by_benchmark(self, benchmark: str) -> list[BenchmarkResult[Any]]:
        """Get results for a specific benchmark."""
        return [r for r in self._results if r.benchmark_name == benchmark]

    def compute_comparisons(self) -> dict[str, dict[str, float]]:
        """Compute speedup comparisons across databases.

        Returns:
            Dict mapping benchmark name to dict of database->speedup ratios.
        """
        comparisons: dict[str, dict[str, float]] = {}

        benchmarks = set(r.benchmark_name for r in self._results)

        for bench in benchmarks:
            bench_results = self.get_results_by_benchmark(bench)
            times = {}
            for r in bench_results:
                if r.ok and r.metrics:
                    times[r.database] = r.metrics.timing.mean_ns

            if not times:
                continue

            min_time = min(times.values())
            speedups = {}
            for db, t in times.items():
                speedups[db] = round(min_time / t if t > 0 else 0.0, 2)

            comparisons[bench] = speedups

        return comparisons

    def to_dict(self) -> dict[str, Any]:
        """Convert collected data to dictionary."""
        return {
            "session": {
                "id": self._session.session_id,
                "started_at": self._session.started_at,
                "completed_at": self._session.completed_at,
                "scale": self._session.scale,
                "databases": self._session.databases,
            },
            "environment": {
                "platform": self._environment.platform,
                "python_version": self._environment.python_version,
                "cpu": self._environment.cpu,
                "memory_gb": self._environment.memory_gb,
            },
            "results": [self._result_to_dict(r) for r in self._results],
            "comparisons": self.compute_comparisons(),
        }

    def _result_to_dict(self, result: BenchmarkResult[Any]) -> dict[str, Any]:
        """Convert a single result to dictionary."""
        data: dict[str, Any] = {
            "benchmark": result.benchmark_name,
            "database": result.database,
            "scale": result.scale.name,
            "status": result.status.name,
        }

        if result.metrics:
            data["metrics"] = {
                "timing": {
                    "min_ns": result.metrics.timing.min_ns,
                    "max_ns": result.metrics.timing.max_ns,
                    "mean_ns": result.metrics.timing.mean_ns,
                    "median_ns": result.metrics.timing.median_ns,
                    "std_ns": result.metrics.timing.std_ns,
                    "p99_ns": result.metrics.timing.p99_ns,
                    "iterations": result.metrics.timing.iterations,
                },
                "throughput": result.metrics.throughput,
                "items_processed": result.metrics.items_processed,
            }
            if result.metrics.memory_bytes:
                data["metrics"]["memory_bytes"] = result.metrics.memory_bytes
            if result.metrics.custom:
                data["metrics"]["custom"] = result.metrics.custom

        if result.error:
            data["error"] = result.error

        return data
