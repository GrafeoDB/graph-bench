r"""
graph-bench: Comprehensive benchmark suite for graph databases.

Benchmarks Memgraph, Neo4j, Kuzu (Ladybug), ArangoDB, and Grafeo
across storage, traversal, algorithm, and query workloads.

    from graph_bench import run_benchmarks, SCALES
    from graph_bench.adapters import Neo4jAdapter, KuzuAdapter

    results = run_benchmarks(
        adapters=[Neo4jAdapter(), KuzuAdapter()],
        scale="medium",
    )
"""

from graph_bench.config import DEFAULT_SCALE, SCALES, get_scale
from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig, Status, TimingStats

__all__ = [
    "BenchmarkResult",
    "DEFAULT_SCALE",
    "Metrics",
    "SCALES",
    "ScaleConfig",
    "Status",
    "TimingStats",
    "get_scale",
]

__version__ = "0.1.0"
