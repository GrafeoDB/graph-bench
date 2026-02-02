r"""
Pattern matching benchmarks for graph databases.

Measures pattern detection operations including triangles and common neighbors.

    from graph_bench.benchmarks.pattern import TriangleCountBenchmark

    bench = TriangleCountBenchmark()
    metrics = bench.run(adapter, scale)
"""

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "TriangleCountBenchmark",
    "CommonNeighborsBenchmark",
]


class PatternBenchmarkBase(BaseBenchmark):
    """Base class for pattern matching benchmarks."""

    category = "pattern"
    _node_ids: list[str] = []
    _node_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._node_count = min(2000, scale.nodes // 50)
        edge_count = min(10000, scale.edges // 50)

        nodes = [{"id": f"person_{i}", "name": f"Person {i}"} for i in range(self._node_count)]
        adapter.insert_nodes(nodes, label="Person")

        # Create edges with clustering to ensure triangles exist
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(edge_count):
            # Create clustered connections to increase triangle probability
            src_idx = i % self._node_count
            # Connect to nearby nodes (creates clusters/triangles)
            tgt_idx = (src_idx + (i % 10) + 1) % self._node_count
            src = f"person_{src_idx}"
            tgt = f"person_{tgt_idx}"
            if src != tgt:
                edges.append((src, tgt, "KNOWS", {}))

        adapter.insert_edges(edges)
        self._node_ids = [f"person_{i}" for i in range(self._node_count)]


@BenchmarkRegistry.register("triangle_count", category="pattern")
class TriangleCountBenchmark(PatternBenchmarkBase):
    """Benchmark triangle counting in the graph.

    Triangles are fundamental graph patterns used for:
    - Clustering coefficient calculation
    - Fraud ring detection
    - Social network analysis
    """

    @property
    def name(self) -> str:
        return "triangle_count"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Count triangles by sampling node neighborhoods."""
        triangles = 0
        sample_size = min(50, len(self._node_ids))

        for _ in range(sample_size):
            node = random.choice(self._node_ids)
            neighbors = adapter.get_neighbors(node)
            if len(neighbors) < 2:
                continue

            # Check pairs of neighbors for edges (triangle completion)
            neighbor_set = set(neighbors)
            for i, n1 in enumerate(neighbors[:10]):  # Limit to avoid explosion
                n1_neighbors = set(adapter.get_neighbors(n1))
                # Count common neighbors that form triangles
                common = neighbor_set & n1_neighbors
                triangles += len(common)

        return triangles // 3  # Each triangle counted 3 times


@BenchmarkRegistry.register("common_neighbors", category="pattern")
class CommonNeighborsBenchmark(PatternBenchmarkBase):
    """Benchmark finding common neighbors between node pairs.

    Common neighbors are used for:
    - Friend recommendations ("mutual friends")
    - Link prediction
    - Similarity scoring
    """

    @property
    def name(self) -> str:
        return "common_neighbors"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find common neighbors between random node pairs."""
        total_common = 0
        pair_count = 30

        for _ in range(pair_count):
            node_a = random.choice(self._node_ids)
            node_b = random.choice(self._node_ids)
            if node_a == node_b:
                continue

            neighbors_a = set(adapter.get_neighbors(node_a))
            neighbors_b = set(adapter.get_neighbors(node_b))
            common = neighbors_a & neighbors_b
            total_common += len(common)

        return total_common
