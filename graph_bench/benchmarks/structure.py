r"""
Graph structure benchmarks for graph databases.

Measures structural analysis operations like connected components and degree distribution.

    from graph_bench.benchmarks.structure import ConnectedComponentsBenchmark

    bench = ConnectedComponentsBenchmark()
    metrics = bench.run(adapter, scale)
"""

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import Metrics, ScaleConfig, TimingStats

__all__ = [
    "ConnectedComponentsBenchmark",
    "DegreeDistributionBenchmark",
    "GraphDensityBenchmark",
]


class StructureBenchmarkBase(BaseBenchmark):
    """Base class for graph structure benchmarks."""

    category = "structure"
    _node_ids: list[str] = []
    _node_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._node_count = min(3000, scale.nodes // 30)
        edge_count = min(15000, scale.edges // 30)

        nodes = [{"id": f"person_{i}", "name": f"Person {i}"} for i in range(self._node_count)]
        adapter.insert_nodes(nodes, label="Person")

        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(edge_count):
            src = f"person_{i % self._node_count}"
            tgt = f"person_{(i * 7 + 3) % self._node_count}"
            if src != tgt:
                edges.append((src, tgt, "CONNECTS", {}))

        adapter.insert_edges(edges)
        self._node_ids = [f"person_{i}" for i in range(self._node_count)]


@BenchmarkRegistry.register("connected_components", category="structure")
class ConnectedComponentsBenchmark(StructureBenchmarkBase):
    """Benchmark finding connected components.

    Connected components identify disconnected subgraphs, useful for:
    - Data quality analysis
    - Network segmentation
    - Identifying isolated clusters
    """

    @property
    def name(self) -> str:
        return "connected_components"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find connected components using BFS from multiple starting points."""
        visited: set[str] = set()
        components = 0

        # Sample nodes to find components
        sample = random.sample(self._node_ids, min(100, len(self._node_ids)))

        for start_node in sample:
            if start_node in visited:
                continue

            # BFS to find all nodes in this component
            component = adapter.traverse_bfs(start_node, max_depth=10)
            visited.update(component)
            components += 1

        return components


@BenchmarkRegistry.register("degree_distribution", category="structure")
class DegreeDistributionBenchmark(StructureBenchmarkBase):
    """Benchmark computing degree distribution.

    Degree distribution analysis is used for:
    - Identifying hubs and influencers
    - Power-law detection
    - Network characterization
    """

    @property
    def name(self) -> str:
        return "degree_distribution"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute degree for sampled nodes and find high-degree nodes."""
        degrees: list[tuple[str, int]] = []
        sample_size = min(200, len(self._node_ids))

        for node_id in random.sample(self._node_ids, sample_size):
            neighbors = adapter.get_neighbors(node_id)
            degrees.append((node_id, len(neighbors)))

        # Sort to find top-k highest degree nodes
        degrees.sort(key=lambda x: x[1], reverse=True)
        top_10 = degrees[:10]

        # Return sum of top-10 degrees as metric
        return sum(d for _, d in top_10)


@BenchmarkRegistry.register("graph_density", category="structure")
class GraphDensityBenchmark(StructureBenchmarkBase):
    """Benchmark computing graph density metrics.

    Graph density helps understand:
    - Network connectivity
    - Sparseness vs density
    - Potential for traversal efficiency
    """

    @property
    def name(self) -> str:
        return "graph_density"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute node and edge counts for density calculation."""
        node_count = adapter.count_nodes(label="Person")
        edge_count = adapter.count_edges(edge_type="CONNECTS")

        # Density = edges / max_possible_edges
        # For directed graph: max = n * (n-1)
        if node_count > 1:
            max_edges = node_count * (node_count - 1)
            # Return density * 10000 for precision in integer return
            density_scaled = int((edge_count / max_edges) * 10000)
            return density_scaled
        return 0


@BenchmarkRegistry.register("reachability", category="structure")
class ReachabilityBenchmark(StructureBenchmarkBase):
    """Benchmark reachability queries between node pairs.

    Reachability is fundamental for:
    - Access control (can user reach resource?)
    - Dependency analysis
    - Impact assessment
    """

    @property
    def name(self) -> str:
        return "reachability"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Check reachability between random node pairs."""
        reachable_count = 0
        pair_count = 20
        max_depth = 5

        for _ in range(pair_count):
            source = random.choice(self._node_ids)
            target = random.choice(self._node_ids)
            if source == target:
                reachable_count += 1
                continue

            # Use BFS with limited depth to check reachability
            visited = adapter.traverse_bfs(source, max_depth=max_depth)
            if target in visited:
                reachable_count += 1

        return reachable_count
