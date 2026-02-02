r"""
Traversal benchmarks for graph databases.

Measures graph traversal operations including hop queries and pattern matching.

    from graph_bench.benchmarks.traversal import Hop1Benchmark

    bench = Hop1Benchmark()
    metrics = bench.run(adapter, scale)
"""

from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "Hop1Benchmark",
    "Hop2Benchmark",
    "BFSBenchmark",
    "DFSBenchmark",
    "ShortestPathBenchmark",
]


class TraversalBenchmarkBase(BaseBenchmark):
    """Base class for traversal benchmarks."""

    category = "traversal"
    _start_nodes: list[str] = []

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        node_count = min(5000, scale.nodes // 20)
        edge_count = min(25000, scale.edges // 20)

        nodes = [{"id": f"person_{i}", "name": f"Person {i}"} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="Person")

        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(edge_count):
            src = f"person_{i % node_count}"
            tgt = f"person_{(i * 7 + 3) % node_count}"
            if src != tgt:
                edges.append((src, tgt, "FOLLOWS", {}))

        adapter.insert_edges(edges)

        self._start_nodes = [f"person_{i}" for i in range(min(100, node_count))]


@BenchmarkRegistry.register("hop_1", category="traversal")
class Hop1Benchmark(TraversalBenchmarkBase):
    """Benchmark 1-hop neighbor expansion."""

    @property
    def name(self) -> str:
        return "hop_1"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        total_neighbors = 0
        for _ in range(50):
            start = random.choice(self._start_nodes)
            neighbors = adapter.get_neighbors(start)
            total_neighbors += len(neighbors)
        return total_neighbors


@BenchmarkRegistry.register("hop_2", category="traversal")
class Hop2Benchmark(TraversalBenchmarkBase):
    """Benchmark 2-hop neighbor expansion."""

    @property
    def name(self) -> str:
        return "hop_2"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        total_visited = 0
        for _ in range(20):
            start = random.choice(self._start_nodes)
            visited = adapter.traverse_bfs(start, max_depth=2)
            total_visited += len(visited)
        return total_visited


@BenchmarkRegistry.register("bfs", category="traversal")
class BFSBenchmark(TraversalBenchmarkBase):
    """Benchmark BFS traversal."""

    @property
    def name(self) -> str:
        return "bfs"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        total_visited = 0
        for _ in range(10):
            start = random.choice(self._start_nodes)
            visited = adapter.traverse_bfs(start, max_depth=3)
            total_visited += len(visited)
        return total_visited


@BenchmarkRegistry.register("dfs", category="traversal")
class DFSBenchmark(TraversalBenchmarkBase):
    """Benchmark DFS traversal."""

    @property
    def name(self) -> str:
        return "dfs"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        total_visited = 0
        for _ in range(10):
            start = random.choice(self._start_nodes)
            visited = adapter.traverse_dfs(start, max_depth=3)
            total_visited += len(visited)
        return total_visited


@BenchmarkRegistry.register("shortest_path", category="traversal")
class ShortestPathBenchmark(TraversalBenchmarkBase):
    """Benchmark shortest path finding."""

    @property
    def name(self) -> str:
        return "shortest_path"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        paths_found = 0
        for _ in range(10):
            src = random.choice(self._start_nodes)
            tgt = random.choice(self._start_nodes)
            if src != tgt:
                path = adapter.shortest_path(src, tgt)
                if path:
                    paths_found += len(path)
        return paths_found
