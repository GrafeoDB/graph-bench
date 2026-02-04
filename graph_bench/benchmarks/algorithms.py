r"""
Algorithm benchmarks for graph databases.

Measures graph algorithm performance including PageRank and community detection.

    from graph_bench.benchmarks.algorithms import PageRankBenchmark

    bench = PageRankBenchmark()
    metrics = bench.run(adapter, scale)
"""

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "PageRankBenchmark",
    "CommunityDetectionBenchmark",
    "BetweennessCentralityBenchmark",
    "ClosenessCentralityBenchmark",
]


class AlgorithmBenchmarkBase(BaseBenchmark):
    """Base class for algorithm benchmarks."""

    category = "algorithm"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        node_count = min(10000, scale.nodes // 10)
        edge_count = min(50000, scale.edges // 10)

        nodes = [{"id": f"person_{i}", "name": f"Person {i}"} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="Person")

        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(edge_count):
            src = f"person_{i % node_count}"
            tgt = f"person_{(i * 7 + 3) % node_count}"
            if src != tgt:
                edges.append((src, tgt, "FOLLOWS", {"weight": 1.0}))

        adapter.insert_edges(edges)


@BenchmarkRegistry.register("pagerank", category="algorithm")
class PageRankBenchmark(AlgorithmBenchmarkBase):
    """Benchmark PageRank computation.

    Databases with native PageRank use their implementation.
    Others use NetworkX fallback (includes graph extraction overhead).
    """

    @property
    def name(self) -> str:
        return "pagerank"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        scores = adapter.pagerank(damping=0.85, max_iterations=20, tolerance=1e-4)
        return len(scores)


@BenchmarkRegistry.register("community_detection", category="algorithm")
class CommunityDetectionBenchmark(AlgorithmBenchmarkBase):
    """Benchmark community detection (Louvain).

    Databases with native Louvain use their implementation.
    Others use NetworkX fallback (includes graph extraction overhead).
    """

    @property
    def name(self) -> str:
        return "community_detection"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        communities = adapter.community_detection(algorithm="louvain")
        return len(communities)


class CentralityBenchmarkBase(BaseBenchmark):
    """Base class for centrality benchmarks."""

    category = "algorithm"
    _node_ids: list[str] = []
    _node_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._node_count = min(1000, scale.nodes // 100)
        edge_count = min(5000, scale.edges // 100)

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


@BenchmarkRegistry.register("betweenness_centrality", category="algorithm")
class BetweennessCentralityBenchmark(CentralityBenchmarkBase):
    """Benchmark betweenness centrality approximation.

    Betweenness centrality identifies nodes that act as bridges:
    - Bottleneck detection
    - Key influencer identification
    - Network flow analysis
    """

    @property
    def name(self) -> str:
        return "betweenness_centrality"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Approximate betweenness by sampling shortest paths through nodes."""
        # Sample pairs and count how often nodes appear on shortest paths
        path_counts: dict[str, int] = {}
        sample_pairs = 20

        for _ in range(sample_pairs):
            src = random.choice(self._node_ids)
            tgt = random.choice(self._node_ids)
            if src == tgt:
                continue

            path = adapter.shortest_path(src, tgt)
            if path and len(path) > 2:
                # Count intermediate nodes (not src/tgt)
                for node in path[1:-1]:
                    path_counts[node] = path_counts.get(node, 0) + 1

        # Return count of unique nodes with non-zero betweenness
        return len([k for k, v in path_counts.items() if v > 0])


@BenchmarkRegistry.register("closeness_centrality", category="algorithm")
class ClosenessCentralityBenchmark(CentralityBenchmarkBase):
    """Benchmark closeness centrality approximation.

    Closeness centrality measures how close a node is to all others:
    - Information spread analysis
    - Optimal location finding
    - Access time optimization
    """

    @property
    def name(self) -> str:
        return "closeness_centrality"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Approximate closeness by measuring BFS distances."""
        closeness_scores: list[tuple[str, float]] = []
        sample_size = min(30, len(self._node_ids))

        for node_id in random.sample(self._node_ids, sample_size):
            # BFS to measure reachable nodes
            visited = adapter.traverse_bfs(node_id, max_depth=5)
            # Closeness approximated by count of reachable nodes
            closeness_scores.append((node_id, len(visited)))

        # Sort and return top-5 highest closeness scores sum
        closeness_scores.sort(key=lambda x: x[1], reverse=True)
        return sum(int(s) for _, s in closeness_scores[:5])
