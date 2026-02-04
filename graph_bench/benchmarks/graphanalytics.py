r"""
LDBC Graphanalytics benchmarks.

Implements the six core algorithms from the LDBC Graphanalytics benchmark:
- BFS: Breadth-First Search (vertex depths from source)
- PR: PageRank (iterative ranking)
- WCC: Weakly Connected Components
- CDLP: Community Detection using Label Propagation
- LCC: Local Clustering Coefficient
- SSSP: Single-Source Shortest Paths (weighted)

Reference: https://github.com/ldbc/ldbc_graphalytics
Spec: https://github.com/ldbc/ldbc_graphalytics_docs

    from graph_bench.benchmarks.graphanalytics import LdbcBfsBenchmark

    bench = LdbcBfsBenchmark()
    metrics = bench.run(adapter, scale)
"""

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "LdbcBfsBenchmark",
    "LdbcPageRankBenchmark",
    "LdbcWccBenchmark",
    "LdbcCdlpBenchmark",
    "LdbcLccBenchmark",
    "LdbcSsspBenchmark",
]


class LdbcBenchmarkBase(BaseBenchmark):
    """Base class for LDBC Graphanalytics benchmarks.

    Uses a consistent graph setup matching LDBC specifications:
    - Directed graph with unique edges
    - No self-loops or multi-edges
    - Vertices identified by unique IDs
    """

    category = "ldbc_graphanalytics"
    _node_ids: list[str] = []
    _node_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup graph for LDBC benchmarks."""
        adapter.clear()

        # Scale appropriately for LDBC-style workloads
        self._node_count = min(10000, scale.nodes // 10)
        edge_count = min(50000, scale.edges // 10)

        # Create nodes
        nodes = [{"id": f"v_{i}", "name": f"Vertex {i}"} for i in range(self._node_count)]
        adapter.insert_nodes(nodes, label="Vertex")

        # Create directed edges (no self-loops, no multi-edges)
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[str, str]] = set()

        for i in range(edge_count):
            src_idx = i % self._node_count
            tgt_idx = (i * 7 + 3) % self._node_count
            if src_idx != tgt_idx:
                src = f"v_{src_idx}"
                tgt = f"v_{tgt_idx}"
                if (src, tgt) not in edge_set:
                    edge_set.add((src, tgt))
                    # Add weight for SSSP benchmark
                    weight = round(random.uniform(0.1, 10.0), 2)
                    edges.append((src, tgt, "EDGE", {"weight": weight}))

        adapter.insert_edges(edges)
        self._node_ids = [f"v_{i}" for i in range(self._node_count)]


@BenchmarkRegistry.register("ldbc_bfs", category="ldbc_graphanalytics")
class LdbcBfsBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics BFS benchmark.

    Breadth-First Search labels each vertex with its distance (depth)
    from a given source vertex. The source has depth 0, its neighbors
    have depth 1, their neighbors depth 2, etc.

    Unreachable vertices should have depth infinity.
    """

    @property
    def name(self) -> str:
        return "ldbc_bfs"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run BFS from multiple random sources."""
        total_visited = 0
        num_runs = 5

        for _ in range(num_runs):
            source = random.choice(self._node_ids)
            levels = adapter.bfs_levels(source)
            total_visited += len(levels)

        return total_visited


@BenchmarkRegistry.register("ldbc_pagerank", category="ldbc_graphanalytics")
class LdbcPageRankBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics PageRank benchmark.

    PageRank iteratively assigns ranking values to vertices.
    Initial value: 1/|V| for all vertices.
    Damping factor: typically 0.85.
    Fixed number of iterations (LDBC spec).
    """

    @property
    def name(self) -> str:
        return "ldbc_pagerank"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute PageRank with LDBC parameters."""
        # LDBC uses fixed iterations, not convergence
        scores = adapter.pagerank(damping=0.85, max_iterations=20, tolerance=1e-8)
        return len(scores)


@BenchmarkRegistry.register("ldbc_wcc", category="ldbc_graphanalytics")
class LdbcWccBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics Weakly Connected Components benchmark.

    Finds all weakly connected components and assigns each vertex
    a label indicating its component. For directed graphs, edges
    are treated as undirected (can traverse in either direction).
    """

    @property
    def name(self) -> str:
        return "ldbc_wcc"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find weakly connected components."""
        components = adapter.weakly_connected_components()
        # Return total vertices assigned to components
        return sum(len(c) for c in components)


@BenchmarkRegistry.register("ldbc_cdlp", category="ldbc_graphanalytics")
class LdbcCdlpBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics Community Detection (Label Propagation) benchmark.

    CDLP assigns labels to vertices based on neighbor label frequencies.
    LDBC uses a deterministic variant:
    - Initially each vertex has label = its ID
    - Each iteration: vertex adopts most frequent neighbor label
    - Ties broken by smallest label (deterministic)
    - Synchronous updates (all vertices update based on previous iteration)
    """

    @property
    def name(self) -> str:
        return "ldbc_cdlp"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run community detection with label propagation."""
        # Use label_propagation as specified by LDBC
        communities = adapter.community_detection(algorithm="label_propagation")
        return len(communities)


@BenchmarkRegistry.register("ldbc_lcc", category="ldbc_graphanalytics")
class LdbcLccBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics Local Clustering Coefficient benchmark.

    LCC measures how close a vertex's neighbors are to being a clique.
    For vertex v with neighbors N(v):
    - LCC(v) = (edges between neighbors) / (max possible edges between neighbors)
    - If |N(v)| <= 1, LCC(v) = 0

    For directed graphs, neighborhood is undirected but edge counting is directed.
    """

    @property
    def name(self) -> str:
        return "ldbc_lcc"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute local clustering coefficients."""
        coefficients = adapter.local_clustering_coefficient()
        return len(coefficients)


@BenchmarkRegistry.register("ldbc_sssp", category="ldbc_graphanalytics")
class LdbcSsspBenchmark(LdbcBenchmarkBase):
    """LDBC Graphanalytics Single-Source Shortest Paths benchmark.

    SSSP computes shortest path distances from a source vertex to all others.
    Path length = sum of edge weights.
    Unreachable vertices have distance infinity.

    Edge weights are non-negative floats (64-bit double precision).
    """

    @property
    def name(self) -> str:
        return "ldbc_sssp"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute SSSP from multiple random sources."""
        total_reachable = 0
        num_runs = 3

        for _ in range(num_runs):
            source = random.choice(self._node_ids)
            distances = adapter.sssp(source)
            total_reachable += len(distances)

        return total_reachable
