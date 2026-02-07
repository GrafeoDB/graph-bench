r"""
Hybrid graph + vector benchmarks.

Combines graph traversal with vector similarity search — the
unique capability of graph databases with built-in vector support.

- Graph-to-Vector: Traverse graph neighbors, then find similar nodes
  via vector search (e.g. "expand my network semantically").
- Vector-to-Graph: Find similar nodes via k-NN, then expand each
  result through graph edges (e.g. "find similar, show context").

    from graph_bench.benchmarks.hybrid import HybridGraphToVectorBenchmark

    bench = HybridGraphToVectorBenchmark()
    metrics = bench.run(adapter, scale)
"""

from __future__ import annotations

import math
import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "HybridGraphToVectorBenchmark",
    "HybridVectorToGraphBenchmark",
]

# Hybrid benchmark constants
VECTOR_DIMENSIONS = 128
VECTOR_CLUSTERS = 10
VECTOR_NOISE = 0.1
INTRA_CLUSTER_EDGES = 5  # SIMILAR edges per node within cluster


class HybridBenchmarkBase(BaseBenchmark):
    """Base class for hybrid graph + vector benchmarks.

    Creates a graph with vector-embedded nodes and edges representing
    relationships. Intra-cluster nodes are connected with SIMILAR edges,
    and cross-cluster RELATED edges provide graph connectivity.

    Instance variables store state to avoid class-level issues.
    """

    category = "hybrid"

    def __init__(self, seed: int = 42) -> None:
        """Initialize with reproducible random seed."""
        self._seed = seed
        self._rng = random.Random(seed)
        self._vector_count: int = 0
        self._dimensions: int = VECTOR_DIMENSIONS
        self._num_clusters: int = VECTOR_CLUSTERS
        self._centroids: list[list[float]] = []
        self._ground_truth: dict[int, list[str]] = {}
        self._node_ids: list[str] = []

    def _generate_clustered_vectors(
        self,
        count: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate vectors with cluster structure.

        Same algorithm as vector benchmarks: random unit centroids
        with gaussian noise, L2-normalized.

        Returns:
            Tuple of (vectors, centroids).
        """
        centroids: list[list[float]] = []
        for _ in range(self._num_clusters):
            raw = [self._rng.gauss(0.0, 1.0) for _ in range(self._dimensions)]
            norm = math.sqrt(sum(x * x for x in raw))
            centroids.append([x / norm for x in raw])

        vectors: list[list[float]] = []
        for i in range(count):
            cluster_idx = i % self._num_clusters
            centroid = centroids[cluster_idx]
            raw = [
                c + self._rng.gauss(0.0, VECTOR_NOISE)
                for c in centroid
            ]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                vectors.append([x / norm for x in raw])
            else:
                vectors.append(raw)

        return vectors, centroids

    def setup(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> None:
        """Create graph with vector nodes and edges.

        - Nodes: VectorNode with {id, embedding, cluster}
        - Edges: SIMILAR (intra-cluster), RELATED (cross-cluster)
        - Vector index on embedding property
        """
        adapter.clear()
        self._rng = random.Random(self._seed)
        # Smaller than pure vector benchmarks — graph ops add overhead
        self._vector_count = min(10_000, max(200, scale.nodes // 5))

        vectors, self._centroids = self._generate_clustered_vectors(
            self._vector_count,
        )

        # Build ground truth and insert nodes
        self._ground_truth = {i: [] for i in range(self._num_clusters)}
        nodes: list[dict[str, Any]] = []
        for i, vec in enumerate(vectors):
            node_id = f"vec_{i}"
            cluster_idx = i % self._num_clusters
            self._ground_truth[cluster_idx].append(node_id)
            nodes.append({
                "id": node_id,
                "embedding": vec,
                "cluster": cluster_idx,
            })
        self._node_ids = [n["id"] for n in nodes]
        adapter.insert_nodes(nodes, label="VectorNode")

        # Create vector index
        adapter.create_vector_index(
            "VectorNode",
            "embedding",
            dimensions=self._dimensions,
            metric="cosine",
        )

        # Create intra-cluster SIMILAR edges
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for cluster_ids in self._ground_truth.values():
            for i, src in enumerate(cluster_ids):
                # Connect to next N nodes in same cluster (circular)
                for j in range(1, INTRA_CLUSTER_EDGES + 1):
                    tgt_idx = (i + j) % len(cluster_ids)
                    tgt = cluster_ids[tgt_idx]
                    if src != tgt:
                        edges.append((src, tgt, "SIMILAR", {}))

        # Create cross-cluster RELATED edges (~1 per 10 nodes)
        cross_count = self._vector_count // 10
        for i in range(cross_count):
            src_cluster = i % self._num_clusters
            tgt_cluster = (i + 1) % self._num_clusters
            src_nodes = self._ground_truth[src_cluster]
            tgt_nodes = self._ground_truth[tgt_cluster]
            if src_nodes and tgt_nodes:
                src = src_nodes[i % len(src_nodes)]
                tgt = tgt_nodes[i % len(tgt_nodes)]
                edges.append((src, tgt, "RELATED", {}))

        adapter.insert_edges(edges)


# =============================================================================
# Graph → Vector
# =============================================================================


@BenchmarkRegistry.register(
    "hybrid_graph_to_vector", category="hybrid",
)
class HybridGraphToVectorBenchmark(HybridBenchmarkBase):
    """Graph traversal followed by vector search.

    Pattern: Start at a node, get its SIMILAR neighbors (1-hop),
    then use the cluster centroid as a query vector to find more
    related nodes via k-NN search beyond the graph neighborhood.

    Simulates: "Find users similar to my network" — traverse graph
    to understand context, then vector search to expand discovery.
    """

    @property
    def name(self) -> str:
        return "hybrid_graph_to_vector"

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Traverse neighbors then vector search. Returns total found."""
        total_found = 0
        query_count = min(50, self._vector_count)
        for i in range(query_count):
            start_id = self._node_ids[i]

            # Step 1: Graph traversal — get 1-hop SIMILAR neighbors
            neighbors = adapter.get_neighbors(
                start_id, edge_type="SIMILAR",
            )

            # Step 2: Vector search for similar nodes beyond graph
            cluster_idx = i % self._num_clusters
            query_vector = self._centroids[cluster_idx]
            results = adapter.vector_search(
                query_vector,
                label="VectorNode",
                property_name="embedding",
                k=10,
                metric="cosine",
            )

            total_found += len(neighbors) + len(results)
        return total_found


# =============================================================================
# Vector → Graph
# =============================================================================


@BenchmarkRegistry.register(
    "hybrid_vector_to_graph", category="hybrid",
)
class HybridVectorToGraphBenchmark(HybridBenchmarkBase):
    """Vector search followed by graph expansion.

    Pattern: Start with a query vector, find k nearest neighbors
    via vector search, then expand each result through SIMILAR
    edges to discover the full neighborhood.

    Simulates: "Find content like this, then show related items" —
    use semantic similarity to find initial matches, then graph
    structure to discover broader context.
    """

    @property
    def name(self) -> str:
        return "hybrid_vector_to_graph"

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Vector search then expand via graph. Returns total found."""
        total_found = 0
        for cluster_idx in range(self._num_clusters):
            query_vector = self._centroids[cluster_idx]

            # Step 1: Vector search — find k nearest neighbors
            nn_results = adapter.vector_search(
                query_vector,
                label="VectorNode",
                property_name="embedding",
                k=10,
                metric="cosine",
            )

            # Step 2: Graph expansion — traverse SIMILAR edges
            for node_id, _score in nn_results:
                neighbors = adapter.get_neighbors(
                    node_id, edge_type="SIMILAR",
                )
                total_found += 1 + len(neighbors)
        return total_found
