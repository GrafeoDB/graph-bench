r"""
Vector search benchmarks.

Measures vector storage and retrieval performance:
- Insert: Bulk insert nodes with high-dimensional embeddings
- k-NN: Single-query nearest neighbor search latency
- Batch Search: Multi-query throughput
- Recall: Search quality (fraction of true nearest neighbors found)

Vectors are 128-dimensional with 10 clusters for meaningful similarity.
Uses only stdlib (math, random) for vector generation — no numpy.

    from graph_bench.benchmarks.vector import VectorKnnBenchmark

    bench = VectorKnnBenchmark()
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
    "VectorInsertBenchmark",
    "VectorKnnBenchmark",
    "VectorBatchSearchBenchmark",
    "VectorRecallBenchmark",
]

# Vector benchmark constants
VECTOR_DIMENSIONS = 128
VECTOR_CLUSTERS = 10
VECTOR_NOISE = 0.1  # Gaussian noise std dev around cluster centroids


class VectorBenchmarkBase(BaseBenchmark):
    """Base class for vector benchmarks.

    Generates L2-normalized vectors with cluster structure so that
    k-NN queries return meaningful results (same-cluster vectors
    should be nearest neighbors). Reuses insert_nodes() for data
    loading since vectors are just node properties.

    Instance variables store state to avoid class-level issues.
    """

    category = "vector"

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
        """Generate vectors with cluster structure for meaningful k-NN.

        Each cluster has a centroid (random unit vector). Vectors are
        generated as centroid + gaussian noise, then L2-normalized.

        Args:
            count: Total number of vectors to generate.

        Returns:
            Tuple of (vectors, centroids).
        """
        # Generate cluster centroids (random unit vectors)
        centroids: list[list[float]] = []
        for _ in range(self._num_clusters):
            raw = [self._rng.gauss(0.0, 1.0) for _ in range(self._dimensions)]
            norm = math.sqrt(sum(x * x for x in raw))
            centroids.append([x / norm for x in raw])

        # Generate vectors around centroids
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

    def _insert_vectors(
        self,
        adapter: GraphDatabaseAdapter,
        vectors: list[list[float]],
    ) -> int:
        """Insert vectors as node properties and build ground truth.

        Args:
            adapter: Database adapter.
            vectors: List of vectors to insert.

        Returns:
            Number of nodes inserted.
        """
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
        return adapter.insert_nodes(nodes, label="VectorNode")

    def setup(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> None:
        """Load vector dataset and create index.

        Vector count is derived from scale.nodes (capped at 50K).
        """
        adapter.clear()
        self._rng = random.Random(self._seed)
        self._vector_count = min(50_000, max(500, scale.nodes))

        vectors, self._centroids = self._generate_clustered_vectors(
            self._vector_count,
        )
        self._insert_vectors(adapter, vectors)

        adapter.create_vector_index(
            "VectorNode",
            "embedding",
            dimensions=self._dimensions,
            metric="cosine",
        )


# =============================================================================
# Vector Insert
# =============================================================================


@BenchmarkRegistry.register("vector_insert", category="vector")
class VectorInsertBenchmark(VectorBenchmarkBase):
    """Vector node insertion benchmark.

    Measures the throughput of inserting nodes with 128-dim vector
    embeddings and creating a vector index. Each iteration clears
    the database and re-inserts all vectors.
    """

    @property
    def name(self) -> str:
        return "vector_insert"

    def setup(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> None:
        """Only initialize parameters — insertion happens in run_iteration."""
        adapter.clear()
        self._rng = random.Random(self._seed)
        self._vector_count = min(50_000, max(500, scale.nodes))

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Insert all vectors and create index. Returns count inserted."""
        adapter.clear()
        self._rng = random.Random(self._seed)

        vectors, self._centroids = self._generate_clustered_vectors(
            self._vector_count,
        )
        count = self._insert_vectors(adapter, vectors)

        adapter.create_vector_index(
            "VectorNode",
            "embedding",
            dimensions=self._dimensions,
            metric="cosine",
        )
        return count


# =============================================================================
# k-NN Search
# =============================================================================


@BenchmarkRegistry.register("vector_knn", category="vector")
class VectorKnnBenchmark(VectorBenchmarkBase):
    """Single k-NN search benchmark.

    Measures the latency of k-NN vector search (k=10) using
    each cluster centroid as a query vector. Returns total
    number of results across all queries.
    """

    @property
    def name(self) -> str:
        return "vector_knn"

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Search with each centroid. Returns total results."""
        total_results = 0
        for centroid in self._centroids:
            results = adapter.vector_search(
                centroid,
                label="VectorNode",
                property_name="embedding",
                k=10,
                metric="cosine",
            )
            total_results += len(results)
        return total_results


# =============================================================================
# Batch Search
# =============================================================================


@BenchmarkRegistry.register("vector_batch_search", category="vector")
class VectorBatchSearchBenchmark(VectorBenchmarkBase):
    """Batch k-NN search benchmark.

    Runs 100 k-NN searches with perturbed centroid queries to
    measure aggregate search throughput. Each query is a slightly
    noisy version of a cluster centroid.
    """

    @property
    def name(self) -> str:
        return "vector_batch_search"

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Run 100 searches with perturbed queries. Returns total results."""
        total_results = 0
        for i in range(100):
            centroid = self._centroids[i % self._num_clusters]
            query = [c + self._rng.gauss(0.0, 0.05) for c in centroid]
            norm = math.sqrt(sum(x * x for x in query))
            if norm > 0:
                query = [x / norm for x in query]

            results = adapter.vector_search(
                query,
                label="VectorNode",
                property_name="embedding",
                k=10,
                metric="cosine",
            )
            total_results += len(results)
        return total_results


# =============================================================================
# Recall@k
# =============================================================================


@BenchmarkRegistry.register("vector_recall", category="vector")
class VectorRecallBenchmark(VectorBenchmarkBase):
    """Recall@k benchmark.

    Measures search quality: what fraction of the k returned results
    belong to the same cluster as the query centroid. High recall
    indicates the HNSW index is finding true nearest neighbors.

    Timing is measured normally; recall value is reported via
    items_processed (relevant hits) vs total queries * k.
    """

    @property
    def name(self) -> str:
        return "vector_recall"

    def run_iteration(
        self,
        adapter: GraphDatabaseAdapter,
        scale: ScaleConfig,
    ) -> int:
        """Search each centroid and count same-cluster hits."""
        total_relevant = 0
        for cluster_idx, centroid in enumerate(self._centroids):
            results = adapter.vector_search(
                centroid,
                label="VectorNode",
                property_name="embedding",
                k=10,
                metric="cosine",
            )
            expected_ids = set(self._ground_truth[cluster_idx])
            returned_ids = {node_id for node_id, _score in results}
            total_relevant += len(returned_ids & expected_ids)
        return total_relevant
