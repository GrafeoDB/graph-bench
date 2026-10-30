r"""
Vector search benchmarks, in two families that are never mixed.

``vector_scan_*`` (category ``vector_scan``): the same brute-force scan for every
adapter (``BaseAdapter.vector_scan``: read every vector node, score it in
Python). Exact, and comparable across all databases.

``vector_ann_*`` (category ``vector_ann``): the database's native approximate
nearest neighbour index (for example HNSW), with no fallback. An adapter
without one raises ``UnsupportedOperation`` and the result is SKIPPED. Every
ANN result reports recall@k against the exact ground truth, computed from the
vectors as stored (the data the scan path reads), next to its latency.

The HNSW parameters (``m``, ``ef_construction``, ``ef``), the quantization mode
and the optional ``ef`` sweep are shared by every adapter in a run and set with
``configure_ann()`` (the CLI's ``--ann-*`` options).

Vectors are 128-dimensional with 10 clusters for meaningful similarity, and are
loaded through ``insert_nodes()`` as lists of floats, which the Grafeo adapters
store as vectors (direct API or a ``vector()`` call on a query parameter).

    from graph_bench.benchmarks.vector import VectorAnnKnnBenchmark, configure_ann

    configure_ann(ef=128, sweep=(16, 32, 64, 128))
    metrics = VectorAnnKnnBenchmark().run(adapter, scale)
    print(metrics.custom["recall_at_k"])
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "DEFAULT_ANN_SWEEP",
    "QUANTIZATIONS",
    "AnnSettings",
    "VectorAnnBuildBenchmark",
    "VectorAnnDeleteBenchmark",
    "VectorAnnKnnBenchmark",
    "VectorBatchSearchBenchmark",
    "VectorInsertBenchmark",
    "VectorKnnBenchmark",
    "VectorRecallBenchmark",
    "VectorScanBatchSearchBenchmark",
    "VectorScanInsertBenchmark",
    "VectorScanKnnBenchmark",
    "VectorScanRecallBenchmark",
    "ann_settings",
    "configure_ann",
    "exact_top_k",
    "recall_at_k",
]

# Vector benchmark constants
VECTOR_DIMENSIONS = 128
VECTOR_CLUSTERS = 10
VECTOR_NOISE = 0.1  # Gaussian noise std dev around cluster centroids
QUERY_NOISE = 0.05  # Noise of the perturbed-centroid queries
LABEL = "VectorNode"
PROPERTY = "embedding"

QUANTIZATIONS = (None, "scalar", "binary", "product")
DEFAULT_ANN_SWEEP = (10, 20, 40, 80, 160, 320)


@dataclass(frozen=True)
class AnnSettings:
    """ANN parameters shared by every adapter in a run.

    Attributes:
        k: Neighbours per query (recall@k).
        m: HNSW links per node.
        ef_construction: HNSW build beam width.
        ef: HNSW search beam width for the timed queries.
        quantization: Index quantization (None, "scalar", "binary", "product").
        queries: Queries per iteration.
        sweep: ef values for the recall-versus-latency curve (empty: no sweep).
        delete_fraction: Share of the vectors vector_ann_delete removes.
    """

    k: int = 10
    m: int = 16
    ef_construction: int = 128
    ef: int = 64
    quantization: str | None = None
    queries: int = 100
    sweep: tuple[int, ...] = ()
    delete_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            msg = f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}"
            raise ValueError(msg)
        if not 0 < self.delete_fraction < 1:
            msg = f"delete_fraction must be between 0 and 1, got {self.delete_fraction}"
            raise ValueError(msg)
        if min(self.k, self.m, self.ef_construction, self.ef, self.queries, *self.sweep) < 1:
            msg = "k, m, ef_construction, ef, queries and the sweep values must be positive"
            raise ValueError(msg)

    def index_metrics(self) -> dict[str, Any]:
        """The index parameters, for a result's custom metrics."""
        settings = asdict(self)
        return {key: settings[key] for key in ("k", "m", "ef_construction", "ef", "quantization")}


_ann = AnnSettings()


def ann_settings() -> AnnSettings:
    """The ANN parameters the vector_ann benchmarks use."""
    return _ann


def configure_ann(**changes: Any) -> AnnSettings:
    """Change the ANN parameters (for every adapter) and return the new settings."""
    global _ann
    _ann = replace(_ann, **changes)
    return _ann


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)


def exact_top_k(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    queries: Sequence[Sequence[float]],
    k: int,
) -> list[list[str]]:
    """Exact top-k IDs by cosine similarity for each query (the ground truth)."""
    import numpy as np

    if not ids:
        return [[] for _ in queries]
    data = np.asarray(vectors, dtype=np.float64)
    data /= np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)
    query = np.asarray(queries, dtype=np.float64)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query @ data.T
    top = min(k, len(ids))
    best = np.argpartition(-similarity, top - 1, axis=1)[:, :top]
    # Order each query's candidates by descending similarity.
    order = np.argsort(-np.take_along_axis(similarity, best, axis=1), axis=1)
    ranked = np.take_along_axis(best, order, axis=1)
    return [[ids[i] for i in row] for row in ranked.tolist()]


def recall_at_k(results: Sequence[Sequence[str]], truth: Sequence[Sequence[str]], k: int) -> float:
    """Share of the true k nearest neighbours that the results contain."""
    expected = sum(min(k, len(t)) for t in truth)
    if expected == 0:
        return 1.0
    found = sum(len(set(list(r)[:k]) & set(t)) for r, t in zip(results, truth, strict=True))
    return found / expected


class VectorBenchmarkBase(BaseBenchmark):
    """Shared data generation for the vector benchmarks.

    Generates L2-normalized vectors with cluster structure so that k-NN queries
    return meaningful results, and loads them through insert_nodes() (vectors
    are node properties).
    """

    category = "vector_scan"

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
        self._queries: list[list[float]] = []
        self._truth: list[list[str]] = []
        self._last_results: list[list[str]] = []
        self._stored: list[tuple[str, list[float]]] = []

    def _generate_clustered_vectors(self, count: int) -> tuple[list[list[float]], list[list[float]]]:
        """Generate unit vectors around random unit centroids (gaussian noise)."""
        centroids = [
            _unit([self._rng.gauss(0.0, 1.0) for _ in range(self._dimensions)]) for _ in range(self._num_clusters)
        ]
        vectors = [
            _unit([c + self._rng.gauss(0.0, VECTOR_NOISE) for c in centroids[i % self._num_clusters]])
            for i in range(count)
        ]
        return vectors, centroids

    def _generate_queries(self, count: int) -> list[list[float]]:
        """Perturbed centroids, the same for every iteration and adapter."""
        rng = random.Random(self._seed + 1)
        return [
            _unit([c + rng.gauss(0.0, QUERY_NOISE) for c in self._centroids[i % self._num_clusters]])
            for i in range(count)
        ]

    def _insert_vectors(self, adapter: GraphDatabaseAdapter, vectors: list[list[float]]) -> int:
        """Insert vectors as node properties and record each cluster's members."""
        self._ground_truth = {i: [] for i in range(self._num_clusters)}
        nodes: list[dict[str, Any]] = []
        for i, vec in enumerate(vectors):
            node_id = f"vec_{i}"
            cluster_idx = i % self._num_clusters
            self._ground_truth[cluster_idx].append(node_id)
            nodes.append({"id": node_id, PROPERTY: vec, "cluster": cluster_idx})
        self._node_ids = [n["id"] for n in nodes]
        return adapter.insert_nodes(nodes, label=LABEL)

    def _load(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Clear the database and load the vectors (capped at 50K)."""
        adapter.clear()
        self._rng = random.Random(self._seed)
        self._vector_count = min(50_000, max(500, scale.nodes))
        vectors, self._centroids = self._generate_clustered_vectors(self._vector_count)
        self._insert_vectors(adapter, vectors)

    def _exact_truth(self, adapter: GraphDatabaseAdapter, queries: Sequence[Sequence[float]], k: int) -> None:
        """Ground truth from the vectors as the database stores them (what the scan reads)."""
        stored = [
            (str(node["id"]), node[PROPERTY])
            for node in adapter.get_nodes_by_label(LABEL, limit=100_000)
            if isinstance(node.get(PROPERTY), list)
        ]
        self._stored = stored
        self._truth = exact_top_k([i for i, _ in stored], [v for _, v in stored], queries, k)

    def _recall(self, k: int) -> float:
        return recall_at_k(self._last_results, self._truth, k)


# =============================================================================
# Scan: the same brute-force search for every adapter
# =============================================================================


class VectorScanBenchmarkBase(VectorBenchmarkBase):
    """Loads the vectors; every query goes through ``vector_scan`` (no index)."""

    category = "vector_scan"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        self._load(adapter, scale)

    def _scan(self, adapter: GraphDatabaseAdapter, queries: Sequence[list[float]], k: int = 10) -> int:
        self._last_results = [
            [node_id for node_id, _ in adapter.vector_scan(q, label=LABEL, property_name=PROPERTY, k=k)]
            for q in queries
        ]
        return sum(len(r) for r in self._last_results)

    def custom_metrics(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> dict[str, Any]:
        # Proves the scan is exact: recall@10 against the stored vectors.
        return {"recall_at_k": self._recall(10), "k": 10, "queries": len(self._truth)}


@BenchmarkRegistry.register("vector_scan_insert", category="vector_scan")
class VectorScanInsertBenchmark(VectorBenchmarkBase):
    """Insert all vector nodes (no index). Each iteration clears and re-inserts."""

    category = "vector_scan"

    @property
    def name(self) -> str:
        return "vector_scan_insert"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._vector_count = min(50_000, max(500, scale.nodes))

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        adapter.clear()
        self._rng = random.Random(self._seed)
        vectors, self._centroids = self._generate_clustered_vectors(self._vector_count)
        return self._insert_vectors(adapter, vectors)


@BenchmarkRegistry.register("vector_scan_knn", category="vector_scan")
class VectorScanKnnBenchmark(VectorScanBenchmarkBase):
    """k=10 scan search with each cluster centroid as the query."""

    @property
    def name(self) -> str:
        return "vector_scan_knn"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        super().setup(adapter, scale)
        self._exact_truth(adapter, self._centroids, 10)

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        return self._scan(adapter, self._centroids)


@BenchmarkRegistry.register("vector_scan_batch_search", category="vector_scan")
class VectorScanBatchSearchBenchmark(VectorScanBenchmarkBase):
    """100 k=10 scan searches with perturbed centroid queries (throughput)."""

    @property
    def name(self) -> str:
        return "vector_scan_batch_search"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        super().setup(adapter, scale)
        self._queries = self._generate_queries(100)
        self._exact_truth(adapter, self._queries, 10)

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        return self._scan(adapter, self._queries)


@BenchmarkRegistry.register("vector_scan_recall", category="vector_scan")
class VectorScanRecallBenchmark(VectorScanBenchmarkBase):
    """Cluster precision: how many of the 10 results share the query centroid's cluster.

    items_processed counts the same-cluster hits; recall_at_k in the custom
    metrics is the recall against the exact nearest neighbours.
    """

    @property
    def name(self) -> str:
        return "vector_scan_recall"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        super().setup(adapter, scale)
        self._exact_truth(adapter, self._centroids, 10)

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        self._scan(adapter, self._centroids)
        return sum(
            len(set(results) & set(self._ground_truth[cluster])) for cluster, results in enumerate(self._last_results)
        )


# The scan benchmarks were called vector_* before the ANN family existed.
for _old, _new in [
    ("vector_insert", "vector_scan_insert"),
    ("vector_knn", "vector_scan_knn"),
    ("vector_batch_search", "vector_scan_batch_search"),
    ("vector_recall", "vector_scan_recall"),
]:
    BenchmarkRegistry.alias(_old, _new)
VectorInsertBenchmark = VectorScanInsertBenchmark
VectorKnnBenchmark = VectorScanKnnBenchmark
VectorBatchSearchBenchmark = VectorScanBatchSearchBenchmark
VectorRecallBenchmark = VectorScanRecallBenchmark


# =============================================================================
# ANN: the database's native index, no fallback
# =============================================================================


class VectorAnnBenchmarkBase(VectorBenchmarkBase):
    """Loads the vectors, builds the native index and computes the exact ground truth."""

    category = "vector_ann"

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._settings = ann_settings()

    def _build_index(self, adapter: GraphDatabaseAdapter) -> None:
        settings = self._settings
        adapter.create_vector_index(
            LABEL,
            PROPERTY,
            dimensions=self._dimensions,
            metric="cosine",
            m=settings.m,
            ef_construction=settings.ef_construction,
            quantization=settings.quantization,
        )

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        self._settings = ann_settings()
        self._load(adapter, scale)
        self._build_index(adapter)
        self._queries = self._generate_queries(self._settings.queries)
        self._exact_truth(adapter, self._queries, self._settings.k)

    def _search(self, adapter: GraphDatabaseAdapter, ef: int) -> list[list[str]]:
        k = self._settings.k
        return [
            [node_id for node_id, _ in adapter.vector_ann_search(q, label=LABEL, property_name=PROPERTY, k=k, ef=ef)]
            for q in self._queries
        ]

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        self._last_results = self._search(adapter, self._settings.ef)
        return sum(len(r) for r in self._last_results)

    def _sweep(self, adapter: GraphDatabaseAdapter) -> list[dict[str, float]]:
        """Recall and latency per query for each ef of the sweep (one warm pass, one timed)."""
        curve = []
        for ef in self._settings.sweep:
            self._search(adapter, ef)
            start = time.perf_counter_ns()
            results = self._search(adapter, ef)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            curve.append(
                {
                    "ef": ef,
                    "recall_at_k": recall_at_k(results, self._truth, self._settings.k),
                    "mean_query_ms": elapsed_ms / len(self._queries),
                }
            )
        return curve

    def custom_metrics(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            **self._settings.index_metrics(),
            "recall_at_k": self._recall(self._settings.k),
            "queries": len(self._queries),
            "vectors": self._vector_count,
        }
        if self._settings.sweep:
            metrics["sweep"] = self._sweep(adapter)
        return metrics


@BenchmarkRegistry.register("vector_ann_build", category="vector_ann")
class VectorAnnBuildBenchmark(VectorAnnBenchmarkBase):
    """Insert all vector nodes, then build the native index. Each iteration starts empty.

    Compare with vector_scan_insert (the same insert without an index).
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._timings: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "vector_ann_build"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        self._settings = ann_settings()
        adapter.clear()
        self._vector_count = min(50_000, max(500, scale.nodes))

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        adapter.clear()
        self._rng = random.Random(self._seed)
        vectors, self._centroids = self._generate_clustered_vectors(self._vector_count)
        start = time.perf_counter_ns()
        count = self._insert_vectors(adapter, vectors)
        inserted = time.perf_counter_ns()
        self._build_index(adapter)
        built = time.perf_counter_ns()
        self._timings = {"insert_ms": (inserted - start) / 1e6, "index_ms": (built - inserted) / 1e6}
        return count

    def custom_metrics(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> dict[str, Any]:
        return {**self._settings.index_metrics(), "vectors": self._vector_count, **self._timings}


@BenchmarkRegistry.register("vector_ann_knn", category="vector_ann")
class VectorAnnKnnBenchmark(VectorAnnBenchmarkBase):
    """Native k-NN queries (perturbed centroids) at the configured ef, with recall@k.

    With a sweep configured, the result also holds recall and latency per ef.
    """

    @property
    def name(self) -> str:
        return "vector_ann_knn"


@BenchmarkRegistry.register("vector_ann_delete", category="vector_ann")
class VectorAnnDeleteBenchmark(VectorAnnBenchmarkBase):
    """Recall after deleting a share of the vectors (delete_fraction, default 10%).

    Setup measures recall, deletes the vectors and recomputes the ground truth
    without them; the timed iterations query the index after the deletes. The
    result reports recall before and after, and how many results named a
    deleted vector or came back short, so a drop is visible, not hidden.
    """

    @property
    def name(self) -> str:
        return "vector_ann_delete"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        super().setup(adapter, scale)
        k = self._settings.k
        self._recall_before = recall_at_k(self._search(adapter, self._settings.ef), self._truth, k)

        rng = random.Random(self._seed + 2)
        count = max(1, round(len(self._node_ids) * self._settings.delete_fraction))
        self._deleted_ids = set(rng.sample(self._node_ids, count))
        start = time.perf_counter_ns()
        self._deleted = adapter.delete_nodes(sorted(self._deleted_ids))
        self._delete_ms = (time.perf_counter_ns() - start) / 1e6

        remaining = [(i, v) for i, v in self._stored if i not in self._deleted_ids]
        self._truth = exact_top_k([i for i, _ in remaining], [v for _, v in remaining], self._queries, k)

    def custom_metrics(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> dict[str, Any]:
        metrics = super().custom_metrics(adapter, scale)
        k = self._settings.k
        metrics.update(
            {
                "recall_before_delete": self._recall_before,
                "recall_drop": self._recall_before - metrics["recall_at_k"],
                "deleted": self._deleted,
                "delete_fraction": self._settings.delete_fraction,
                "delete_ms": self._delete_ms,
                "deleted_returned": sum(1 for r in self._last_results for i in r if i in self._deleted_ids),
                "short_results": sum(1 for r in self._last_results if len(r) < k),
            }
        )
        return metrics
