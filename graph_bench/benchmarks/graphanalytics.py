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
Spec: https://ldbcouncil.org/ldbc_graphalytics_docs/graphalytics_spec.pdf

Scale Definition: s(V, E) = log10(|V| + |E|), rounded to one decimal place

Validation Methods:
- Exact Match: BFS, CDLP (output values must be identical)
- Equivalence Match: WCC (label mappings must be logically equivalent)
- Epsilon Match: PR, LCC, SSSP (tolerance: |ref - sys| <= 0.0001 * |ref|)

    from graph_bench.benchmarks.graphanalytics import LdbcBfsBenchmark

    bench = LdbcBfsBenchmark()
    metrics = bench.run(adapter, scale)
"""

from __future__ import annotations

import math
import random
from abc import abstractmethod
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

# LDBC Graphanalytics constants
LDBC_DAMPING_FACTOR = 0.85
LDBC_PAGERANK_ITERATIONS = 20
LDBC_EPSILON = 0.0001  # 0.01% relative error tolerance
LDBC_INFINITY = 9223372036854775807  # Sentinel for unreachable vertices


def compute_ldbc_scale(num_vertices: int, num_edges: int) -> float:
    """Compute LDBC scale factor: s(V, E) = log10(|V| + |E|)."""
    return round(math.log10(num_vertices + num_edges), 1)


def epsilon_match(reference: float, system: float, epsilon: float = LDBC_EPSILON) -> bool:
    """Check if two float values match within epsilon tolerance.

    LDBC spec: |reference - system| <= epsilon * |reference|
    Special case: equality required when both values are zero.
    """
    if reference == 0 and system == 0:
        return True
    if reference == 0:
        return abs(system) <= epsilon
    return abs(reference - system) <= epsilon * abs(reference)


class LdbcGraphalyticsBenchmarkBase(BaseBenchmark):
    """Base class for LDBC Graphanalytics benchmarks.

    Uses a consistent graph setup matching LDBC specifications:
    - Directed graph with unique edges
    - No self-loops or multi-edges
    - Vertices identified by unique IDs
    - Edge weights for SSSP (non-negative floats)

    Instance variables are used to avoid class-level state issues.
    """

    category = "ldbc_graphanalytics"

    def __init__(self, seed: int = 42) -> None:
        """Initialize benchmark with reproducible random seed."""
        self._seed = seed
        self._node_ids: list[str] = []
        self._node_count: int = 0
        self._edge_count: int = 0
        self._rng = random.Random(seed)

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup graph for LDBC Graphanalytics benchmarks.

        Creates a directed graph with the following characteristics:
        - Scale-appropriate number of vertices and edges
        - No self-loops or multi-edges
        - Edge weights in range [0.1, 10.0] for SSSP
        - Reproducible structure via seeded random generation
        """
        adapter.clear()

        # Reset RNG for reproducibility
        self._rng = random.Random(self._seed)

        # Scale vertices and edges based on scale config
        # Use a portion of the full scale for reasonable benchmark times
        self._node_count = min(10000, max(100, scale.nodes // 10))
        self._edge_count = min(50000, max(500, scale.edges // 10))

        # Create vertices
        nodes = [{"id": f"v_{i}", "name": f"Vertex {i}"} for i in range(self._node_count)]
        adapter.insert_nodes(nodes, label="Vertex")

        # Create directed edges (no self-loops, no multi-edges)
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[str, str]] = set()

        # Use deterministic edge generation for reproducibility
        for i in range(self._edge_count * 2):  # Generate extra to account for duplicates
            if len(edges) >= self._edge_count:
                break

            # Deterministic but varied edge selection
            src_idx = i % self._node_count
            tgt_idx = (i * 7 + 3) % self._node_count

            if src_idx != tgt_idx:  # No self-loops
                src = f"v_{src_idx}"
                tgt = f"v_{tgt_idx}"

                if (src, tgt) not in edge_set:  # No multi-edges
                    edge_set.add((src, tgt))
                    # Non-negative weight for SSSP (LDBC spec)
                    weight = round(self._rng.uniform(0.1, 10.0), 2)
                    edges.append((src, tgt, "EDGE", {"weight": weight}))

        adapter.insert_edges(edges)
        self._node_ids = [f"v_{i}" for i in range(self._node_count)]
        self._edge_count = len(edges)

        # Log scale factor for reference
        scale_factor = compute_ldbc_scale(self._node_count, self._edge_count)
        # Scale factor ~4.0 for 10K nodes + 50K edges

    def _get_source_vertex(self, index: int = 0) -> str:
        """Get a deterministic source vertex for algorithms that need one.

        Uses index to allow multiple runs with different sources while
        maintaining reproducibility.
        """
        if not self._node_ids:
            return "v_0"
        return self._node_ids[index % len(self._node_ids)]


# =============================================================================
# BFS - Breadth-First Search
# =============================================================================


@BenchmarkRegistry.register("ldbc_bfs", category="ldbc_graphanalytics")
class LdbcBfsBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics BFS benchmark.

    Breadth-First Search labels each vertex with its distance (depth)
    from a given source vertex:
    - Source vertex has depth 0
    - Direct neighbors have depth 1
    - Their neighbors have depth 2, etc.
    - Unreachable vertices have depth = LDBC_INFINITY (sentinel value)

    Validation: Exact match (output values must be identical to reference).
    """

    @property
    def name(self) -> str:
        return "ldbc_bfs"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run BFS from a deterministic source vertex.

        Returns the number of vertices with assigned depth levels.
        """
        # Use first vertex as source (deterministic)
        source = self._get_source_vertex(0)
        levels = adapter.bfs_levels(source)
        return len(levels)


# =============================================================================
# PageRank
# =============================================================================


@BenchmarkRegistry.register("ldbc_pagerank", category="ldbc_graphanalytics")
class LdbcPageRankBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics PageRank benchmark.

    PageRank iteratively assigns ranking values to vertices:
    - Initial value: 1/|V| for all vertices
    - Damping factor: 0.85 (LDBC spec)
    - Fixed number of iterations: 20 (LDBC spec uses iteration count, not convergence)
    - Sink vertices (no outgoing edges) redistribute their rank

    Uses 64-bit IEEE 754 double-precision floating-point.
    Validation: Epsilon match (tolerance: 0.0001 relative error).
    """

    @property
    def name(self) -> str:
        return "ldbc_pagerank"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute PageRank with LDBC parameters.

        Returns the number of vertices with PageRank scores.
        """
        scores = adapter.pagerank(
            damping=LDBC_DAMPING_FACTOR,
            max_iterations=LDBC_PAGERANK_ITERATIONS,
            tolerance=1e-10,  # Very tight tolerance to force full iterations
        )
        return len(scores)


# =============================================================================
# WCC - Weakly Connected Components
# =============================================================================


@BenchmarkRegistry.register("ldbc_wcc", category="ldbc_graphanalytics")
class LdbcWccBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics Weakly Connected Components benchmark.

    Finds all weakly connected components and assigns each vertex
    a component label:
    - Two vertices are in the same component if connected by any path
    - For directed graphs, edges are treated as undirected
    - Component labels need not be sequential

    Validation: Equivalence match (label mappings must be logically equivalent,
    actual label values may differ).
    """

    @property
    def name(self) -> str:
        return "ldbc_wcc"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find weakly connected components.

        Returns total number of vertices assigned to components.
        """
        components = adapter.weakly_connected_components()
        return sum(len(c) for c in components)


# =============================================================================
# CDLP - Community Detection via Label Propagation
# =============================================================================


@BenchmarkRegistry.register("ldbc_cdlp", category="ldbc_graphanalytics")
class LdbcCdlpBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics Community Detection (Label Propagation) benchmark.

    CDLP assigns community labels to vertices through iterative propagation:
    - Initially each vertex has label = its ID
    - Each iteration: vertex adopts most frequent neighbor label
    - Ties broken by smallest label (deterministic variant)
    - Synchronous updates (all vertices update based on previous iteration)
    - Vertices without neighbors retain current labels
    - Bidirectional edges counted twice

    Validation: Exact match (output values must be identical).
    """

    @property
    def name(self) -> str:
        return "ldbc_cdlp"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run community detection with label propagation.

        Returns the number of communities detected.
        """
        communities = adapter.community_detection(algorithm="label_propagation")
        return len(communities)


# =============================================================================
# LCC - Local Clustering Coefficient
# =============================================================================


@BenchmarkRegistry.register("ldbc_lcc", category="ldbc_graphanalytics")
class LdbcLccBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics Local Clustering Coefficient benchmark.

    LCC measures how close a vertex's neighbors are to being a clique:
    - LCC(v) = (edges between neighbors) / (max possible edges between neighbors)
    - For vertex v with neighbors N(v):
      - If |N(v)| <= 1: LCC(v) = 0
      - Otherwise: LCC(v) = 2 * |edges in N(v)| / (|N(v)| * (|N(v)| - 1))
    - For directed graphs, neighborhood is undirected but edge counting is directed

    Uses 64-bit IEEE 754 double-precision floating-point.
    Validation: Epsilon match (tolerance: 0.0001 relative error).
    """

    @property
    def name(self) -> str:
        return "ldbc_lcc"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute local clustering coefficients.

        Returns the number of vertices with computed coefficients.
        """
        coefficients = adapter.local_clustering_coefficient()
        return len(coefficients)


# =============================================================================
# SSSP - Single-Source Shortest Paths
# =============================================================================


@BenchmarkRegistry.register("ldbc_sssp", category="ldbc_graphanalytics")
class LdbcSsspBenchmark(LdbcGraphalyticsBenchmarkBase):
    """LDBC Graphanalytics Single-Source Shortest Paths benchmark.

    SSSP computes shortest path distances from a source vertex to all others:
    - Path length = sum of edge weights
    - Unreachable vertices have distance = infinity
    - Edge weights are non-negative floats (zero allowed, negative prohibited)

    Uses 64-bit IEEE 754 double-precision floating-point.
    Validation: Epsilon match (tolerance: 0.0001 relative error).
    """

    @property
    def name(self) -> str:
        return "ldbc_sssp"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Compute SSSP from a deterministic source vertex.

        Returns the number of vertices with computed distances.
        """
        # Use first vertex as source (deterministic)
        source = self._get_source_vertex(0)
        distances = adapter.sssp(source)
        return len(distances)
