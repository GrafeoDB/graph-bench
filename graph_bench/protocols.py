r"""
Protocol definitions for graph database adapters and benchmarks.

All adapters must implement GraphDatabaseAdapter protocol.
All benchmarks must implement Benchmark protocol.

    from graph_bench.protocols import GraphDatabaseAdapter, Benchmark

    class MyAdapter(GraphDatabaseAdapter):
        ...
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig

__all__ = [
    "GraphDatabaseAdapter",
    "Benchmark",
    "DatasetLoader",
    "ResultCollector",
]


@runtime_checkable
class GraphDatabaseAdapter(Protocol):
    """Protocol for graph database adapters.

    Each database (Neo4j, Memgraph, Kuzu, ArangoDB, Grafeo) must implement
    this protocol to be compatible with the benchmark suite.
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    @property
    def version(self) -> str:
        """Database version string."""
        ...

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        """Establish connection to the database."""
        ...

    def disconnect(self) -> None:
        """Close connection to the database."""
        ...

    def clear(self) -> None:
        """Remove all data from the database."""
        ...

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        """Insert nodes and return count inserted."""
        ...

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a node by ID."""
        ...

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve nodes by label."""
        ...

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        """Insert edges (src, tgt, type, props) and return count inserted."""
        ...

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        """Get neighbor node IDs."""
        ...

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """BFS traversal returning visited node IDs."""
        ...

    def traverse_dfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """DFS traversal returning visited node IDs."""
        ...

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        """Find shortest path between nodes."""
        ...

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a native query (Cypher, AQL, etc.)."""
        ...

    def count_nodes(self, *, label: str | None = None) -> int:
        """Count nodes, optionally filtered by label."""
        ...

    def count_edges(self, *, edge_type: str | None = None) -> int:
        """Count edges, optionally filtered by type."""
        ...

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """Compute PageRank scores."""
        ...

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Detect communities in the graph."""
        ...

    def create_vector_index(
        self,
        label: str,
        property_name: str,
        *,
        dimensions: int = 128,
        metric: str = "cosine",
    ) -> None:
        """Create a vector similarity index on a node property."""
        ...

    def vector_search(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors to a query vector."""
        ...


@runtime_checkable
class Benchmark(Protocol):
    """Protocol for benchmark implementations."""

    @property
    def name(self) -> str:
        """Benchmark name."""
        ...

    @property
    def category(self) -> str:
        """Benchmark category (storage, traversal, algorithm, query)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Prepare benchmark (load data, create indices, etc.)."""
        ...

    def run(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> Metrics:
        """Execute the benchmark and return metrics."""
        ...

    def teardown(self, adapter: GraphDatabaseAdapter) -> None:
        """Clean up after benchmark."""
        ...


@runtime_checkable
class DatasetLoader(Protocol):
    """Protocol for dataset loaders."""

    @property
    def name(self) -> str:
        """Dataset name."""
        ...

    def generate(self, scale: ScaleConfig) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate or load dataset, returning (nodes, edges)."""
        ...

    def load_into(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Load dataset directly into adapter."""
        ...


@runtime_checkable
class ResultCollector(Protocol):
    """Protocol for collecting and exporting results."""

    def add_result(self, result: BenchmarkResult[Any]) -> None:
        """Add a benchmark result."""
        ...

    def export_json(self, path: str) -> None:
        """Export results to JSON."""
        ...

    def export_csv(self, path: str) -> None:
        """Export results to CSV."""
        ...

    def export_markdown(self, path: str) -> None:
        """Export results as Markdown report."""
        ...
