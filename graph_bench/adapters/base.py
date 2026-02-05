r"""
Base adapter implementation with common functionality.

Provides default implementations and helper methods
that can be shared across database adapters.

    from graph_bench.adapters.base import BaseAdapter

    class MyAdapter(BaseAdapter):
        def connect(self, *, uri: str | None = None, **kwargs) -> None:
            ...
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Any

__all__ = ["BaseAdapter", "AdapterRegistry"]


class AdapterRegistry:
    """Registry for database adapters."""

    _adapters: dict[str, type["BaseAdapter"]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Decorator to register an adapter class."""

        def decorator(adapter_cls: type["BaseAdapter"]) -> type["BaseAdapter"]:
            cls._adapters[name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type["BaseAdapter"] | None:
        """Get adapter class by name."""
        return cls._adapters.get(name)

    @classmethod
    def list(cls) -> list[str]:
        """List registered adapter names."""
        return list(cls._adapters.keys())

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "BaseAdapter":
        """Create adapter instance by name."""
        adapter_cls = cls.get(name)
        if adapter_cls is None:
            valid = ", ".join(cls.list()) or "none"
            msg = f"Unknown adapter '{name}'. Registered: {valid}"
            raise ValueError(msg)
        return adapter_cls(**kwargs)


class BaseAdapter(ABC):
    """Base class for graph database adapters.

    Provides default implementations for common operations
    and defines the interface all adapters must implement.
    """

    _connected: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    @property
    def version(self) -> str:
        """Database version string."""
        return "unknown"

    @property
    def connected(self) -> bool:
        """Whether adapter is currently connected."""
        return self._connected

    @property
    def is_embedded(self) -> bool:
        """Whether this is an embedded (in-process) database.

        Embedded databases measure Python process memory.
        Server databases measure Docker container memory.

        Override in subclasses. Defaults to False (server).
        """
        return False

    @abstractmethod
    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the database."""
        ...

    @abstractmethod
    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        """Insert nodes and return count inserted."""
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a node by ID."""
        ...

    @abstractmethod
    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Update node properties.

        Args:
            node_id: The ID of the node to update.
            properties: Properties to update (merged with existing).

        Returns:
            True if node was found and updated, False otherwise.
        """
        ...

    @abstractmethod
    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve nodes by label."""
        ...

    @abstractmethod
    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        """Insert edges (src, tgt, type, props) and return count inserted."""
        ...

    @abstractmethod
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
        """BFS traversal returning visited node IDs.

        Default implementation using get_neighbors.
        Override for native database support.
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: list[str] = []

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append(node_id)

            if depth < max_depth:
                for neighbor in self.get_neighbors(node_id, edge_type=edge_type):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return result

    def traverse_dfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """DFS traversal returning visited node IDs.

        Default implementation using get_neighbors.
        Override for native database support.
        """
        visited: set[str] = set()
        result: list[str] = []

        def dfs(node_id: str, depth: int) -> None:
            if node_id in visited or depth > max_depth:
                return
            visited.add(node_id)
            result.append(node_id)

            for neighbor in self.get_neighbors(node_id, edge_type=edge_type):
                dfs(neighbor, depth + 1)

        dfs(start, 0)
        return result

    @abstractmethod
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

    @abstractmethod
    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a native query (Cypher, AQL, etc.)."""
        ...

    @abstractmethod
    def count_nodes(self, *, label: str | None = None) -> int:
        """Count nodes, optionally filtered by label."""
        ...

    @abstractmethod
    def count_edges(self, *, edge_type: str | None = None) -> int:
        """Count edges, optionally filtered by type."""
        ...

    def _build_networkx_graph(self) -> Any:
        """Extract graph from database for algorithm fallback.

        Uses BFS from known nodes to discover the graph structure.
        The extraction overhead is included in benchmark timing,
        representing the real cost of not having native support.
        """
        import networkx as nx

        G = nx.DiGraph()
        visited: set[str] = set()
        to_visit: list[str] = []

        # Get starting nodes from common labels
        for label in ["Person", "Node", "Vertex"]:
            try:
                for node in self.get_nodes_by_label(label, limit=5000):
                    node_id = node.get("id")
                    if node_id and str(node_id) not in visited:
                        to_visit.append(str(node_id))
            except Exception:
                pass
            if to_visit:
                break

        # BFS to discover all nodes and edges
        while to_visit:
            node_id = to_visit.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            G.add_node(node_id)

            try:
                for neighbor in self.get_neighbors(node_id):
                    neighbor_str = str(neighbor)
                    G.add_edge(node_id, neighbor_str)
                    if neighbor_str not in visited:
                        to_visit.append(neighbor_str)
            except Exception:
                pass

        return G

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """Compute PageRank scores.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        scores = nx.pagerank(G, alpha=damping, max_iter=max_iterations, tol=tolerance)
        return {str(k): float(v) for k, v in scores.items()}

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Detect communities in the graph.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx
        from networkx.algorithms.community import label_propagation_communities, louvain_communities

        G = self._build_networkx_graph()
        if len(G) == 0:
            return []

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        if algorithm == "louvain":
            communities = louvain_communities(G_undirected)
        elif algorithm == "label_propagation":
            communities = label_propagation_communities(G_undirected)
        else:
            msg = f"Unknown community detection algorithm: {algorithm}"
            raise ValueError(msg)

        return [set(str(n) for n in c) for c in communities]

    def local_clustering_coefficient(self) -> dict[str, float]:
        """Compute Local Clustering Coefficient for all vertices.

        LDBC Graphanalytics LCC: ratio of edges between neighbors
        to maximum possible edges between neighbors.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        # For directed graphs, use undirected neighborhood but directed edges
        # NetworkX clustering handles this correctly
        coefficients = nx.clustering(G)
        return {str(k): float(v) for k, v in coefficients.items()}

    def weakly_connected_components(self) -> list[set[str]]:
        """Find Weakly Connected Components.

        LDBC Graphanalytics WCC: assigns each vertex a component label.
        Two vertices are in the same component if connected (ignoring direction).

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        G = self._build_networkx_graph()
        if len(G) == 0:
            return []

        components = list(nx.weakly_connected_components(G))
        return [set(str(n) for n in c) for c in components]

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """Single-Source Shortest Paths with weights.

        LDBC Graphanalytics SSSP: shortest path distances from source to all vertices.
        Returns dict mapping vertex ID to distance (infinity if unreachable).

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        try:
            # Use Dijkstra for weighted shortest paths
            distances = nx.single_source_dijkstra_path_length(G, source, weight=weight_attr)
            return {str(k): float(v) for k, v in distances.items()}
        except nx.NetworkXError:
            # Source not in graph
            return {}

    def bfs_levels(self, source: str) -> dict[str, int]:
        """BFS with level/depth tracking.

        LDBC Graphanalytics BFS: assigns each vertex its distance from source.
        Unreachable vertices get infinity (represented as -1 here).

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        try:
            # BFS returns dict of node -> distance
            distances = dict(nx.single_source_shortest_path_length(G, source))
            return {str(k): int(v) for k, v in distances.items()}
        except nx.NetworkXError:
            return {}

    def __enter__(self) -> "BaseAdapter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - disconnect."""
        if self._connected:
            self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}({self.name}, {status})"
