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
from typing import Any, ClassVar

from graph_bench.types import UnsupportedOperation

__all__ = ["AdapterRegistry", "BaseAdapter", "ldbc_directed_lcc"]


def ldbc_directed_lcc(graph: Any) -> dict[str, float]:
    """LDBC Graphalytics local clustering coefficient of a directed NetworkX graph.

    For vertex v, N(v) is the set of in- and out-neighbours (v excluded). The
    coefficient is the number of directed edges (u, w) with u and w in N(v),
    divided by |N(v)| * (|N(v)| - 1); it is 0 when |N(v)| < 2. An edge in each
    direction between two neighbours counts twice. This is not the Fagiolo
    coefficient that ``nx.clustering`` computes on a DiGraph.
    """
    successors = {v: set(graph.successors(v)) - {v} for v in graph}
    result: dict[str, float] = {}
    for v in graph:
        neighbours = (successors[v] | set(graph.predecessors(v))) - {v}
        degree = len(neighbours)
        if degree < 2:
            result[str(v)] = 0.0
            continue
        edges = sum(len(successors[u] & neighbours) for u in neighbours)
        result[str(v)] = edges / (degree * (degree - 1))
    return result


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

    # Clustering coefficient definitions ("directed", "undirected") the database
    # computes natively; local_clustering_coefficient() falls back for the others.
    native_lcc_definitions: ClassVar[frozenset[str]] = frozenset()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    # ── Fallback tracking ───────────────────────────────────────

    def _record_fallback(self, operation: str) -> None:
        """Note that ``operation`` ran in Python instead of in the database.

        Every BaseAdapter fallback calls it, and so must an adapter's own Python
        path (for example a BFS over get_neighbors). Benchmarks report what
        they measured from it (see ``take_fallbacks``).
        """
        self.__dict__.setdefault("_fallbacks", set()).add(operation)

    def take_fallbacks(self) -> set[str]:
        """Return the operations that used a fallback since the last call, and reset."""
        return self.__dict__.pop("_fallbacks", set())

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

    @property
    def child_pids(self) -> list[int]:
        """PIDs of child/daemon processes owned by this adapter.

        Override in subclasses that spawn external processes (e.g. redislite
        spawns a detached redis-server). These PIDs are included in memory
        measurement for embedded adapters.
        """
        return []

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
        self._record_fallback("traverse_bfs")
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
        self._record_fallback("traverse_dfs")
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
        A failing read raises: an empty graph would be timed as a fast answer.
        """
        import networkx as nx

        G = nx.DiGraph()
        visited: set[str] = set()
        to_visit: list[str] = []

        # Get starting nodes from common labels
        for label in ["Person", "Node", "Vertex"]:
            for node in self.get_nodes_by_label(label, limit=5000):
                node_id = node.get("id")
                if node_id and str(node_id) not in visited:
                    to_visit.append(str(node_id))
            if to_visit:
                break

        # BFS to discover all nodes and edges
        while to_visit:
            node_id = to_visit.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            G.add_node(node_id)

            for neighbor in self.get_neighbors(node_id):
                neighbor_str = str(neighbor)
                G.add_edge(node_id, neighbor_str)
                if neighbor_str not in visited:
                    to_visit.append(neighbor_str)

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

        Runs the power iteration of ``nx.pagerank`` (unweighted, the rank of
        sink nodes spread evenly), but returns the scores after
        ``max_iterations`` when they have not converged: LDBC Graphalytics
        PageRank runs a fixed number of iterations, while ``nx.pagerank``
        raises ``PowerIterationFailedConvergence`` instead.
        """
        import networkx as nx
        import numpy as np

        self._record_fallback("pagerank")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        nodes = list(G)
        n = len(nodes)
        adjacency = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, dtype=float)
        out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
        is_sink = out_degree == 0
        inverse_degree = np.divide(1.0, out_degree, out=np.zeros(n), where=~is_sink)

        scores = np.full(n, 1.0 / n)
        for _ in range(max_iterations):
            previous = scores
            spread = (previous * inverse_degree) @ adjacency
            scores = damping * (spread + previous[is_sink].sum() / n) + (1 - damping) / n
            # Same convergence test as nx.pagerank
            if np.abs(scores - previous).sum() < n * tolerance:
                break
        return {str(node): float(score) for node, score in zip(nodes, scores, strict=True)}

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Detect communities in the graph.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.

        For label_propagation: uses LDBC-compliant synchronous algorithm
        where all nodes update based on previous iteration's labels
        (not NetworkX's asynchronous variant).
        """
        from networkx.algorithms.community import louvain_communities

        self._record_fallback("community_detection")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return []

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        if algorithm == "louvain":
            communities = louvain_communities(G_undirected)
            return [set(str(n) for n in c) for c in communities]
        elif algorithm == "label_propagation":
            return self._synchronous_label_propagation(G_undirected)
        else:
            msg = f"Unknown community detection algorithm: {algorithm}"
            raise ValueError(msg)

    @staticmethod
    def _synchronous_label_propagation(
        G: Any, max_iterations: int = 10,
    ) -> list[set[str]]:
        """LDBC-compliant synchronous Community Detection Label Propagation.

        Per LDBC Graphanalytics spec: all nodes update simultaneously
        based on the previous iteration's labels. Ties broken by
        selecting the smallest label.
        """
        # Initialize: each node's label = its own id
        labels: dict[Any, Any] = {node: node for node in G.nodes()}

        for _ in range(max_iterations):
            new_labels: dict[Any, Any] = {}
            for node in G.nodes():
                neighbor_labels = [labels[n] for n in G.neighbors(node)]
                if not neighbor_labels:
                    new_labels[node] = labels[node]
                    continue
                # Count label frequencies
                freq: dict[Any, int] = {}
                for lbl in neighbor_labels:
                    freq[lbl] = freq.get(lbl, 0) + 1
                max_freq = max(freq.values())
                # Break ties by smallest label
                candidates = [
                    lbl for lbl, cnt in freq.items() if cnt == max_freq
                ]
                new_labels[node] = min(candidates)
            if new_labels == labels:
                break  # converged
            labels = new_labels

        # Group nodes by label
        communities: dict[Any, set[str]] = {}
        for node, lbl in labels.items():
            communities.setdefault(lbl, set()).add(str(node))
        return list(communities.values())

    def local_clustering_coefficient(self, *, directed: bool = True) -> dict[str, float]:
        """Compute the Local Clustering Coefficient of every vertex.

        ``directed=True`` is the LDBC Graphalytics definition for a directed
        graph (see ``ldbc_directed_lcc``). ``directed=False`` treats every edge
        as undirected: edges between neighbours over d * (d - 1) / 2.

        Default uses the NetworkX graph extracted from the database. Adapters
        list the definitions they compute natively in ``native_lcc_definitions``
        and call this for the others.
        """
        import networkx as nx

        self._record_fallback("local_clustering_coefficient")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}
        if directed:
            return ldbc_directed_lcc(G)
        coefficients = nx.clustering(nx.Graph(G))
        return {str(k): float(v) for k, v in coefficients.items()}

    def weakly_connected_components(self) -> list[set[str]]:
        """Find Weakly Connected Components.

        LDBC Graphanalytics WCC: assigns each vertex a component label.
        Two vertices are in the same component if connected (ignoring direction).

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        self._record_fallback("weakly_connected_components")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return []

        components = list(nx.weakly_connected_components(G))
        return [set(str(n) for n in c) for c in components]

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """Single-Source Shortest Paths with weights.

        LDBC Graphanalytics SSSP: shortest path distances from source to all vertices.
        Unreachable vertices get infinity per spec.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        self._record_fallback("sssp")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        try:
            distances = nx.single_source_dijkstra_path_length(
                G, source, weight=weight_attr,
            )
            # Include all vertices; unreachable ones get infinity
            result: dict[str, float] = {}
            for node in G.nodes():
                result[str(node)] = float(distances.get(node, float("inf")))
            return result
        except nx.NetworkXError:
            return {}

    def bfs_levels(self, source: str) -> dict[str, int]:
        """BFS with level/depth tracking.

        LDBC Graphanalytics BFS: assigns each vertex its distance from source.
        Unreachable vertices get LDBC_INFINITY (2^63 - 1) per spec.

        Default uses NetworkX fallback (extracts graph from database).
        Override for native database support.
        """
        import networkx as nx

        LDBC_INFINITY = 9223372036854775807

        self._record_fallback("bfs_levels")
        G = self._build_networkx_graph()
        if len(G) == 0:
            return {}

        try:
            distances = dict(nx.single_source_shortest_path_length(G, source))
            # Include all vertices; unreachable ones get LDBC_INFINITY
            result: dict[str, int] = {}
            for node in G.nodes():
                result[str(node)] = distances.get(node, LDBC_INFINITY)
            return result
        except nx.NetworkXError:
            return {}

    # ── Vector search ───────────────────────────────────────────
    #
    # Two separate paths, so a result always says which one it measured:
    # vector_scan() is the same brute-force scan for every adapter (exact),
    # create_vector_index() / vector_ann_search() are the database's native
    # approximate nearest neighbour search and have no fallback.

    def create_vector_index(
        self,
        label: str,
        property_name: str,
        *,
        dimensions: int = 128,
        metric: str = "cosine",
        m: int | None = None,
        ef_construction: int | None = None,
        quantization: str | None = None,
    ) -> None:
        """Build the database's native ANN index (for example HNSW) on a node property.

        ``m`` and ``ef_construction`` are the HNSW parameters; ``quantization``
        is a database-specific compression mode. Raises UnsupportedOperation
        when the database has no native vector index: there is no fallback.
        """
        raise UnsupportedOperation(f"{self.name} has no native vector index")

    def vector_ann_search(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
        ef: int | None = None,
    ) -> list[tuple[str, float]]:
        """Approximate k nearest neighbours through the native index, best first.

        Returns ``(id, similarity)`` pairs; ``ef`` is the search beam width.
        Raises UnsupportedOperation when the database has no native vector
        search: there is no fallback.
        """
        raise UnsupportedOperation(f"{self.name} has no native vector search")

    def delete_nodes(self, node_ids: Sequence[str]) -> int:
        """Delete nodes (and their edges) by ``id`` and return how many were deleted.

        Used by the ANN delete benchmark; raises UnsupportedOperation by default.
        """
        raise UnsupportedOperation(f"the {self.name} adapter does not implement delete_nodes")

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def vector_scan(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Exact k nearest neighbours by cosine similarity, with a brute-force scan.

        Reads every node of ``label`` from the database and scores it in
        Python: the same work for every adapter, so it must not be overridden.
        """
        self._record_fallback("vector_scan")
        nodes = self.get_nodes_by_label(label, limit=100_000)
        scored: list[tuple[str, float]] = []
        for node in nodes:
            node_id = node.get("id")
            vec = node.get(property_name)
            if node_id is not None and isinstance(vec, list):
                score = self._cosine_similarity(query_vector, vec)
                scored.append((str(node_id), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def vector_search(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors to a query vector (the hybrid benchmarks use it).

        Default is the brute-force ``vector_scan``.
        """
        return self.vector_scan(query_vector, label=label, property_name=property_name, k=k)

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
