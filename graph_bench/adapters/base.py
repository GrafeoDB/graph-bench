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

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """Compute PageRank scores.

        Default: raise NotImplementedError.
        Override for native database support.
        """
        raise NotImplementedError(f"{self.name} does not support native PageRank")

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Detect communities in the graph.

        Default: raise NotImplementedError.
        Override for native database support.
        """
        raise NotImplementedError(f"{self.name} does not support native community detection")

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
