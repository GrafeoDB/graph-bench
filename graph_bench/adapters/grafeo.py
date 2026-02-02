r"""
Grafeo database adapter using GQL (ISO graph query language).

Grafeo is an embedded graph database with Python bindings.
Uses GQL as the default query language (ISO/IEC 39075).

Requires: pip install grafeo

Environment variables:
    GRAPH_BENCH_GRAFEO_PATH: Database path (default: ./data/grafeo)

    from graph_bench.adapters.grafeo import GrafeoAdapter

    adapter = GrafeoAdapter()
    adapter.connect(path="./data/grafeo")

    # GQL queries
    adapter.execute_query("MATCH (n:Person) RETURN n.name")
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["GrafeoAdapter"]


@AdapterRegistry.register("grafeo")
class GrafeoAdapter(BaseAdapter):
    """Grafeo embedded graph database adapter."""

    def __init__(self) -> None:
        self._db: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "Grafeo"

    @property
    def version(self) -> str:
        try:
            import grafeo

            return grafeo.__version__
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from grafeo import GrafeoDB
        except ImportError as e:
            msg = "grafeo package not installed. Install with: pip install grafeo"
            raise ImportError(msg) from e

        path = uri or kwargs.get("path") or get_env("GRAFEO_PATH")

        # Handle in-memory mode (empty path, None, or :memory:)
        if path and path != ":memory:":
            Path(path).mkdir(parents=True, exist_ok=True)
            self._db = GrafeoDB(path)
        else:
            self._db = GrafeoDB()  # In-memory

        self._connected = True

    def disconnect(self) -> None:
        self._db = None
        self._connected = False

    def clear(self) -> None:
        self._db.execute("MATCH (n) DETACH DELETE n")

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            for node in batch:
                props = dict(node)
                self._db.create_node([label], props)
                count += 1
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        # Query returns internal node ID, use get_node() to fetch properties
        result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) AS nid", {"id": node_id})
        for row in result:
            nid = row["nid"]
            node_obj = self._db.get_node(nid)
            return node_obj.properties()
        return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        result = self._db.execute(f"MATCH (n:{label}) RETURN id(n) AS nid LIMIT {limit}")
        nodes = []
        for row in result:
            nid = row["nid"]
            node_obj = self._db.get_node(nid)
            nodes.append(node_obj.properties())
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for src, tgt, edge_type, props in edges:
            src_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": src})
            tgt_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": tgt})

            src_id = None
            tgt_id = None
            for row in src_result:
                src_id = row["nid"]
            for row in tgt_result:
                tgt_id = row["nid"]

            if src_id is not None and tgt_id is not None:
                self._db.create_edge(src_id, tgt_id, edge_type, props)
                count += 1
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if edge_type:
            query = f"MATCH (n {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
        else:
            query = "MATCH (n {id: $id})-[]->(m) RETURN m.id AS id"

        result = self._db.execute(query, {"id": node_id})
        return [row["id"] for row in result if row["id"]]

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        try:
            if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "bfs"):
                start_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": start})
                for row in start_result:
                    start_nid = row["nid"]
                    result = self._db.algorithms.bfs(start_nid, max_depth=max_depth)
                    return [str(n) for n in result]
        except Exception:
            pass

        return super().traverse_bfs(start, max_depth=max_depth, edge_type=edge_type)

    def traverse_dfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        try:
            if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "dfs"):
                start_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": start})
                for row in start_result:
                    start_nid = row["nid"]
                    result = self._db.algorithms.dfs(start_nid, max_depth=max_depth)
                    return [str(n) for n in result]
        except Exception:
            pass

        return super().traverse_dfs(start, max_depth=max_depth, edge_type=edge_type)

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        try:
            if hasattr(self._db, "algorithms"):
                src_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": source})
                tgt_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": target})

                src_nid = None
                tgt_nid = None
                for row in src_result:
                    src_nid = row["nid"]
                for row in tgt_result:
                    tgt_nid = row["nid"]

                if src_nid is not None and tgt_nid is not None:
                    if weighted and hasattr(self._db.algorithms, "dijkstra"):
                        result = self._db.algorithms.dijkstra(src_nid, tgt_nid)
                    elif hasattr(self._db.algorithms, "shortest_path"):
                        result = self._db.algorithms.shortest_path(src_nid, tgt_nid)
                    else:
                        result = None

                    if result:
                        return [str(n) for n in result]
        except Exception:
            pass

        from collections import deque

        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.get_neighbors(current, edge_type=edge_type):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._db.execute(query, params or {})
        return [dict(row) for row in result]

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"

        result = self._db.execute(query)
        for row in result:
            return row["count"]
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"

        result = self._db.execute(query)
        for row in result:
            return row["count"]
        return 0

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "pagerank"):
            result = self._db.algorithms.pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)
            return {str(k): v for k, v in result.items()}
        raise NotImplementedError(f"{self.name} does not support native PageRank")

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        if hasattr(self._db, "algorithms"):
            if algorithm == "louvain" and hasattr(self._db.algorithms, "louvain"):
                result = self._db.algorithms.louvain()
                return [{str(n) for n in community} for community in result]
            elif algorithm == "label_propagation" and hasattr(self._db.algorithms, "label_propagation"):
                result = self._db.algorithms.label_propagation()
                return [{str(n) for n in community} for community in result]
        raise NotImplementedError(f"{self.name} does not support native community detection with {algorithm}")
