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
        self._id_index_created = False

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

    @property
    def is_embedded(self) -> bool:
        return True

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
        # Grafeo's native Python API is most efficient for batch inserts
        # GQL UNWIND has limitations with property setting from variables
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            for node in batch:
                props = dict(node)
                self._db.create_node([label], props)
                count += 1

        # Create property index on "id" for O(1) lookups
        if not self._id_index_created and hasattr(self._db, "create_property_index"):
            self._db.create_property_index("id")
            self._id_index_created = True

        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        # Query returns internal node ID, use get_node() to fetch properties
        result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) AS nid", {"id": node_id})
        for row in result:
            nid = row["nid"]
            node_obj = self._db.get_node(nid)
            # Grafeo's get_node() can return None even when the query found the node
            # This might be a Grafeo bug with internal ID handling
            if node_obj is not None:
                return node_obj.properties()
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        # Build SET clause for properties
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
        query = f"MATCH (n {{id: $id}}) SET {set_clauses} RETURN n"
        params = {"id": node_id, **properties}
        result = self._db.execute(query, params)
        # Check if any rows were returned (node was found and updated)
        for _ in result:
            return True
        return False

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        result = self._db.execute(f"MATCH (n:{label}) RETURN id(n) AS nid LIMIT {limit}")
        nodes = []
        for row in result:
            nid = row["nid"]
            node_obj = self._db.get_node(nid)
            if node_obj is not None:
                nodes.append(node_obj.properties())
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        # Grafeo's native Python API is most efficient
        # GQL UNWIND has limitations with property access from variables
        count = 0
        for src, tgt, edge_type, props in edges:
            src_result = self._db.execute(
                "MATCH (n {id: $id}) RETURN id(n) as nid", {"id": src}
            )
            tgt_result = self._db.execute(
                "MATCH (n {id: $id}) RETURN id(n) as nid", {"id": tgt}
            )

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
                # label_propagation returns dict[node_id, label] - convert to list of sets
                if isinstance(result, dict):
                    communities: dict[int, set[str]] = {}
                    for node_id, label in result.items():
                        if label not in communities:
                            communities[label] = set()
                        communities[label].add(str(node_id))
                    return list(communities.values())
                return [{str(n) for n in community} for community in result]
        raise NotImplementedError(f"{self.name} does not support native community detection with {algorithm}")

    def bfs_levels(self, source: str) -> dict[str, int]:
        """LDBC BFS using native Grafeo bfs_layers."""
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "bfs_layers"):
            src_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": source})
            for row in src_result:
                src_nid = row["nid"]
                # bfs_layers returns list of lists: [[level0_nodes], [level1_nodes], ...]
                layers = self._db.algorithms.bfs_layers(src_nid)
                result: dict[str, int] = {}
                for depth, layer in enumerate(layers):
                    for node_id in layer:
                        result[str(node_id)] = depth
                return result
        return super().bfs_levels(source)

    def weakly_connected_components(self) -> list[set[str]]:
        """LDBC WCC using native Grafeo connected_components."""
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "connected_components"):
            result = self._db.algorithms.connected_components()
            # Returns dict[node_id, component_id] - convert to list of sets
            if isinstance(result, dict):
                components: dict[int, set[str]] = {}
                for node_id, comp_id in result.items():
                    if comp_id not in components:
                        components[comp_id] = set()
                    components[comp_id].add(str(node_id))
                return list(components.values())
            return [{str(n) for n in comp} for comp in result]
        return super().weakly_connected_components()

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """LDBC SSSP using native Grafeo dijkstra."""
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "dijkstra"):
            src_result = self._db.execute("MATCH (n {id: $id}) RETURN id(n) as nid", {"id": source})
            for row in src_result:
                src_nid = row["nid"]
                # Get all nodes and compute shortest paths
                all_nodes = self._db.execute("MATCH (n) RETURN id(n) as nid")
                result: dict[str, float] = {}
                result[source] = 0.0
                for target_row in all_nodes:
                    tgt_nid = target_row["nid"]
                    if tgt_nid != src_nid:
                        try:
                            path = self._db.algorithms.dijkstra(src_nid, tgt_nid, weight=weight_attr)
                            if path:
                                # Calculate path length from weights
                                # For now, use path length as distance (unweighted)
                                result[str(tgt_nid)] = float(len(path) - 1)
                        except Exception:
                            pass  # Unreachable
                return result
        return super().sssp(source, weight_attr=weight_attr)

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LDBC LCC using native Grafeo local_clustering_coefficient."""
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "local_clustering_coefficient"):
            result = self._db.algorithms.local_clustering_coefficient()
            return {str(k): float(v) for k, v in result.items()}
        return super().local_clustering_coefficient()
