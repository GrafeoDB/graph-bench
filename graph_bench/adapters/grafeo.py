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
        # Reinitialize to avoid storage degradation from repeated delete/insert
        # cycles (Grafeo's internal ID space grows after DETACH DELETE, causing
        # progressively slower MATCH lookups).
        from grafeo import GrafeoDB
        self._db = GrafeoDB()

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        if not nodes:
            return 0
        count = 0
        prop_keys = list(nodes[0].keys())
        prop_map = ", ".join(f"{k}: props.{k}" for k in prop_keys)
        query = f"UNWIND $nodes AS props CREATE (:{label}:Node {{{prop_map}}})"

        for i in range(0, len(nodes), batch_size):
            batch = list(nodes[i : i + batch_size])
            try:
                self._db.execute(query, {"nodes": batch})
                count += len(batch)
            except Exception:
                # Fallback to individual create_node() API
                for node in batch:
                    self._db.create_node([label, "Node"], dict(node))
                    count += 1

        if hasattr(self._db, "create_property_index"):
            try:
                self._db.create_property_index("id")
            except Exception:
                pass  # Index already exists

        return count

    @staticmethod
    def _strip_internal(node: Any) -> dict[str, Any]:
        """Strip internal fields (_id, _labels) from a node map."""
        if isinstance(node, dict):
            return {k: v for k, v in node.items()
                    if k not in ("_id", "_labels")}
        return {}

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        # Grafeo >=0.5.6: RETURN n yields full node map {_id, _labels, ...props}
        result = self._db.execute(
            "MATCH (n {id: $id}) RETURN n", {"id": node_id},
        )
        for row in result:
            return self._strip_internal(row["n"])
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

    def get_nodes_by_label(
        self, label: str, *, limit: int = 100,
    ) -> list[dict[str, Any]]:
        result = self._db.execute(
            f"MATCH (n:{label}) RETURN n LIMIT {limit}"
        )
        return [self._strip_internal(row["n"]) for row in result]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        if not edges:
            return 0

        # Group by edge type for UNWIND (Cypher/GQL requires static rel types)
        by_type: dict[str, list[dict[str, Any]]] = {}
        for src, tgt, edge_type, props in edges:
            if edge_type not in by_type:
                by_type[edge_type] = []
            by_type[edge_type].append({"src": src, "tgt": tgt, **props})

        count = 0
        for edge_type, edge_list in by_type.items():
            # Build property map from first edge (excluding src/tgt)
            prop_keys = [k for k in edge_list[0] if k not in ("src", "tgt")]
            prop_map = ", ".join(f"{k}: e.{k}" for k in prop_keys)
            prop_str = f" {{{prop_map}}}" if prop_map else ""
            query = (
                f"UNWIND $edges AS e "
                f"MATCH (a {{id: e.src}}), (b {{id: e.tgt}}) "
                f"CREATE (a)-[:{edge_type}{prop_str}]->(b)"
            )
            for i in range(0, len(edge_list), batch_size):
                batch = edge_list[i : i + batch_size]
                try:
                    self._db.execute(query, {"edges": batch})
                    count += len(batch)
                except Exception:
                    # Fallback to individual create_edge() API
                    result = self._db.execute("MATCH (n) RETURN n.id AS id, id(n) AS nid")
                    nid_cache = {row["id"]: row["nid"] for row in result}
                    for e in batch:
                        src_nid = nid_cache.get(e["src"])
                        tgt_nid = nid_cache.get(e["tgt"])
                        if src_nid is not None and tgt_nid is not None:
                            props = {k: e[k] for k in prop_keys}
                            try:
                                self._db.create_edge(src_nid, tgt_nid, edge_type, props)
                                count += 1
                            except Exception:
                                pass
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

    def sssp(
        self, source: str, *, weight_attr: str = "weight",
    ) -> dict[str, float]:
        """LDBC SSSP using native Grafeo sssp (v0.5.6+)."""
        if not (
            hasattr(self._db, "algorithms")
            and hasattr(self._db.algorithms, "sssp")
        ):
            return super().sssp(source, weight_attr=weight_attr)

        # Resolve app ID → internal node ID
        src_result = self._db.execute(
            "MATCH (n {id: $id}) RETURN id(n) AS nid",
            {"id": source},
        )
        src_nid = None
        for row in src_result:
            src_nid = row["nid"]
        if src_nid is None:
            return super().sssp(source, weight_attr=weight_attr)

        # Build internal ID → app ID mapping
        mapping = self._db.execute(
            "MATCH (n) RETURN id(n) AS nid, n.id AS app_id"
        )
        nid_to_app = {
            row["nid"]: str(row["app_id"]) for row in mapping
        }

        distances = self._db.algorithms.sssp(
            source=str(src_nid), weight_attr=weight_attr,
        )
        if isinstance(distances, dict):
            return {
                nid_to_app.get(k, str(k)): float(v)
                for k, v in distances.items()
            }
        # Handle iterable of (node_id, distance) tuples
        return {
            nid_to_app.get(k, str(k)): float(v)
            for k, v in distances
        }

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LDBC LCC using native Grafeo local_clustering_coefficient."""
        if hasattr(self._db, "algorithms") and hasattr(self._db.algorithms, "local_clustering_coefficient"):
            result = self._db.algorithms.local_clustering_coefficient()
            return {str(k): float(v) for k, v in result.items()}
        return super().local_clustering_coefficient()

    # NOTE: create_vector_index() and vector_search() use the brute-force
    # fallback from BaseAdapter. Grafeo has native HNSW vector support in
    # Rust, but the Python bindings don't yet expose vector type conversion
    # or CREATE VECTOR INDEX DDL. Once the bindings are updated (see
    # grafeo/.claude/todo/4_language_bindings/improve-vector-support.md),
    # override these methods with native GQL implementations.
