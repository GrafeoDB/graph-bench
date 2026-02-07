r"""
TuGraph database adapter.

TuGraph uses the Bolt protocol (Neo4j-compatible) with Cypher queries.
Requires: pip install neo4j

TuGraph Cypher dialect differences from Neo4j:
- Schema-ful: vertex/edge labels must be defined before data insertion
- No SET n = node (must set properties individually or inline in CREATE)
- No parameterized $limit (must use literal integer)
- No shortestPath() function (use algo.shortestPath procedure)
- Sessions require database='default' parameter

Environment variables:
    GRAPH_BENCH_TUGRAPH_URI: Connection URI (default: bolt://localhost:7689)
    GRAPH_BENCH_TUGRAPH_USER: Username (default: admin)
    GRAPH_BENCH_TUGRAPH_PASSWORD: Password (default: 73@TuGraph)

    from graph_bench.adapters.tugraph import TuGraphAdapter

    adapter = TuGraphAdapter()
    adapter.connect(uri="bolt://localhost:7689")
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["TuGraphAdapter"]

# TuGraph default graph name
_DEFAULT_GRAPH = "default"


@AdapterRegistry.register("tugraph")
class TuGraphAdapter(BaseAdapter):
    """TuGraph graph database adapter.

    TuGraph is schema-ful: vertex/edge labels must be defined before
    data insertion. This adapter lazily creates labels on first use.
    """

    def __init__(self) -> None:
        self._driver: Any = None
        self._connected = False
        self._known_vertex_labels: set[str] = set()
        self._known_edge_labels: set[str] = set()

    def _session(self) -> Any:
        """Open a session with the default graph context."""
        return self._driver.session(database=_DEFAULT_GRAPH)

    @property
    def name(self) -> str:
        return "TuGraph"

    @property
    def version(self) -> str:
        if not self._connected or self._driver is None:
            return "unknown"
        try:
            with self._session() as session:
                result = session.run("CALL dbms.system.info()")
                for record in result:
                    for key in ("lgraph_version", "version"):
                        if key in record:
                            return str(record[key])
                return "unknown"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            msg = "neo4j package not installed. Install with: pip install neo4j"
            raise ImportError(msg) from e

        uri = uri or get_env("TUGRAPH_URI", default="bolt://localhost:7689")
        user = kwargs.get("user") or get_env("TUGRAPH_USER", default="admin")
        password = kwargs.get("password") or get_env("TUGRAPH_PASSWORD", default="73@TuGraph")

        if uri is None:
            msg = "TuGraph URI required"
            raise ValueError(msg)

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._connected = True

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
        self._connected = False
        self._known_vertex_labels.clear()
        self._known_edge_labels.clear()

    # -- Schema helpers (TuGraph is schema-ful) --

    def _ensure_vertex_label(self, session: Any, label: str, sample_props: dict[str, Any] | None = None) -> None:
        """Create vertex label if not already known."""
        if label in self._known_vertex_labels:
            return
        try:
            fields = self._infer_fields(sample_props or {})
            if "id" not in fields:
                fields["id"] = "STRING"

            field_specs = ""
            for fname, ftype in fields.items():
                optional = "false" if fname == "id" else "true"
                field_specs += f", '{fname}', '{ftype}', {optional}"

            query = f"CALL db.createVertexLabel('{label}', 'id'{field_specs})"
            session.run(query)
        except Exception:
            pass  # Label may already exist
        self._known_vertex_labels.add(label)

    def _ensure_edge_label(self, session: Any, edge_type: str, sample_props: dict[str, Any] | None = None) -> None:
        """Create edge label if not already known."""
        if edge_type in self._known_edge_labels:
            return
        try:
            fields = self._infer_fields(sample_props or {"weight": 1.0})
            field_specs = ""
            for fname, ftype in fields.items():
                field_specs += f", '{fname}', '{ftype}', true"

            session.run(f"CALL db.createEdgeLabel('{edge_type}', '[]'{field_specs})")
        except Exception:
            pass  # Edge type may already exist
        self._known_edge_labels.add(edge_type)

    @staticmethod
    def _infer_fields(props: dict[str, Any]) -> dict[str, str]:
        """Infer TuGraph field types from Python values."""
        type_map = {
            str: "STRING",
            int: "INT64",
            float: "DOUBLE",
            bool: "BOOL",
        }
        fields: dict[str, str] = {}
        for key, value in props.items():
            if isinstance(value, list):
                fields[key] = "STRING"
            else:
                fields[key] = type_map.get(type(value), "STRING")
        return fields

    # -- CRUD operations --

    def clear(self) -> None:
        # db.dropDB() cleanly removes all data AND schema in one call
        with self._session() as session:
            try:
                session.run("CALL db.dropDB()")
            except Exception:
                try:
                    session.run("MATCH (n) DETACH DELETE n")
                except Exception:
                    pass
        self._known_vertex_labels.clear()
        self._known_edge_labels.clear()

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
        with self._session() as session:
            sample = nodes[0]
            self._ensure_vertex_label(session, label, sample)
            prop_keys = list(sample.keys())

            # TuGraph's UNWIND only inserts the first row; use individual CREATEs
            create_query = f"CREATE (n:{label} {{{', '.join(f'{k}: ${k}' for k in prop_keys)}}})"
            for node in nodes:
                try:
                    session.run(create_query, **node)
                    count += 1
                except Exception:
                    pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            result = session.run("MATCH (n {id: $id}) RETURN n", id=node_id)
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        with self._session() as session:
            # TuGraph doesn't support SET n += $props; use per-property SET
            set_clauses = ", ".join(f"n.{k} = ${k}" for k in properties)
            query = f"MATCH (n {{id: $id}}) SET {set_clauses} RETURN n"
            result = session.run(query, id=node_id, **properties)
            return result.single() is not None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as session:
            # TuGraph doesn't support parameterized $limit
            result = session.run(f"MATCH (n:{label}) RETURN n LIMIT {int(limit)}")
            return [dict(record["n"]) for record in result]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        # Group by edge type to ensure labels exist
        by_type: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for src, tgt, edge_type, props in edges:
            if edge_type not in by_type:
                by_type[edge_type] = []
            by_type[edge_type].append((src, tgt, props))

        with self._session() as session:
            for edge_type, edge_list in by_type.items():
                sample_props = edge_list[0][2] if edge_list else {"weight": 1.0}
                self._ensure_edge_label(session, edge_type, sample_props)

                # Build per-property SET clause
                prop_keys = list(sample_props.keys())
                if prop_keys:
                    set_clause = " SET " + ", ".join(f"r.{k} = ${k}" for k in prop_keys)
                else:
                    set_clause = ""

                # TuGraph's UNWIND only processes first row; use individual CREATEs
                query = (
                    f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                    f"CREATE (a)-[r:{edge_type}]->(b){set_clause}"
                )
                for src, tgt, props in edge_list:
                    try:
                        session.run(query, src=src, tgt=tgt, **props)
                        count += 1
                    except Exception:
                        pass
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        with self._session() as session:
            if edge_type:
                query = f"MATCH (n {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
            else:
                query = "MATCH (n {id: $id})-->(m) RETURN m.id AS id"
            result = session.run(query, id=node_id)
            return [record["id"] for record in result if record["id"]]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        # TuGraph doesn't support Cypher shortestPath(); use native algo.shortestPath
        with self._session() as session:
            try:
                result = session.run(
                    "MATCH (s {id: $src}), (t {id: $tgt}) "
                    "CALL algo.shortestPath(s, t) YIELD nodeCount, totalCost, path "
                    "RETURN path",
                    src=source,
                    tgt=target,
                )
                record = result.single()
                if record and record["path"]:
                    path_data = record["path"]
                    if isinstance(path_data, list):
                        return [
                            str(n.get("id", n)) if isinstance(n, dict) else str(n)
                            for n in path_data
                        ]
            except Exception:
                pass

        # Fallback: BFS via base class traverse_bfs
        return self._bfs_shortest_path(source, target, edge_type=edge_type)

    def _bfs_shortest_path(
        self, source: str, target: str, *, edge_type: str | None = None
    ) -> list[str] | None:
        """BFS-based shortest path fallback."""
        visited: dict[str, str | None] = {source: None}
        queue = [source]
        while queue:
            current = queue.pop(0)
            if current == target:
                path = []
                node: str | None = target
                while node is not None:
                    path.append(node)
                    node = visited[node]
                return list(reversed(path))
            for neighbor in self.get_neighbors(current, edge_type=edge_type):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def count_nodes(self, *, label: str | None = None) -> int:
        with self._session() as session:
            if label:
                query = f"MATCH (n:{label}) RETURN count(n) AS count"
            else:
                query = "MATCH (n) RETURN count(n) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        with self._session() as session:
            if edge_type:
                query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
            else:
                query = "MATCH ()-[r]->() RETURN count(r) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    # -- Native algorithm overrides (fallback to NetworkX via super()) --

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank using TuGraph built-in algo.pagerank."""
        try:
            with self._session() as session:
                result = session.run(
                    "CALL algo.pagerank($num_iteration, $damping_factor)",
                    num_iteration=max_iterations,
                    damping_factor=damping,
                )
                scores = {}
                for record in result:
                    node = record.get("node", {})
                    nid = node.get("id") if isinstance(node, dict) else record.get("node.id")
                    score = record.get("score", record.get("pr", 0.0))
                    if nid is not None:
                        scores[str(nid)] = float(score)
                if scores:
                    return scores
        except Exception:
            pass
        return super().pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection using TuGraph built-in LPA or Louvain."""
        try:
            with self._session() as session:
                if algorithm == "label_propagation":
                    result = session.run("CALL algo.lpa($max_iteration)", max_iteration=20)
                else:
                    result = session.run("CALL algo.louvain()")
                communities: dict[int, set[str]] = {}
                for record in result:
                    cid = record.get("community_id", record.get("label", 0))
                    node = record.get("node", {})
                    nid = node.get("id") if isinstance(node, dict) else record.get("node.id")
                    if nid is not None:
                        if cid not in communities:
                            communities[cid] = set()
                        communities[cid].add(str(nid))
                if communities:
                    return list(communities.values())
        except Exception:
            pass
        return super().community_detection(algorithm=algorithm)

    def weakly_connected_components(self) -> list[set[str]]:
        """WCC using TuGraph built-in algo.wcc."""
        try:
            with self._session() as session:
                result = session.run("CALL algo.wcc()")
                components: dict[int, set[str]] = {}
                for record in result:
                    cid = record.get("component_id", record.get("label", 0))
                    node = record.get("node", {})
                    nid = node.get("id") if isinstance(node, dict) else record.get("node.id")
                    if nid is not None:
                        if cid not in components:
                            components[cid] = set()
                        components[cid].add(str(nid))
                if components:
                    return list(components.values())
        except Exception:
            pass
        return super().weakly_connected_components()

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LCC using TuGraph built-in algo.lcc."""
        try:
            with self._session() as session:
                result = session.run("CALL algo.lcc()")
                coeffs = {}
                for record in result:
                    node = record.get("node", {})
                    nid = node.get("id") if isinstance(node, dict) else record.get("node.id")
                    coeff = record.get("coefficient", record.get("lcc", 0.0))
                    if nid is not None:
                        coeffs[str(nid)] = float(coeff)
                if coeffs:
                    return coeffs
        except Exception:
            pass
        return super().local_clustering_coefficient()

    def bfs_levels(self, source: str) -> dict[str, int]:
        """BFS levels using TuGraph built-in algo.bfs."""
        try:
            with self._session() as session:
                result = session.run(
                    "MATCH (n {id: $id}) CALL algo.bfs(n) YIELD node, level "
                    "RETURN node.id AS id, level",
                    id=source,
                )
                levels = {}
                for record in result:
                    if record["id"] is not None:
                        levels[str(record["id"])] = int(record["level"])
                if levels:
                    return levels
        except Exception:
            pass
        return super().bfs_levels(source)

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """SSSP using TuGraph built-in algo.sssp."""
        try:
            with self._session() as session:
                result = session.run(
                    "MATCH (n {id: $id}) CALL algo.sssp(n, $weight_attr) "
                    "YIELD node, distance "
                    "RETURN node.id AS id, distance",
                    id=source,
                    weight_attr=weight_attr,
                )
                distances = {}
                for record in result:
                    if record["id"] is not None:
                        distances[str(record["id"])] = float(record["distance"])
                if distances:
                    return distances
        except Exception:
            pass
        return super().sssp(source, weight_attr=weight_attr)
