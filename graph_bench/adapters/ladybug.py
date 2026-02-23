r"""
LadybugDB database adapter.

LadybugDB is an embedded graph database built for query speed and scalability.
Uses Cypher as its query language with a schema-rigid table model.

Requires: pip install real_ladybug

Environment variables:
    GRAPH_BENCH_LADYBUG_PATH: Database path (default: :memory:)

    from graph_bench.adapters.ladybug import LadybugAdapter

    adapter = LadybugAdapter()
    adapter.connect()  # In-memory by default
"""

import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["LadybugAdapter"]

# Properties stored as dedicated columns (not JSON overflow).
# These cover the most common benchmark properties across all categories.
_KNOWN_NODE_COLS = {"id", "label", "name", "age", "city", "score",
                    "firstName", "lastName"}
_KNOWN_EDGE_COLS = {"edge_type", "weight"}


def _node_to_row(node: dict[str, Any], idx: int, label: str) -> dict[str, Any]:
    """Extract a node dict into dedicated columns + JSON overflow."""
    overflow = {
        k: v for k, v in node.items()
        if k not in _KNOWN_NODE_COLS
    }
    return {
        "id": str(node.get("id", idx)),
        "label": node.get("label", label),
        "name": node.get("name", ""),
        "age": int(node["age"]) if "age" in node else 0,
        "city": node.get("city", ""),
        "score": int(node["score"]) if "score" in node else 0,
        "firstName": node.get("firstName", ""),
        "lastName": node.get("lastName", ""),
        "props": json.dumps(overflow) if overflow else "",
    }


def _row_to_node(row: tuple) -> dict[str, Any]:
    """Convert a query row back to a node dict, merging overflow."""
    # Expected column order:
    #   n.id, n.label, n.name, n.age, n.city, n.score,
    #   n.firstName, n.lastName, n.props
    node: dict[str, Any] = {"id": row[0]}
    if row[1]:
        node["label"] = row[1]
    if row[2]:
        node["name"] = row[2]
    if row[3]:
        node["age"] = row[3]
    if row[4]:
        node["city"] = row[4]
    if row[5]:
        node["score"] = row[5]
    if row[6]:
        node["firstName"] = row[6]
    if row[7]:
        node["lastName"] = row[7]
    if row[8]:
        try:
            node.update(json.loads(row[8]))
        except (json.JSONDecodeError, TypeError):
            pass
    return node


_NODE_RETURN = (
    "n.id, n.label, n.name, n.age, n.city, n.score, "
    "n.firstName, n.lastName, n.props"
)


@AdapterRegistry.register("ladybug")
class LadybugAdapter(BaseAdapter):
    """LadybugDB embedded graph database adapter."""

    def __init__(self) -> None:
        self._db: Any = None
        self._conn: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "LadybugDB"

    @property
    def version(self) -> str:
        try:
            import real_ladybug

            return real_ladybug.__version__
        except Exception:
            return "unknown"

    @property
    def is_embedded(self) -> bool:
        return True

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from real_ladybug import Connection, Database
        except ImportError as e:
            msg = (
                "real_ladybug package not installed. "
                "Install with: pip install real_ladybug"
            )
            raise ImportError(msg) from e

        path = (
            uri
            or kwargs.get("path")
            or get_env("LADYBUG_PATH", default=":memory:")
        )

        if path == ":memory:" or path is None:
            self._db = Database()
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(path)

        self._conn = Connection(self._db)
        self._connected = True
        self._setup_schema()
        self._setup_algo()

    def _setup_schema(self) -> None:
        """Create node and edge tables with dedicated columns."""
        try:
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Node(
                    id STRING PRIMARY KEY,
                    label STRING,
                    name STRING,
                    age INT64,
                    city STRING,
                    score INT64,
                    firstName STRING,
                    lastName STRING,
                    props STRING
                )
            """)
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS Edge(
                    FROM Node TO Node,
                    edge_type STRING,
                    weight DOUBLE,
                    props STRING
                )
            """)
        except Exception:
            pass

    def _setup_algo(self) -> None:
        """Install and load the ALGO extension for native graph algorithms."""
        self._has_algo = False
        try:
            self._conn.execute("INSTALL ALGO")
            self._conn.execute("LOAD EXTENSION ALGO")
            self._has_algo = True
        except Exception:
            pass

    def _ensure_projected_graph(self) -> None:
        """Create or recreate the projected graph for algo calls."""
        try:
            self._conn.execute("CALL drop_projected_graph('bench')")
        except Exception:
            pass
        self._conn.execute("CALL project_graph('bench', ['Node'], ['Edge'])")

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
        self._conn = None
        self._db = None
        self._connected = False

    def clear(self) -> None:
        try:
            self._conn.execute("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

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
            node_data = [
                _node_to_row(node, i + j, label)
                for j, node in enumerate(batch)
            ]
            try:
                self._conn.execute(
                    """
                    UNWIND $nodes AS n
                    CREATE (:Node {
                        id: n.id, label: n.label,
                        name: n.name, age: n.age,
                        city: n.city, score: n.score,
                        firstName: n.firstName,
                        lastName: n.lastName,
                        props: n.props
                    })
                    """,
                    {"nodes": node_data},
                )
                count += len(batch)
            except Exception:
                for j, node in enumerate(batch):
                    row = _node_to_row(node, i + j, label)
                    try:
                        self._conn.execute(
                            """
                            CREATE (:Node {
                                id: $id, label: $label,
                                name: $name, age: $age,
                                city: $city, score: $score,
                                firstName: $firstName,
                                lastName: $lastName,
                                props: $props
                            })
                            """,
                            row,
                        )
                        count += 1
                    except Exception:
                        pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        try:
            result = self._conn.execute(
                f"MATCH (n:Node {{id: $id}}) RETURN {_NODE_RETURN}",
                {"id": node_id},
            )
            for row in result:
                return _row_to_node(row)
        except Exception:
            pass
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        try:
            # SET known columns directly — no JSON read-modify-write
            set_parts = []
            params: dict[str, Any] = {"id": node_id}
            overflow: dict[str, Any] = {}

            for k, v in properties.items():
                if k in _KNOWN_NODE_COLS and k != "id":
                    set_parts.append(f"n.{k} = ${k}")
                    params[k] = v
                else:
                    overflow[k] = v

            if overflow:
                # Read existing overflow, merge, write back
                result = self._conn.execute(
                    "MATCH (n:Node {id: $id}) RETURN n.props",
                    {"id": node_id},
                )
                existing = {}
                for row in result:
                    if row[0]:
                        try:
                            existing = json.loads(row[0])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    break
                else:
                    return False
                existing.update(overflow)
                set_parts.append("n.props = $props")
                params["props"] = json.dumps(existing)

            if not set_parts:
                return False

            query = (
                f"MATCH (n:Node {{id: $id}}) "
                f"SET {', '.join(set_parts)} RETURN n.id"
            )
            result = self._conn.execute(query, params)
            for _ in result:
                return True
            return False
        except Exception:
            return False

    def get_nodes_by_label(
        self, label: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        nodes = []
        try:
            result = self._conn.execute(
                "MATCH (n:Node) WHERE n.label = $label "
                f"RETURN {_NODE_RETURN} LIMIT {limit}",
                {"label": label},
            )
            for row in result:
                nodes.append(_row_to_node(row))
        except Exception:
            pass
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            edge_data = [
                {
                    "src": src,
                    "tgt": tgt,
                    "type": edge_type,
                    "weight": float(props.get("weight", 0.0)),
                    "props": json.dumps(
                        {k: v for k, v in props.items()
                         if k not in _KNOWN_EDGE_COLS}
                    ) if any(
                        k not in _KNOWN_EDGE_COLS for k in props
                    ) else "",
                }
                for src, tgt, edge_type, props in batch
            ]
            try:
                self._conn.execute(
                    """
                    UNWIND $edges AS e
                    MATCH (a:Node {id: e.src}), (b:Node {id: e.tgt})
                    CREATE (a)-[:Edge {
                        edge_type: e.type,
                        weight: e.weight,
                        props: e.props
                    }]->(b)
                    """,
                    {"edges": edge_data},
                )
                count += len(batch)
            except Exception:
                for src, tgt, edge_type, props in batch:
                    weight = float(props.get("weight", 0.0))
                    overflow = {
                        k: v for k, v in props.items()
                        if k not in _KNOWN_EDGE_COLS
                    }
                    try:
                        self._conn.execute(
                            """
                            MATCH (a:Node {id: $src}),
                                  (b:Node {id: $tgt})
                            CREATE (a)-[:Edge {
                                edge_type: $type,
                                weight: $weight,
                                props: $props
                            }]->(b)
                            """,
                            {
                                "src": src,
                                "tgt": tgt,
                                "type": edge_type,
                                "weight": weight,
                                "props": json.dumps(overflow)
                                if overflow else "",
                            },
                        )
                        count += 1
                    except Exception:
                        pass
        return count

    def get_neighbors(
        self, node_id: str, *, edge_type: str | None = None
    ) -> list[str]:
        neighbors = []
        try:
            if edge_type:
                result = self._conn.execute(
                    """
                    MATCH (n:Node {id: $id})-[e:Edge]->(m:Node)
                    WHERE e.edge_type = $type
                    RETURN m.id
                    """,
                    {"id": node_id, "type": edge_type},
                )
            else:
                result = self._conn.execute(
                    "MATCH (n:Node {id: $id})-[:Edge]->(m:Node) "
                    "RETURN m.id",
                    {"id": node_id},
                )
            for row in result:
                if row[0]:
                    neighbors.append(row[0])
        except Exception:
            pass
        return neighbors

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque(
            [(source, [source])]
        )

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.get_neighbors(
                current, edge_type=edge_type
            ):
                if neighbor not in visited:
                    queue.append((neighbor, [*path, neighbor]))

        return None

    def execute_query(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query."""
        results = []
        try:
            result = self._conn.execute(query, params or {})
            if hasattr(result, "get_column_names"):
                columns = result.get_column_names()
            else:
                columns = [f"col_{i}" for i in range(10)]

            for row in result:
                if isinstance(row, (list, tuple)):
                    results.append(
                        dict(zip(columns, row, strict=False))
                    )
                else:
                    results.append({"value": row})
        except Exception:
            pass
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        try:
            if label:
                result = self._conn.execute(
                    "MATCH (n:Node) WHERE n.label = $label "
                    "RETURN count(n)",
                    {"label": label},
                )
            else:
                result = self._conn.execute(
                    "MATCH (n:Node) RETURN count(n)"
                )

            for row in result:
                return row[0] if row else 0
        except Exception:
            pass
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        try:
            if edge_type:
                result = self._conn.execute(
                    "MATCH ()-[e:Edge]->() WHERE e.edge_type = $type "
                    "RETURN count(e)",
                    {"type": edge_type},
                )
            else:
                result = self._conn.execute(
                    "MATCH ()-[e:Edge]->() RETURN count(e)"
                )

            for row in result:
                return row[0] if row else 0
        except Exception:
            pass
        return 0

    # --- Native graph algorithms via ALGO extension ---

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank via LadybugDB native ALGO extension."""
        if not self._has_algo:
            return super().pagerank(
                damping=damping, max_iterations=max_iterations, tolerance=tolerance
            )
        try:
            self._ensure_projected_graph()
            result = self._conn.execute(
                "CALL page_rank('bench', dampingFactor := $d, "
                "maxIterations := $i, tolerance := $t) "
                "RETURN node.id, rank",
                {"d": damping, "i": max_iterations, "t": tolerance},
            )
            return {
                str(row[0]): float(row[1])
                for row in result
                if row[0] is not None
            }
        except Exception:
            return super().pagerank(
                damping=damping, max_iterations=max_iterations, tolerance=tolerance
            )

    def weakly_connected_components(self) -> list[set[str]]:
        """WCC via LadybugDB native ALGO extension."""
        if not self._has_algo:
            return super().weakly_connected_components()
        try:
            self._ensure_projected_graph()
            result = self._conn.execute(
                "CALL weakly_connected_components('bench') "
                "RETURN node.id, group_id"
            )
            components: dict[int, set[str]] = {}
            for row in result:
                if row[0] is not None:
                    gid = int(row[1])
                    if gid not in components:
                        components[gid] = set()
                    components[gid].add(str(row[0]))
            return list(components.values())
        except Exception:
            return super().weakly_connected_components()

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection via LadybugDB native ALGO extension (Louvain)."""
        if not self._has_algo:
            return super().community_detection(algorithm=algorithm)
        try:
            self._ensure_projected_graph()
            result = self._conn.execute(
                "CALL louvain('bench') RETURN node.id, louvain_id"
            )
            communities: dict[int, set[str]] = {}
            for row in result:
                if row[0] is not None:
                    cid = int(row[1])
                    if cid not in communities:
                        communities[cid] = set()
                    communities[cid].add(str(row[0]))
            return list(communities.values())
        except Exception:
            return super().community_detection(algorithm=algorithm)
