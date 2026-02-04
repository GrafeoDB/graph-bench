r"""
LadybugDB database adapter.

LadybugDB is an embedded graph database built for query speed and scalability.
Uses Cypher as its query language.

Requires: pip install real_ladybug

Environment variables:
    GRAPH_BENCH_LADYBUG_PATH: Database path (default: :memory:)

    from graph_bench.adapters.ladybug import LadybugAdapter

    adapter = LadybugAdapter()
    adapter.connect()  # In-memory by default
"""

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["LadybugAdapter"]


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

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from real_ladybug import Connection, Database
        except ImportError as e:
            msg = "real_ladybug package not installed. Install with: pip install real_ladybug"
            raise ImportError(msg) from e

        path = uri or kwargs.get("path") or get_env("LADYBUG_PATH", default=":memory:")

        # LadybugDB uses :memory: or a path
        if path == ":memory:" or path is None:
            self._db = Database()
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(path)

        self._conn = Connection(self._db)
        self._connected = True
        self._setup_schema()

    def _setup_schema(self) -> None:
        """Create node and edge tables if they don't exist."""
        try:
            # Create a generic Node table
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Node(
                    id STRING PRIMARY KEY,
                    label STRING,
                    props STRING
                )
            """)
            # Create a generic Edge relationship
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS Edge(
                    FROM Node TO Node,
                    edge_type STRING,
                    props STRING
                )
            """)
        except Exception:
            pass

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
        import json

        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            for node in batch:
                node_id = str(node.get("id", count))
                props = json.dumps({k: v for k, v in node.items() if k not in ("id", "label")})
                node_label = node.get("label", label)
                try:
                    self._conn.execute(
                        "CREATE (:Node {id: $id, label: $label, props: $props})",
                        {"id": node_id, "label": node_label, "props": props},
                    )
                    count += 1
                except Exception:
                    pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        import json

        try:
            result = self._conn.execute(
                "MATCH (n:Node {id: $id}) RETURN n.id, n.label, n.props",
                {"id": node_id},
            )
            for row in result:
                props = json.loads(row[2]) if row[2] else {}
                return {"id": row[0], "label": row[1], **props}
        except Exception:
            pass
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        import json

        try:
            # Get existing props
            result = self._conn.execute(
                "MATCH (n:Node {id: $id}) RETURN n.props",
                {"id": node_id},
            )
            existing_props = {}
            for row in result:
                existing_props = json.loads(row[0]) if row[0] else {}
                break
            else:
                return False  # Node not found

            # Merge and update
            existing_props.update(properties)
            self._conn.execute(
                "MATCH (n:Node {id: $id}) SET n.props = $props",
                {"id": node_id, "props": json.dumps(existing_props)},
            )
            return True
        except Exception:
            return False

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        import json

        nodes = []
        try:
            result = self._conn.execute(
                f"MATCH (n:Node) WHERE n.label = $label RETURN n.id, n.label, n.props LIMIT {limit}",
                {"label": label},
            )
            for row in result:
                props = json.loads(row[2]) if row[2] else {}
                nodes.append({"id": row[0], "label": row[1], **props})
        except Exception:
            pass
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        import json

        count = 0
        for src, tgt, edge_type, props in edges:
            props_json = json.dumps(props)
            try:
                self._conn.execute(
                    """
                    MATCH (a:Node {id: $src}), (b:Node {id: $tgt})
                    CREATE (a)-[:Edge {edge_type: $type, props: $props}]->(b)
                    """,
                    {"src": src, "tgt": tgt, "type": edge_type, "props": props_json},
                )
                count += 1
            except Exception:
                pass
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
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
                    "MATCH (n:Node {id: $id})-[:Edge]->(m:Node) RETURN m.id",
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
        # Use BFS fallback
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
        """Execute a Cypher query."""
        results = []
        try:
            result = self._conn.execute(query, params or {})
            # Get column names if available
            if hasattr(result, "get_column_names"):
                columns = result.get_column_names()
            else:
                columns = [f"col_{i}" for i in range(10)]

            for row in result:
                if isinstance(row, (list, tuple)):
                    results.append(dict(zip(columns, row, strict=False)))
                else:
                    results.append({"value": row})
        except Exception:
            pass
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        try:
            if label:
                result = self._conn.execute(
                    "MATCH (n:Node) WHERE n.label = $label RETURN count(n)",
                    {"label": label},
                )
            else:
                result = self._conn.execute("MATCH (n:Node) RETURN count(n)")

            for row in result:
                return row[0] if row else 0
        except Exception:
            pass
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        try:
            if edge_type:
                result = self._conn.execute(
                    "MATCH ()-[e:Edge]->() WHERE e.edge_type = $type RETURN count(e)",
                    {"type": edge_type},
                )
            else:
                result = self._conn.execute("MATCH ()-[e:Edge]->() RETURN count(e)")

            for row in result:
                return row[0] if row else 0
        except Exception:
            pass
        return 0
