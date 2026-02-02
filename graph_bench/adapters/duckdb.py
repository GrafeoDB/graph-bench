r"""
DuckDB database adapter with SQL/PGQ graph extension.

DuckDB supports graph queries via the SQL/PGQ standard (ISO GQL compatible)
using the property graph extension.

Requires: pip install duckdb

Environment variables:
    GRAPH_BENCH_DUCKDB_PATH: Database path (default: :memory:)

    from graph_bench.adapters.duckdb import DuckDBAdapter

    adapter = DuckDBAdapter()
    adapter.connect()  # In-memory by default
"""

from collections import deque
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["DuckDBAdapter"]


@AdapterRegistry.register("duckdb")
class DuckDBAdapter(BaseAdapter):
    """DuckDB embedded database adapter with SQL/PGQ graph extension."""

    def __init__(self) -> None:
        self._conn: Any = None
        self._connected = False
        self._graph_name = "benchmark_graph"

    @property
    def name(self) -> str:
        return "DuckDB"

    @property
    def version(self) -> str:
        try:
            import duckdb

            return duckdb.__version__
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            import duckdb
        except ImportError as e:
            msg = "duckdb package not installed. Install with: pip install duckdb"
            raise ImportError(msg) from e

        path = uri or kwargs.get("path") or get_env("DUCKDB_PATH", default=":memory:")

        self._conn = duckdb.connect(path)
        self._connected = True
        self._setup_schema()

    def _setup_schema(self) -> None:
        """Create graph tables and property graph."""
        # Create node and edge tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id VARCHAR PRIMARY KEY,
                label VARCHAR,
                properties JSON
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                src VARCHAR,
                tgt VARCHAR,
                edge_type VARCHAR,
                properties JSON,
                PRIMARY KEY (src, tgt, edge_type)
            )
        """)

        # Create indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(tgt)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")

        # Create property graph using SQL/PGQ
        self._create_property_graph()

    def _create_property_graph(self) -> None:
        """Create or replace the property graph definition."""
        try:
            # Drop existing graph if any
            self._conn.execute(f"DROP PROPERTY GRAPH IF EXISTS {self._graph_name}")

            # Create property graph with SQL/PGQ syntax
            self._conn.execute(f"""
                CREATE PROPERTY GRAPH {self._graph_name}
                VERTEX TABLES (
                    nodes PROPERTIES (id, label, properties) LABEL Node
                )
                EDGE TABLES (
                    edges
                        SOURCE KEY (src) REFERENCES nodes (id)
                        DESTINATION KEY (tgt) REFERENCES nodes (id)
                        PROPERTIES (edge_type, properties)
                        LABEL Edge
                )
            """)
        except Exception:
            # Property graph extension might not be available in all versions
            pass

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._connected = False

    def clear(self) -> None:
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM nodes")
        # Recreate property graph after clearing
        self._create_property_graph()

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
            values = []
            for node in batch:
                node_id = str(node.get("id", count))
                props = {k: v for k, v in node.items() if k not in ("id", "label")}
                node_label = node.get("label", label)
                values.append((node_id, node_label, json.dumps(props)))
                count += 1

            stmt = "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)"
            self._conn.executemany(stmt, values)

        # Recreate property graph to pick up new nodes
        self._create_property_graph()
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        import json

        result = self._conn.execute(
            "SELECT id, label, properties FROM nodes WHERE id = ?", [node_id]
        ).fetchone()

        if result:
            props = json.loads(result[2]) if result[2] else {}
            return {"id": result[0], "label": result[1], **props}
        return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        import json

        results = self._conn.execute(
            "SELECT id, label, properties FROM nodes WHERE label = ? LIMIT ?",
            [label, limit],
        ).fetchall()

        nodes = []
        for row in results:
            props = json.loads(row[2]) if row[2] else {}
            nodes.append({"id": row[0], "label": row[1], **props})
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        import json

        count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            values = []
            for src, tgt, edge_type, props in batch:
                values.append((src, tgt, edge_type, json.dumps(props)))
                count += 1

            stmt = "INSERT OR REPLACE INTO edges (src, tgt, edge_type, properties) VALUES (?, ?, ?, ?)"
            self._conn.executemany(stmt, values)

        # Recreate property graph to pick up new edges
        self._create_property_graph()
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        # Try SQL/PGQ MATCH query first
        try:
            if edge_type:
                query = f"""
                    FROM GRAPH_TABLE ({self._graph_name}
                        MATCH (a:Node WHERE a.id = ?)
                              -[e:Edge WHERE e.edge_type = ?]->(b:Node)
                        COLUMNS (b.id AS neighbor_id)
                    )
                """
                result = self._conn.execute(query, [node_id, edge_type]).fetchall()
            else:
                query = f"""
                    FROM GRAPH_TABLE ({self._graph_name}
                        MATCH (a:Node WHERE a.id = ?)-[e:Edge]->(b:Node)
                        COLUMNS (b.id AS neighbor_id)
                    )
                """
                result = self._conn.execute(query, [node_id]).fetchall()
            return [row[0] for row in result]
        except Exception:
            # Fallback to regular SQL
            if edge_type:
                results = self._conn.execute(
                    "SELECT tgt FROM edges WHERE src = ? AND edge_type = ?",
                    [node_id, edge_type],
                ).fetchall()
            else:
                results = self._conn.execute(
                    "SELECT tgt FROM edges WHERE src = ?", [node_id]
                ).fetchall()
            return [row[0] for row in results]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        # Try SQL/PGQ path query first
        try:
            # SQL/PGQ supports SHORTEST path patterns
            result = self._conn.execute(f"""
                FROM GRAPH_TABLE ({self._graph_name}
                    MATCH SHORTEST (a:Node WHERE a.id = ?)-[e:Edge]->{"{1,10}"}(b:Node WHERE b.id = ?)
                    COLUMNS (a.id AS start_id, b.id AS end_id)
                )
            """, [source, target]).fetchone()
            if result:
                # Reconstruct path - for now use BFS fallback
                pass
        except Exception:
            pass

        # Fallback to BFS
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

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """BFS traversal using SQL/PGQ reachability or fallback."""
        # Try SQL/PGQ variable-length path
        try:
            result = self._conn.execute(f"""
                FROM GRAPH_TABLE ({self._graph_name}
                    MATCH (a:Node WHERE a.id = ?)-[e:Edge]->{"{0," + str(max_depth) + "}"}(b:Node)
                    COLUMNS (DISTINCT b.id AS node_id)
                )
            """, [start]).fetchall()
            return [row[0] for row in result]
        except Exception:
            # Fallback to manual BFS
            return super().traverse_bfs(start, max_depth=max_depth, edge_type=edge_type)

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute SQL or SQL/PGQ query."""
        if params:
            result = self._conn.execute(query, list(params.values()))
        else:
            result = self._conn.execute(query)

        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()

        return [dict(zip(columns, row, strict=False)) for row in rows]

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE label = ?", [label]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()

        return result[0] if result else 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE edge_type = ?", [edge_type]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()

        return result[0] if result else 0

    def execute_graph_query(self, match_pattern: str, columns: str) -> list[dict[str, Any]]:
        """Execute a SQL/PGQ graph query.

        Args:
            match_pattern: The MATCH pattern (e.g., "(a:Node)-[e:Edge]->(b:Node)")
            columns: The COLUMNS clause (e.g., "a.id AS src, b.id AS tgt")

        Returns:
            List of result dictionaries.

        Example:
            results = adapter.execute_graph_query(
                "(a:Node)-[e:Edge]->(b:Node)",
                "a.id AS src, b.id AS tgt, e.edge_type AS rel"
            )
        """
        query = f"""
            FROM GRAPH_TABLE ({self._graph_name}
                MATCH {match_pattern}
                COLUMNS ({columns})
            )
        """
        return self.execute_query(query)
