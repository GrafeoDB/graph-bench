r"""
Grafeo Server database adapter (GQL over HTTP).

Grafeo Server is an HTTP-based graph database built on the Grafeo engine.
This adapter uses GQL (ISO/IEC 39075) via the /query endpoint, compatible
with the minimal "lite" image (grafeo-server:lite) which only supports GQL.

Requires: pip install requests

Environment variables:
    GRAPH_BENCH_GRAFEO_SERVER_URI: Connection URI (default: http://localhost:7474)

    from graph_bench.adapters.grafeo_server import GrafeoServerAdapter

    adapter = GrafeoServerAdapter()
    adapter.connect(uri="http://localhost:7474")
"""

import math
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["GrafeoServerAdapter"]


@AdapterRegistry.register("grafeo-server")
class GrafeoServerAdapter(BaseAdapter):
    """Grafeo Server graph database adapter (HTTP/GQL)."""

    def __init__(self) -> None:
        self._base_url: str = ""
        self._db_name: str = "bench"
        self._connected = False
        self._session: Any = None  # requests.Session

    @property
    def name(self) -> str:
        return "Grafeo Server"

    @property
    def version(self) -> str:
        if not self._connected:
            return "unknown"
        try:
            resp = self._session.get(f"{self._base_url}/health")
            data = resp.json()
            return data.get("version", "unknown")
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            import requests
        except ImportError as e:
            msg = (
                "requests package not installed. "
                "Install with: pip install requests"
            )
            raise ImportError(msg) from e

        default_uri = "http://localhost:7474"
        self._base_url = (
            uri
            or get_env("GRAFEO_SERVER_URI", default=default_uri)
            or default_uri
        )
        self._base_url = self._base_url.rstrip("/")

        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json"}
        )

        # Verify connectivity
        resp = self._session.get(f"{self._base_url}/health")
        resp.raise_for_status()

        # Create benchmark database (ignore if exists)
        try:
            self._session.post(
                f"{self._base_url}/db",
                json={
                    "name": self._db_name,
                    "database_type": "Lpg",
                },
            )
        except Exception:
            pass

        self._connected = True

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False

    # ── Query helpers (GQL via /query) ────────────────────────────

    def _query(self, gql: str) -> list[dict[str, Any]]:
        """Execute a GQL query (auto-commit) and return rows."""
        resp = self._session.post(
            f"{self._base_url}/query",
            json={
                "query": gql,
                "language": "gql",
                "database": self._db_name,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        return [
            {col: row[i] for i, col in enumerate(columns)}
            for row in rows
        ]

    def _exec(self, gql: str) -> Any:
        """Execute a GQL query and return raw response JSON."""
        resp = self._session.post(
            f"{self._base_url}/query",
            json={
                "query": gql,
                "language": "gql",
                "database": self._db_name,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def _tx_begin(self) -> str:
        """Begin a transaction, return session_id."""
        resp = self._session.post(
            f"{self._base_url}/tx/begin",
            json={"database": self._db_name},
        )
        resp.raise_for_status()
        return resp.json()["session_id"]

    def _tx_query(self, session_id: str, gql: str) -> Any:
        """Execute a query within a transaction."""
        resp = self._session.post(
            f"{self._base_url}/tx/query",
            headers={"X-Session-Id": session_id},
            json={"query": gql, "language": "gql"},
        )
        resp.raise_for_status()
        return resp.json()

    def _tx_commit(self, session_id: str) -> None:
        """Commit a transaction."""
        resp = self._session.post(
            f"{self._base_url}/tx/commit",
            headers={"X-Session-Id": session_id},
        )
        resp.raise_for_status()

    # ── Value formatting (no parameterized GQL) ──────────────────

    @staticmethod
    def _gql_literal(value: Any) -> str:
        """Convert a Python value to a GQL literal string."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (
                math.isnan(value) or math.isinf(value)
            ):
                return "null"
            return repr(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(
                GrafeoServerAdapter._gql_literal(v) for v in value
            )
            return "[" + inner + "]"
        s = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s}'"

    @staticmethod
    def _format_props(props: dict[str, Any]) -> str:
        """Format a dict as a GQL property map {key: value, ...}."""
        if not props:
            return ""
        parts = [
            f"{k}: {GrafeoServerAdapter._gql_literal(v)}"
            for k, v in props.items()
        ]
        return "{" + ", ".join(parts) + "}"

    # ── Core operations ──────────────────────────────────────────

    def clear(self) -> None:
        try:
            self._session.delete(
                f"{self._base_url}/db/{self._db_name}"
            )
        except Exception:
            pass
        try:
            self._session.post(
                f"{self._base_url}/db",
                json={
                    "name": self._db_name,
                    "database_type": "Lpg",
                },
            )
        except Exception:
            pass

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 200,
    ) -> int:
        count = 0
        sid = self._tx_begin()
        try:
            for i in range(0, len(nodes), batch_size):
                batch = list(nodes[i : i + batch_size])
                patterns = [
                    f"(:{label} {self._format_props(node)})"
                    for node in batch
                ]
                query = "CREATE " + ", ".join(patterns)
                self._tx_query(sid, query)
                count += len(batch)
            self._tx_commit(sid)
        except Exception:
            count = 0
            sid = self._tx_begin()
            for node in nodes:
                try:
                    self._tx_query(
                        sid,
                        f"CREATE (:{label} "
                        f"{self._format_props(node)})",
                    )
                    count += 1
                except Exception:
                    pass
            self._tx_commit(sid)
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        lit = self._gql_literal(node_id)
        rows = self._query(
            f"MATCH (n {{id: {lit}}}) RETURN n"
        )
        if rows:
            node = rows[0].get("n")
            if isinstance(node, dict):
                return node
            return rows[0]
        return None

    def update_node(
        self, node_id: str, properties: dict[str, Any]
    ) -> bool:
        lit = self._gql_literal(node_id)
        set_parts = [
            f"n.{k} = {self._gql_literal(v)}"
            for k, v in properties.items()
        ]
        set_clause = ", ".join(set_parts)
        sid = self._tx_begin()
        try:
            result = self._tx_query(
                sid,
                f"MATCH (n {{id: {lit}}}) "
                f"SET {set_clause} RETURN n.id AS id",
            )
            self._tx_commit(sid)
            return bool(result.get("rows"))
        except Exception:
            return False

    def get_nodes_by_label(
        self, label: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = self._query(
            f"MATCH (n:{label}) RETURN n LIMIT {limit}"
        )
        result = []
        for row in rows:
            node = row.get("n")
            if isinstance(node, dict):
                result.append(node)
            else:
                result.append(row)
        return result

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 100,
    ) -> int:
        by_type: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for src, tgt, etype, props in edges:
            by_type.setdefault(etype, []).append((src, tgt, props))

        count = 0
        COMMIT_INTERVAL = 2000
        edges_in_tx = 0
        sid = self._tx_begin()

        for etype, type_edges in by_type.items():
            for i in range(0, len(type_edges), batch_size):
                batch = type_edges[i : i + batch_size]
                try:
                    match_parts = []
                    create_parts = []
                    for j, (src, tgt, _props) in enumerate(batch):
                        src_lit = self._gql_literal(src)
                        tgt_lit = self._gql_literal(tgt)
                        match_parts.append(
                            f"(a{j} {{id: {src_lit}}}), "
                            f"(b{j} {{id: {tgt_lit}}})"
                        )
                        create_parts.append(
                            f"(a{j})-[:{etype}]->(b{j})"
                        )
                    query = (
                        "MATCH "
                        + ", ".join(match_parts)
                        + " CREATE "
                        + ", ".join(create_parts)
                    )
                    self._tx_query(sid, query)
                    count += len(batch)
                    edges_in_tx += len(batch)
                except Exception:
                    for src, tgt, _props in batch:
                        src_lit = self._gql_literal(src)
                        tgt_lit = self._gql_literal(tgt)
                        try:
                            self._tx_query(
                                sid,
                                f"MATCH (a {{id: {src_lit}}}), "
                                f"(b {{id: {tgt_lit}}}) "
                                f"CREATE (a)-[:{etype}]->(b)",
                            )
                            count += 1
                            edges_in_tx += 1
                        except Exception:
                            pass

                if edges_in_tx >= COMMIT_INTERVAL:
                    self._tx_commit(sid)
                    sid = self._tx_begin()
                    edges_in_tx = 0

        self._tx_commit(sid)
        return count

    def get_neighbors(
        self, node_id: str, *, edge_type: str | None = None
    ) -> list[str]:
        lit = self._gql_literal(node_id)
        if edge_type:
            query = (
                f"MATCH (n {{id: {lit}}})-[:{edge_type}]->(m) "
                f"RETURN m.id AS id"
            )
        else:
            query = (
                f"MATCH (n {{id: {lit}}})-[r]->(m) "
                f"RETURN m.id AS id"
            )
        rows = self._query(query)
        return [
            str(r["id"]) for r in rows if r.get("id") is not None
        ]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        src_lit = self._gql_literal(source)
        tgt_lit = self._gql_literal(target)
        rel = f":{edge_type}*" if edge_type else "*"
        try:
            rows = self._query(
                f"MATCH (start {{id: {src_lit}}}), "
                f"(end {{id: {tgt_lit}}}), "
                f"path = shortestPath((start)-[{rel}]->(end)) "
                f"RETURN [n IN nodes(path) | n.id] AS path"
            )
            if rows and rows[0].get("path"):
                return [str(n) for n in rows[0]["path"]]
        except Exception:
            pass

        # BFS fallback
        from collections import deque

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
        if params:
            for k, v in sorted(
                params.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):
                query = query.replace(
                    f"${k}", self._gql_literal(v)
                )

        q_upper = query.strip().upper()
        is_write = any(
            kw in q_upper
            for kw in ["CREATE", "DELETE", "SET ", "REMOVE", "MERGE"]
        )
        if is_write:
            sid = self._tx_begin()
            try:
                result = self._tx_query(sid, query)
                self._tx_commit(sid)
                columns = result.get("columns", [])
                rows = result.get("rows", [])
                return [
                    {col: row[i] for i, col in enumerate(columns)}
                    for row in rows
                ]
            except Exception:
                return []
        return self._query(query)

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"
        rows = self._query(query)
        if rows:
            return int(rows[0]["count"])
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = (
                f"MATCH ()-[r:{edge_type}]->() "
                f"RETURN count(r) AS count"
            )
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"
        rows = self._query(query)
        if rows:
            return int(rows[0]["count"])
        return 0
