r"""
TuringDB database adapter.

TuringDB is a high-performance in-memory column-oriented graph database
written in C++23, accessed via HTTP REST API with OpenCypher queries.

Requires: pip install turingdb

Environment variables:
    GRAPH_BENCH_TURING_URI: Connection URI (default: http://localhost:6666)

    from graph_bench.adapters.turing import TuringDBAdapter

    adapter = TuringDBAdapter()
    adapter.connect(uri="http://localhost:6666")
"""

import math
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["TuringDBAdapter"]


@AdapterRegistry.register("turing")
class TuringDBAdapter(BaseAdapter):
    """TuringDB graph database adapter (HTTP/OpenCypher)."""

    def __init__(self) -> None:
        self._db: Any = None
        self._connected = False
        self._graph_name = "bench"
        self._in_change = False  # Track whether we're inside a write change

    @property
    def name(self) -> str:
        return "TuringDB"

    @property
    def version(self) -> str:
        try:
            import turingdb
            return getattr(turingdb, "__version__", "unknown")
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from turingdb import TuringDB
        except ImportError as e:
            msg = "turingdb package not installed. Install with: pip install turingdb"
            raise ImportError(msg) from e

        host = uri or get_env("TURING_URI", default="http://localhost:6666")
        if host is None:
            host = "http://localhost:6666"

        self._db = TuringDB(host=host, timeout=120)
        self._db.try_reach()

        # Create or load graph for benchmarking
        try:
            self._db.create_graph(self._graph_name)
        except Exception:
            try:
                self._db.load_graph(self._graph_name, raise_if_loaded=False)
            except Exception:
                pass
        self._db.set_graph(self._graph_name)
        self._connected = True

    def disconnect(self) -> None:
        self._db = None
        self._connected = False

    def clear(self) -> None:
        # _begin_write handles stale change cleanup (COMMIT+SUBMIT) internally
        self._begin_write()
        try:
            self._db.query("MATCH (n) DETACH DELETE n")
        except Exception:
            # Fallback: delete edges then nodes separately
            try:
                self._db.query("MATCH ()-[r]->() DELETE r")
            except Exception:
                pass
            try:
                self._db.query("MATCH (n) DELETE n")
            except Exception:
                pass
        self._end_write()

    # ── Write transaction management ───────────────────────────────
    # TuringDB uses git-like versioning: writes must happen inside a
    # "change" (branch). Pattern: new_change → writes → COMMIT → CHANGE SUBMIT

    def _new_change(self) -> None:
        """Create a new change, fixing the client's hex-encoding bug."""
        self._db.new_change()
        # Fix: TuringDB client stores raw uint64 but server expects hex string
        raw = self._db._params.get("change")
        if raw is not None:
            self._db._params["change"] = f"{int(raw):x}"

    def _begin_write(self) -> None:
        """Start a write change, discarding any stale change first."""
        if self._in_change:
            # Discard stale change from a previous timeout/error
            try:
                self._db.query("CHANGE DELETE")
            except Exception:
                # Fallback: try commit+submit if delete fails
                for cmd in ("COMMIT", "CHANGE SUBMIT"):
                    try:
                        self._db.query(cmd)
                    except Exception:
                        pass
            self._in_change = False

        try:
            self._db.checkout("main")
        except Exception:
            pass
        self._new_change()
        self._in_change = True

    def _end_write(self) -> None:
        """Commit and submit the current write change."""
        if self._in_change:
            try:
                self._db.query("COMMIT")
            except Exception:
                pass
            try:
                self._db.query("CHANGE SUBMIT")
            except Exception:
                pass
            try:
                self._db.checkout("main")
            except Exception:
                pass
            self._in_change = False

    # ── Value formatting (no parameterized queries) ──────────────

    @staticmethod
    def _cypher_literal(value: Any) -> str:
        """Convert a Python value to a Cypher literal string."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return "null"
            return repr(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(TuringDBAdapter._cypher_literal(v) for v in value) + "]"
        # String: escape backslashes and single quotes
        s = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s}'"

    @staticmethod
    def _format_props(props: dict[str, Any]) -> str:
        """Format a dict as a Cypher property map {key: value, ...}."""
        if not props:
            return ""
        parts = [f"{k}: {TuringDBAdapter._cypher_literal(v)}" for k, v in props.items()]
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _df_val(val: Any) -> Any:
        """Convert a pandas value to a plain Python value (NaN → None)."""
        try:
            import pandas as pd
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        # Convert numpy/pandas int types to plain int
        if hasattr(val, "item"):
            return val.item()
        return val

    def _df_to_dicts(self, df: Any) -> list[dict[str, Any]]:
        """Convert a pandas DataFrame to a list of plain dicts."""
        if df is None or df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            records.append({col: self._df_val(row[col]) for col in df.columns})
        return records

    # ── Core operations ──────────────────────────────────────────

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 50,
    ) -> int:
        self._begin_write()
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = list(nodes[i : i + batch_size])
            # Multi-pattern CREATE: CREATE (:L {p}), (:L {p}), ...
            patterns = [f"(:{label} {self._format_props(node)})" for node in batch]
            query = "CREATE " + ", ".join(patterns)
            self._db.query(query)
            count += len(batch)
        self._end_write()
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        lit = self._cypher_literal(node_id)
        result = self._db.query(f"MATCH (n {{id: {lit}}}) RETURN n.id AS id")
        if result.empty:
            return None
        return {"id": self._df_val(result.iloc[0]["id"])}

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        self._begin_write()
        set_parts = [f"n.{k} = {self._cypher_literal(v)}" for k, v in properties.items()]
        set_clause = ", ".join(set_parts)
        lit = self._cypher_literal(node_id)
        result = self._db.query(f"MATCH (n {{id: {lit}}}) SET {set_clause} RETURN n.id AS id")
        self._end_write()
        return not result.empty

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        result = self._db.query(f"MATCH (n:{label}) RETURN n.id AS id LIMIT {limit}")
        return self._df_to_dicts(result)

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 15,
    ) -> int:
        count = 0

        # Group edges by type for batching
        by_type: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for src, tgt, etype, props in edges:
            by_type.setdefault(etype, []).append((src, tgt, props))

        # Commit every N edges to avoid huge changesets
        COMMIT_INTERVAL = 2000
        edges_in_change = 0
        self._begin_write()

        for etype, type_edges in by_type.items():
            for i in range(0, len(type_edges), batch_size):
                batch = type_edges[i : i + batch_size]

                # Multi-MATCH batch: one HTTP request for N edges
                # MATCH (a0 {id: 'x'}), (b0 {id: 'y'}), ... CREATE (a0)-[:T]->(b0), ...
                try:
                    match_parts = []
                    create_parts = []
                    for j, (src, tgt, _props) in enumerate(batch):
                        src_lit = self._cypher_literal(src)
                        tgt_lit = self._cypher_literal(tgt)
                        match_parts.append(f"(a{j} {{id: {src_lit}}}), (b{j} {{id: {tgt_lit}}})")
                        create_parts.append(f"(a{j})-[:{etype}]->(b{j})")
                    query = "MATCH " + ", ".join(match_parts) + " CREATE " + ", ".join(create_parts)
                    self._db.query(query)
                    count += len(batch)
                    edges_in_change += len(batch)
                except Exception:
                    # Fallback to individual inserts for this batch
                    for src, tgt, _props in batch:
                        src_lit = self._cypher_literal(src)
                        tgt_lit = self._cypher_literal(tgt)
                        try:
                            self._db.query(
                                f"MATCH (a {{id: {src_lit}}}), (b {{id: {tgt_lit}}}) "
                                f"CREATE (a)-[:{etype}]->(b)"
                            )
                            count += 1
                            edges_in_change += 1
                        except Exception:
                            pass

                # Periodic commit to avoid huge changesets
                if edges_in_change >= COMMIT_INTERVAL:
                    self._end_write()
                    self._begin_write()
                    edges_in_change = 0

        self._end_write()
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        lit = self._cypher_literal(node_id)
        if edge_type:
            query = f"MATCH (n {{id: {lit}}})-[:{edge_type}]->(m) RETURN m.id AS id"
        else:
            query = f"MATCH (n {{id: {lit}}})-->(m) RETURN m.id AS id"
        result = self._db.query(query)
        if result.empty:
            return []
        return [str(self._df_val(row["id"])) for _, row in result.iterrows() if self._df_val(row["id"]) is not None]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        src_lit = self._cypher_literal(source)
        tgt_lit = self._cypher_literal(target)
        rel = f":{edge_type}*" if edge_type else "*"
        try:
            result = self._db.query(
                f"MATCH (start {{id: {src_lit}}}), (end {{id: {tgt_lit}}}), "
                f"path = shortestPath((start)-[{rel}]->(end)) "
                f"RETURN [n IN nodes(path) | n.id] AS path"
            )
            if result.empty:
                return None
            path = self._df_val(result.iloc[0]["path"])
            if path:
                return [str(n) for n in path]
        except Exception:
            pass

        # BFS fallback
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
                    queue.append((neighbor, [*path, neighbor]))
        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if params:
            # Substitute $param with literal values (longest key first to avoid prefix collisions)
            for k, v in sorted(params.items(), key=lambda x: len(x[0]), reverse=True):
                query = query.replace(f"${k}", self._cypher_literal(v))

        # Detect write queries and wrap in a change if needed
        q_upper = query.strip().upper()
        is_write = any(kw in q_upper for kw in ["CREATE", "DELETE", "SET ", "REMOVE", "MERGE"])
        if is_write:
            self._begin_write()
        try:
            result = self._db.query(query)
        finally:
            if is_write:
                self._end_write()
        return self._df_to_dicts(result)

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN COUNT(n) AS count"
        else:
            query = "MATCH (n) RETURN COUNT(n) AS count"
        result = self._db.query(query)
        if result.empty:
            return 0
        return int(self._df_val(result.iloc[0]["count"]))

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = f"MATCH ()-[r:{edge_type}]->() RETURN COUNT(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN COUNT(r) AS count"
        result = self._db.query(query)
        if result.empty:
            return 0
        return int(self._df_val(result.iloc[0]["count"]))
