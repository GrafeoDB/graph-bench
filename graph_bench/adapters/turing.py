r"""
TuringDB database adapter.

TuringDB is a high-performance in-memory column-oriented graph database
written in C++23, accessed via HTTP REST API with OpenCypher queries.

Write pattern: CHANGE NEW → writes → CHANGE COMMIT (git-like versioning).
The Python client's new_change() has a pandas bug, so we use raw
CHANGE NEW queries and extract the change_id from the result DataFrame.

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
        import time
        self._graph_prefix = f"b{int(time.time())}"
        self._graph_seq = 0

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

        # Create a fresh graph with unique name
        gname = f"{self._graph_prefix}_0"
        self._db.create_graph(gname)
        self._db.set_graph(gname)
        self._connected = True

    def disconnect(self) -> None:
        if self._db and self._has_open_change:
            self._flush()
        self._db = None
        self._connected = False

    # ── Write transaction management ───────────────────────────────
    # TuringDB corrupts its change state after ~10 commits on a graph.
    # Workaround: keep a persistent open change and batch writes into it.
    # _flush() submits the change; _ensure_read() flushes before reads.
    # clear() rotates to a fresh graph to guarantee a clean slate.

    _has_open_change: bool = False

    def _rotate_graph(self) -> None:
        """Switch to a fresh graph to avoid change-state corruption."""
        if self._has_open_change:
            self._flush()
        self._graph_seq += 1
        name = f"{self._graph_prefix}_{self._graph_seq}"
        self._db.create_graph(name)
        self._db.set_graph(name)
        self._has_open_change = False

    def _ensure_change(self) -> None:
        """Ensure a write change is open (create one if needed)."""
        if self._has_open_change:
            return
        try:
            result = self._db.query("CHANGE NEW")
            change_id = str(result.iloc[0, 0])
            self._db._params["change"] = change_id
            self._has_open_change = True
        except Exception:
            # Change state corrupted — rotate and retry
            self._rotate_graph()
            result = self._db.query("CHANGE NEW")
            change_id = str(result.iloc[0, 0])
            self._db._params["change"] = change_id
            self._has_open_change = True

    def _flush(self) -> None:
        """Submit the current change and checkout main."""
        if not self._has_open_change:
            return
        try:
            self._db.query("CHANGE SUBMIT")
        except Exception:
            pass
        self._db._params.pop("change", None)
        self._has_open_change = False
        try:
            self._db.checkout("main")
        except Exception:
            pass

    def _ensure_read(self) -> None:
        """Flush any pending writes before a read operation."""
        self._flush()

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
        # Use pandas C-level conversion instead of slow iterrows()
        import numpy as np
        records = df.where(df.notna(), other=None).to_dict("records")
        # Convert numpy types to plain Python
        return [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in records
        ]

    def clear(self) -> None:
        """Clear by switching to a fresh graph (avoids change-state corruption)."""
        self._rotate_graph()

    # ── Core operations ──────────────────────────────────────────

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 200,
    ) -> int:
        if not nodes:
            return 0
        self._ensure_change()
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = list(nodes[i : i + batch_size])
            # Multi-pattern CREATE: CREATE (:L {p}), (:L {p}), ...
            patterns = [f"(:{label} {self._format_props(node)})" for node in batch]
            query = "CREATE " + ", ".join(patterns)
            self._db.query(query)
            count += len(batch)
        self._flush()
        # Create index on id property for faster lookups (if supported)
        try:
            self._ensure_change()
            self._db.query("CREATE INDEX FOR (n:Node) ON (n.id)")
            self._flush()
        except Exception:
            self._flush()
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        self._ensure_read()
        lit = self._cypher_literal(node_id)
        result = self._db.query(f"MATCH (n {{id: {lit}}}) RETURN n")
        if result.empty:
            return None
        raw = self._df_val(result.iloc[0]["n"])
        if isinstance(raw, dict):
            return raw
        return {"id": node_id}

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        self._ensure_change()
        set_parts = [f"n.{k} = {self._cypher_literal(v)}" for k, v in properties.items()]
        set_clause = ", ".join(set_parts)
        lit = self._cypher_literal(node_id)
        result = self._db.query(f"MATCH (n {{id: {lit}}}) SET {set_clause} RETURN n.id AS id")
        return not result.empty

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_read()
        result = self._db.query(f"MATCH (n:{label}) RETURN n LIMIT {limit}")
        if result.empty:
            return []
        nodes = []
        for _, row in result.iterrows():
            raw = self._df_val(row["n"])
            if isinstance(raw, dict):
                nodes.append(raw)
            else:
                nodes.append({"id": str(raw)})
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 50,
    ) -> int:
        if not edges:
            return 0

        # Group edges by type for batching
        by_type: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for src, tgt, etype, props in edges:
            by_type.setdefault(etype, []).append((src, tgt, props))

        self._ensure_change()
        count = 0

        for etype, type_edges in by_type.items():
            for i in range(0, len(type_edges), batch_size):
                batch = type_edges[i : i + batch_size]

                # Multi-MATCH batch: one HTTP request for N edges
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
                        except Exception:
                            pass

        self._flush()
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        self._ensure_read()
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
        self._ensure_read()
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

        # Detect write queries and wrap in a change (don't flush per-query to batch writes)
        q_upper = query.strip().upper()
        is_write = any(kw in q_upper for kw in ["CREATE", "DELETE", "SET ", "REMOVE", "MERGE"])
        if is_write:
            self._ensure_change()
        else:
            self._ensure_read()
        result = self._db.query(query)
        return self._df_to_dicts(result)

    def count_nodes(self, *, label: str | None = None) -> int:
        self._ensure_read()
        if label:
            query = f"MATCH (n:{label}) RETURN COUNT(n) AS count"
        else:
            query = "MATCH (n) RETURN COUNT(n) AS count"
        result = self._db.query(query)
        if result.empty:
            return 0
        return int(self._df_val(result.iloc[0]["count"]))

    def count_edges(self, *, edge_type: str | None = None) -> int:
        self._ensure_read()
        if edge_type:
            query = f"MATCH ()-[r:{edge_type}]->() RETURN COUNT(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN COUNT(r) AS count"
        result = self._db.query(query)
        if result.empty:
            return 0
        return int(self._df_val(result.iloc[0]["count"]))
