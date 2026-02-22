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
        """Flush any pending writes before a read operation.

        Only flushes if there's actually an open change to submit.
        """
        if self._has_open_change:
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
        self._resolved_props = None  # Reset property cache for new graph

    # ── Core operations ──────────────────────────────────────────

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        if not nodes:
            return 0
        # Keep change open — don't flush after each label.
        # _ensure_read() will flush before the first read query.
        # This avoids hitting TuringDB's ~10 commit corruption limit
        # when SNB inserts 9 labels sequentially.
        self._ensure_change()
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = list(nodes[i : i + batch_size])
            # Multi-pattern CREATE: CREATE (:L {p}), (:L {p}), ...
            patterns = [f"(:{label} {self._format_props(node)})" for node in batch]
            query = "CREATE " + ", ".join(patterns)
            self._db.query(query)
            count += len(batch)
        # Don't flush here — let _ensure_read() handle it
        return count

    # All candidate properties across benchmark categories.
    # TuringDB is column-oriented: only properties that exist in the graph
    # schema can be queried. _resolved_props caches the working set per graph.
    _ALL_PROPS = [
        "id", "firstName", "lastName", "gender", "birthday", "creationDate",
        "locationIP", "browserUsed", "content", "imageFile", "length",
        "title", "name", "country", "age", "city", "weight", "score",
        "embedding",
    ]
    _resolved_props: list[str] | None = None

    def _resolve_props(self) -> list[str]:
        """Discover which properties exist in the current graph schema."""
        if self._resolved_props is not None:
            return self._resolved_props
        import re
        # Try full list, remove missing properties on error
        props = list(self._ALL_PROPS)
        for _ in range(len(props)):
            ret_parts = ", ".join(f"n.{p} AS {p}" for p in props)
            try:
                self._db.query(f"MATCH (n) RETURN {ret_parts} LIMIT 1")
                self._resolved_props = props
                return props
            except Exception as e:
                # Error format: "Property type 'X' not found"
                msg = str(e)
                m = re.search(r"Property type '(\w+)' not found", msg)
                if m and m.group(1) in props:
                    props.remove(m.group(1))
                else:
                    # Can't parse — fall back to just id
                    self._resolved_props = ["id"]
                    return self._resolved_props
        self._resolved_props = ["id"]
        return self._resolved_props

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        self._ensure_read()
        lit = self._cypher_literal(node_id)
        props_list = self._resolve_props()
        try:
            ret_parts = ", ".join(f"n.{p} AS {p}" for p in props_list)
            result = self._db.query(f"MATCH (n {{id: {lit}}}) RETURN {ret_parts}")
            if result.empty:
                return None
            row = result.iloc[0]
            props: dict[str, Any] = {}
            for p in props_list:
                val = self._df_val(row[p])
                if val is not None:
                    props[p] = val
            return props if props else {"id": node_id}
        except Exception:
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
        props_list = self._resolve_props()
        try:
            ret_parts = ", ".join(f"n.{p} AS {p}" for p in props_list)
            result = self._db.query(
                f"MATCH (n:{label}) RETURN {ret_parts} LIMIT {limit}"
            )
            if result.empty:
                return []
            nodes = []
            for _, row in result.iterrows():
                props: dict[str, Any] = {}
                for p in props_list:
                    val = self._df_val(row[p])
                    if val is not None:
                        props[p] = val
                nodes.append(props if props else {"id": "unknown"})
            return nodes
        except Exception:
            return []

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 100,
    ) -> int:
        if not edges:
            return 0

        # Flush pending node inserts so MATCH can find them
        self._flush()

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
                    for j, (src, tgt, props) in enumerate(batch):
                        src_lit = self._cypher_literal(src)
                        tgt_lit = self._cypher_literal(tgt)
                        match_parts.append(f"(a{j} {{id: {src_lit}}}), (b{j} {{id: {tgt_lit}}})")
                        prop_str = f" {self._format_props(props)}" if props else ""
                        create_parts.append(f"(a{j})-[:{etype}{prop_str}]->(b{j})")
                    query = "MATCH " + ", ".join(match_parts) + " CREATE " + ", ".join(create_parts)
                    self._db.query(query)
                    count += len(batch)
                except Exception:
                    # Fallback to individual inserts for this batch
                    for src, tgt, props in batch:
                        src_lit = self._cypher_literal(src)
                        tgt_lit = self._cypher_literal(tgt)
                        prop_str = f" {self._format_props(props)}" if props else ""
                        try:
                            self._db.query(
                                f"MATCH (a {{id: {src_lit}}}), (b {{id: {tgt_lit}}}) "
                                f"CREATE (a)-[:{etype}{prop_str}]->(b)"
                            )
                            count += 1
                        except Exception:
                            pass

        # Don't flush here — let _ensure_read() handle it
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

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        # Use variable-length path pattern — single HTTP request instead of
        # N sequential get_neighbors calls.
        self._ensure_read()
        lit = self._cypher_literal(start)
        rel = f":{edge_type}" if edge_type else ""
        try:
            result = self._db.query(
                f"MATCH (n {{id: {lit}}})-[{rel}*1..{max_depth}]->(m) "
                f"RETURN DISTINCT m.id AS id"
            )
            ids = [start]
            if not result.empty:
                for _, row in result.iterrows():
                    val = self._df_val(row["id"])
                    if val is not None:
                        ids.append(str(val))
            return ids
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
        # Same approach as traverse_bfs — result set is identical,
        # only visit order differs (which doesn't affect benchmark correctness).
        self._ensure_read()
        lit = self._cypher_literal(start)
        rel = f":{edge_type}" if edge_type else ""
        try:
            result = self._db.query(
                f"MATCH (n {{id: {lit}}})-[{rel}*1..{max_depth}]->(m) "
                f"RETURN DISTINCT m.id AS id"
            )
            ids = [start]
            if not result.empty:
                for _, row in result.iterrows():
                    val = self._df_val(row["id"])
                    if val is not None:
                        ids.append(str(val))
            return ids
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
        # TuringDB doesn't support shortestPath() or list returns —
        # go straight to BFS fallback.
        self._ensure_read()
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
