r"""
FalkorDB Lite embedded database adapter.

FalkorDB Lite is an embedded version of FalkorDB using redislite
(embedded Redis with FalkorDB module). Same Cypher query API as FalkorDB
server but runs in-process via a Unix domain socket.

Requires: pip install falkordblite
Note: Linux/macOS only (WSL2 on Windows). The package installs as
``redislite`` module, not ``falkordblite``.

Environment variables:
    GRAPH_BENCH_FALKORDBLITE_PATH: Database file path (default: ./falkordblite_bench.db)

    from graph_bench.adapters.falkordblite import FalkorDBLiteAdapter

    adapter = FalkorDBLiteAdapter()
    adapter.connect()
"""

import os
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["FalkorDBLiteAdapter"]


@AdapterRegistry.register("falkordblite")
class FalkorDBLiteAdapter(BaseAdapter):
    """FalkorDB Lite embedded graph database adapter."""

    def __init__(self) -> None:
        self._client: Any = None
        self._graph: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "FalkorDB Lite"

    @property
    def is_embedded(self) -> bool:
        return True

    @property
    def child_pids(self) -> list[int]:
        """Return redis-server PID spawned by redislite."""
        if self._client is not None:
            try:
                pid = self._client.connection.pid
                if pid:
                    return [pid]
            except Exception:
                pass
        return []

    @property
    def version(self) -> str:
        if not self._connected or self._client is None:
            return "unknown"
        try:
            info = self._client.connection.info()
            return info.get("falkordb_version", info.get("redis_version", "unknown"))
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        # Monkey-patch redis.connection for falkordblite 0.8.0 + redis-py 7.2.0
        # UnixDomainSocketConnection is missing .port attribute
        try:
            import redis.connection

            if not hasattr(redis.connection.UnixDomainSocketConnection, "port"):
                redis.connection.UnixDomainSocketConnection.port = 0  # type: ignore[attr-defined]
        except ImportError:
            pass

        try:
            from redislite.falkordb_client import FalkorDB
        except ImportError as e:
            msg = "falkordblite package not installed. Install with: pip install falkordblite"
            raise ImportError(msg) from e

        db_path = (
            kwargs.get("path")
            or get_env("FALKORDBLITE_PATH", default=None)
            or "./falkordblite_bench.db"
        )

        self._client = FalkorDB(db_path)
        self._graph = self._client.select_graph("benchmark")
        self._connected = True

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.connection.close()
            except Exception:
                pass
        self._client = None
        self._graph = None
        self._connected = False

    def clear(self) -> None:
        try:
            self._graph.delete()
            self._graph = self._client.select_graph("benchmark")
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
            batch = list(nodes[i : i + batch_size])
            try:
                query = f"UNWIND $nodes AS node CREATE (n:{label}:Node) SET n = node"
                self._graph.query(query, {"nodes": batch})
                count += len(batch)
            except Exception:
                for node in batch:
                    props = ", ".join(f"{k}: ${k}" for k in node.keys())
                    query = f"CREATE (n:{label}:Node {{{props}}})"
                    self._graph.query(query, node)
                    count += 1
        try:
            self._graph.query("CREATE INDEX FOR (n:Node) ON (n.id)")
        except Exception:
            pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        query = "MATCH (n:Node {id: $id}) RETURN n"
        result = self._graph.query(query, {"id": node_id})
        if result.result_set:
            node = result.result_set[0][0]
            return dict(node.properties)
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
        query = f"MATCH (n:Node {{id: $id}}) SET {set_clauses} RETURN n"
        params = {"id": node_id, **properties}
        result = self._graph.query(query, params)
        return len(result.result_set) > 0

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        query = f"MATCH (n:{label}) RETURN n LIMIT $limit"
        result = self._graph.query(query, {"limit": limit})
        return [dict(row[0].properties) for row in result.result_set]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            by_type: dict[str, list[dict[str, Any]]] = {}
            for src, tgt, edge_type, props in batch:
                if edge_type not in by_type:
                    by_type[edge_type] = []
                by_type[edge_type].append({"src": src, "tgt": tgt, "props": props})

            for edge_type, edge_list in by_type.items():
                try:
                    query = f"""
                    UNWIND $edges AS e
                    MATCH (a:Node {{id: e.src}}), (b:Node {{id: e.tgt}})
                    CREATE (a)-[r:{edge_type}]->(b)
                    SET r = e.props
                    """
                    self._graph.query(query, {"edges": edge_list})
                    count += len(edge_list)
                except Exception:
                    for edge in edge_list:
                        props = edge["props"]
                        props_str = ", ".join(f"{k}: ${k}" for k in props.keys())
                        props_clause = f" {{{props_str}}}" if props_str else ""
                        query = f"""
                        MATCH (a:Node {{id: $src}}), (b:Node {{id: $tgt}})
                        CREATE (a)-[r:{edge_type}{props_clause}]->(b)
                        """
                        params = {"src": edge["src"], "tgt": edge["tgt"], **props}
                        self._graph.query(query, params)
                        count += 1
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if edge_type:
            query = f"MATCH (n:Node {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
        else:
            query = "MATCH (n:Node {id: $id})-->(m) RETURN m.id AS id"

        result = self._graph.query(query, {"id": node_id})
        return [row[0] for row in result.result_set if row[0]]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        if edge_type:
            rel = f":{edge_type}*"
        else:
            rel = "*"

        query = f"""
        MATCH (start:Node {{id: $src}}), (end:Node {{id: $tgt}}),
              path = shortestPath((start)-[{rel}]->(end))
        RETURN [n IN nodes(path) | n.id] AS path
        """
        result = self._graph.query(query, {"src": source, "tgt": target})
        if result.result_set:
            return result.result_set[0][0]
        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._graph.query(query, params or {})
        results = []
        for row in result.result_set:
            if len(row) == 1:
                val = row[0]
                if hasattr(val, "properties"):
                    results.append(dict(val.properties))
                else:
                    results.append({"value": val})
            else:
                results.append({f"col{i}": v for i, v in enumerate(row)})
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"

        result = self._graph.query(query)
        if result.result_set:
            return result.result_set[0][0]
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"

        result = self._graph.query(query)
        if result.result_set:
            return result.result_set[0][0]
        return 0

    # --- Native graph algorithms (3/6 LDBC) ---

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank via FalkorDB native algo.pageRank procedure."""
        try:
            result = self._graph.query(
                "CALL algo.pageRank(null, null) YIELD node, score "
                "RETURN node.id AS id, score"
            )
            return {
                str(row[0]): float(row[1])
                for row in result.result_set
                if row[0] is not None
            }
        except Exception:
            return super().pagerank(
                damping=damping,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )

    def weakly_connected_components(self) -> list[set[str]]:
        """WCC via FalkorDB native algo.WCC procedure."""
        try:
            result = self._graph.query(
                "CALL algo.WCC(null, null) YIELD node, componentId "
                "RETURN node.id AS id, componentId"
            )
            components: dict[int, set[str]] = {}
            for row in result.result_set:
                if row[0] is not None:
                    cid = int(row[1])
                    if cid not in components:
                        components[cid] = set()
                    components[cid].add(str(row[0]))
            return list(components.values())
        except Exception:
            return super().weakly_connected_components()

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection via FalkorDB native algo.labelPropagation."""
        if algorithm == "label_propagation":
            try:
                result = self._graph.query(
                    "CALL algo.labelPropagation(null, null) YIELD node, communityId "
                    "RETURN node.id AS id, communityId"
                )
                communities: dict[int, set[str]] = {}
                for row in result.result_set:
                    if row[0] is not None:
                        cid = int(row[1])
                        if cid not in communities:
                            communities[cid] = set()
                        communities[cid].add(str(row[0]))
                return list(communities.values())
            except Exception:
                return super().community_detection(algorithm=algorithm)
        return super().community_detection(algorithm=algorithm)
