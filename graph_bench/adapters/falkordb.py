r"""
FalkorDB database adapter.

FalkorDB is a graph database built on Redis, using Cypher query language.
Successor to RedisGraph.

Requires: pip install FalkorDB

Environment variables:
    GRAPH_BENCH_FALKORDB_HOST: Host (default: localhost)
    GRAPH_BENCH_FALKORDB_PORT: Port (default: 6379)
    GRAPH_BENCH_FALKORDB_PASSWORD: Password (optional)

    from graph_bench.adapters.falkordb import FalkorDBAdapter

    adapter = FalkorDBAdapter()
    adapter.connect(host="localhost", port=6379)
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["FalkorDBAdapter"]


@AdapterRegistry.register("falkordb")
class FalkorDBAdapter(BaseAdapter):
    """FalkorDB graph database adapter."""

    def __init__(self) -> None:
        self._client: Any = None
        self._graph: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "FalkorDB"

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
        try:
            from falkordb import FalkorDB
        except ImportError as e:
            msg = "FalkorDB package not installed. Install with: pip install FalkorDB"
            raise ImportError(msg) from e

        host = kwargs.get("host") or get_env("FALKORDB_HOST", default="localhost")
        port = int(kwargs.get("port") or get_env("FALKORDB_PORT", default="6379"))
        password = kwargs.get("password") or get_env("FALKORDB_PASSWORD")

        self._client = FalkorDB(host=host, port=port, password=password)
        self._graph = self._client.select_graph("benchmark")
        self._connected = True

    def disconnect(self) -> None:
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
                # Use UNWIND for batch insert
                query = f"UNWIND $nodes AS node CREATE (n:{label}) SET n = node"
                self._graph.query(query, {"nodes": batch})
                count += len(batch)
            except Exception:
                # Fall back to individual inserts if UNWIND fails
                for node in batch:
                    props = ", ".join(f"{k}: ${k}" for k in node.keys())
                    query = f"CREATE (n:{label} {{{props}}})"
                    self._graph.query(query, node)
                    count += 1
        # Create index on id for this label to speed up MATCH in insert_edges
        try:
            self._graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.id)")
        except Exception:
            pass  # Index may already exist
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        query = "MATCH (n {id: $id}) RETURN n"
        result = self._graph.query(query, {"id": node_id})
        if result.result_set:
            node = result.result_set[0][0]
            return dict(node.properties)
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        # Build SET clause dynamically
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
        query = f"MATCH (n {{id: $id}}) SET {set_clauses} RETURN n"
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
            # Group edges by type for UNWIND (Cypher requires static relationship types)
            by_type: dict[str, list[dict[str, Any]]] = {}
            for src, tgt, edge_type, props in batch:
                if edge_type not in by_type:
                    by_type[edge_type] = []
                by_type[edge_type].append({"src": src, "tgt": tgt, "props": props})

            for edge_type, edge_list in by_type.items():
                try:
                    query = f"""
                    UNWIND $edges AS e
                    MATCH (a {{id: e.src}}), (b {{id: e.tgt}})
                    CREATE (a)-[r:{edge_type}]->(b)
                    SET r = e.props
                    """
                    self._graph.query(query, {"edges": edge_list})
                    count += len(edge_list)
                except Exception:
                    # Fall back to individual inserts
                    for edge in edge_list:
                        props = edge["props"]
                        props_str = ", ".join(f"{k}: ${k}" for k in props.keys())
                        props_clause = f" {{{props_str}}}" if props_str else ""
                        query = f"""
                        MATCH (a {{id: $src}}), (b {{id: $tgt}})
                        CREATE (a)-[r:{edge_type}{props_clause}]->(b)
                        """
                        params = {"src": edge["src"], "tgt": edge["tgt"], **props}
                        self._graph.query(query, params)
                        count += 1
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if edge_type:
            query = f"MATCH (n {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
        else:
            query = "MATCH (n {id: $id})-->(m) RETURN m.id AS id"

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
        MATCH (start {{id: $src}}), (end {{id: $tgt}}),
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
