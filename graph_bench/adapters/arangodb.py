r"""
ArangoDB database adapter.

Requires: pip install python-arango

Environment variables:
    GRAPH_BENCH_ARANGO_URI: Connection URI (default: http://localhost:8529)
    GRAPH_BENCH_ARANGO_USER: Username (default: root)
    GRAPH_BENCH_ARANGO_PASSWORD: Password
    GRAPH_BENCH_ARANGO_DATABASE: Database name (default: benchmark)

    from graph_bench.adapters.arangodb import ArangoDBAdapter

    adapter = ArangoDBAdapter()
    adapter.connect(uri="http://localhost:8529", database="benchmark")
"""

from collections import deque
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["ArangoDBAdapter"]


@AdapterRegistry.register("arangodb")
class ArangoDBAdapter(BaseAdapter):
    """ArangoDB multi-model database adapter."""

    def __init__(self) -> None:
        self._client: Any = None
        self._db: Any = None
        self._graph: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "ArangoDB"

    @property
    def version(self) -> str:
        if not self._connected or self._db is None:
            return "unknown"
        try:
            return self._db.version()["version"]
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from arango import ArangoClient
        except ImportError as e:
            msg = "python-arango package not installed. Install with: pip install python-arango"
            raise ImportError(msg) from e

        uri = uri or get_env("ARANGO_URI", default="http://localhost:8529")
        user = kwargs.get("user") or get_env("ARANGO_USER", default="root")
        password = kwargs.get("password") or get_env("ARANGO_PASSWORD", default="")
        database = kwargs.get("database") or get_env("ARANGO_DATABASE", default="benchmark")

        if uri is None:
            msg = "ArangoDB URI required"
            raise ValueError(msg)

        self._client = ArangoClient(hosts=uri)
        sys_db = self._client.db("_system", username=user, password=password)

        if not sys_db.has_database(database):
            sys_db.create_database(database)

        self._db = self._client.db(database, username=user, password=password)
        self._ensure_collections()
        self._connected = True

    def _ensure_collections(self) -> None:
        """Create default collections if not exist."""
        if not self._db.has_collection("nodes"):
            self._db.create_collection("nodes")
        if not self._db.has_collection("edges"):
            self._db.create_collection("edges", edge=True)

        if not self._db.has_graph("benchmark_graph"):
            edge_def = {
                "edge_collection": "edges",
                "from_vertex_collections": ["nodes"],
                "to_vertex_collections": ["nodes"],
            }
            self._graph = self._db.create_graph("benchmark_graph", edge_definitions=[edge_def])
        else:
            self._graph = self._db.graph("benchmark_graph")

    def disconnect(self) -> None:
        self._client = None
        self._db = None
        self._graph = None
        self._connected = False

    def clear(self) -> None:
        if self._db.has_collection("nodes"):
            self._db.collection("nodes").truncate()
        if self._db.has_collection("edges"):
            self._db.collection("edges").truncate()

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        collection = self._db.collection("nodes")
        count = 0

        for i in range(0, len(nodes), batch_size):
            batch = []
            for node in nodes[i : i + batch_size]:
                doc = dict(node)
                doc["_key"] = str(doc.get("id", count))
                doc["_label"] = label
                batch.append(doc)
                count += 1
            collection.insert_many(batch, overwrite=True)

        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        try:
            collection = self._db.collection("nodes")
            doc = collection.get(node_id)
            if doc:
                return {k: v for k, v in doc.items() if not k.startswith("_")}
            return None
        except Exception:
            return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        query = "FOR n IN nodes FILTER n._label == @label LIMIT @limit RETURN n"
        cursor = self._db.aql.execute(query, bind_vars={"label": label, "limit": limit})
        return [{k: v for k, v in doc.items() if not k.startswith("_")} for doc in cursor]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        collection = self._db.collection("edges")
        count = 0

        for i in range(0, len(edges), batch_size):
            batch = []
            for src, tgt, edge_type, props in edges[i : i + batch_size]:
                doc = dict(props)
                doc["_from"] = f"nodes/{src}"
                doc["_to"] = f"nodes/{tgt}"
                doc["_type"] = edge_type
                batch.append(doc)
                count += 1
            collection.insert_many(batch)

        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if edge_type:
            query = """
            FOR e IN edges
                FILTER e._from == @from AND e._type == @type
                RETURN PARSE_IDENTIFIER(e._to).key
            """
            cursor = self._db.aql.execute(query, bind_vars={"from": f"nodes/{node_id}", "type": edge_type})
        else:
            query = """
            FOR e IN edges
                FILTER e._from == @from
                RETURN PARSE_IDENTIFIER(e._to).key
            """
            cursor = self._db.aql.execute(query, bind_vars={"from": f"nodes/{node_id}"})

        return list(cursor)

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        if weighted:
            query = """
            FOR v, e IN OUTBOUND SHORTEST_PATH @source TO @target GRAPH 'benchmark_graph'
                OPTIONS {weightAttribute: 'weight'}
                RETURN v._key
            """
        else:
            query = """
            FOR v IN OUTBOUND SHORTEST_PATH @source TO @target GRAPH 'benchmark_graph'
                RETURN v._key
            """

        try:
            cursor = self._db.aql.execute(query, bind_vars={"source": f"nodes/{source}", "target": f"nodes/{target}"})
            path = list(cursor)
            return path if path else None
        except Exception:
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
        cursor = self._db.aql.execute(query, bind_vars=params or {})
        results = []
        for doc in cursor:
            if isinstance(doc, dict):
                results.append(doc)
            else:
                results.append({"value": doc})
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = "FOR n IN nodes FILTER n._label == @label COLLECT WITH COUNT INTO count RETURN count"
            cursor = self._db.aql.execute(query, bind_vars={"label": label})
        else:
            query = "FOR n IN nodes COLLECT WITH COUNT INTO count RETURN count"
            cursor = self._db.aql.execute(query)

        for count in cursor:
            return count
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = "FOR e IN edges FILTER e._type == @type COLLECT WITH COUNT INTO count RETURN count"
            cursor = self._db.aql.execute(query, bind_vars={"type": edge_type})
        else:
            query = "FOR e IN edges COLLECT WITH COUNT INTO count RETURN count"
            cursor = self._db.aql.execute(query)

        for count in cursor:
            return count
        return 0
