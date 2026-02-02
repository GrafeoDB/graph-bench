r"""
NebulaGraph database adapter.

NebulaGraph is a distributed graph database using nGQL (Nebula Graph Query Language).

Requires: pip install nebula3-python

Environment variables:
    GRAPH_BENCH_NEBULA_HOST: Host (default: localhost)
    GRAPH_BENCH_NEBULA_PORT: Port (default: 9669)
    GRAPH_BENCH_NEBULA_USER: Username (default: root)
    GRAPH_BENCH_NEBULA_PASSWORD: Password (default: nebula)

    from graph_bench.adapters.nebula import NebulaGraphAdapter

    adapter = NebulaGraphAdapter()
    adapter.connect(host="localhost", port=9669)
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["NebulaGraphAdapter"]


@AdapterRegistry.register("nebula")
class NebulaGraphAdapter(BaseAdapter):
    """NebulaGraph distributed graph database adapter."""

    def __init__(self) -> None:
        self._pool: Any = None
        self._session: Any = None
        self._connected = False
        self._space = "benchmark"

    @property
    def name(self) -> str:
        return "NebulaGraph"

    @property
    def version(self) -> str:
        if not self._connected or self._session is None:
            return "unknown"
        try:
            result = self._session.execute("SHOW HOSTS")
            return "3.x"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from nebula3.gclient.net import ConnectionPool
            from nebula3.Config import Config
        except ImportError as e:
            msg = "nebula3-python package not installed. Install with: pip install nebula3-python"
            raise ImportError(msg) from e

        host = kwargs.get("host") or get_env("NEBULA_HOST", default="localhost")
        port = int(kwargs.get("port") or get_env("NEBULA_PORT", default="9669"))
        user = kwargs.get("user") or get_env("NEBULA_USER", default="root")
        password = kwargs.get("password") or get_env("NEBULA_PASSWORD", default="nebula")

        config = Config()
        config.max_connection_pool_size = 10

        self._pool = ConnectionPool()
        if not self._pool.init([(host, port)], config):
            msg = f"Failed to connect to NebulaGraph at {host}:{port}"
            raise ConnectionError(msg)

        self._session = self._pool.get_session(user, password)
        self._ensure_space()
        self._connected = True

    def _ensure_space(self) -> None:
        """Create space if not exists and use it."""
        # Create space
        create_space = f"""
        CREATE SPACE IF NOT EXISTS {self._space}
        (vid_type=FIXED_STRING(64), partition_num=1, replica_factor=1)
        """
        self._session.execute(create_space)

        # Use space
        self._session.execute(f"USE {self._space}")

        # Create tag (node label)
        self._session.execute(
            "CREATE TAG IF NOT EXISTS Node(id string, name string, value int)"
        )

        # Create edge type
        self._session.execute(
            "CREATE EDGE IF NOT EXISTS CONNECTS(weight double)"
        )

    def disconnect(self) -> None:
        if self._session:
            self._session.release()
        if self._pool:
            self._pool.close()
        self._session = None
        self._pool = None
        self._connected = False

    def clear(self) -> None:
        try:
            self._session.execute(f"CLEAR SPACE {self._space}")
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
            values = []
            for node in batch:
                node_id = node.get("id", f"n{count}")
                name = node.get("name", "")
                value = node.get("value", 0)
                values.append(f'"{node_id}":({label} {{id: "{node_id}", name: "{name}", value: {value}}})')
                count += 1

            if values:
                query = f"INSERT VERTEX {label}(id, name, value) VALUES {', '.join(values)}"
                self._session.execute(query)

        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        query = f'FETCH PROP ON Node "{node_id}" YIELD properties(vertex) AS props'
        result = self._session.execute(query)
        if result.is_succeeded() and result.row_size() > 0:
            props = result.row_values(0)[0].as_map()
            return {k: v.as_string() if hasattr(v, "as_string") else v for k, v in props.items()}
        return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        query = f"MATCH (n:{label}) RETURN n LIMIT {limit}"
        result = self._session.execute(query)
        nodes = []
        if result.is_succeeded():
            for i in range(result.row_size()):
                node = result.row_values(i)[0].as_node()
                props = node.properties(label)
                nodes.append({k: v.as_string() if hasattr(v, "as_string") else v for k, v in props.items()})
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
            values = []
            for src, tgt, edge_type, props in batch:
                weight = props.get("weight", 1.0)
                values.append(f'"{src}"->"{tgt}"@0:(CONNECTS {{weight: {weight}}})')
                count += 1

            if values:
                query = f"INSERT EDGE CONNECTS(weight) VALUES {', '.join(values)}"
                self._session.execute(query)

        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        query = f'GO FROM "{node_id}" OVER CONNECTS YIELD dst(edge) AS id'
        result = self._session.execute(query)
        neighbors = []
        if result.is_succeeded():
            for i in range(result.row_size()):
                val = result.row_values(i)[0]
                neighbors.append(val.as_string() if hasattr(val, "as_string") else str(val))
        return neighbors

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        query = f'FIND SHORTEST PATH FROM "{source}" TO "{target}" OVER * YIELD path AS p'
        result = self._session.execute(query)
        if result.is_succeeded() and result.row_size() > 0:
            path = result.row_values(0)[0].as_path()
            nodes = [path.start_node().get_id().as_string()]
            for step in path.steps():
                nodes.append(step.dst_id().as_string())
            return nodes
        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._session.execute(query)
        results = []
        if result.is_succeeded():
            cols = result.keys()
            for i in range(result.row_size()):
                row = {}
                for j, col in enumerate(cols):
                    val = result.row_values(i)[j]
                    row[col] = val.as_string() if hasattr(val, "as_string") else val
                results.append(row)
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"

        result = self._session.execute(query)
        if result.is_succeeded() and result.row_size() > 0:
            return result.row_values(0)[0].as_int()
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        query = "MATCH ()-[e]->() RETURN count(e) AS count"
        result = self._session.execute(query)
        if result.is_succeeded() and result.row_size() > 0:
            return result.row_values(0)[0].as_int()
        return 0
