r"""
Grafeo Server database adapter (GQL over GWP/gRPC).

Grafeo Server uses the GQL Wire Protocol (GWP) — a gRPC-based binary protocol.
This adapter communicates directly via GWP using the bundled proto stubs.

Uses parameterized queries throughout (same approach as the Neo4j adapter).
Multi-CREATE with numbered params for node batching; per-edge parameterized
MATCH+CREATE for edges. RETURN n yields full node maps via GWP node_value.
Native algorithms via CALL procedures (pagerank, SSSP, WCC, etc.).

Works with grafeo-server:lite (GWP-only, no HTTP).

Requires: pip install grpcio

Environment variables:
    GRAPH_BENCH_GRAFEO_SERVER_URI: GWP URI (default: localhost:7687)

    from graph_bench.adapters.grafeo_server import GrafeoServerAdapter

    adapter = GrafeoServerAdapter()
    adapter.connect(uri="localhost:7687")
"""

import math
import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import grpc

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.adapters.proto import gql_service_pb2 as pb
from graph_bench.adapters.proto import gql_service_pb2_grpc as pb_grpc
from graph_bench.adapters.proto import gql_types_pb2 as types_pb
from graph_bench.config import get_env

__all__ = ["GrafeoServerAdapter"]


# ── Proto conversion helpers ────────────────────────────────────


def _proto_to_python(value: types_pb.Value) -> Any:
    """Convert a GWP proto Value to a Python value."""
    kind = value.WhichOneof("kind")
    if kind is None or kind == "null_value":
        return None
    if kind == "boolean_value":
        return value.boolean_value
    if kind == "integer_value":
        return value.integer_value
    if kind == "unsigned_integer_value":
        return value.unsigned_integer_value
    if kind == "float_value":
        return value.float_value
    if kind == "string_value":
        return value.string_value
    if kind == "bytes_value":
        return value.bytes_value
    if kind == "list_value":
        return [_proto_to_python(e) for e in value.list_value.elements]
    if kind == "record_value":
        return {
            f.name: _proto_to_python(f.value)
            for f in value.record_value.fields
        }
    if kind == "node_value":
        node = value.node_value
        props = {k: _proto_to_python(v) for k, v in node.properties.items()}
        props["_labels"] = list(node.labels)
        return props
    if kind == "edge_value":
        edge = value.edge_value
        props = {k: _proto_to_python(v) for k, v in edge.properties.items()}
        props["_labels"] = list(edge.labels)
        return props
    if kind == "path_value":
        path = value.path_value
        return {
            "nodes": [_proto_to_python(types_pb.Value(node_value=n)) for n in path.nodes],
            "edges": [_proto_to_python(types_pb.Value(edge_value=e)) for e in path.edges],
        }
    # Temporal types, big integer, etc. — return as string
    return str(value)


def _python_to_proto(value: Any) -> types_pb.Value:
    """Convert a Python value to a GWP proto Value."""
    if value is None:
        return types_pb.Value(null_value=types_pb.NullValue())
    if isinstance(value, bool):
        return types_pb.Value(boolean_value=value)
    if isinstance(value, int):
        return types_pb.Value(integer_value=value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return types_pb.Value(null_value=types_pb.NullValue())
        return types_pb.Value(float_value=value)
    if isinstance(value, str):
        return types_pb.Value(string_value=value)
    if isinstance(value, bytes):
        return types_pb.Value(bytes_value=value)
    if isinstance(value, (list, tuple)):
        elements = [_python_to_proto(v) for v in value]
        return types_pb.Value(list_value=types_pb.GqlList(elements=elements))
    if isinstance(value, dict):
        fields = [
            types_pb.Field(name=str(k), value=_python_to_proto(v))
            for k, v in value.items()
        ]
        return types_pb.Value(record_value=types_pb.Record(fields=fields))
    return types_pb.Value(string_value=str(value))


def _execute_and_collect(
    gql_stub: pb_grpc.GqlServiceStub,
    session_id: str,
    statement: str,
    *,
    transaction_id: str = "",
    parameters: dict[str, Any] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Execute a GQL statement via GWP and collect all result frames.

    Returns (columns, rows) where rows are lists of Python values.
    """
    params = {}
    if parameters:
        params = {k: _python_to_proto(v) for k, v in parameters.items()}

    req = pb.ExecuteRequest(
        session_id=session_id,
        statement=statement,
        parameters=params,
        transaction_id=transaction_id,
    )
    response_stream = gql_stub.Execute(req)

    columns: list[str] = []
    rows: list[list[Any]] = []

    for frame in response_stream:
        which = frame.WhichOneof("frame")
        if which == "header":
            columns = [col.name for col in frame.header.columns]
        elif which == "row_batch":
            for row in frame.row_batch.rows:
                rows.append([_proto_to_python(v) for v in row.values])

    return columns, rows


# ── Adapter ─────────────────────────────────────────────────────


@AdapterRegistry.register("grafeo-server")
class GrafeoServerAdapter(BaseAdapter):
    """Grafeo Server graph database adapter (GWP/gRPC).

    Communicates with grafeo-server via the GQL Wire Protocol.
    Uses parameterized queries throughout — same approach as the Neo4j adapter.
    """

    def __init__(self) -> None:
        self._db_name: str = "bench"
        self._connected = False
        self._gwp_uri: str = ""
        self._channel: grpc.Channel | None = None
        self._session_stub: pb_grpc.SessionServiceStub | None = None
        self._gql_stub: pb_grpc.GqlServiceStub | None = None
        self._db_stub: pb_grpc.DatabaseServiceStub | None = None
        # Session-per-thread pool: each thread gets its own GWP session
        self._thread_sessions: dict[int, str] = {}
        self._session_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "Grafeo Server"

    @property
    def version(self) -> str:
        if not self._connected or not self._session_stub:
            return "unknown"
        try:
            self._session_stub.Ping(
                pb.PingRequest(session_id=self._get_session_id())
            )
            return "0.4.2"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        default_uri = "localhost:7687"
        self._gwp_uri = (
            uri
            or get_env("GRAFEO_SERVER_URI", default=default_uri)
            or default_uri
        )
        # Strip protocol prefix if present
        for prefix in ("http://", "https://", "grpc://"):
            if self._gwp_uri.startswith(prefix):
                self._gwp_uri = self._gwp_uri[len(prefix):]
        self._gwp_uri = self._gwp_uri.rstrip("/")

        self._channel = grpc.insecure_channel(
            self._gwp_uri,
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        self._session_stub = pb_grpc.SessionServiceStub(self._channel)
        self._gql_stub = pb_grpc.GqlServiceStub(self._channel)
        self._db_stub = pb_grpc.DatabaseServiceStub(self._channel)

        # Create benchmark database via DatabaseService (ignore if exists)
        try:
            self._db_stub.CreateDatabase(
                pb.CreateDatabaseRequest(
                    name=self._db_name,
                    database_type="Lpg",
                    storage_mode="InMemory",
                )
            )
        except grpc.RpcError:
            pass  # Already exists

        # Create initial GWP session for the current (main) thread
        resp = self._session_stub.Handshake(
            pb.HandshakeRequest(
                protocol_version=1,
                client_info={"client": "graph-bench", "version": "1.0"},
            )
        )
        self._session_stub.Configure(
            pb.ConfigureRequest(
                session_id=resp.session_id,
                graph=self._db_name,
            )
        )
        self._thread_sessions[threading.get_ident()] = resp.session_id
        self._connected = True

    def disconnect(self) -> None:
        self._close_all_sessions()
        if self._channel:
            self._channel.close()
            self._channel = None
        self._session_stub = None
        self._gql_stub = None
        self._db_stub = None
        self._connected = False

    def _get_session_id(self) -> str:
        """Get or create a GWP session for the current thread.

        Each thread gets its own independent GWP session, enabling
        true concurrent operations (separate transactions, streams).
        """
        tid = threading.get_ident()
        if tid not in self._thread_sessions:
            with self._session_lock:
                if tid not in self._thread_sessions:
                    resp = self._session_stub.Handshake(
                        pb.HandshakeRequest(
                            protocol_version=1,
                            client_info={"client": "graph-bench", "version": "1.0"},
                        )
                    )
                    self._session_stub.Configure(
                        pb.ConfigureRequest(
                            session_id=resp.session_id,
                            graph=self._db_name,
                        )
                    )
                    self._thread_sessions[tid] = resp.session_id
        return self._thread_sessions[tid]

    def _close_all_sessions(self) -> None:
        """Close all thread-local GWP sessions."""
        with self._session_lock:
            for sid in self._thread_sessions.values():
                try:
                    self._session_stub.Close(
                        pb.CloseRequest(session_id=sid)
                    )
                except Exception:
                    pass
            self._thread_sessions.clear()

    # ── GWP query helpers ───────────────────────────────────────

    def _query(
        self, gql: str, *, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a GQL query via GWP and return rows as dicts."""
        columns, rows = _execute_and_collect(
            self._gql_stub, self._get_session_id(), gql,
            parameters=parameters,
        )
        return [
            {col: row[i] for i, col in enumerate(columns)}
            for row in rows
        ]

    def _exec(
        self, gql: str, *, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a GQL query via GWP and return raw result."""
        columns, rows = _execute_and_collect(
            self._gql_stub, self._get_session_id(), gql,
            parameters=parameters,
        )
        return {"columns": columns, "rows": rows}

    def _tx_begin(self) -> str:
        """Begin a transaction via GWP, return transaction_id."""
        resp = self._gql_stub.BeginTransaction(
            pb.BeginRequest(
                session_id=self._get_session_id(),
                mode=pb.READ_WRITE,
            )
        )
        return resp.transaction_id

    def _tx_query(
        self, tx_id: str, gql: str, *, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a query within a GWP transaction."""
        columns, rows = _execute_and_collect(
            self._gql_stub,
            self._get_session_id(),
            gql,
            transaction_id=tx_id,
            parameters=parameters,
        )
        return {"columns": columns, "rows": rows}

    def _tx_commit(self, tx_id: str) -> None:
        """Commit a GWP transaction."""
        self._gql_stub.Commit(
            pb.CommitRequest(
                session_id=self._get_session_id(),
                transaction_id=tx_id,
            )
        )

    # ── Database lifecycle ──────────────────────────────────────

    def clear(self) -> None:
        """Drop and recreate the benchmark database.

        Closes ALL thread sessions (database is being dropped), then
        creates a fresh session for the current thread. Other threads
        will lazily create new sessions on next access.
        """
        self._close_all_sessions()

        try:
            self._db_stub.DeleteDatabase(
                pb.DeleteDatabaseRequest(name=self._db_name)
            )
        except grpc.RpcError:
            pass

        self._db_stub.CreateDatabase(
            pb.CreateDatabaseRequest(
                name=self._db_name,
                database_type="Lpg",
                storage_mode="InMemory",
            )
        )

        # Create a fresh session for the current thread
        resp = self._session_stub.Handshake(
            pb.HandshakeRequest(
                protocol_version=1,
                client_info={"client": "graph-bench", "version": "1.0"},
            )
        )
        self._session_stub.Configure(
            pb.ConfigureRequest(
                session_id=resp.session_id,
                graph=self._db_name,
            )
        )
        self._thread_sessions[threading.get_ident()] = resp.session_id

    # ── Core operations ─────────────────────────────────────────

    @staticmethod
    def _strip_internal(node: Any) -> dict[str, Any]:
        """Strip internal fields (_id, _labels) from a node map."""
        if isinstance(node, dict):
            return {k: v for k, v in node.items()
                    if k not in ("_id", "_labels")}
        return {}

    @staticmethod
    def _gql_literal(value: Any) -> str:
        """Convert a Python value to a GQL inline literal."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "null"
            return repr(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(
                GrafeoServerAdapter._gql_literal(v) for v in value
            ) + "]"
        # String: escape backslashes and double quotes
        s = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    @staticmethod
    def _gql_map(props: dict[str, Any]) -> str:
        """Format a dict as a GQL inline map {key: value, ...}."""
        parts = [
            f"{k}: {GrafeoServerAdapter._gql_literal(v)}"
            for k, v in props.items()
        ]
        return "{" + ", ".join(parts) + "}"

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        """Insert nodes using UNWIND with inline map literals.

        GWP's proto encoding breaks list-of-maps params, so we
        inline the map values directly in the GQL statement.
        UNWIND+CREATE works with inline maps on local engine build.
        """
        if not nodes:
            return 0
        # Get property keys from first node (all nodes have same schema)
        prop_keys = list(nodes[0].keys())

        count = 0
        sid = self._tx_begin()
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            # Build inline list of maps
            maps = ", ".join(self._gql_map(node) for node in batch)
            # Build property assignment: {id: props.id, name: props.name, ...}
            assigns = ", ".join(f"{k}: props.{k}" for k in prop_keys)
            query = (
                f"UNWIND [{maps}] AS props "
                f"CREATE (:{label}:Node {{{assigns}}})"
            )
            self._tx_query(sid, query)
            count += len(batch)
        self._tx_commit(sid)
        try:
            self._exec("CREATE INDEX FOR (n:Node) ON (n.id)")
        except Exception:
            pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "MATCH (n:Node {id: $id}) RETURN n",
            parameters={"id": node_id},
        )
        if rows:
            return self._strip_internal(rows[0]["n"])
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        # SET n += $props is not supported; set each property individually
        set_parts = []
        params: dict[str, Any] = {"id": node_id}
        for k, v in properties.items():
            pname = f"prop_{k}"
            params[pname] = v
            set_parts.append(f"n.{k} = ${pname}")
        sid = self._tx_begin()
        try:
            result = self._tx_query(
                sid,
                f"MATCH (n:Node {{id: $id}}) SET {', '.join(set_parts)} RETURN n.id AS id",
                parameters=params,
            )
            self._tx_commit(sid)
            return bool(result.get("rows"))
        except Exception:
            return False

    def get_nodes_by_label(
        self, label: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = self._query(
            f"MATCH (n:{label}) RETURN n LIMIT {int(limit)}",
        )
        return [self._strip_internal(row["n"]) for row in rows]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 200,
    ) -> int:
        """Insert edges using batched UNWIND MATCH+CREATE.

        Groups edges by type, then batches each group into UNWIND
        statements with inline map literals.
        """
        by_type: dict[
            str, list[tuple[str, str, dict[str, Any]]]
        ] = defaultdict(list)
        for src, tgt, etype, props in edges:
            by_type[etype].append((src, tgt, props))

        count = 0
        commit_interval = 2000
        edges_in_tx = 0
        sid = self._tx_begin()

        for etype, type_edges in by_type.items():
            # Check if edges have extra properties
            has_props = any(props for _, _, props in type_edges)
            prop_keys = []
            if has_props:
                for _, _, props in type_edges:
                    if props:
                        prop_keys = list(props.keys())
                        break

            for i in range(0, len(type_edges), batch_size):
                batch = type_edges[i : i + batch_size]
                # Build inline list of edge maps
                edge_maps = []
                for src, tgt, props in batch:
                    m: dict[str, Any] = {"src": src, "tgt": tgt}
                    if props:
                        m.update(props)
                    edge_maps.append(self._gql_map(m))
                maps_str = ", ".join(edge_maps)

                if has_props and prop_keys:
                    prop_assigns = " {" + ", ".join(
                        f"{k}: e.{k}" for k in prop_keys
                    ) + "}"
                else:
                    prop_assigns = ""

                query = (
                    f"UNWIND [{maps_str}] AS e "
                    f"MATCH (a:Node {{id: e.src}}), "
                    f"(b:Node {{id: e.tgt}}) "
                    f"CREATE (a)-[:{etype}{prop_assigns}]->(b)"
                )
                self._tx_query(sid, query)
                count += len(batch)
                edges_in_tx += len(batch)

                if edges_in_tx >= commit_interval:
                    self._tx_commit(sid)
                    sid = self._tx_begin()
                    edges_in_tx = 0

        self._tx_commit(sid)
        return count

    def get_neighbors(
        self, node_id: str, *, edge_type: str | None = None
    ) -> list[str]:
        if edge_type:
            query = f"MATCH (n:Node {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
        else:
            query = "MATCH (n:Node {id: $id})-[r]->(m) RETURN m.id AS id"
        rows = self._query(query, parameters={"id": node_id})
        return [str(r["id"]) for r in rows if r.get("id") is not None]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        # BFS fallback (same as base adapter)
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

    def _nid_map(self) -> dict[int, str]:
        """Build internal node ID → app ID mapping."""
        mapping = self._query(
            "MATCH (n) RETURN id(n) AS nid, n.id AS app_id"
        )
        return {row["nid"]: str(row["app_id"]) for row in mapping}

    def _resolve_nid(self, app_id: str) -> int | None:
        """Resolve app ID to internal node ID."""
        rows = self._query(
            "MATCH (n:Node {id: $id}) RETURN id(n) AS nid",
            parameters={"id": app_id},
        )
        return rows[0]["nid"] if rows else None

    def sssp(
        self, source: str, *, weight_attr: str = "weight",
    ) -> dict[str, float]:
        """LDBC SSSP using CALL grafeo.sssp()."""
        src_nid = self._resolve_nid(source)
        if src_nid is None:
            return super().sssp(source, weight_attr=weight_attr)
        nid_to_app = self._nid_map()
        try:
            results = self._query(
                f"CALL grafeo.sssp('{src_nid}', "
                f"'{weight_attr}') "
                f"YIELD node_id, distance",
            )
            return {
                nid_to_app.get(r["node_id"], str(r["node_id"])):
                    float(r["distance"])
                for r in results
            }
        except Exception:
            return super().sssp(source, weight_attr=weight_attr)

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank via CALL grafeo.pagerank()."""
        nid_to_app = self._nid_map()
        try:
            results = self._query(
                "CALL grafeo.pagerank() YIELD node_id, score"
            )
            return {
                nid_to_app.get(r["node_id"], str(r["node_id"])):
                    float(r["score"])
                for r in results
            }
        except Exception:
            raise NotImplementedError(
                f"{self.name} pagerank procedure failed"
            )

    def community_detection(
        self, *, algorithm: str = "louvain"
    ) -> list[set[str]]:
        """Community detection via CALL grafeo.louvain/label_propagation."""
        nid_to_app = self._nid_map()
        try:
            if algorithm == "louvain":
                results = self._query(
                    "CALL grafeo.louvain() YIELD node_id, community_id"
                )
            else:
                results = self._query(
                    "CALL grafeo.label_propagation() "
                    "YIELD node_id, community_id"
                )
            communities: dict[int, set[str]] = {}
            for r in results:
                cid = r["community_id"]
                app_id = nid_to_app.get(
                    r["node_id"], str(r["node_id"])
                )
                if cid not in communities:
                    communities[cid] = set()
                communities[cid].add(app_id)
            return list(communities.values())
        except Exception:
            raise NotImplementedError(
                f"{self.name} community detection failed"
            )

    def bfs_levels(self, source: str) -> dict[str, int]:
        """LDBC BFS via CALL grafeo.bfs()."""
        src_nid = self._resolve_nid(source)
        if src_nid is None:
            return super().bfs_levels(source)
        nid_to_app = self._nid_map()
        try:
            results = self._query(
                f"CALL grafeo.bfs({src_nid}) "
                f"YIELD node_id, depth"
            )
            return {
                nid_to_app.get(r["node_id"], str(r["node_id"])):
                    int(r["depth"])
                for r in results
            }
        except Exception:
            return super().bfs_levels(source)

    def weakly_connected_components(self) -> list[set[str]]:
        """LDBC WCC via CALL grafeo.connected_components()."""
        nid_to_app = self._nid_map()
        try:
            results = self._query(
                "CALL grafeo.connected_components() "
                "YIELD node_id, component_id"
            )
            components: dict[int, set[str]] = {}
            for r in results:
                cid = r["component_id"]
                app_id = nid_to_app.get(
                    r["node_id"], str(r["node_id"])
                )
                if cid not in components:
                    components[cid] = set()
                components[cid].add(app_id)
            return list(components.values())
        except Exception:
            return super().weakly_connected_components()

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LDBC LCC via CALL grafeo.clustering_coefficient()."""
        nid_to_app = self._nid_map()
        try:
            results = self._query(
                "CALL grafeo.clustering_coefficient() "
                "YIELD node_id, coefficient"
            )
            return {
                nid_to_app.get(r["node_id"], str(r["node_id"])):
                    float(r["coefficient"])
                for r in results
            }
        except Exception:
            return super().local_clustering_coefficient()

    def execute_query(
        self, query: str, *, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        q_upper = query.strip().upper()
        is_write = any(
            kw in q_upper
            for kw in ["CREATE", "DELETE", "SET ", "REMOVE", "MERGE"]
        )
        if is_write:
            sid = self._tx_begin()
            try:
                result = self._tx_query(sid, query, parameters=params)
                self._tx_commit(sid)
                columns = result.get("columns", [])
                rows = result.get("rows", [])
                return [
                    {col: row[i] for i, col in enumerate(columns)}
                    for row in rows
                ]
            except Exception:
                return []
        return self._query(query, parameters=params)

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
            query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"
        rows = self._query(query)
        if rows:
            return int(rows[0]["count"])
        return 0
