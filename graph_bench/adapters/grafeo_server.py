r"""
Grafeo Server database adapter (GQL over GWP/gRPC).

Grafeo Server speaks the GQL Wire Protocol (GWP), a gRPC protocol. This adapter
uses the bundled stubs in ``graph_bench.adapters.proto`` (gwp 0.2.1, the
protocol of grafeo-server 0.5.44): ``CatalogService`` creates the benchmark
graph, ``GqlService`` runs parameterized GQL, and native algorithms run as
``CALL grafeo.<algorithm>()`` procedures.

Works with the GWP-enabled images (``grafeo/grafeo-server:<version>-gwp`` and
``-full``) and with a local ``grafeo-server`` binary.

Requires: uv sync --extra grafeo-server (grpcio, protobuf)

Environment variables:
    GRAPH_BENCH_GRAFEO_SERVER_URI: GWP address (default: localhost:7688)

    from graph_bench.adapters.grafeo_server import GrafeoServerAdapter

    adapter = GrafeoServerAdapter()
    adapter.connect(uri="localhost:7688")

GWP reports query errors in the result summary instead of as gRPC errors.
The adapter checks every summary and raises ``GrafeoServerError``, so a failing
query fails its benchmark instead of being timed as an empty result.
"""

from __future__ import annotations

import functools
import math
import threading
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env
from graph_bench.types import UnsupportedOperation

__all__ = ["DEFAULT_URI", "GrafeoServerAdapter", "GrafeoServerError", "gwp_address"]

DEFAULT_URI = "localhost:7688"
_WRITE_ATTEMPTS = 5
_NO_VECTOR_INDEX = "this grafeo-server has no vector index (it needs the vector-index feature, as in the -full image)"
_SCHEMA = "default"
_INSTALL_HINT = "grafeo-server adapter needs grpcio and protobuf. Install with: uv sync --extra grafeo-server"


class GrafeoServerError(RuntimeError):
    """A GQL statement or transaction failed on the server (GQLSTATUS exception)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    @property
    def rolled_back(self) -> bool:
        """GQLSTATUS class 40 (transaction rollback): the server aborted it, a retry may succeed."""
        return self.code[:2] == "40"


@functools.cache
def _gwp() -> SimpleNamespace:
    """Import grpc and the GWP stubs on first use."""
    try:
        import grpc

        from graph_bench.adapters.proto import gql_service_pb2, gql_service_pb2_grpc, gql_types_pb2
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return SimpleNamespace(grpc=grpc, pb=gql_service_pb2, stubs=gql_service_pb2_grpc, types=gql_types_pb2)


def gwp_address(uri: str | None = None) -> str:
    """The GWP address to use: ``uri``, else GRAPH_BENCH_GRAFEO_SERVER_URI, else the default."""
    address = uri or get_env("GRAFEO_SERVER_URI", default=DEFAULT_URI) or DEFAULT_URI
    for prefix in ("http://", "https://", "grpc://"):
        address = address.removeprefix(prefix)
    return address.rstrip("/")


# ── GQLSTATUS ───────────────────────────────────────────────────


def _is_exception(code: str) -> bool:
    """Whether a GQLSTATUS code is an exception.

    Classes 00 (success), 01 (warning), 02 (no data) and 03 (information) are
    not; every other class is (ISO/IEC 39075, section 23).
    """
    status_class = code[:2]
    if len(status_class) < 2:
        return False
    return status_class[0].isalpha() or status_class >= "08"


def _check_status(message: Any) -> None:
    """Raise GrafeoServerError when a response's ``status`` is an exception."""
    if message.HasField("status") and _is_exception(message.status.code):
        raise GrafeoServerError(message.status.code, message.status.message)


# ── Value conversion ────────────────────────────────────────────


def _entity_to_python(entity: Any) -> dict[str, Any]:
    props = {k: _proto_to_python(v) for k, v in entity.properties.items()}
    props["_labels"] = list(entity.labels)
    return props


def _proto_to_python(value: Any) -> Any:
    """Convert a GWP proto Value to a Python value."""
    kind = value.WhichOneof("kind")
    if kind is None or kind == "null_value":
        return None
    if kind in ("boolean_value", "integer_value", "unsigned_integer_value", "float_value"):
        return getattr(value, kind)
    if kind in ("string_value", "bytes_value"):
        return getattr(value, kind)
    if kind == "list_value":
        return [_proto_to_python(e) for e in value.list_value.elements]
    if kind == "record_value":
        return {f.name: _proto_to_python(f.value) for f in value.record_value.fields}
    if kind == "node_value":
        return _entity_to_python(value.node_value)
    if kind == "edge_value":
        return _entity_to_python(value.edge_value)
    if kind == "path_value":
        path = value.path_value
        return {
            "nodes": [_entity_to_python(n) for n in path.nodes],
            "edges": [_entity_to_python(e) for e in path.edges],
        }
    # Temporal types, big integers and the like: keep their text form.
    return str(value)


def _python_to_proto(value: Any) -> Any:
    """Convert a Python value to a GWP proto Value."""
    types = _gwp().types
    if value is None:
        return types.Value(null_value=types.NullValue())
    if isinstance(value, bool):
        return types.Value(boolean_value=value)
    if isinstance(value, int):
        return types.Value(integer_value=value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return types.Value(null_value=types.NullValue())
        return types.Value(float_value=value)
    if isinstance(value, str):
        return types.Value(string_value=value)
    if isinstance(value, bytes):
        return types.Value(bytes_value=value)
    if isinstance(value, (list, tuple)):
        return types.Value(list_value=types.GqlList(elements=[_python_to_proto(v) for v in value]))
    if isinstance(value, dict):
        fields = [types.Field(name=str(k), value=_python_to_proto(v)) for k, v in value.items()]
        return types.Value(record_value=types.Record(fields=fields))
    return types.Value(string_value=str(value))


def _all_number_lists(values: Any) -> bool:
    """Whether every value given is a non-empty list of numbers (an embedding)."""
    seen = False
    for value in values:
        if value is None:
            continue
        if not (
            isinstance(value, list)
            and value
            and all(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
        ):
            return False
        seen = True
    return seen


@dataclass
class _Result:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=False)) for row in self.rows]


def _execute_and_collect(
    gql_stub: Any,
    session_id: str,
    statement: str,
    *,
    transaction_id: str = "",
    parameters: dict[str, Any] | None = None,
    timeout: int = 300,
) -> _Result:
    """Execute a GQL statement via GWP, collect every frame and check the summary."""
    pb = _gwp().pb
    request = pb.ExecuteRequest(
        session_id=session_id,
        statement=statement,
        parameters={k: _python_to_proto(v) for k, v in (parameters or {}).items()},
    )
    if transaction_id:
        # An empty but present transaction_id names a transaction that does not exist.
        request.transaction_id = transaction_id

    result = _Result()
    for frame in gql_stub.Execute(request, timeout=timeout):
        which = frame.WhichOneof("frame")
        if which == "header":
            result.columns = [col.name for col in frame.header.columns]
        elif which == "row_batch":
            result.rows.extend([_proto_to_python(v) for v in row.values] for row in frame.row_batch.rows)
        elif which == "summary":
            _check_status(frame.summary)
            result.counters = dict(frame.summary.counters)
    return result


# ── Adapter ─────────────────────────────────────────────────────


@AdapterRegistry.register("grafeo-server")
class GrafeoServerAdapter(BaseAdapter):
    """Grafeo Server graph database adapter (GWP/gRPC).

    Uses parameterized queries throughout, like the Neo4j adapter. Each thread
    gets its own GWP session over one shared HTTP/2 channel. Native vector
    search goes through the GWP AdminService (index) and SearchService (k-NN),
    which need a server built with the vector-index feature (the -full image).

    Runs in Python (and is labelled a fallback): shortest paths, DFS, BFS
    restricted to one edge type, and the directed (LDBC) clustering coefficient.
    """

    # CALL grafeo.clustering_coefficient() ignores edge direction.
    native_lcc_definitions = frozenset({"undirected"})

    def __init__(self, *, graph: str = "bench") -> None:
        self._graph = graph
        self._connected = False
        self._gwp_uri = ""
        self._server_version = "unknown"
        self._channel: Any = None
        self._session_stub: Any = None
        self._gql_stub: Any = None
        self._catalog_stub: Any = None
        self._admin_stub: Any = None
        self._search_stub: Any = None
        self._procedures: frozenset[str] = frozenset()
        self._vector_metric = "cosine"
        # Session-per-thread pool: each thread gets its own GWP session
        self._thread_sessions: dict[int, str] = {}
        self._session_lock = threading.Lock()
        # Shared thread pool for traversals that fan out over the channel
        self._pool: ThreadPoolExecutor | None = None

    @property
    def name(self) -> str:
        return "Grafeo Server"

    @property
    def version(self) -> str:
        """Version from the GWP handshake.

        grafeo-server 0.5.44 reports the version of its GWP library there
        (``gwp 0.2.1``), not its own.
        """
        return self._server_version

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        gwp = _gwp()
        self._gwp_uri = gwp_address(uri)

        self._channel = gwp.grpc.insecure_channel(
            self._gwp_uri,
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.optimization_target", "latency"),
                ("grpc.max_concurrent_streams", 100),
                ("grpc.initial_reconnect_backoff_ms", 100),
            ],
        )
        self._session_stub = gwp.stubs.SessionServiceStub(self._channel)
        self._gql_stub = gwp.stubs.GqlServiceStub(self._channel)
        self._catalog_stub = gwp.stubs.CatalogServiceStub(self._channel)
        self._admin_stub = gwp.stubs.AdminServiceStub(self._channel)
        self._search_stub = gwp.stubs.SearchServiceStub(self._channel)

        self._ensure_graph(replace=False)
        self._connected = True
        self._procedures = frozenset(row["name"] for row in self._query("CALL grafeo.procedures() YIELD name"))
        self._pool = ThreadPoolExecutor(max_workers=16)

    def disconnect(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._session_stub is not None:
            self._close_all_sessions()
        if self._channel:
            self._channel.close()
            self._channel = None
        self._session_stub = None
        self._gql_stub = None
        self._catalog_stub = None
        self._admin_stub = None
        self._search_stub = None
        self._connected = False

    # ── Sessions and graph lifecycle ────────────────────────────

    def _ensure_graph(self, *, replace: bool) -> None:
        """Create the benchmark graph, dropping an existing one when ``replace``."""
        pb = _gwp().pb
        listing = self._catalog_stub.ListGraphs(pb.ListGraphsRequest(schema=_SCHEMA))
        if self._graph in {graph.name for graph in listing.graphs}:
            if not replace:
                return
            self._catalog_stub.DropGraph(pb.DropGraphRequest(schema=_SCHEMA, name=self._graph))
        self._catalog_stub.CreateGraph(
            pb.CreateGraphRequest(schema=_SCHEMA, name=self._graph, open_type=True, storage_mode="InMemory")
        )

    def _new_session(self) -> str:
        pb = _gwp().pb
        resp = self._session_stub.Handshake(
            pb.HandshakeRequest(protocol_version=1, client_info={"client": "graph-bench"})
        )
        if resp.HasField("server_info"):
            self._server_version = f"gwp {resp.server_info.version}"
        self._session_stub.Configure(pb.ConfigureRequest(session_id=resp.session_id, graph=self._graph))
        return resp.session_id

    def _get_session_id(self) -> str:
        """Get or create the GWP session of the current thread."""
        tid = threading.get_ident()
        if tid not in self._thread_sessions:
            with self._session_lock:
                if tid not in self._thread_sessions:
                    self._thread_sessions[tid] = self._new_session()
        return self._thread_sessions[tid]

    def _close_all_sessions(self) -> None:
        """Close every thread's GWP session (closing also rolls back its transaction)."""
        gwp = _gwp()
        with self._session_lock:
            sessions = list(self._thread_sessions.values())
            self._thread_sessions.clear()
        for sid in sessions:
            try:
                self._session_stub.CloseSession(gwp.pb.CloseSessionRequest(session_id=sid))
            except gwp.grpc.RpcError as error:
                # The server already ended it (idle timeout); there is nothing left to close.
                if error.code() != gwp.grpc.StatusCode.NOT_FOUND:
                    raise

    def clear(self) -> None:
        """Drop and recreate the benchmark graph.

        Dropping the graph ends every session on it, so all thread sessions are
        closed first; each thread opens a new one on its next query.
        """
        self._close_all_sessions()
        self._ensure_graph(replace=True)

    # ── Query helpers ───────────────────────────────────────────

    def _run(self, gql: str, *, parameters: dict[str, Any] | None = None, transaction_id: str = "") -> _Result:
        return _execute_and_collect(
            self._gql_stub,
            self._get_session_id(),
            gql,
            transaction_id=transaction_id,
            parameters=parameters,
        )

    def _query(self, gql: str, *, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a GQL query (auto-commit) and return rows as dicts."""
        return self._run(gql, parameters=parameters).records()

    @contextmanager
    def _transaction(self) -> Iterator[str]:
        """Run the block in a read-write transaction: commit on success, roll back on error."""
        pb = _gwp().pb
        sid = self._get_session_id()
        begin = self._gql_stub.BeginTransaction(pb.BeginRequest(session_id=sid, mode=pb.READ_WRITE))
        _check_status(begin)
        tx_id = begin.transaction_id
        try:
            yield tx_id
        except BaseException as error:
            try:
                self._gql_stub.Rollback(pb.RollbackRequest(session_id=sid, transaction_id=tx_id))
            except Exception as rollback_error:
                error.add_note(f"rollback also failed: {rollback_error}")
            raise
        _check_status(self._gql_stub.Commit(pb.CommitRequest(session_id=sid, transaction_id=tx_id)))

    @staticmethod
    def _strip_internal(node: Any) -> dict[str, Any]:
        """Strip internal fields (_id, _labels) from a node map."""
        if isinstance(node, dict):
            return {k: v for k, v in node.items() if k not in ("_id", "_labels")}
        return {}

    # ── Core operations ─────────────────────────────────────────

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        """Insert nodes with a parameterized UNWIND, one transaction per call."""
        if not nodes:
            return 0
        labels = f"{label}:Node" if label != "Node" else "Node"
        # The union of keys, so a property missing from the first node is kept.
        keys = sorted({key for node in nodes for key in node})
        # GWP has no vector type: a list parameter arrives as a list, which the
        # vector index leaves out, so lists of numbers are stored with vector().
        vectors = {key for key in keys if _all_number_lists(node.get(key) for node in nodes)}
        assigns = ", ".join(f"{key}: vector(props.{key})" if key in vectors else f"{key}: props.{key}" for key in keys)
        query = f"UNWIND $nodes AS props CREATE (:{labels} {{{assigns}}})"

        count = 0
        with self._transaction() as tx_id:
            for start in range(0, len(nodes), batch_size):
                batch = [dict(node) for node in nodes[start : start + batch_size]]
                result = self._run(query, parameters={"nodes": batch}, transaction_id=tx_id)
                count += result.counters.get("nodes_created", 0)
        self._ensure_id_index()
        return count

    def _ensure_id_index(self) -> None:
        # FAIRNESS.md allows one index on `id`, created after the insert.
        indexes = self._query("SHOW INDEXES")
        if not any(ix.get("label") == "Node" and ix.get("property") == "id" for ix in indexes):
            self._query("CREATE INDEX idx_node_id FOR (n:Node) ON (n.id)")

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._query("MATCH (n:Node {id: $id}) RETURN n", parameters={"id": node_id})
        if rows:
            return self._strip_internal(rows[0]["n"])
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Merge properties in one auto-commit statement, like the embedded and Neo4j adapters.

        Concurrent updates of one node (the concurrent and ACID benchmarks) can
        abort with a write-write conflict; a rolled-back statement is retried,
        as a driver retries a transient error. grafeo-server 0.5.44 reports the
        conflict as 42001 instead of class 40, so it is not retried there.
        """
        for attempt in range(_WRITE_ATTEMPTS):
            try:
                rows = self._query(
                    "MATCH (n:Node {id: $id}) SET n += $props RETURN n.id AS id",
                    parameters={"id": node_id, "props": dict(properties)},
                )
                return bool(rows)
            except GrafeoServerError as error:
                if not error.rolled_back or attempt == _WRITE_ATTEMPTS - 1:
                    raise
        return False

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query(f"MATCH (n:{label}) RETURN n LIMIT $limit", parameters={"limit": int(limit)})
        return [self._strip_internal(row["n"]) for row in rows]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 500,
    ) -> int:
        """Insert edges with UNWIND + MATCH + CREATE and return how many were created.

        Endpoints are found through the `id` index. An edge whose endpoint does
        not exist is not created and not counted (the server's write counters).
        """
        if not edges:
            return 0
        by_type: dict[str, list[dict[str, Any]]] = {}
        for src, tgt, edge_type, props in edges:
            by_type.setdefault(edge_type, []).append({**props, "src": src, "dst": tgt})

        count = 0
        with self._transaction() as tx_id:
            for edge_type, rows in by_type.items():
                keys = sorted({key for row in rows for key in row} - {"src", "dst"})
                props = " {" + ", ".join(f"{key}: e.{key}" for key in keys) + "}" if keys else ""
                query = (
                    "UNWIND $edges AS e "
                    "MATCH (a:Node {id: e.src}) MATCH (b:Node {id: e.dst}) "
                    f"CREATE (a)-[:{edge_type}{props}]->(b)"
                )
                for start in range(0, len(rows), batch_size):
                    batch = rows[start : start + batch_size]
                    result = self._run(query, parameters={"edges": batch}, transaction_id=tx_id)
                    count += result.counters.get("edges_created", 0)
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        rel = f"[:{edge_type}]" if edge_type else "[]"
        rows = self._query(f"MATCH (n:Node {{id: $id}})-{rel}->(m) RETURN m.id AS id", parameters={"id": node_id})
        return [str(r["id"]) for r in rows if r.get("id") is not None]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        # BFS over get_neighbors (unweighted, as in the base adapter)
        self._record_fallback("shortest_path")
        visited: set[str] = set()
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            neighbors = self.get_neighbors(current, edge_type=edge_type)
            queue.extend([*path, neighbor] for neighbor in neighbors if neighbor not in visited)
        return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q_upper = query.strip().upper()
        is_write = any(kw in q_upper for kw in ["CREATE", "DELETE", "SET ", "REMOVE", "MERGE"])
        if not is_write:
            return self._query(query, parameters=params)
        with self._transaction() as tx_id:
            return self._run(query, parameters=params, transaction_id=tx_id).records()

    def count_nodes(self, *, label: str | None = None) -> int:
        query = f"MATCH (n:{label}) RETURN count(n) AS count" if label else "MATCH (n) RETURN count(n) AS count"
        rows = self._query(query)
        return int(rows[0]["count"]) if rows else 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        rel = f"[r:{edge_type}]" if edge_type else "[r]"
        rows = self._query(f"MATCH ()-{rel}->() RETURN count(r) AS count")
        return int(rows[0]["count"]) if rows else 0

    # ── Native algorithms (CALL procedures) ─────────────────────

    # grafeo 0.5.44 ignores query parameters passed as CALL arguments (optional
    # ones silently fall back to their defaults), so the procedures below get
    # their arguments as literals built from validated numbers and names.

    def _has_procedure(self, name: str) -> bool:
        return f"grafeo.{name}" in self._procedures

    def _nid_map(self) -> dict[int, str]:
        """Build internal node ID → app ID mapping."""
        rows = self._query("MATCH (n) RETURN id(n) AS nid, n.id AS app_id")
        return {row["nid"]: str(row["app_id"]) for row in rows}

    def _resolve_nid(self, app_id: str) -> int | None:
        """Resolve app ID to internal node ID."""
        rows = self._query("MATCH (n:Node {id: $id}) RETURN id(n) AS nid", parameters={"id": app_id})
        return rows[0]["nid"] if rows else None

    def _groups(self, rows: list[dict[str, Any]], column: str) -> list[set[str]]:
        nid_to_app = self._nid_map()
        groups: dict[Any, set[str]] = {}
        for row in rows:
            app_id = nid_to_app.get(row["node_id"], str(row["node_id"]))
            groups.setdefault(row[column], set()).add(app_id)
        return list(groups.values())

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """LDBC SSSP via CALL grafeo.dijkstra(); unreachable vertices are left out."""
        src_nid = self._resolve_nid(source)
        if src_nid is None or not self._has_procedure("dijkstra"):
            return super().sssp(source, weight_attr=weight_attr)
        nid_to_app = self._nid_map()
        if not weight_attr.isidentifier():
            msg = f"weight_attr must be a property name, got {weight_attr!r}"
            raise ValueError(msg)
        rows = self._query(
            f"CALL grafeo.dijkstra({{source: {int(src_nid)}, weight: '{weight_attr}'}}) YIELD node_id, distance"
        )
        return {nid_to_app.get(r["node_id"], str(r["node_id"])): float(r["distance"]) for r in rows}

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank via CALL grafeo.pagerank()."""
        if not self._has_procedure("pagerank"):
            return super().pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)
        nid_to_app = self._nid_map()
        args = f"damping: {float(damping)!r}, max_iterations: {int(max_iterations)}, tolerance: {float(tolerance)!r}"
        rows = self._query(f"CALL grafeo.pagerank({{{args}}}) YIELD node_id, score")
        return {nid_to_app.get(r["node_id"], str(r["node_id"])): float(r["score"]) for r in rows}

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection via CALL grafeo.louvain() / grafeo.label_propagation()."""
        if algorithm not in ("louvain", "label_propagation") or not self._has_procedure(algorithm):
            return super().community_detection(algorithm=algorithm)
        rows = self._query(f"CALL grafeo.{algorithm}() YIELD node_id, community_id")
        return self._groups(rows, "community_id")

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """BFS via CALL grafeo.bfs(), or a level-by-level fan-out for one edge type.

        With an edge type, all neighbor queries of a BFS level run concurrently,
        each thread on its own GWP session over the shared HTTP/2 channel.
        """
        if edge_type:
            if self._pool is None:
                return super().traverse_bfs(start, max_depth=max_depth, edge_type=edge_type)
            self._record_fallback("traverse_bfs")
            visited: set[str] = set()
            current_level = {start}
            result: list[str] = []
            for depth in range(max_depth + 1):
                new_nodes = current_level - visited
                visited.update(new_nodes)
                result.extend(new_nodes)
                if depth == max_depth:
                    break
                futures = [self._pool.submit(self.get_neighbors, nid, edge_type=edge_type) for nid in new_nodes]
                current_level = {n for future in as_completed(futures) for n in future.result() if n not in visited}
            return result

        src_nid = self._resolve_nid(start)
        if src_nid is None or not self._has_procedure("bfs"):
            return super().traverse_bfs(start, max_depth=max_depth, edge_type=edge_type)
        nid_to_app = self._nid_map()
        rows = self._query(f"CALL grafeo.bfs({int(src_nid)}) YIELD node_id, depth")
        return [nid_to_app.get(r["node_id"], str(r["node_id"])) for r in rows if int(r["depth"]) <= max_depth]

    def traverse_dfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        """Depth-limited DFS with concurrent neighbor expansion.

        ``CALL grafeo.dfs()`` has no depth limit and its ``depth`` column is the
        node's finishing position, not its depth, so it cannot be used here.
        """
        pool = self._pool
        if pool is None:
            return super().traverse_dfs(start, max_depth=max_depth, edge_type=edge_type)

        self._record_fallback("traverse_dfs")
        visited: set[str] = set()
        stack = [(start, 0)]
        result: list[str] = []
        while stack:
            # Take the whole stack and expand its nodes concurrently
            batch = []
            while stack:
                nid, depth = stack.pop()
                if nid not in visited:
                    visited.add(nid)
                    result.append(nid)
                    if depth < max_depth:
                        batch.append((nid, depth))
            if not batch:
                break
            futures = {pool.submit(self.get_neighbors, nid, edge_type=edge_type): depth for nid, depth in batch}
            for future in as_completed(futures):
                depth = futures[future]
                stack.extend((n, depth + 1) for n in future.result() if n not in visited)
        return result

    def bfs_levels(self, source: str) -> dict[str, int]:
        """LDBC BFS via CALL grafeo.bfs(); unreachable vertices are left out."""
        src_nid = self._resolve_nid(source)
        if src_nid is None or not self._has_procedure("bfs"):
            return super().bfs_levels(source)
        nid_to_app = self._nid_map()
        rows = self._query(f"CALL grafeo.bfs({int(src_nid)}) YIELD node_id, depth")
        return {nid_to_app.get(r["node_id"], str(r["node_id"])): int(r["depth"]) for r in rows}

    def weakly_connected_components(self) -> list[set[str]]:
        """LDBC WCC via CALL grafeo.connected_components()."""
        if not self._has_procedure("connected_components"):
            return super().weakly_connected_components()
        rows = self._query("CALL grafeo.connected_components() YIELD node_id, component_id")
        return self._groups(rows, "component_id")

    def local_clustering_coefficient(self, *, directed: bool = True) -> dict[str, float]:
        """LCC via CALL grafeo.clustering_coefficient() (undirected only, like Neo4j GDS)."""
        definition = "directed" if directed else "undirected"
        if definition not in self.native_lcc_definitions or not self._has_procedure("clustering_coefficient"):
            return super().local_clustering_coefficient(directed=directed)
        nid_to_app = self._nid_map()
        rows = self._query("CALL grafeo.clustering_coefficient() YIELD node_id, coefficient")
        return {nid_to_app.get(r["node_id"], str(r["node_id"])): float(r["coefficient"]) for r in rows}

    # ── Native vector search (GWP AdminService / SearchService) ──

    def _has_vector_index(self) -> bool:
        # The search procedures exist only when the server has the vector-index feature.
        return self._has_procedure("search.vector")

    def create_vector_index(
        self,
        label: str,
        property_name: str,
        *,
        dimensions: int = 128,
        metric: str = "cosine",
        m: int | None = None,
        ef_construction: int | None = None,
        quantization: str | None = None,
    ) -> None:
        """Build the HNSW index with AdminService.CreateIndex (m, ef_construction)."""
        if not self._has_vector_index():
            raise UnsupportedOperation(_NO_VECTOR_INDEX)
        if quantization is not None:
            raise UnsupportedOperation("GWP CreateIndex has no quantization option")
        pb = _gwp().pb
        index = pb.VectorIndexDef(label=label, property=property_name, dimensions=dimensions, metric=metric)
        if m is not None:
            index.m = m
        if ef_construction is not None:
            index.ef_construction = ef_construction
        self._admin_stub.CreateIndex(pb.CreateIndexRequest(graph=self._graph, vector_index=index))
        self._vector_metric = metric

    def vector_ann_search(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
        ef: int | None = None,
    ) -> list[tuple[str, float]]:
        """HNSW k-NN with SearchService.VectorSearch, then one query for the hits' IDs.

        The search returns internal node IDs and distances; mapping them to the
        ``id`` property costs a second round trip, included in the timing.
        """
        if not self._has_vector_index():
            raise UnsupportedOperation(_NO_VECTOR_INDEX)
        pb = _gwp().pb
        request = pb.VectorSearchRequest(
            graph=self._graph, label=label, property=property_name, query_vector=query_vector, k=k
        )
        if ef is not None:
            request.ef = ef
        hits = self._search_stub.VectorSearch(request).hits
        rows = self._query(
            "UNWIND $nids AS nid MATCH (n) WHERE id(n) = nid RETURN nid, n.id AS app_id",
            parameters={"nids": [hit.node_id for hit in hits]},
        )
        app = {row["nid"]: str(row["app_id"]) for row in rows}
        cosine = self._vector_metric == "cosine"
        return [(app[hit.node_id], 1.0 - hit.score if cosine else -hit.score) for hit in hits]

    def delete_nodes(self, node_ids: Sequence[str]) -> int:
        """Delete nodes and their edges (GQL DETACH DELETE); returns how many were deleted."""
        with self._transaction() as tx_id:
            result = self._run(
                "UNWIND $keys AS key MATCH (n:Node {id: key}) DETACH DELETE n",
                parameters={"keys": list(node_ids)},
                transaction_id=tx_id,
            )
        return result.counters.get("nodes_deleted", 0)
