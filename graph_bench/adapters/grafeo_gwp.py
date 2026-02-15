r"""
Grafeo Server adapter using GQL Wire Protocol (GWP/gRPC).

Extends GrafeoServerAdapter for GQL query logic (insert_nodes, get_node, etc.)
but replaces all HTTP transport with GWP:
  - DatabaseService for database lifecycle (create/delete)
  - SessionService for session management
  - GqlService for query execution and transactions

Works with grafeo-server:lite (GWP-only, no HTTP).

Requires: pip install grpcio

Environment variables:
    GRAPH_BENCH_GRAFEO_GWP_URI: GWP URI (default: localhost:7687)
"""

import math
from collections.abc import Sequence
from typing import Any

import grpc

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.adapters.grafeo_server import GrafeoServerAdapter
from graph_bench.adapters.proto import gql_service_pb2 as pb
from graph_bench.adapters.proto import gql_service_pb2_grpc as pb_grpc
from graph_bench.adapters.proto import gql_types_pb2 as types_pb
from graph_bench.config import get_env

__all__ = ["GrafeoGwpAdapter"]


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
        # summary frame: ignore for now

    return columns, rows


@AdapterRegistry.register("grafeo-gwp")
class GrafeoGwpAdapter(GrafeoServerAdapter):
    """Grafeo Server adapter using GWP (gRPC) for all operations.

    No HTTP dependency — works with grafeo-server:lite.
    Uses DatabaseService for lifecycle, GqlService for queries.
    """

    def __init__(self) -> None:
        # Initialize parent for GQL query logic (_gql_literal, _format_props)
        # but we won't use any of its HTTP transport
        BaseAdapter.__init__(self)
        self._db_name: str = "bench"
        self._connected = False
        self._gwp_uri: str = ""
        self._channel: grpc.Channel | None = None
        self._session_stub: pb_grpc.SessionServiceStub | None = None
        self._gql_stub: pb_grpc.GqlServiceStub | None = None
        self._db_stub: pb_grpc.DatabaseServiceStub | None = None
        self._gwp_session_id: str = ""

    @property
    def name(self) -> str:
        return "Grafeo Server (GWP)"

    @property
    def version(self) -> str:
        if not self._connected or not self._session_stub:
            return "unknown"
        try:
            self._session_stub.Ping(
                pb.PingRequest(session_id=self._gwp_session_id)
            )
            return "0.4.0"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        # GWP connection — no HTTP needed
        default_gwp = "localhost:7687"
        self._gwp_uri = (
            uri
            or get_env("GRAFEO_GWP_URI", default=default_gwp)
            or default_gwp
        )

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

        # Create GWP session
        resp = self._session_stub.Handshake(
            pb.HandshakeRequest(
                protocol_version=1,
                client_info={"client": "graph-bench", "version": "1.0"},
            )
        )
        self._gwp_session_id = resp.session_id

        # Configure session to use benchmark database
        self._session_stub.Configure(
            pb.ConfigureRequest(
                session_id=self._gwp_session_id,
                graph=self._db_name,
            )
        )
        self._connected = True

    def disconnect(self) -> None:
        if self._session_stub and self._gwp_session_id:
            try:
                self._session_stub.Close(
                    pb.CloseRequest(session_id=self._gwp_session_id)
                )
            except Exception:
                pass
        if self._channel:
            self._channel.close()
            self._channel = None
        self._gwp_session_id = ""
        self._session_stub = None
        self._gql_stub = None
        self._db_stub = None
        self._connected = False

    # ── GWP query helpers (override HTTP methods) ────────────────

    def _query(self, gql: str) -> list[dict[str, Any]]:
        """Execute a GQL query via GWP and return rows as dicts."""
        columns, rows = _execute_and_collect(
            self._gql_stub, self._gwp_session_id, gql
        )
        return [
            {col: row[i] for i, col in enumerate(columns)}
            for row in rows
        ]

    def _exec(self, gql: str) -> Any:
        """Execute a GQL query via GWP and return raw result."""
        columns, rows = _execute_and_collect(
            self._gql_stub, self._gwp_session_id, gql
        )
        return {"columns": columns, "rows": rows}

    def _tx_begin(self) -> str:
        """Begin a transaction via GWP, return transaction_id."""
        resp = self._gql_stub.BeginTransaction(
            pb.BeginRequest(
                session_id=self._gwp_session_id,
                mode=pb.READ_WRITE,
            )
        )
        return resp.transaction_id

    def _tx_query(self, tx_id: str, gql: str) -> Any:
        """Execute a query within a GWP transaction."""
        columns, rows = _execute_and_collect(
            self._gql_stub,
            self._gwp_session_id,
            gql,
            transaction_id=tx_id,
        )
        return {"columns": columns, "rows": rows}

    def _tx_commit(self, tx_id: str) -> None:
        """Commit a GWP transaction."""
        self._gql_stub.Commit(
            pb.CommitRequest(
                session_id=self._gwp_session_id,
                transaction_id=tx_id,
            )
        )

    # ── Database lifecycle via GWP DatabaseService ────────────────

    def clear(self) -> None:
        """Drop and recreate the benchmark database via GWP DatabaseService."""
        # Close GWP session before dropping the database
        if self._session_stub and self._gwp_session_id:
            try:
                self._session_stub.Close(
                    pb.CloseRequest(session_id=self._gwp_session_id)
                )
            except Exception:
                pass

        # Drop database via GWP DatabaseService
        try:
            self._db_stub.DeleteDatabase(
                pb.DeleteDatabaseRequest(name=self._db_name)
            )
        except grpc.RpcError:
            pass  # Might not exist

        # Recreate database via GWP DatabaseService
        self._db_stub.CreateDatabase(
            pb.CreateDatabaseRequest(
                name=self._db_name,
                database_type="Lpg",
                storage_mode="InMemory",
            )
        )

        # Re-establish GWP session on the new database
        resp = self._session_stub.Handshake(
            pb.HandshakeRequest(
                protocol_version=1,
                client_info={"client": "graph-bench", "version": "1.0"},
            )
        )
        self._gwp_session_id = resp.session_id
        self._session_stub.Configure(
            pb.ConfigureRequest(
                session_id=self._gwp_session_id,
                graph=self._db_name,
            )
        )
