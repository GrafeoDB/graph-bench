r"""
GWP handling of the grafeo-server adapter, tested without a server.

GWP returns a failing statement as a GQLSTATUS exception in the result summary,
not as a gRPC error. These tests feed the adapter such responses and check that
it raises, so a failing server query fails its benchmark instead of being timed.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("grpc", reason="the grafeo-server adapter needs grpcio (uv sync --extra grafeo-server)")

from graph_bench.adapters.grafeo_server import (
    DEFAULT_URI,
    GrafeoServerAdapter,
    GrafeoServerError,
    _execute_and_collect,
    _is_exception,
    _proto_to_python,
    _python_to_proto,
    gwp_address,
)
from graph_bench.adapters.proto import gql_service_pb2 as pb
from graph_bench.adapters.proto import gql_service_pb2_grpc as pb_grpc
from graph_bench.adapters.proto import gql_types_pb2 as types_pb
from graph_bench.types import UnsupportedOperation

SUCCESS = pb.ResultSummary(status=types_pb.GqlStatus(code="00000", message="successful completion"))


def summary(code: str, message: str = "", **counters: int) -> pb.ExecuteResponse:
    status = types_pb.GqlStatus(code=code, message=message)
    return pb.ExecuteResponse(summary=pb.ResultSummary(status=status, counters=counters))


def header(*columns: str) -> pb.ExecuteResponse:
    return pb.ExecuteResponse(header=pb.ResultHeader(columns=[pb.ColumnDescriptor(name=c) for c in columns]))


def rows(*values: list[Any]) -> pb.ExecuteResponse:
    batch = pb.RowBatch(rows=[pb.Row(values=[_python_to_proto(v) for v in row]) for row in values])
    return pb.ExecuteResponse(row_batch=batch)


class FakeGql:
    """A GqlService stub that answers each Execute with the next scripted frames."""

    def __init__(self, *responses: list[pb.ExecuteResponse], begin: str = "00000", commit: str = "00000") -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []
        self.calls: list[str] = []
        self.begin_code = begin
        self.commit_code = commit

    def Execute(self, request: Any, timeout: int | None = None) -> Iterator[pb.ExecuteResponse]:
        self.calls.append("Execute")
        self.requests.append(request)
        return iter(self.responses.pop(0))

    def BeginTransaction(self, request: Any) -> pb.BeginResponse:
        self.calls.append("Begin")
        tx_id = "tx-1" if self.begin_code == "00000" else ""
        return pb.BeginResponse(transaction_id=tx_id, status=types_pb.GqlStatus(code=self.begin_code, message="b"))

    def Commit(self, request: Any) -> pb.CommitResponse:
        self.calls.append("Commit")
        return pb.CommitResponse(status=types_pb.GqlStatus(code=self.commit_code, message="conflict"))

    def Rollback(self, request: Any) -> pb.RollbackResponse:
        self.calls.append("Rollback")
        return pb.RollbackResponse(status=types_pb.GqlStatus(code="00000"))


def offline_adapter(gql: FakeGql) -> GrafeoServerAdapter:
    """An adapter wired to a fake GqlService, with a session already open."""
    import threading

    adapter = GrafeoServerAdapter()
    adapter._gql_stub = gql
    adapter._thread_sessions[threading.get_ident()] = "session-1"
    adapter._connected = True
    return adapter


@pytest.mark.parametrize(
    ("code", "is_exception"),
    [
        ("00000", False),  # success
        ("00001", False),  # omitted result
        ("01000", False),  # warning
        ("02000", False),  # no data
        ("03000", False),  # information
        ("08000", True),  # connection exception
        ("22000", True),  # data exception
        ("25G01", True),  # invalid transaction state
        ("40000", True),  # transaction rollback
        ("42001", True),  # syntax error or access rule violation
        ("G2000", True),  # graph type violation
        ("", False),  # no status
    ],
)
def test_gqlstatus_classes(code: str, is_exception: bool) -> None:
    assert _is_exception(code) is is_exception


def test_rows_and_counters_are_collected() -> None:
    gql = FakeGql([header("id", "n"), rows(["a", 1], ["b", 2]), summary("00000", nodes_created=2)])
    result = _execute_and_collect(gql, "s", "MATCH (n) RETURN n")
    assert result.records() == [{"id": "a", "n": 1}, {"id": "b", "n": 2}]
    assert result.counters == {"nodes_created": 2}


def test_summary_exception_raises() -> None:
    # Regression: the summary frame was ignored, so this came back as zero rows.
    gql = FakeGql([summary("42001", "syntax error: multiple statements")])
    with pytest.raises(GrafeoServerError, match="42001: syntax error") as raised:
        _execute_and_collect(gql, "s", "MATCH (n) RETURN n; MATCH (m) RETURN m")
    assert raised.value.code == "42001"


def test_warning_and_no_data_do_not_raise() -> None:
    gql = FakeGql([summary("01000", "warning")], [summary("02000", "no data")])
    assert _execute_and_collect(gql, "s", "q1").rows == []
    assert _execute_and_collect(gql, "s", "q2").rows == []


def test_auto_commit_sends_no_transaction_id() -> None:
    # transaction_id is optional in gwp 0.2.1: an empty one names a transaction that does not exist.
    gql = FakeGql([summary("00000")], [summary("00000")])
    _execute_and_collect(gql, "s", "RETURN 1")
    _execute_and_collect(gql, "s", "RETURN 1", transaction_id="tx-9")
    assert not gql.requests[0].HasField("transaction_id")
    assert gql.requests[1].transaction_id == "tx-9"


def test_failing_read_raises_from_adapter_methods() -> None:
    adapter = offline_adapter(FakeGql([summary("42001", "bad query")]))
    with pytest.raises(GrafeoServerError, match="bad query"):
        adapter.get_neighbors("p0")


def test_failing_write_rolls_back_and_raises() -> None:
    # Regression: execute_query returned [] for a failing write.
    gql = FakeGql([summary("22000", "write failed")])
    adapter = offline_adapter(gql)
    with pytest.raises(GrafeoServerError, match="write failed"):
        adapter.execute_query("MATCH (n {id: $id}) SET n.x = 1", params={"id": "a"})
    assert gql.calls == ["Begin", "Execute", "Rollback"]


def test_failed_begin_raises() -> None:
    gql = FakeGql(begin="25G01")
    adapter = offline_adapter(gql)
    with pytest.raises(GrafeoServerError, match="25G01"):
        adapter.execute_query("MATCH (n) SET n.x = 1")
    assert gql.calls == ["Begin"]


def test_failed_commit_raises() -> None:
    gql = FakeGql([summary("00000", nodes_created=1)], commit="40000")
    adapter = offline_adapter(gql)
    with pytest.raises(GrafeoServerError, match="40000"):
        adapter.insert_nodes([{"id": "a"}])
    assert gql.calls == ["Begin", "Execute", "Commit"]


def test_insert_counts_come_from_the_server() -> None:
    # Edges whose endpoints do not exist are not created, so the count comes from
    # the write counters, not from the number of rows sent.
    gql = FakeGql([summary("00000", edges_created=1)])
    adapter = offline_adapter(gql)
    created = adapter.insert_edges([("a", "b", "E", {"w": 1}), ("a", "missing", "E", {"w": 2})])
    assert created == 1
    statement = gql.requests[0].statement
    assert "MATCH (a:Node {id: e.src}) MATCH (b:Node {id: e.dst})" in statement
    assert "CREATE (a)-[:E {w: e.w}]->(b)" in statement


def test_insert_nodes_keeps_every_property_key() -> None:
    # Regression: only the first node's keys were written.
    gql = FakeGql([summary("00000", nodes_created=2)], [header("label", "property"), rows(["Node", "id"])])
    adapter = offline_adapter(gql)
    adapter.insert_nodes([{"id": "a", "content": "x"}, {"id": "b", "imageFile": "y.png"}], label="Post")
    assigns = "content: props.content, id: props.id, imageFile: props.imageFile"
    assert gql.requests[0].statement == f"UNWIND $nodes AS props CREATE (:Post:Node {{{assigns}}})"


def test_values_round_trip() -> None:
    values = [None, True, 3, 2.5, "s", [1, "x"], {"a": 1, "b": [2.0]}]
    assert [_proto_to_python(_python_to_proto(v)) for v in values] == values


def test_default_address_is_the_gwp_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_BENCH_GRAFEO_SERVER_URI", raising=False)
    assert DEFAULT_URI == "localhost:7688"  # grafeo-server's GWP port (7687 is Bolt)
    assert gwp_address() == "localhost:7688"
    assert gwp_address("grpc://host:1234/") == "host:1234"
    monkeypatch.setenv("GRAPH_BENCH_GRAFEO_SERVER_URI", "http://other:7688")
    assert gwp_address() == "other:7688"


def test_stubs_speak_gwp_0_2_1() -> None:
    # grafeo-server 0.5.44 replaced DatabaseService with CatalogService, renamed
    # Close to CloseSession and made ExecuteRequest.transaction_id optional.
    assert hasattr(pb_grpc, "CatalogServiceStub")
    assert not hasattr(pb_grpc, "DatabaseServiceStub")
    assert hasattr(pb, "CloseSessionRequest")
    # HasField raises for a field without presence, so this also checks that it is optional.
    assert not pb.ExecuteRequest(session_id="s").HasField("transaction_id")


def test_adapters_import_without_grpc() -> None:
    # Regression: graph_bench.adapters imported grpc, so it failed without the
    # grafeo-server extra. The adapter now imports it when it connects.
    code = (
        "import sys; sys.modules['grpc'] = None\n"
        "import graph_bench.adapters\n"
        "from graph_bench.adapters.grafeo_server import GrafeoServerAdapter\n"
        "try:\n"
        "    GrafeoServerAdapter().connect(uri='localhost:1')\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "uv sync --extra grafeo-server" in result.stdout


# ── Native vector search over GWP ───────────────────────────────


class FakeVectorServices:
    """AdminService and SearchService stubs that record their requests."""

    def __init__(self, hits: list[tuple[int, float]]) -> None:
        self.hits = hits
        self.requests: list[Any] = []

    def CreateIndex(self, request: Any) -> pb.CreateIndexResponse:
        self.requests.append(request)
        return pb.CreateIndexResponse()

    def VectorSearch(self, request: Any) -> pb.VectorSearchResponse:
        self.requests.append(request)
        return pb.VectorSearchResponse(hits=[pb.SearchHit(node_id=n, score=d) for n, d in self.hits])


def vector_adapter(gql: FakeGql, services: FakeVectorServices, *, vector_index: bool = True) -> GrafeoServerAdapter:
    adapter = offline_adapter(gql)
    adapter._admin_stub = services
    adapter._search_stub = services
    adapter._procedures = frozenset({"grafeo.search.vector"} if vector_index else {"grafeo.pagerank"})
    return adapter


def test_create_vector_index_passes_the_hnsw_parameters() -> None:
    services = FakeVectorServices([])
    adapter = vector_adapter(FakeGql(), services)
    adapter.create_vector_index("VectorNode", "embedding", dimensions=128, m=24, ef_construction=200)
    index = services.requests[0].vector_index
    assert (index.label, index.property, index.dimensions, index.metric) == ("VectorNode", "embedding", 128, "cosine")
    assert (index.m, index.ef_construction) == (24, 200)


def test_quantization_is_unsupported_over_gwp() -> None:
    adapter = vector_adapter(FakeGql(), FakeVectorServices([]))
    with pytest.raises(UnsupportedOperation, match="quantization"):
        adapter.create_vector_index("VectorNode", "embedding", quantization="scalar")


def test_server_without_vector_index_is_unsupported() -> None:
    adapter = vector_adapter(FakeGql(), FakeVectorServices([]), vector_index=False)
    with pytest.raises(UnsupportedOperation, match="vector-index feature"):
        adapter.create_vector_index("VectorNode", "embedding")
    with pytest.raises(UnsupportedOperation, match="vector-index feature"):
        adapter.vector_ann_search([1.0, 0.0])


def test_vector_ann_search_maps_hits_to_ids() -> None:
    services = FakeVectorServices([(7, 0.0), (3, 0.25)])
    gql = FakeGql([header("nid", "app_id"), rows([3, "b"], [7, "a"]), summary("00000")])
    adapter = vector_adapter(gql, services)
    hits = adapter.vector_ann_search([1.0, 0.0], k=2, ef=32)
    assert hits == [("a", 1.0), ("b", 0.75)]
    request = services.requests[0]
    assert (request.k, request.ef, list(request.query_vector)) == (2, 32, [1.0, 0.0])


def test_embeddings_are_stored_as_vectors() -> None:
    # A list parameter would be stored as a list, which the vector index leaves out.
    gql = FakeGql([summary("00000", nodes_created=2)], [header("label", "property"), rows(["Node", "id"])])
    adapter = offline_adapter(gql)
    adapter.insert_nodes([{"id": "a", "embedding": [0.1, 0.2]}, {"id": "b", "embedding": [1, 2]}], label="V")
    assigns = "embedding: vector(props.embedding), id: props.id"
    assert gql.requests[0].statement == f"UNWIND $nodes AS props CREATE (:V:Node {{{assigns}}})"


def test_delete_nodes_counts_from_the_server() -> None:
    gql = FakeGql([summary("00000", nodes_deleted=1)])
    adapter = offline_adapter(gql)
    assert adapter.delete_nodes(["a", "missing"]) == 1
    assert gql.calls == ["Begin", "Execute", "Commit"]
    assert "DETACH DELETE" in gql.requests[0].statement


def test_rolled_back_update_is_retried() -> None:
    gql = FakeGql([summary("40001", "serialization failure")], [header("id"), rows(["a"]), summary("00000")])
    assert offline_adapter(gql).update_node("a", {"x": 1}) is True
    assert gql.calls == ["Execute", "Execute"]


def test_other_update_errors_are_not_retried() -> None:
    gql = FakeGql([summary("42001", "GRAFEO-T001: Write conflict")])
    with pytest.raises(GrafeoServerError, match="42001"):
        offline_adapter(gql).update_node("a", {"x": 1})
    assert gql.calls == ["Execute"]
