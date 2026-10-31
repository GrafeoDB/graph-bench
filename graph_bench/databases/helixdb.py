"""HelixDB embedded (helix-db + helix-db-embedded): an in-process, in-memory graph queried with the Python DSL."""

import time
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any, ClassVar

from helixdb import (
    Client,
    EdgeRef,
    IndexSpec,
    InMemory,
    NodeRef,
    g,
    parse_index_operation_status,
    read_batch,
    write_batch,
)

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.databases.helix_queries import BUILDERS
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model

# Entries per `write_batch`. The docs give no batch size or limit ("All entries in the write batch commit or roll
# back together", https://docs.helix-db.com/database/helix-db/query-guides/writing-data), so this is our
# choice: a sweep on snb-small loaded in 10.2 s (100), 8.2 s (500) and 7.8 s (2000), so 500 is a middle value
# past the steep part of the curve, not a tuned optimum.
BATCH_SIZE = 500
# Seconds to wait for an index build; the indexes are built on an empty database, so this is a safety bound.
INDEX_TIMEOUT_S = 60.0


class HelixDatabase(Database):
    """An embedded HelixDB (`Client.embedded(InMemory(...))`), loaded with `write_batch` chunks.

    HelixDB has no bulk loader and no text query language in the Python SDK: data is written with `add_n` / `add_e`
    entries (https://docs.helix-db.com/database/helix-db/query-guides/writing-data) and read with operation trees
    built by the DSL. A query's `helix` dialect text is the name of a builder in `helix_queries`.
    """

    name: ClassVar[str] = "helixdb"
    display_name: ClassVar[str] = "HelixDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "helix"

    client: Any = None

    def info(self) -> DatabaseInfo:
        # The embedded runtime is a Rust async runtime with its own worker threads; the docs
        # (https://docs.helix-db.com/database/helix-db/start-here/local-development/embedded-database) show no setting.
        note = "HelixDB exposes no thread setting"
        return DatabaseInfo(self.name, version("helix-db"), "in-process", False, note, "in-memory")

    def _wait_index(self, operation_id: str) -> None:
        """Poll the index build until it succeeds; the docs say to poll the DDL receipt's operation until active
        (https://docs.helix-db.com/database/helix-db/query-guides/secondary-indexes)."""
        deadline = time.monotonic() + INDEX_TIMEOUT_S
        while True:
            raw = self.client.query(
                read_batch().var_as("s", g().get_index_operation(operation_id)).returning(["s"]).to_query_request()
            )
            status = parse_index_operation_status(raw["s"])
            state = type(status).__name__
            if state == "IndexOperationSucceeded":
                return
            if state in ("IndexOperationAborted", "IndexOperationBlocked") or time.monotonic() > deadline:
                raise RuntimeError(f"index build did not finish: {status}")
            time.sleep(0.01)

    def load(self, dataset: Dataset) -> None:
        self.client = Client.embedded(InMemory("bench"))
        # A unique equality index on `id` per label, created before the data (the lookups of every query use it),
        # and awaited: the load clock covers the index being ready.
        # https://docs.helix-db.com/database/helix-db/query-guides/secondary-indexes
        for table in dataset.nodes:
            spec = IndexSpec.node_unique_equality(table.label, "id")
            receipt = self.client.query(
                write_batch().var_as("i", g().create_index_if_not_exists(spec)).returning(["i"]).to_query_request()
            )["i"]
            if receipt.get("kind") == "accepted":
                self._wait_index(receipt["operation_id"])
        # Nodes: one `add_n` entry per row; the returned internal ids (`$id`) map (label, csv id) to the node.
        ids: dict[str, dict[Any, Any]] = {}
        for table in dataset.nodes:
            label_ids = ids.setdefault(table.label, {})
            for chunk in _chunks(dataset.rows(table)):
                batch = write_batch()
                for i, row in enumerate(chunk):
                    batch = batch.var_as(f"n{i}", g().add_n(table.label, _properties(row)))
                result = self.client.query(batch.returning([f"n{i}" for i in range(len(chunk))]).to_query_request())
                for i, row in enumerate(chunk):
                    label_ids[row["id"]] = result[f"n{i}"][0]["$id"]
        # Edges: `n(<internal id>).add_e(type, <internal id>, props)`.
        for edge in dataset.edges:
            src_ids, dst_ids = ids[edge.src], ids[edge.dst]
            for chunk in _chunks(dataset.rows(edge)):
                batch = write_batch()
                for i, row in enumerate(chunk):
                    props = _properties({k: v for k, v in row.items() if k not in ("src", "dst")})
                    batch = batch.var_as(
                        f"e{i}", g().n(src_ids[row["src"]]).add_e(edge.type, dst_ids[row["dst"]], props)
                    )
                self.client.query(batch.returning(["e0"]).to_query_request())

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        builder = BUILDERS[text]
        if isinstance(builder, str):
            raise NotSupported(builder)
        request, rows = builder(params)
        return rows(self.client.query(request))

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with the DSL's `count` step over all nodes and all edges
        (https://docs.helix-db.com/database/helix-db/query-guides/projections lists `count`)."""
        result = self.client.query(
            read_batch()
            .var_as("n", g().n(NodeRef.all()).count())
            .var_as("e", g().e(EdgeRef.all()).count())
            .returning(["n", "e"])
            .to_query_request()
        )
        return int(result["n"]), int(result["e"])

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        # BFS: the SDK's `NativeGraph.traverse` could give hop distances, but the docs only name `Client.graph(...)` in
        # one sentence (https://docs.helix-db.com/database/helix-db/start-here/sdk-setup/python-project-setup) and
        # never describe the snapshot or its traversal, so it is not a documented path. The query language has no
        # BFS levels (`repeat` yields paths, not distances) and no WCC or PageRank.
        reasons = {
            "bfs": "BFS levels: no documented algorithm API (NativeGraph is not described in the docs)",
            "wcc": "WCC: HelixDB has no weakly connected components",
            "pr": "PageRank: HelixDB has no PageRank",
        }
        raise NotSupported(reasons.get(name, name))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def _chunks(rows: Any) -> Any:
    chunk: list[Any] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == BATCH_SIZE:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _properties(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row without its empty strings: an empty CSV field is a missing property, as in the other adapters."""
    return {k: v for k, v in row.items() if v != ""}
