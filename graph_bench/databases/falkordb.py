"""FalkorDB through falkordblite: a local Redis server subprocess with the FalkorDB module, in memory."""

from collections.abc import Iterator, Mapping
from importlib.metadata import version
from typing import Any, ClassVar, Literal

from redislite.falkordb_client import FalkorDB

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset, EdgeTable, NodeTable
from graph_bench.types import Model

GRAPH = "bench"
# Rows per UNWIND call. FalkorDB's docs give no batch size for UNWIND (https://docs.falkordb.com/cypher/unwind.html,
# https://docs.falkordb.com/integration/bulk-loader.html), so this is our choice: large enough that the per-call
# round trip is small, small enough to stay far below Redis' request limits.
BATCH_SIZE = 10_000
# WCC: https://docs.falkordb.com/algorithms/wcc.html (component ids are arbitrary; the answer comparison relabels).
WCC = "CALL algo.WCC({nodeLabels: ['V'], relationshipTypes: ['E']}) YIELD node, componentId RETURN node.id, componentId"
# PageRank: https://docs.falkordb.com/algorithms/pagerank.html. The procedure takes only a label and a relationship
# type; damping, iteration bound and tolerance are fixed inside FalkorDB and not documented, so none is passed.
PAGE_RANK = "CALL algo.pageRank('V', 'E') YIELD node, score RETURN node.id, score"
# BFS levels: the documented `shortestPath` function, in a WITH clause, gives the fewest-edges path from the source to
# each vertex (https://docs.falkordb.com/cypher/functions.html, "shortestPath"). Its endpoints must be bound before the
# call and the pattern may carry no property filters; a minimum of 0 includes the source (depth 0), and no path gives
# NULL, which the WHERE drops (the reference omits unreachable vertices). Outgoing E edges, as in the other adapters.
# `algo.BFS` yields no levels and `algo.SSpaths` is a weighted, path-count-limited search, so neither is used.
BFS = (
    "MATCH (s:V {id: $s}), (b:V) WITH b, shortestPath((s)-[:E*0..]->(b)) AS p WHERE p IS NOT NULL "
    "RETURN b.id, length(p)"
)
# The load check (`counts`): every node and every relationship, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n)"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r)"


def _batches(rows: Iterator[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


class FalkorDatabase(Database):
    """A falkordblite server (a local `redis-server` process) holding one graph, loaded with UNWIND batches.

    Per node label an index on `id` is created first (https://docs.falkordb.com/cypher/indexing/index.html), then the
    typed CSV rows go in as `UNWIND $rows AS r CREATE ...` batches, nodes before edges, edges matched to their
    endpoints by that index. All of it is inside `load()`, so the load clock covers the server start, the indexes
    and every batch. Queries are the shared openCypher texts; a text FalkorDB cannot run fails with its error.
    """

    name: ClassVar[str] = "falkordb"
    display_name: ClassVar[str] = "FalkorDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    process: ClassVar[Literal["in-process", "local process"]] = "local process"

    server: Any = None
    graph: Any = None

    def info(self) -> DatabaseInfo:
        # THREAD_COUNT and OMP_THREAD_COUNT are load-time module arguments (https://docs.falkordb.com/configuration.html)
        # and falkordblite starts the module without any, so both stay at the core count.
        note = "THREAD_COUNT and OMP_THREAD_COUNT are module load arguments that falkordblite does not expose"
        return DatabaseInfo(self.name, version("falkordblite"), "local process", False, note, "in-memory")

    def _node_query(self, table: NodeTable) -> str:
        return f"UNWIND $rows AS r CREATE (n:{table.label}) SET n = r"

    def _edge_query(self, table: EdgeTable) -> str:
        sets = ", ".join(f"e.{c.name} = r.{c.name}" for c in table.columns)
        tail = f" SET {sets}" if sets else ""
        return (
            f"UNWIND $rows AS r MATCH (a:{table.src} {{id: r.src}}), (b:{table.dst} {{id: r.dst}}) "
            f"CREATE (a)-[e:{table.type}]->(b){tail}"
        )

    def load(self, dataset: Dataset) -> None:
        # No path: a temporary directory. An empty `save` removes the periodic RDB snapshots, so the data stays in
        # memory only (https://docs.falkordb.com/operations/durability.html, the `save ""` development setting).
        self.server = FalkorDB(None, serverconfig={"save": ""})
        self.graph = self.server.select_graph(GRAPH)
        for table in dataset.nodes:
            self.graph.query(f"CREATE INDEX FOR (n:{table.label}) ON (n.id)")
        for table in dataset.nodes:
            query = self._node_query(table)
            for batch in _batches(dataset.rows(table)):
                self.graph.query(query, {"rows": batch})
        for edge in dataset.edges:
            query = self._edge_query(edge)
            for batch in _batches(dataset.rows(edge)):
                self.graph.query(query, {"rows": batch})

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.graph.query(text, dict(params))
        return [tuple(row) for row in result.result_set]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES`."""
        nodes = self.query(COUNT_NODES, {})[0][0]
        edges = self.query(COUNT_EDGES, {})[0][0]
        return int(nodes), int(edges)

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name == "wcc":
            return dict(self.query(WCC, {}))
        if name == "pr":
            return dict(self.query(PAGE_RANK, {}))
        if name == "bfs":
            return dict(self.query(BFS, {"s": params["source"]}))
        raise NotSupported(name)

    def close(self) -> None:
        self.graph = None
        if self.server is not None:
            self.server.close()
            self.server = None
