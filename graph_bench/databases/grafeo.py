"""Grafeo, embedded and in memory: batch loading, Cypher queries and the native algorithms."""

from collections.abc import Iterator, Mapping
from itertools import islice
from typing import Any, ClassVar

import grafeo

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset, EdgeTable, NodeTable
from graph_bench.types import Model

CHUNK = 10_000


def _chunks[T](items: Iterator[T], size: int) -> Iterator[list[T]]:
    while chunk := list(islice(items, size)):
        yield chunk


def _rows(dataset: Dataset, table: NodeTable | EdgeTable) -> Iterator[dict[str, Any]]:
    # Dataset contract: an empty CSV field means "no property" (LadybugDB's COPY and Grafeo's LOAD CSV give null).
    return ({k: v for k, v in row.items() if v != ""} for row in dataset.rows(table))


class GrafeoDatabase(Database):
    """An in-memory `grafeo.GrafeoDB`, loaded with the batch APIs and queried with `execute_cypher`.

    The batch APIs are Grafeo's fastest documented bulk path (CSV parsing included, in Python). Node ids from the
    dataset are kept as the `id` property (with a property index). WCC and PageRank key their results by it
    (`key="id"`); BFS takes and returns internal node ids, mapped through the ids the batch call returns.
    """

    name: ClassVar[str] = "grafeo"
    display_name: ClassVar[str] = "Grafeo"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "cypher"

    db: Any = None

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        """Grafeo's parallel work runs on the rayon pool; one thread means a one-thread pool."""
        return {"RAYON_NUM_THREADS": "1"} if threads == 1 else {}

    def info(self) -> DatabaseInfo:
        if self.threads == 1:
            note = "single-threaded executor"
        else:
            note = "query executor single-threaded (parallel queries from 0.6.4); rayon pool = all cores"
        return DatabaseInfo(self.name, grafeo.__version__, "in-process", True, note, "in-memory")

    def load(self, dataset: Dataset) -> None:
        db = self.db = grafeo.GrafeoDB()
        # label -> dataset id -> internal node id, for the edge endpoints.
        ids: dict[str, dict[Any, int]] = {}
        for table in dataset.nodes:
            label_ids = ids.setdefault(table.label, {})
            for rows in _chunks(_rows(dataset, table), CHUNK):
                created = db.batch_create_nodes_with_props(table.label, rows)
                label_ids.update(zip((row["id"] for row in rows), created, strict=True))
        for table in dataset.edges:
            src, dst = ids[table.src], ids[table.dst]
            edges = ((src[row.pop("src")], dst[row.pop("dst")], table.type, row) for row in _rows(dataset, table))
            for chunk in _chunks(edges, CHUNK):
                db.batch_create_edges(chunk)
        db.create_property_index("id")
        # bfs_layers has no `key=`: the graphalytics vertices keep their internal ids, and the way back.
        self.ids: dict[Any, int] = {}
        self.keys: dict[int, Any] = {}
        if dataset.name == "graphalytics":
            self.ids = ids["V"]
            self.keys = {nid: key for key, nid in self.ids.items()}

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.db.execute_cypher(text, dict(params))
        columns = result.columns
        return [tuple(row[c] for c in columns) for row in result]

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        algorithms = self.db.algorithms
        if name == "bfs":
            keys = self.keys
            layers = algorithms.bfs_layers(self.ids[params["source"]])
            return {keys[nid]: depth for depth, layer in enumerate(layers) for nid in layer}
        if name == "wcc":
            return dict(algorithms.connected_components(key="id"))
        if name == "pr":
            # `key="id"` keys the result by the dataset id (grafeo docs/algorithms/index.md, "Results independent of
            # the load order: `key=`").
            return dict(algorithms.pagerank(damping=0.85, max_iterations=1000, tolerance=1e-10, key="id"))
        raise NotSupported(name)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
