"""Grafeo, embedded and in memory: batch loading, Cypher queries and the native algorithms."""

from collections.abc import Iterator, Mapping
from itertools import islice
from typing import Any, ClassVar

import grafeo

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model

CHUNK = 10_000


def _chunks[T](items: Iterator[T], size: int) -> Iterator[list[T]]:
    while chunk := list(islice(items, size)):
        yield chunk


class GrafeoDatabase(Database):
    """An in-memory `grafeo.GrafeoDB`, loaded with the batch APIs and queried with `execute_cypher`.

    Node ids from the dataset are kept as the `id` property (with a property index); the native algorithms
    return internal node ids, which are mapped back to the dataset ids through the map built while loading.
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
        # (label, dataset id) -> internal node id, and back.
        self.ids: dict[tuple[str, Any], int] = {}
        self.keys: dict[int, Any] = {}
        for table in dataset.nodes:
            for rows in _chunks(dataset.rows(table), CHUNK):
                created = db.batch_create_nodes_with_props(table.label, rows)
                for row, nid in zip(rows, created, strict=True):
                    self.ids[(table.label, row["id"])] = nid
                    self.keys[nid] = row["id"]
        ids = self.ids
        for table in dataset.edges:
            edges = (
                (ids[(table.src, row.pop("src"))], ids[(table.dst, row.pop("dst"))], table.type, row)
                for row in dataset.rows(table)
            )
            for chunk in _chunks(edges, CHUNK):
                db.batch_create_edges(chunk)
        db.create_property_index("id")

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.db.execute_cypher(text, dict(params))
        columns = result.columns
        return [tuple(row[c] for c in columns) for row in result]

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        algorithms = self.db.algorithms
        keys = self.keys
        if name == "bfs":
            layers = algorithms.bfs_layers(self.ids[("V", params["source"])])
            return {keys[nid]: depth for depth, layer in enumerate(layers) for nid in layer}
        if name == "wcc":
            return dict(algorithms.connected_components(key="id"))
        if name == "pr":
            ranks = algorithms.pagerank(damping=0.85, max_iterations=1000, tolerance=1e-10)
            return {keys[nid]: rank for nid, rank in ranks.items()}
        raise NotSupported(name)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
