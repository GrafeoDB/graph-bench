"""Grafeo, embedded and in memory: batch loading, Cypher, SQL/PGQ and SPARQL queries and the native algorithms."""

from collections.abc import Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

import grafeo
from grafeo.terms import Literal

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets import rdf
from graph_bench.datasets.schema import Dataset, EdgeTable, NodeTable
from graph_bench.types import Model, Store

CHUNK = 10_000

# Triples per `INSERT DATA` statement of the transitional RDF load (`_insert_data`). Grafeo's loading guide calls
# 10,000 triples per batch "a good default" for its streaming loaders (docs/user-guide/sparql/loading.md, "Turtle
# streaming load").
RDF_CHUNK = 10_000

# The load check (`counts`): every node and every relationship, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n) AS c"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r) AS c"


def _chunks[T](items: Iterator[T], size: int) -> Iterator[list[T]]:
    while chunk := list(islice(items, size)):
        yield chunk


def _rows(dataset: Dataset, table: NodeTable | EdgeTable) -> Iterator[dict[str, Any]]:
    # Dataset contract: an empty CSV field means "no property" (LadybugDB's COPY and Grafeo's LOAD CSV give null).
    return ({k: v for k, v in row.items() if v != ""} for row in dataset.rows(table))


def _term(value: Any) -> Any:
    """A SPARQL result term as a Python value: a literal by its datatype (`rdf.python_value`), an IRI as its text.

    `grafeo.terms.Literal` and `IRI` subclass `str`; `str()` is the lexical form or the IRI (grafeo
    docs/user-guide/sparql, research-b section 3.7).
    """
    if value is None:
        return None
    if isinstance(value, Literal):
        return rdf.python_value(str(value), value.datatype)
    return str(value)


def _insert_data(db: Any, path: Path) -> None:
    """The transitional RDF load for a grafeo without `load_rdf` (0.6.0): the file's N-Triples lines in SPARQL
    `INSERT DATA` statements of `RDF_CHUNK` lines. Removed once a Grafeo 0.6.1 build is installed."""
    with path.open(encoding="utf-8") as fh:
        for lines in _chunks((line.rstrip("\n") for line in fh if line.strip()), RDF_CHUNK):
            db.execute_sparql("INSERT DATA {\n" + "\n".join(lines) + "\n}")


class GrafeoDatabase(Database):
    """An in-memory `grafeo.GrafeoDB` in one of two stores, queried in three models.

    Graph store (property graph and SQL/PGQ): loaded with the batch APIs, Grafeo's fastest documented bulk path
    (CSV parsing included, in Python), into a named graph called after the dataset (`snb`, `graphalytics`), which
    `set_graph` selects for every later call (docs/user-guide/named-graphs.md; docs/user-guide/python/database.md:
    "the direct API, `db.algorithms`, CALL procedures ... follow it"). SQL/PGQ's `GRAPH_TABLE (snb MATCH ...)`
    names that graph, as the standard requires. Node ids from the dataset are kept as the `id` property (with a
    property index). WCC and PageRank key their results by it (`key="id"`); BFS takes and returns internal node
    ids, mapped through the ids the batch call returns. Property graph queries run with `execute_cypher`, SQL/PGQ
    with `execute_sql` (docs/user-guide/sql-pgq/index.md).

    RDF store: the dataset's N-Triples file (`rdf.ntriples_path`, written untimed at prepare time by the shared
    mapping) loaded with `db.load_rdf(path, format="ntriples")`, Grafeo's RDF file loader: one bulk write into the
    default graph (new in 0.6.1, grafeo release/0.6.1 commit 61c4388d; its docstring in
    crates/bindings/python/src/database.rs, experimental per docs/stability.md). A grafeo build without `load_rdf`
    (0.6.0) gets the file's lines as SPARQL `INSERT DATA` statements of `RDF_CHUNK` triples, the path grafeo
    documents for bindings without a file loader (docs/user-guide/sparql/loading.md, "Loading via SPARQL"). That
    fallback is transitional: the load switches to `load_rdf` by itself once a Grafeo 0.6.1 build is installed, and
    is then removed. SPARQL queries run with `execute_sparql`, their parameters inlined as literals
    (`rdf.bind`). The RDF store lives in the same `GrafeoDB` type but apart from the property graph, so
    it gets its own instance and its own load (`load_snb_rdf`).
    """

    name: ClassVar[str] = "grafeo"
    display_name: ClassVar[str] = "Grafeo"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph", "sql_pgq", "rdf"})
    dialect: ClassVar[str] = "cypher"
    dialects: ClassVar[Mapping[Model, str]] = {"property_graph": "cypher", "sql_pgq": "sql_pgq", "rdf": "sparql"}

    db: Any = None
    store: Store = "graph"

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
        db.create_graph(dataset.name)
        db.set_graph(dataset.name)
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

    def load_store(self, store: Store, dataset: Dataset) -> None:
        self.store = store
        if store != "rdf":
            self.load(dataset)
            return
        path = rdf.ntriples_path(dataset)
        db = self.db = grafeo.GrafeoDB()
        load_rdf = getattr(db, "load_rdf", None)
        if load_rdf is not None:
            load_rdf(str(path), format="ntriples")
        else:
            _insert_data(db, path)

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.db.execute_cypher(text, dict(params))
        columns = result.columns
        return [tuple(row[c] for c in columns) for row in result]

    def query_model(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        if model == "sql_pgq":
            result = self.db.execute_sql(text, dict(params))
            columns = result.columns
            return [tuple(row[c] for c in columns) for row in result]
        if model == "rdf":
            result = self.db.execute_sparql(rdf.bind(text, params))
            columns = result.columns
            return [tuple(_term(row.get(c)) for c in columns) for row in result]
        return self.query(text, params)

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded: `COUNT_NODES` and `COUNT_EDGES` with Cypher, or the RDF mapping's with SPARQL."""
        if self.store == "rdf":
            nodes = self.query_model("rdf", rdf.COUNT_NODES, {})[0][0]
            edges = self.query_model("rdf", rdf.COUNT_EDGES, {})[0][0]
        else:
            nodes = self.query(COUNT_NODES, {})[0][0]
            edges = self.query(COUNT_EDGES, {})[0][0]
        return int(nodes), int(edges)

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
