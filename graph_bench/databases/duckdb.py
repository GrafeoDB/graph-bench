"""DuckDB with the DuckPGQ extension, in process and in memory: SQL/PGQ queries over a property graph on tables."""

import collections
from collections.abc import Mapping
from typing import Any, ClassVar

import duckdb

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Column, Dataset
from graph_bench.types import Model

# DuckDB column types of the dataset's column types (https://duckdb.org/docs/current/sql/data_types/overview).
TYPES = {"INT64": "BIGINT", "DOUBLE": "DOUBLE", "STRING": "VARCHAR"}
# Set once `provision` has installed DuckPGQ in this process (the worker calls it twice: before the Startup children
# and in `prepare_environment`).
_PROVISIONED: list[bool] = []

# Algorithms: the DuckPGQ table functions `pagerank` and `weakly_connected_component`
# (https://duckpgq.org/documentation/graph_functions/). The arguments are the property graph and the vertex and edge
# labels. PageRank has a fixed damping factor (0.85) and tolerance (1e-6) that cannot be set; the answer is checked
# against the same epsilon as every database, so a result the tolerance leaves too far from the reference is ≠.
PAGE_RANK = "SELECT id, pagerank FROM pagerank({graph}, V, E)"
WCC = "SELECT id, componentId FROM weakly_connected_component({graph}, V, E)"
# BFS levels as a shortest-path query in the database's own language, like LadybugDB's and FalkorDB's: ANY SHORTEST
# gives one path per reachable vertex, whose length is the hop distance; `->*` includes the source at length 0
# (https://duckpgq.org/documentation/sql_pgq/, path-finding).
BFS = (
    "SELECT vid, d FROM GRAPH_TABLE ({graph} MATCH p = ANY SHORTEST (a:V WHERE a.id = $source)-[e:E]->*(b:V) "
    "COLUMNS (b.id AS vid, PATH_LENGTH(p) AS d))"
)


def _path(dataset: Dataset, file: str) -> str:
    path = (dataset.root / file).resolve().as_posix()
    if "'" in path:
        raise ValueError(f"cannot quote a path with a single quote in COPY: {path}")
    return path


def _columns(columns: tuple[Column, ...]) -> str:
    return ", ".join(f"{c.name} {TYPES[c.type]}" for c in columns)


class DuckDatabase(Database):
    """An in-memory DuckDB with DuckPGQ: typed tables loaded with `COPY`, a `CREATE PROPERTY GRAPH` over them.

    Node tables are keyed by their `id` column. Each edge table is a table `<TYPE>__<src>__<dst>` with `src`, `dst`
    and its properties; the property graph labels it with the edge type when that type has one table, and with the
    table name otherwise, because DuckPGQ needs unique labels (https://duckpgq.org/documentation/property_graph/).
    So a type spread over several node pairs (REPLY_OF between comments and between a comment and a post) is not one
    label, and `Post|Comment` is not a label expression DuckPGQ parses: those SQL/PGQ cases fail with DuckPGQ's error
    and are recorded in docs/known-disagreements.md; no union table is built to hide it.

    The property graph is named after the dataset (`snb`, `graphalytics`). The extension is installed once per
    process (`provision`); each connection loads it and activates its grammar, which DuckDB 2.0 needs
    per connection (https://github.com/cwida/duckpgq-extension/blob/main/docs/GRAMMAR_EXTENSION.md).
    """

    name: ClassVar[str] = "duckdb"
    display_name: ClassVar[str] = "DuckDB + DuckPGQ"
    models: ClassVar[frozenset[Model]] = frozenset({"sql_pgq"})
    dialect: ClassVar[str] = "duckpgq"

    con: Any = None
    graph: str = ""
    nodes: tuple[str, ...] = ()
    edges: tuple[str, ...] = ()

    @classmethod
    def provision(cls) -> None:
        """Install DuckPGQ from the community repository (it downloads), once per process, so that no clock includes
        it (`prepare_environment` is this alone).

        https://duckdb.org/community_extensions/extensions/duckpgq.html
        """
        if _PROVISIONED:
            return
        con = duckdb.connect(":memory:")
        try:
            con.execute("INSTALL duckpgq FROM community")
        finally:
            con.close()
        _PROVISIONED.append(True)

    def info(self) -> DatabaseInfo:
        note = "SET threads=1" if self.threads == 1 else "default threads"
        return DatabaseInfo(self.name, duckdb.__version__, "in-process", True, note, "in-memory")

    def open_empty(self) -> None:
        """An empty in-memory database with DuckPGQ loaded and its grammar active: what `load` does before its
        first table. `LOAD` reads the extension `provision` installed; it never installs it."""
        # ":memory:" is the in-memory database (https://duckdb.org/docs/current/clients/python/dbapi).
        self.con = duckdb.connect(":memory:")
        if self.threads == 1:
            # https://duckdb.org/docs/current/configuration/overview ("threads")
            self.con.execute("SET threads=1")
        self.con.execute("LOAD duckpgq")
        # Its own statement, per connection (research: required since DuckDB 2.0).
        self.con.execute("SET active_grammar_extensions=['duckpgq']")

    def load(self, dataset: Dataset) -> None:
        self.open_empty()
        ids = {t.label: t.columns[0].type for t in dataset.nodes}
        for table in dataset.nodes:
            self.con.execute(f"CREATE TABLE {table.label}({_columns(table.columns)})")
            # https://duckdb.org/docs/current/data/csv/overview
            self.con.execute(f"COPY {table.label} FROM '{_path(dataset, table.file)}' (HEADER true)")
        per_type = collections.Counter(e.type for e in dataset.edges)
        edge_defs = []
        edge_tables = []
        for t in dataset.edges:
            table = f"{t.type}__{t.src}__{t.dst}"
            endpoints = f"src {TYPES[ids[t.src]]}, dst {TYPES[ids[t.dst]]}"
            properties = "".join(f", {c.name} {TYPES[c.type]}" for c in t.columns)
            self.con.execute(f"CREATE TABLE {table}({endpoints}{properties})")
            self.con.execute(f"COPY {table} FROM '{_path(dataset, t.file)}' (HEADER true)")
            label = t.type if per_type[t.type] == 1 else table
            edge_defs.append(
                f"{table} SOURCE KEY (src) REFERENCES {t.src} (id) DESTINATION KEY (dst) REFERENCES {t.dst} (id) "
                f"LABEL {label}"
            )
            edge_tables.append(table)
        # The shared SQL/PGQ texts name the graph `snb`, the SNB dataset's name.
        self.graph = dataset.name
        vertices = ", ".join(t.label for t in dataset.nodes)
        # https://duckpgq.org/documentation/property_graph/
        self.con.execute(
            f"CREATE PROPERTY GRAPH {self.graph} VERTEX TABLES ({vertices}) EDGE TABLES ({', '.join(edge_defs)})"
        )
        self.nodes = tuple(t.label for t in dataset.nodes)
        self.edges = tuple(edge_tables)

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        # The Python API binds `$name` parameters from a dict, also inside a graph pattern WHERE.
        return [tuple(row) for row in self.con.execute(text, dict(params)).fetchall()]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded: the rows of the node tables and of the edge tables, counted with plain SQL."""
        nodes = " + ".join(f"(SELECT count(*) FROM {t})" for t in self.nodes)
        edges = " + ".join(f"(SELECT count(*) FROM {t})" for t in self.edges)
        row = self.con.execute(f"SELECT {nodes}, {edges}").fetchone()
        return int(row[0]), int(row[1])

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name not in ("bfs", "wcc", "pr"):
            raise NotSupported(name)
        if self.graph != "graphalytics":
            raise RuntimeError("the algorithms run on the graphalytics dataset")
        if name == "pr":
            return dict(self.query(PAGE_RANK.format(graph=self.graph), {}))
        if name == "wcc":
            return dict(self.query(WCC.format(graph=self.graph), {}))
        return dict(self.query(BFS.format(graph=self.graph), {"source": params["source"]}))

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None
