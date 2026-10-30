"""NeuG, embedded and in memory: CSV bulk loading, Cypher queries and the bundled GDS extension."""

from collections.abc import Mapping
from typing import Any, ClassVar

import neug

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Column, Dataset, EdgeTable
from graph_bench.types import Model

# Algorithms run through the GDS extension (https://neug.io/docs/extensions/load_gds/) on the projected graph `g`.
# The options map of a call is mandatory and cannot be empty, so each call passes at least `concurrency: 1`.
# The graphalytics edge list holds both directions of every edge, so `directed: true` is the graph as stored.
PROJECT = "CALL project_graph('g', ['V'], ['[V, E, V]'])"
# PageRank has no tolerance option, only an iteration bound (default 20): 1000 reaches the converged ranks, the
# same bound the other adapters use.
PAGE_RANK = (
    "CALL page_rank('g', {damping_factor: 0.85, max_iterations: 1000, directed: true, concurrency: 1}) "
    "RETURN node.id, rank"
)
WCC = "CALL wcc('g', {concurrency: 1}) RETURN node.id, comp"
# BFS reports -1 as the distance of an unreachable vertex; the reference omits those vertices (see `algorithm`).
# The options map takes literals only (a `$parameter` inside it is a binder error), so the source is inlined as an int.
BFS = "CALL bfs('g', {{source: {source}, directed: true}}) RETURN node.id, distance"
# The load check (`counts`): every node and every relationship, over all tables, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n)"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r)"


def _path(dataset: Dataset, file: str) -> str:
    path = (dataset.root / file).resolve().as_posix()
    if '"' in path:
        raise ValueError(f"cannot quote a path with a double quote in COPY: {path}")
    return path


def _columns(columns: tuple[Column, ...]) -> str:
    return ", ".join(f"{c.name} {c.type}" for c in columns)


class NeugDatabase(Database):
    """An in-memory `neug.Database`, loaded with `COPY FROM` CSV and queried with parameterized Cypher.

    Node tables are keyed by their `id` column. Each edge type is one rel table with a `FROM a TO b` pair per
    EdgeTable. The GDS extension (bundled in the wheel) is installed once per process (`prepare_environment`);
    the first algorithm loads it and projects the graph `g`, both outside the load clock (the untimed preflight
    call), so a broken extension fails only the algorithms (✗), not the load or the SNB queries.
    """

    name: ClassVar[str] = "neug"
    display_name: ClassVar[str] = "NeuG"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "neug"

    db: Any = None
    conn: Any = None
    gds_ready: bool = False

    @classmethod
    def prepare_environment(cls) -> None:
        """Install the GDS extension on a throwaway in-memory database (it may download, so not in a load clock).

        https://neug.io/docs/extensions/load_gds/
        """
        db = neug.Database("", max_thread_num=1)
        conn = db.connect()
        try:
            list(conn.execute("INSTALL gds"))
        finally:
            conn.close()
            db.close()

    @property
    def _max_thread_num(self) -> int:
        """1 for one thread, 0 for NeuG's default (hardware concurrency)."""
        return 1 if self.threads == 1 else 0

    def info(self) -> DatabaseInfo:
        note = f"max_thread_num={self._max_thread_num}; embedded queries are single-threaded"
        return DatabaseInfo(self.name, neug.__version__, "in-process", True, note, "in-memory")

    def _run(self, text: str) -> None:
        list(self.conn.execute(text))

    def load(self, dataset: Dataset) -> None:
        # An empty path opens an in-memory database (the `Database` docstring, https://neug.io/docs/reference/python_api/).
        self.db = neug.Database("", max_thread_num=self._max_thread_num)
        self.conn = self.db.connect()
        for table in dataset.nodes:
            self._run(f"CREATE NODE TABLE {table.label}({_columns(table.columns)}, PRIMARY KEY(id))")
        groups: dict[str, list[EdgeTable]] = {}
        for edge in dataset.edges:
            groups.setdefault(edge.type, []).append(edge)
        for rel_type, tables in groups.items():
            if len({t.columns for t in tables}) > 1:
                raise ValueError(f"edge type {rel_type} has different properties per node pair")
            pairs = ", ".join(f"FROM {t.src} TO {t.dst}" for t in tables)
            columns = "".join(f", {c.name} {c.type}" for c in tables[0].columns)
            self._run(f"CREATE REL TABLE {rel_type}({pairs}{columns})")
        # The default delimiter is `|`, so `delim` is passed; nodes are loaded before edges
        # (https://neug.io/docs/data_io/import_data/). One COPY per statement; grouping them in a transaction did not
        # make the load measurably faster.
        for table in dataset.nodes:
            self._run(f'COPY {table.label} FROM "{_path(dataset, table.file)}" (header=true, delim=",")')
        for t in dataset.edges:
            # The first two CSV columns are the endpoints, matched to the primary keys of the `from`/`to` tables.
            options = f'from="{t.src}", to="{t.dst}", header=true, delim=","'
            self._run(f'COPY {t.type} FROM "{_path(dataset, t.file)}" ({options})')

    def _load_gds(self) -> None:
        """Load the GDS extension (installed by `prepare_environment`) and project `g`, once, before an algorithm.

        It runs in the untimed preflight call. A failure raises, so that algorithm is ✗ with NeuG's error.
        https://neug.io/docs/extensions/load_gds/
        """
        if not self.gds_ready:
            self._run("LOAD gds")
            self._run(PROJECT)
            self.gds_ready = True

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.conn.execute(text, parameters=dict(params))
        return [tuple(row) for row in result]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES` through `execute()`."""
        nodes = self.query(COUNT_NODES, {})[0][0]
        edges = self.query(COUNT_EDGES, {})[0][0]
        return int(nodes), int(edges)

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name not in ("bfs", "wcc", "pr"):
            raise NotSupported(name)
        self._load_gds()
        if name == "pr":
            return dict(self.query(PAGE_RANK, {}))
        if name == "wcc":
            return dict(self.query(WCC, {}))
        return {
            vertex: depth for vertex, depth in self.query(BFS.format(source=int(params["source"])), {}) if depth >= 0
        }

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.db is not None:
            self.db.close()
            self.db = None
