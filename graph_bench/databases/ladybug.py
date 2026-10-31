"""LadybugDB (a Kuzu fork), embedded and in memory: CSV bulk loading, Cypher queries and the ALGO extension."""

from collections.abc import Mapping
from typing import Any, ClassVar

import ladybug as lb

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported, provision_once
from graph_bench.datasets.schema import Column, Dataset, EdgeTable
from graph_bench.types import Model

# The maximum depth of a recursive (variable-length) pattern. LadybugDB rejects an upper bound above this setting,
# whose default is 30, and an unbounded pattern stops at it, so BFS would silently miss vertices deeper than 30.
# Raised with the documented connection setting (https://docs.ladybugdb.com/cypher/configuration, "Configure
# maximum depth of recursive extend") so that unbounded patterns mean what they mean in openCypher.
MAX_DEPTH = 10_000

PAGE_RANK = "CALL page_rank('g', dampingFactor := 0.85, maxIterations := 1000, tolerance := 1e-10) RETURN node.id, rank"
# WCC stops after `maxIterations` (default 100, https://docs.ladybugdb.com/extensions/algo/wcc), which a component
# with a long diameter could need more of; raised to PageRank's bound so that the result is the converged one.
WCC = "CALL weakly_connected_components('g', maxIterations := 1000) RETURN node.id, group_id"
BFS = f"MATCH p = (a:V {{id: $s}})-[:E* SHORTEST 1..{MAX_DEPTH}]->(b:V) RETURN b.id, length(p)"
# The load check (`counts`): every node and every relationship, over all tables, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n)"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r)"


def _path(dataset: Dataset, file: str) -> str:
    path = (dataset.root / file).resolve().as_posix()
    if "'" in path:
        raise ValueError(f"cannot quote a path with a single quote in COPY: {path}")
    return path


def _columns(columns: tuple[Column, ...]) -> str:
    return ", ".join(f"{c.name} {c.type}" for c in columns)


class LadybugDatabase(Database):
    """An in-memory `ladybug.Database`, loaded with `COPY FROM` CSV and queried with parameterized Cypher.

    Node tables are keyed by their `id` column. Each edge type is one rel table with a `FROM a TO b` pair per
    EdgeTable. On the graphalytics dataset the graph is projected as `g` in the load. The ALGO extension is
    installed once per process (`provision`) and loaded by the first algorithm that needs it, both outside the
    load clock, so a broken extension fails only those algorithms (✗), not the load or BFS. `open_empty` (the
    Startup row) is the load's first step: the empty database, its connection and the recursion depth setting.
    """

    name: ClassVar[str] = "ladybug"
    display_name: ClassVar[str] = "LadybugDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "ladybug"

    db: Any = None
    conn: Any = None
    projected: bool = False
    algo_loaded: bool = False

    @classmethod
    def provision(cls) -> None:
        """Install the ALGO extension (INSTALL may download it) on a throwaway in-memory database, once per process.

        Installing is a one-time download into LadybugDB's extension directory, so it stays out of every clock
        (`prepare_environment` is this alone). A failure raises: the Startup entry and the loads of this process
        fail (✗) with it. https://docs.ladybugdb.com/get-started/graph-algorithms
        """
        provision_once(cls.name, cls._install)

    @staticmethod
    def _install() -> None:
        db = lb.Database()
        conn = lb.Connection(db)
        try:
            conn.execute("INSTALL ALGO").close()
        finally:
            conn.close()
            db.close()

    @property
    def _num_threads(self) -> int:
        """1 for one thread, 0 for LadybugDB's default (all cores)."""
        return 1 if self.threads == 1 else 0

    def info(self) -> DatabaseInfo:
        note = f"max_num_threads={self._num_threads}"
        return DatabaseInfo(self.name, lb.__version__, "in-process", True, note, "in-memory")

    def _run(self, text: str) -> None:
        self.conn.execute(text).close()

    def open_empty(self) -> None:
        """An empty in-memory database and its connection, with the recursion depth setting: what `load` does
        before its first table. The ALGO extension is not loaded (only the first algorithm loads it)."""
        threads = self._num_threads
        # No path: an in-memory database (the `Database` docstring: "If the path is not specified, or empty, or
        # equal to `:memory:`, the database will be created in memory").
        self.db = lb.Database(max_num_threads=threads)
        self.conn = lb.Connection(self.db, num_threads=threads)
        self._run(f"CALL var_length_extend_max_depth={MAX_DEPTH}")

    def load(self, dataset: Dataset) -> None:
        self.open_empty()
        for table in dataset.nodes:
            self._run(f"CREATE NODE TABLE {table.label}({_columns(table.columns)}, PRIMARY KEY(id))")
            self._run(f"COPY {table.label} FROM '{_path(dataset, table.file)}' (HEADER=true)")
        groups: dict[str, list[EdgeTable]] = {}
        for edge in dataset.edges:
            groups.setdefault(edge.type, []).append(edge)
        for rel_type, tables in groups.items():
            if len({t.columns for t in tables}) > 1:
                raise ValueError(f"edge type {rel_type} has different properties per node pair")
            pairs = ", ".join(f"FROM {t.src} TO {t.dst}" for t in tables)
            columns = "".join(f", {c.name} {c.type}" for c in tables[0].columns)
            self._run(f"CREATE REL TABLE {rel_type}({pairs}{columns})")
            # The first two CSV columns are the endpoints; `from`/`to` pick the pair to copy into
            # (https://docs.ladybugdb.com/import, relationship tables with multiple from-to pairs).
            for t in tables:
                options = f"from='{t.src}', to='{t.dst}', HEADER=true"
                self._run(f"COPY {rel_type} FROM '{_path(dataset, t.file)}' ({options})")
        if dataset.name == "graphalytics":
            # A core function, so it works without the ALGO extension.
            self._run("CALL project_graph('g', ['V'], ['E'])")
            self.projected = True

    def _load_algo(self) -> None:
        """Load the ALGO extension (installed by `prepare_environment`) once, before the first PageRank or WCC.

        It runs in the untimed preflight call. A failure raises, so that algorithm is ✗ with LadybugDB's error.
        https://docs.ladybugdb.com/get-started/graph-algorithms
        """
        if not self.algo_loaded:
            self._run("LOAD EXTENSION ALGO")
            self.algo_loaded = True

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.conn.execute(text, parameters=dict(params))
        try:
            return [tuple(row) for row in result.get_all()]
        finally:
            result.close()

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES` through `execute()`."""
        nodes = self.query(COUNT_NODES, {})[0][0]
        edges = self.query(COUNT_EDGES, {})[0][0]
        return int(nodes), int(edges)

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name not in ("bfs", "wcc", "pr"):
            raise NotSupported(name)
        if name != "bfs":
            if not self.projected:
                raise RuntimeError("the graph `g` is projected only for the graphalytics dataset")
            self._load_algo()
        if name == "pr":
            return dict(self.query(PAGE_RANK, {}))
        if name == "wcc":
            return dict(self.query(WCC, {}))
        source = params["source"]
        # The pattern starts at length 1, so the source is added at depth 0.
        return {source: 0, **dict(self.query(BFS, {"s": source}))}

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.db is not None:
            self.db.close()
            self.db = None
