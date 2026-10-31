"""ArcadeDB embedded (arcadedb-embedded, with its bundled JRE): an on-disk database in the instance's workdir.

The engine runs in this process's JVM, started once per process by `prepare_environment` through the shared JVM
helper (`graph_bench.databases.jvm`), with the shared heap cap. Python API docs (package version 26.10.2.post1):
https://github.com/humemai/arcadedb-embedded-python/tree/26.10.2.post1/bindings/python/docs/api (`jvm.md`,
`graph_batch.md`); engine settings: https://docs.arcadedb.com/arcadedb/reference/settings.
"""

import atexit
import os
import shutil
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar, Literal

import arcadedb_embedded as adb

from graph_bench.databases import jvm
from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Column, Dataset, EdgeTable
from graph_bench.types import Model

PACKAGE = "arcadedb-embedded"
# The single-thread settings, as JVM options. ArcadeDB documents them as engine settings
# (https://docs.arcadedb.com/arcadedb/reference/settings), read from `-D` system properties at JVM start:
# - queryParallelismPoolThreads: the JVM-wide pool behind query-time parallelism (graph algorithms, parallel index
#   scans); 0 (default) is the core count.
# - parallelScanProducerPoolThreads: the pool of the producer tasks of parallel bucket scans; 0 is the core count.
# - queryParallelScan: scan the buckets of a type in parallel during a full scan (default true).
# - asyncWorkerThreads: the asynchronous executor, which also runs GraphBatch's edge flush; 0 is cores minus 1.
# The JDK's common ForkJoinPool gets parallelism 1 as well (`start_jvm(common_pool_parallelism=...)` in jvm.md sets
# the same property). They go in through ARCADEDB_JVM_ARGS, the package's documented environment variable for JVM
# options (jvm.md), set by `process_env` so that both the worker and the CodSpeed session start with them.
SINGLE_THREAD = {
    "arcadedb.queryParallelismPoolThreads": "1",
    "arcadedb.parallelScanProducerPoolThreads": "1",
    "arcadedb.queryParallelScan": "false",
    "arcadedb.asyncWorkerThreads": "1",
    "java.util.concurrent.ForkJoinPool.common.parallelism": "1",
}
JVM_ARGS_ENV = "ARCADEDB_JVM_ARGS"
_ENGINE: dict[str, bool] = {}
# Schema types (https://docs.arcadedb.com/arcadedb/reference/sql/sql-create-property, "Supported types").
_TYPES = {"INT64": "LONG", "DOUBLE": "DOUBLE", "STRING": "STRING"}
# Native algorithms through openCypher procedures (https://docs.arcadedb.com/arcadedb/reference/graph-algorithms/chapter),
# run with `command` (a CALL is not idempotent, so `query` refuses it). The graphalytics edge list holds both
# directions of every edge, so outgoing `E` edges are the graph as stored, as in the other adapters.
# BFS (path-finding.html#algo-bfs): "The start node is excluded from the results", so `algorithm` adds it at depth 0.
BFS = "MATCH (s:V {id: $source}) CALL algo.bfs(s, 'E', 'OUT') YIELD node, depth RETURN node.id AS id, depth"
# WCC (community-detection.html#algo-wcc): edge direction is ignored; component ids are arbitrary (the answer
# comparison relabels).
WCC = "CALL algo.wcc('E') YIELD node, componentId RETURN node.id AS id, componentId"
# PageRank (centrality.html#algo-page-rank): the defaults are 20 iterations and tolerance 1e-4, so damping 0.85,
# an iteration bound of 1000 and tolerance 1e-10 are passed, the values the other adapters use.
PAGE_RANK = (
    "CALL algo.pagerank({dampingFactor: 0.85, maxIterations: 1000, tolerance: 1e-10, direction: 'OUT'}) "
    "YIELD node, score RETURN node.id AS id, score"
)
# The load check (`counts`): every node and every relationship, counted with openCypher.
COUNT_NODES = "MATCH (n) RETURN count(n) AS n"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r) AS n"
# The engine's `arcadedb-log.properties` (in arcadedb-engine.jar) without its FileHandler.pattern, which `_log_config`
# sets: the same handlers, levels and formatters.
_LOG_CONFIG = (
    "handlers = java.util.logging.ConsoleHandler, java.util.logging.FileHandler",
    ".level = INFO",
    "io.undertow.level = WARNING",
    "org.xnio.level = WARNING",
    "org.jboss.level = WARNING",
    "org.apache.ratis.grpc.server.GrpcLogAppender.level = SEVERE",
    "com.arcadedb.level = INFO",
    "java.util.logging.ConsoleHandler.level = INFO",
    "java.util.logging.ConsoleHandler.formatter = com.arcadedb.utility.AnsiLogFormatter",
    "java.util.logging.FileHandler.level = INFO",
    "java.util.logging.FileHandler.formatter = com.arcadedb.log.LogFormatter",
    "java.util.logging.FileHandler.limit = 100000000",
    "java.util.logging.FileHandler.count = 10",
)


def _jre_home() -> Path:
    """The bundled JRE: two levels above its JVM library (`bin/server/jvm.dll`, `lib/server/libjvm.so`), jvm.md."""
    return Path(adb.jvm.get_bundled_jre_lib_path()).parents[2]


def _log_config(logs: Path) -> Path:
    """A java.util.logging configuration: the engine's own `arcadedb-log.properties`, its log file moved to `logs`."""
    pattern = (logs / "arcadedb.log").as_posix().replace("%", "%%")
    lines = [*_LOG_CONFIG, f"java.util.logging.FileHandler.pattern = {pattern}", ""]
    path = logs / "logging.properties"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cleanup(logs: Path) -> None:
    """At exit: shut the JVM down (it holds the log file open, which blocks its removal on Windows), then remove."""
    try:
        adb.jvm.shutdown_jvm()
    finally:
        shutil.rmtree(logs, ignore_errors=True)


def _start() -> None:
    """Start the JVM with the shared heap cap, the engine log in a per-process temporary directory.

    The engine's built-in logging configuration (`arcadedb-log.properties` in the engine jar) writes
    `./log/arcadedb.log` into the current directory, and the JVM's crash file goes to `./log` as well (jvm.md). The
    JDK's `java.util.logging.config.file` property replaces that configuration
    (https://docs.oracle.com/en/java/javase/25/docs/api/java.logging/java/util/logging/LogManager.html; ArcadeDB's
    logging page names it as the full control: https://docs.arcadedb.com/arcadedb-python/how-to/logs.html), and
    `ARCADEDB_JVM_ERROR_FILE` moves the crash file (jvm.md). The directory is removed at process exit.
    """
    logs = Path(tempfile.mkdtemp(prefix="graph-bench-arcadedb-"))
    atexit.register(_cleanup, logs)
    os.environ["ARCADEDB_JVM_ERROR_FILE"] = str(logs / "hs_err_pid%p.log")
    # jvm.md: "a heap_size other than "4g" replaces every -Xmx"; with "4g" itself an -Xmx in ARCADEDB_JVM_ARGS is
    # kept, so `jvm.start_once` checks the effective cap after the start.
    adb.jvm.start_jvm(heap_size=jvm.MAX_HEAP, jvm_args=[f"-Djava.util.logging.config.file={_log_config(logs)}"])


def _check_columns(rel_type: str, tables: list[EdgeTable]) -> tuple[Column, ...]:
    if len({t.columns for t in tables}) > 1:
        raise ValueError(f"edge type {rel_type} has different properties per node pair")
    return tables[0].columns


class ArcadeDatabase(Database):
    """An embedded ArcadeDB database in `workdir/db`, loaded with GraphBatch and queried with openCypher.

    ArcadeDB has no in-memory mode, so it runs on disk. Each node label is a vertex type with typed properties and
    a unique index on `id`; each edge type is an edge type. `load()` creates them, then loads every node CSV with
    `GraphBatch.create_vertices`, keeping the returned RIDs in an (label, id) -> RID map, and every edge CSV with
    `GraphBatch.new_edges` from that map (graph_batch.md, the documented bulk path). The faster Graph Importer is not
    used: its CSV reader does not honour quotes, and the SNB text columns contain commas and quotes (see
    docs/known-disagreements.md). Queries are the shared openCypher texts, run natively (no translation).
    """

    name: ClassVar[str] = "arcadedb"
    display_name: ClassVar[str] = "ArcadeDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    db: Any = None

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        if threads != 1:
            return {}
        return {JVM_ARGS_ENV: " ".join(f"-D{key}={value}" for key, value in SINGLE_THREAD.items())}

    @classmethod
    def prepare_environment(cls) -> None:
        """Start the JVM with the shared heap cap (and the single-thread options of `process_env`), unclocked.

        Then create and drop one empty database, once per process: the engine's one-time start (its classes and
        its polyglot engine, about 0.25 s) belongs to the JVM start, which is not timed (fairness rule 2), not to
        the first load. It holds no data and runs no load, so no load path is warmed up; other adapters likewise
        open a throwaway database here (NeuG, LadybugDB).
        """
        jvm.start_once(cls.name, _start)
        if _ENGINE.get("ready"):
            return
        with tempfile.TemporaryDirectory(prefix="graph-bench-arcadedb-") as scratch:
            empty = adb.create_database(str(Path(scratch) / "db"))
            empty.drop()
        _ENGINE["ready"] = True

    def info(self) -> DatabaseInfo:
        if self.threads == 1:
            # What is and is not capped: docs/known-disagreements.md, "ArcadeDB: threads".
            note = (
                "configured to 1 thread: "
                + ", ".join(f"{k}={v}" for k, v in SINGLE_THREAD.items())
                + ", GraphBatch parallel_flush=False; "
                "the JVM's garbage collector and JIT compiler threads are not capped"
            )
        else:
            note = "ArcadeDB defaults: thread pools sized to the core count"
        return DatabaseInfo(
            self.name, version(PACKAGE), "in-process", True, note, "on-disk", java=jvm.java_version(_jre_home())
        )

    def _check_threads(self) -> None:
        """With threads=1, the JVM must carry the single-thread options (`process_env` sets them before the start)."""
        if self.threads != 1:
            return
        wrong = {k: jvm.system_property(k) for k, v in SINGLE_THREAD.items() if jvm.system_property(k) != v}
        if wrong:
            raise RuntimeError(f"the JVM lacks the single-thread options (set {JVM_ARGS_ENV}); got {wrong}")

    def load(self, dataset: Dataset) -> None:
        # The benchmark paths (worker, CodSpeed session) call prepare_environment first, so the JVM start is never in
        # a load clock; this lazy start only serves direct use such as the tests.
        self.prepare_environment()
        self._check_threads()
        self.db = adb.create_database(str(self.workdir / "db"))
        schema = self.db.schema
        groups: dict[str, list[EdgeTable]] = {}
        for edge in dataset.edges:
            groups.setdefault(edge.type, []).append(edge)
        with self.db.transaction():
            for table in dataset.nodes:
                schema.create_vertex_type(table.label)
                for col in table.columns:
                    schema.create_property(table.label, col.name, _TYPES[col.type])
                schema.create_index(table.label, ["id"], unique=True)
            for rel_type, tables in groups.items():
                schema.create_edge_type(rel_type)
                for col in _check_columns(rel_type, tables):
                    schema.create_property(rel_type, col.name, _TYPES[col.type])
        rids: dict[tuple[str, Any], str] = {}
        # Outside any transaction (graph_batch.md, "Transactions"); `expected_edge_count` tunes the batch size, and
        # `parallel_flush` (on by default) is turned off for single-threaded runs.
        options: dict[str, Any] = {"expected_edge_count": dataset.edge_count}
        if self.threads == 1:
            options["parallel_flush"] = False
        with self.db.graph_batch(**options) as batch:
            for table in dataset.nodes:
                rows = list(dataset.rows(table))
                created = batch.create_vertices(table.label, rows)
                if len(created) != len(rows):
                    raise RuntimeError(f"{table.label}: {len(created)} RIDs for {len(rows)} vertices")
                for row, rid in zip(rows, created, strict=True):
                    rids[(table.label, row["id"])] = rid
            for edge in dataset.edges:
                rows = list(dataset.rows(edge))
                sources = [rids[(edge.src, row["src"])] for row in rows]
                targets = [rids[(edge.dst, row["dst"])] for row in rows]
                properties = [{c.name: row[c.name] for c in edge.columns} for row in rows] if edge.columns else None
                batch.new_edges(sources, edge.type, targets, properties)

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        # A dict binds by name (`$name`); an empty one is passed too, since None is not a parameter map.
        return [tuple(row.values()) for row in self.db.query("opencypher", text, dict(params)).to_list()]

    def _command(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        result = self.db.command("opencypher", text, dict(params))
        return [] if result is None else [tuple(row.values()) for row in result.to_list()]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES`."""
        return int(self.query(COUNT_NODES, {})[0][0]), int(self.query(COUNT_EDGES, {})[0][0])

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name == "wcc":
            return dict(self._command(WCC, {}))
        if name == "pr":
            return dict(self._command(PAGE_RANK, {}))
        if name == "bfs":
            source = params["source"]
            return {source: 0, **dict(self._command(BFS, {"source": source}))}
        raise NotSupported(name)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
