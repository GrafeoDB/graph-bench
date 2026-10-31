"""Neo4j Community, embedded through JPype: an on-disk store in the instance's workdir, Cypher and GDS Community.

Nothing is installed from PyPI except JPype: `prepare_environment` fetches the Neo4j Community tarball and a Temurin
JRE into the data cache (`jvm.fetch`, SHA-256 checked), extracts them there, and starts this process's JVM with the
tarball's jars on the classpath, all outside every clock. Each load writes the store with `neo4j-admin database
import full` (Neo4j's offline bulk importer) and then opens it with the embedded `DatabaseManagementServiceBuilder`
in this process.

Docs (Neo4j 2026.10):
- Embedding: https://neo4j.com/docs/java-reference/current/java-embedded/
- Importer: https://neo4j.com/docs/operations-manual/current/import/ and
  https://neo4j.com/docs/operations-manual/current/tools/neo4j-admin/neo4j-admin-import/ (`--help` lists the
  options used below); neo4j-admin's environment variables (NEO4J_CONF, HEAP_SIZE):
  https://neo4j.com/docs/operations-manual/current/neo4j-admin-neo4j-cli/
- Settings: https://neo4j.com/docs/operations-manual/current/configuration/configuration-settings/
- GDS: https://neo4j.com/docs/graph-data-science/current/ (algorithms/wcc, algorithms/page-rank,
  algorithms/dijkstra-single-source, management-ops/graph-creation/graph-project).

Licences: Neo4j Community is GPLv3 (`LICENSE.txt` in the tarball; https://neo4j.com/licensing/). The GDS jar ships
in the same tarball (`products/`); its `LICENSE.txt` is GPLv3 and its `NOTICE.txt` says the Neo4j-owned software is
licensed under GPLv3 unless a commercial agreement supersedes it. Without a licence file GDS runs as its Community
Edition ("By default, GDS will operate as the Community Edition", concurrency capped at 4; `gds.license.state()`
reports "No valid GDS license specified."), https://neo4j.com/docs/graph-data-science/current/introduction/.
Nothing is redistributed: the downloads stay in the data cache.
"""

import contextlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import jpype
import jpype.imports

from graph_bench.databases import jvm
from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.databases.pins import (
    GDS_JAR,
    JRE_RELEASE,
    JRE_VERSION,
    JRES,
    NEO4J_HOME_NAME,
    NEO4J_SHA256,
    NEO4J_URL,
    NEO4J_VERSION,
)
from graph_bench.datasets.schema import Column, Dataset
from graph_bench.types import Model

# The pinned Neo4j tarball and Temurin JRE (versions, URLs, SHA-256) are in `pins.py`, which `data key` hashes.
# The JVM options Neo4j's default `conf/neo4j.conf` gives under "Open modules for neo4j" and "Enable access to
# JDK incubator module", plus the netty reflection flag from the same file: the embedded engine needs them as well.
NEO4J_JVM_ARGS = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--enable-native-access=ALL-UNNAMED",
    "--add-modules=jdk.incubator.vector",
    "-Dorg.neo4j.shaded.lucene9.vectorization.upperJavaFeatureVersion=25",
    "-Dio.netty.tryReflectionSetAccessible=true",
)
# The page cache (`server.memory.pagecache.size`, off heap). Unset, Neo4j sizes it from the machine's RAM, so it is
# fixed here: 1 GiB, the page cache ArcadeDB gets with the shared heap (a quarter of `jvm.MAX_HEAP`). The importer
# gets the same budget (`--max-off-heap-memory`, whose default is 90% of the free memory).
PAGE_CACHE_MIB = 1024
# Importer column types (the header format of `neo4j-admin database import`).
_TYPES = {"INT64": "long", "DOUBLE": "double", "STRING": "string"}
DATABASE = "neo4j"
GRAPH = "g"
# The load check (`counts`): every node and every relationship, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n) AS n"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r) AS n"
# GDS on the projection `g` of the graphalytics dataset (made in `load`). The edge list holds both directions of
# every edge, so the natural orientation is the undirected graph, as in the other adapters. `concurrency` is set to
# 1 for single-threaded runs (`_concurrency`, substituted for `%s`). Node ids go back to the app `id` with
# `gds.util.asNode`.
# WCC (algorithms/wcc): component ids are arbitrary (the answer comparison relabels).
WCC = "CALL gds.wcc.stream('g', {%s}) YIELD nodeId, componentId RETURN gds.util.asNode(nodeId).id AS id, componentId"
# PageRank (algorithms/page-rank): the defaults are 20 iterations and tolerance 1e-7, so damping 0.85, the bound of
# 1000 iterations and tolerance 1e-10 of the other adapters are passed. Unscaled, GDS scores do not sum to 1; the
# documented scaler L1Norm "creates a probability distribution where the sum of scores equals 1", the definition
# the reference (NetworkX) uses.
PAGE_RANK = (
    "CALL gds.pageRank.stream('g', {dampingFactor: 0.85, maxIterations: 1000, tolerance: 1e-10, scaler: 'L1NORM'%s}) "
    "YIELD nodeId, score RETURN gds.util.asNode(nodeId).id AS id, score"
)
# BFS levels: GDS has no BFS with depths (`gds.bfs.stream` yields the visit order only), so the hop distances come
# from the single-source shortest path procedure run unweighted (algorithms/dijkstra-single-source:
# relationshipWeightProperty "If unspecified, the algorithm runs unweighted", every relationship costs 1). Its result
# includes the source with cost 0 and no row for an unreachable vertex, the reference's definition.
BFS = (
    "MATCH (s:V {id: $source}) CALL gds.allShortestPaths.dijkstra.stream('g', {sourceNode: s%s}) "
    "YIELD targetNode, totalCost RETURN gds.util.asNode(targetNode).id AS id, totalCost"
)
_STATE: dict[str, Any] = {}


def _platform_key() -> tuple[str, str]:
    return sys.platform, platform.machine().lower()


def install_root() -> Path:
    """Where the archives are extracted: `unpacked` next to them, in `jvm.downloads_dir()`."""
    return jvm.downloads_dir() / "unpacked"


def neo4j_home() -> Path:
    return install_root() / NEO4J_HOME_NAME


def jre_home() -> Path:
    return install_root() / f"{JRE_RELEASE}-jre"


def plugins_dir() -> Path:
    """The plugins directory of every instance: a copy of the GDS jar alone."""
    return install_root() / f"plugins-{NEO4J_VERSION}"


def jvm_library(home: Path) -> Path:
    """The JVM library inside a JRE (the Temurin layout)."""
    return home / "bin" / "server" / "jvm.dll" if os.name == "nt" else home / "lib" / "server" / "libjvm.so"


def _inside(root: Path, name: str) -> bool:
    target = (root / name).resolve()
    return target == root.resolve() or root.resolve() in target.parents


def extract(archive: Path, top: str, dest_root: Path) -> Path:
    """Extract `archive` (.tar.gz or .zip) whose single top directory is `top` into `dest_root/top`, once.

    The archive is extracted into a temporary directory next to the target and renamed into place only when it is
    complete, so an interrupted extraction never leaves a partial `top`. Members outside the archive's root are
    refused (tar: the `data` filter of `tarfile`; zip: a path check).
    """
    target = dest_root / top
    if target.is_dir():
        return target
    dest_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{top}.", dir=dest_root))
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    if not _inside(scratch, name):
                        raise ValueError(f"{archive.name}: member {name!r} is outside the archive root")
                zf.extractall(scratch)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(scratch, filter="data")
        if not (scratch / top).is_dir():
            raise ValueError(f"{archive.name} has no top directory {top!r}")
        (scratch / top).replace(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def install() -> tuple[Path, Path]:
    """Fetch (SHA-256 checked) and extract the Neo4j Community tarball and the Temurin JRE; their homes."""
    key = _platform_key()
    if key not in JRES:
        raise RuntimeError(f"no pinned Temurin JRE for {key[0]}/{key[1]} (supported: {sorted(JRES)})")
    tarball = jvm.fetch(NEO4J_URL, NEO4J_SHA256, timeout_s=600.0)
    jre_url, jre_sha = JRES[key]
    jre_archive = jvm.fetch(jre_url, jre_sha, timeout_s=600.0)
    home = extract(tarball, NEO4J_HOME_NAME, install_root())
    jre = extract(jre_archive, jre_home().name, install_root())
    # GDS loads from the plugins directory (and from the classpath, see `_start`); this one holds the GDS jar alone,
    # so the tarball's other product (the GenAI plugin) is not loaded.
    plugins = plugins_dir()
    if not (plugins / GDS_JAR).is_file():
        plugins.mkdir(parents=True, exist_ok=True)
        part = plugins / f".{GDS_JAR}.part"
        shutil.copyfile(home / "products" / GDS_JAR, part)
        part.replace(plugins / GDS_JAR)
    return home, jre


def _start() -> None:
    """Start the JVM: the shared options (`jvm.jvm_args`), then Neo4j's own, the tarball's jars and the GDS jar.

    GDS needs its jar on the classpath (else its procedures fail to inject) and in the plugins directory (else they
    are not registered), research-c section 2.6.
    """
    home = neo4j_home()
    classpath = [*sorted(str(p) for p in (home / "lib").glob("*.jar")), str(home / "products" / GDS_JAR)]
    jpype.startJVM(str(jvm_library(jre_home())), *jvm.jvm_args(), *NEO4J_JVM_ARGS, classpath=classpath)


def _builder(home: Path) -> Any:
    """A `DatabaseManagementServiceBuilder` on `home`, with the settings every instance uses."""
    from java.lang import Boolean
    from java.nio.file import Path as JPath
    from java.util import List
    from org.neo4j.configuration import GraphDatabaseSettings
    from org.neo4j.dbms.api import DatabaseManagementServiceBuilder
    from org.neo4j.io import ByteUnit

    builder = DatabaseManagementServiceBuilder(JPath.of(str(home)))
    builder = builder.setConfig(GraphDatabaseSettings.pagecache_memory, jpype.JLong(ByteUnit.mebiBytes(PAGE_CACHE_MIB)))
    builder = builder.setConfig(GraphDatabaseSettings.plugin_dir, JPath.of(str(plugins_dir())))
    builder = builder.setConfig(GraphDatabaseSettings.procedure_unrestricted, List.of("gds.*"))
    builder = builder.setConfig(GraphDatabaseSettings.procedure_allowlist, List.of("gds.*"))
    # IC13 binds a person with itself once. `dbms.cypher.forbid_shortestpath_common_nodes` (default true) makes
    # shortestPath throw "The shortest path algorithm does not work when the start and end nodes are the same";
    # with false "no path will be returned when that happens" (the settings reference), so the shared text's
    # `CASE WHEN p1.id = p2.id THEN 0` gives 0. The setting changes nothing for distinct nodes.
    return builder.setConfig(GraphDatabaseSettings.forbid_shortestpath_common_nodes, Boolean.FALSE)


def _to_java(value: Any) -> Any:
    """A parameter value as Java: ints as Long (the stored integers are Long), lists as ArrayList."""
    from java.lang import Boolean, Double, String
    from java.util import ArrayList

    if value is None:
        return None
    if isinstance(value, bool):
        return Boolean(value)
    if isinstance(value, int):
        return jpype.JLong(value)
    if isinstance(value, float):
        return Double(value)
    if isinstance(value, str):
        return String(value)
    if isinstance(value, (list, tuple)):
        items = ArrayList()
        for item in value:
            items.add(_to_java(item))
        return items
    raise TypeError(f"unsupported parameter type {type(value).__name__}")


def _params(params: Mapping[str, Any]) -> Any:
    from java.util import HashMap

    result = HashMap()
    for key, value in params.items():
        result.put(key, _to_java(value))
    return result


def _to_python(value: Any) -> Any:
    """A boxed Java result value as Python: Long/Integer to int, Double to float, Boolean, String, List, Map."""
    from java.lang import Boolean, Byte, Double, Float, Integer, Long, Short, String
    from java.util import List, Map

    if value is None:
        return None
    if isinstance(value, Boolean):
        return bool(value.booleanValue())
    if isinstance(value, (Long, Integer, Short, Byte)):
        return int(value.longValue())
    if isinstance(value, (Double, Float)):
        return float(value.doubleValue())
    if isinstance(value, String):
        return str(value)
    if isinstance(value, List):
        return [_to_python(item) for item in value]
    if isinstance(value, Map):
        return {str(k): _to_python(value.get(k)) for k in value.keySet()}
    raise TypeError(f"unsupported result type {type(value).__name__}")


def _header(columns: tuple[Column, ...]) -> list[str]:
    return [f"{c.name}:{_TYPES[c.type]}" for c in columns]


def _adapt(source: Path, expected: list[str], header: list[str], target: Path) -> None:
    """Copy a dataset CSV with its header line replaced by the importer's typed header.

    The importer reads the header from the first line of a data source's first file, so the CSV's own header line
    is swapped for the typed one; the rows are copied byte for byte.
    """
    with source.open("rb") as src, target.open("wb") as dst:
        first = src.readline().decode("utf-8").rstrip("\r\n")
        if first.split(",") != expected:
            raise ValueError(f"{source.name}: header {first!r}, expected {','.join(expected)!r}")
        dst.write((",".join(header) + "\n").encode("utf-8"))
        shutil.copyfileobj(src, dst, 1 << 20)


def _plain_path(path: Path) -> str:
    """A path for an importer file list, which separates files with commas."""
    text = str(path)
    if "," in text:
        raise ValueError(f"the importer separates files with commas, so a path cannot contain one: {text}")
    return text


def _id_type(dataset: Dataset) -> str:
    """`--id-type`: integer when every node id column is INT64, string when every one is STRING."""
    kinds = {table.columns[0].type for table in dataset.nodes}
    if kinds == {"INT64"}:
        return "integer"
    if kinds == {"STRING"}:
        return "string"
    raise ValueError(f"node id columns of mixed types {sorted(kinds)}; the importer takes one --id-type")


def import_arguments(dataset: Dataset, import_dir: Path, threads: int | None) -> list[str]:
    """Write the header-adapted CSVs into `import_dir`; the `neo4j-admin database import full` arguments.

    Each node table is a node source with its label and an `id:ID(<label>)` column (an id space per label, the
    property `id` is stored too); each edge table a relationship source with its type and
    `:START_ID(<src label>),:END_ID(<dst label>)`. Typed property columns follow, in the CSV's order.
    """
    import_dir.mkdir(parents=True, exist_ok=True)
    args = ["database", "import", "full", DATABASE, f"--id-type={_id_type(dataset)}", "--path-pattern-style=none"]
    for table in dataset.nodes:
        target = import_dir / f"nodes-{table.file}"
        expected = [c.name for c in table.columns]
        _adapt(dataset.root / table.file, expected, [f"id:ID({table.label})", *_header(table.columns[1:])], target)
        args.append(f"--nodes={table.label}={_plain_path(target)}")
    for edge in dataset.edges:
        target = import_dir / f"edges-{edge.file}"
        expected = ["src", "dst", *(c.name for c in edge.columns)]
        header = [f":START_ID({edge.src})", f":END_ID({edge.dst})", *_header(edge.columns)]
        _adapt(dataset.root / edge.file, expected, header, target)
        args.append(f"--relationships={edge.type}={_plain_path(target)}")
    args.append(f"--max-off-heap-memory={PAGE_CACHE_MIB}m")
    if threads == 1:
        # "(advanced) Max number of worker threads used by the importer."
        args.append("--threads=1")
    return args


def admin_command(home: Path, jre: Path, args: list[str]) -> list[str]:
    """The `neo4j-admin` command line.

    POSIX: the tarball's `bin/neo4j-admin` script. Windows: the unix tarball has no `.bat`/`.ps1` launcher, so the
    script's own `exec` line is run directly (the same JVM options, classpath, properties and main class).
    """
    if os.name != "nt":
        return ["sh", str(home / "bin" / "neo4j-admin"), *args]
    return [
        str(jre / "bin" / "java.exe"),
        "-Xmx128m",
        "-XX:+IgnoreUnrecognizedVMOptions",
        "--sun-misc-unsafe-memory-access=allow",
        "-classpath",
        str(home / "lib" / "*"),
        "-Dapp.name=neo4j-admin",
        f"-Dapp.home={home}",
        f"-Dbasedir={home}",
        "org.neo4j.server.startup.NeoAdminBoot",
        *args,
    ]


def admin_conf(home: Path, workdir: Path) -> str:
    """The importer's `neo4j.conf`: the tarball's default file, the instance's directories and the thread caps.

    Neo4j's default configuration carries the JVM options the importer needs (`--add-opens` and so on, as
    `server.jvm.additional` lines). The data and log directories point into the instance's workdir, so the store
    lands where the embedded engine opens it (`<workdir>/data`, its default for home = workdir) and nothing is
    written into the cached tarball. In a single-threaded run `jvm.SINGLE_THREAD_ARGS` are added the same way.
    """
    lines = [
        f"server.directories.data={(workdir / 'data').as_posix()}",
        f"server.directories.logs={(workdir / 'logs').as_posix()}",
    ]
    if jvm.single_threaded():
        lines += [f"server.jvm.additional={arg}" for arg in jvm.SINGLE_THREAD_ARGS]
    default = (home / "conf" / "neo4j.conf").read_text(encoding="utf-8")
    return default.rstrip("\n") + "\n\n# graph-bench\n" + "\n".join(lines) + "\n"


# The caller's environment variables the importer keeps: what a process needs to run (the search path for the
# launcher script's tools, the Windows system directories, temporary directories, locale, home). Everything else,
# notably JVM option variables (JAVA_TOOL_OPTIONS, JDK_JAVA_OPTIONS, _JAVA_OPTIONS, JAVA_OPTS) and CLASSPATH, stays
# out, so the importer's JVM gets only the configuration below.
_ADMIN_ENV_KEEP = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)
# A hung import is killed after this long. A load reports nothing until it is done, so the supervisor's silence
# cap (`--case-timeout`, default 600 s) would otherwise kill the worker first and leave the importer's JVMs
# running; 540 s stays inside that default with room for the rest of the load. The push loads take about 5 s.
IMPORT_TIMEOUT_S = 540.0


def admin_env(jre: Path, conf: Path) -> dict[str, str]:
    """The importer's environment: `_ADMIN_ENV_KEEP` from this process, the JRE, the configuration and the heap.

    neo4j-admin reads NEO4J_CONF (the directory of neo4j.conf) and HEAP_SIZE (the maximum heap); "If set, HEAP_SIZE
    and JAVA_OPTS override all relevant settings specified in the configuration file", so JAVA_OPTS is never set
    (https://neo4j.com/docs/operations-manual/current/neo4j-admin-neo4j-cli/).
    """
    upper = {key.upper(): value for key, value in os.environ.items()}
    env = {key: upper[key] for key in _ADMIN_ENV_KEEP if key in upper}
    return {**env, "JAVA_HOME": str(jre), "NEO4J_CONF": str(conf), "HEAP_SIZE": jvm.MAX_HEAP}


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill a process and every process it started: the launcher JVM starts the import JVM as its child."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # the process group of `start_new_session`
    proc.kill()
    proc.communicate()


def run_admin(command: list[str], cwd: Path, env: dict[str, str], timeout_s: float = IMPORT_TIMEOUT_S) -> str:
    """Run `command` (neo4j-admin) to completion; its output. A failure or a timeout raises RuntimeError.

    On POSIX the command gets a session (and process group) of its own; on a timeout, or any interruption, the
    whole tree is killed (`_kill_tree`), so no import JVM outlives the call.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise RuntimeError(f"neo4j-admin did not finish in {timeout_s:g} s; its processes were killed") from None
    except BaseException:
        _kill_tree(proc)
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"neo4j-admin failed (exit {proc.returncode}): {output.strip()[-2000:]}")
    return output


class Neo4jDatabase(Database):
    """An embedded Neo4j Community database in `workdir`, written by `neo4j-admin database import full`.

    The load clock covers: the header-adapted CSV copies, the importer (a subprocess with the fetched JRE, Neo4j's
    fastest documented bulk path for an empty database), opening the store embedded, a unique constraint on `id`
    per node label (each backed by a range index, awaited online), and on the graphalytics dataset the GDS
    projection `g`. Queries are the shared openCypher texts, run natively; algorithms are GDS procedures.
    """

    name: ClassVar[str] = "neo4j"
    display_name: ClassVar[str] = "Neo4j"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    dbms: Any = None
    db: Any = None
    projected: bool = False

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        return jvm.process_env(threads)

    @classmethod
    def prepare_environment(cls) -> None:
        """Fetch and extract the tarball and the JRE, start the JVM, then open and shut down one empty DBMS.

        All of it is outside every clock. The empty DBMS covers the engine's one-time start (class loading, GDS
        registration, several seconds) that would otherwise land in the first load; it holds no data and runs no
        query, so no load or query path is warmed up (ArcadeDB, NeuG and LadybugDB likewise open a throwaway
        database here).
        """
        if _STATE.get("ready"):
            # Once per process: `load` calls this too (for direct use), and a load must not fetch, hash or start
            # anything (`jvm.fetch` re-hashes the cached archives).
            return
        install()
        jvm.start_once(cls.name, _start)
        scratch = Path(tempfile.mkdtemp(prefix="graph-bench-neo4j-"))
        try:
            _builder(scratch).build().shutdown()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        _STATE["ready"] = True

    @property
    def _concurrency(self) -> str:
        """GDS options: `concurrency: 1` for single-threaded runs; else the default (4, Community's maximum)."""
        return ", concurrency: 1" if self.threads == 1 else ""

    def info(self) -> DatabaseInfo:
        java = jvm.java_version(jre_home()) if (jre_home() / "release").is_file() else JRE_VERSION
        if self.threads == 1:
            note = (
                "configured to 1 thread: Cypher runs each query on the calling thread (Community has no parallel "
                "runtime); GDS concurrency=1 and readConcurrency=1; neo4j-admin import --threads=1; JVM (embedded and "
                "importer): "
                + ", ".join(jvm.SINGLE_THREAD_ARGS)
                + "; not capped: Neo4j's background threads (page cache flushing, checkpointing, index population) "
                "and the JVM's other service threads, which have no documented setting"
            )
        else:
            note = "Neo4j defaults; GDS concurrency 4 (the Community maximum)"
        return DatabaseInfo(self.name, NEO4J_VERSION, "in-process", True, note, "on-disk", java=f"Temurin {java}")

    def _import(self, dataset: Dataset) -> None:
        home, jre, work = neo4j_home(), jre_home(), self.workdir.resolve()
        conf = work / "conf"
        conf.mkdir(parents=True, exist_ok=True)
        (conf / "neo4j.conf").write_text(admin_conf(home, work), encoding="utf-8")
        args = import_arguments(dataset, work / "import", self.threads)
        # The report file defaults to `import.report` in the current directory.
        args.append(f"--report-file={work / 'import.report'}")
        run_admin(admin_command(home, jre, args), work, admin_env(jre, conf))

    def load(self, dataset: Dataset) -> None:
        # The benchmark paths (worker, CodSpeed session) call prepare_environment first, so the downloads and the JVM
        # start are never in a load clock: this call returns at once then, and only serves direct use (tests).
        self.prepare_environment()
        jvm.check_threads(self.threads)
        self._import(dataset)
        self.dbms = _builder(self.workdir.resolve()).build()
        self.db = self.dbms.database(DATABASE)
        for table in dataset.nodes:
            # https://neo4j.com/docs/cypher-manual/current/schema/constraints/create-constraints/ (property
            # uniqueness; it creates the range index that the `{id: ...}` lookups use).
            self.db.executeTransactionally(f"CREATE CONSTRAINT FOR (n:`{table.label}`) REQUIRE n.id IS UNIQUE")
        # https://neo4j.com/docs/operations-manual/current/procedures/ (db.awaitIndexes: wait until all are online).
        self.db.executeTransactionally("CALL db.awaitIndexes(300)")
        if dataset.name == "graphalytics":
            # https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/
            read = ", {readConcurrency: 1}" if self.threads == 1 else ""
            self._run(f"CALL gds.graph.project('{GRAPH}', 'V', 'E'{read}) YIELD nodeCount", {}, commit=True)
            self.projected = True

    def _run(self, text: str, params: Mapping[str, Any], *, commit: bool = False) -> list[tuple[Any, ...]]:
        with self.db.beginTx() as tx:
            result = tx.execute(text, _params(params))
            columns = list(result.columns())
            rows = []
            while result.hasNext():
                row = result.next()
                rows.append(tuple(_to_python(row.get(c)) for c in columns))
            if commit:
                tx.commit()
            return rows

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        return self._run(text, params)

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES`."""
        return int(self.query(COUNT_NODES, {})[0][0]), int(self.query(COUNT_EDGES, {})[0][0])

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name not in ("bfs", "wcc", "pr"):
            raise NotSupported(name)
        if not self.projected:
            raise RuntimeError("the GDS graph `g` is projected only for the graphalytics dataset")
        if name == "wcc":
            return dict(self._run(WCC % self._concurrency.removeprefix(", "), {}))
        if name == "pr":
            return dict(self._run(PAGE_RANK % self._concurrency, {}))
        depths = {}
        for vertex, cost in self._run(BFS % self._concurrency, {"source": params["source"]}):
            if not float(cost).is_integer():
                raise RuntimeError(f"unweighted shortest path cost {cost} for vertex {vertex} is not a hop count")
            depths[vertex] = int(cost)
        return depths

    def close(self) -> None:
        if self.dbms is None:
            return
        try:
            if self.projected:
                # The GDS graph catalog outlives a DBMS in the JVM, so the projection is dropped with its instance.
                self._run(f"CALL gds.graph.drop('{GRAPH}', false) YIELD graphName", {})
                self.projected = False
        finally:
            self.dbms.shutdown()
            self.dbms = None
            self.db = None
