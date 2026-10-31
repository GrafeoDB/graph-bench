"""Neo4j adapter: the push selection agrees except the recorded known disagreements; import files, JVM and load.

Neo4j starts a JVM, and a process hosts one JVM only, so everything that opens a database runs in a subprocess (the
bench workers, or `_run` below) and the pytest process stays free for other JVM databases. The Neo4j tarball and
the JRE are fetched into the shared downloads (`GRAPH_BENCH_DOWNLOADS_DIR`, set to the normal cache's), so the
temporary data cache of `push_data_dir` does not download them again.
"""

import io
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("jpype")

from graph_bench.databases import get_database, jvm
from graph_bench.databases import neo4j as neo
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")
# Where the downloads live outside the tests: the normal data cache, read before any fixture overrides it.
SHARED_DOWNLOADS = os.environ.get(jvm.DOWNLOADS_ENV) or str(
    Path(os.environ.get("GRAPH_BENCH_DATA_DIR") or Path.home() / ".cache" / "graph-bench") / jvm.DOWNLOADS
)

pytestmark = pytest.mark.neo4j


@pytest.fixture
def shared_downloads(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv(jvm.DOWNLOADS_ENV, SHARED_DOWNLOADS)
    yield SHARED_DOWNLOADS


def test_neo4j_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_downloads: str,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["neo4j"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=300)
    known = known_disagreements("neo4j")
    entries = [e for e in results.entries if e.database == "neo4j"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate; any other ⊘ (a missing capability) must be allowlisted in
        # docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model:
            continue
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)
    # Nothing lands in the working directory (the importer's report file, Neo4j's logs).
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "work"]
    info = results.databases["neo4j"]
    assert info.java == f"Temurin {neo.JRE_VERSION}"
    assert info.storage == "on-disk"
    assert info.version == neo.NEO4J_VERSION


def test_neo4j_info_and_env(tmp_path: Path) -> None:
    """Without starting the JVM: the info probe and the worker environment."""
    cls = get_database("neo4j")
    info = cls(tmp_path, 1).info()
    assert info.process == "in-process"
    assert info.storage == "on-disk"
    assert info.threads_configurable
    assert "GDS concurrency=1" in info.threads_note
    assert "-XX:ParallelGCThreads=1, -XX:ConcGCThreads=1, -XX:CICompilerCount=2" in info.threads_note
    assert "not capped" in info.threads_note
    assert re.fullmatch(r"Temurin \d+(\.\d+)*", info.java or "")
    assert cls.process_env(None) == {"GRAPH_BENCH_JVM_THREADS": ""}
    assert cls.process_env(1) == {"GRAPH_BENCH_JVM_THREADS": "1"}


def _dataset(root: Path, header: str = "id,name,score") -> Dataset:
    (root / "P.csv").write_text(f'{header}\n1,"Smith, John",0.5\n2,"say ""hi""",1.5\n', encoding="utf-8")
    (root / "K.csv").write_text("src,dst,since\n1,2,7\n", encoding="utf-8")
    person = NodeTable("P", (Column("id", "INT64"), Column("name", "STRING"), Column("score", "DOUBLE")), "P.csv")
    knows = EdgeTable("K", "P", "P", (Column("since", "INT64"),), "K.csv")
    return Dataset("t", "small", root, (person,), (knows,), 2, 1)


def test_import_arguments_and_header_adapted_files(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    args = neo.import_arguments(ds, tmp_path / "import", 1)
    nodes, edges = tmp_path / "import" / "nodes-P.csv", tmp_path / "import" / "edges-K.csv"
    assert args == [
        "database",
        "import",
        "full",
        "neo4j",
        "--id-type=integer",
        "--path-pattern-style=none",
        f"--nodes=P={nodes}",
        f"--relationships=K={edges}",
        f"--max-off-heap-memory={neo.PAGE_CACHE_MIB}m",
        "--threads=1",
    ]
    # The header line is replaced; the rows (quoting included) are copied byte for byte.
    rows = '1,"Smith, John",0.5\n2,"say ""hi""",1.5\n'
    assert nodes.read_text(encoding="utf-8") == "id:ID(P),name:string,score:double\n" + rows
    assert edges.read_text(encoding="utf-8") == ":START_ID(P),:END_ID(P),since:long\n1,2,7\n"
    assert "--threads=1" not in neo.import_arguments(ds, tmp_path / "import2", None)


def test_import_arguments_refuse_a_wrong_header_and_mixed_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="header"):
        neo.import_arguments(_dataset(tmp_path, "id,score,name"), tmp_path / "import", 1)
    ds = _dataset(tmp_path)
    city = NodeTable("C", (Column("id", "STRING"),), "P.csv")
    mixed = Dataset("t", "small", tmp_path, (*ds.nodes, city), ds.edges, 2, 1)
    with pytest.raises(ValueError, match="mixed types"):
        neo.import_arguments(mixed, tmp_path / "import", 1)
    with pytest.raises(ValueError, match="cannot contain one"):
        neo._plain_path(tmp_path / "a,b.csv")


def test_admin_conf_and_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, jre, work = tmp_path / "home", tmp_path / "jre", tmp_path / "work"
    (home / "conf").mkdir(parents=True)
    (home / "conf" / "neo4j.conf").write_text("server.jvm.additional=--add-opens=java.base/java.nio=ALL-UNNAMED\n")
    monkeypatch.setenv(jvm.THREADS_ENV, "1")
    conf = neo.admin_conf(home, work)
    assert conf.startswith("server.jvm.additional=--add-opens=java.base/java.nio=ALL-UNNAMED\n")
    assert f"server.directories.data={(work / 'data').as_posix()}" in conf
    assert all(f"server.jvm.additional={arg}" in conf for arg in jvm.SINGLE_THREAD_ARGS)
    monkeypatch.delenv(jvm.THREADS_ENV)
    assert "ParallelGCThreads" not in neo.admin_conf(home, work)
    command = neo.admin_command(home, jre, ["database", "import"])
    assert command[-2:] == ["database", "import"]
    if os.name == "nt":
        assert command[0] == str(jre / "bin" / "java.exe")
        assert "org.neo4j.server.startup.NeoAdminBoot" in command
    else:
        assert command[:2] == ["sh", str(home / "bin" / "neo4j-admin")]


def test_extract_once_and_refuse_escaping_members(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"x"
        member = tarfile.TarInfo("top/bin/file")
        member.size = len(data)
        tf.addfile(member, io.BytesIO(data))
    dest = tmp_path / "out"
    assert neo.extract(archive, "top", dest) == dest / "top"
    assert (dest / "top" / "bin" / "file").read_bytes() == b"x"
    archive.unlink()  # a second call reuses the extracted directory
    assert neo.extract(archive, "top", dest) == dest / "top"
    with pytest.raises(ValueError, match="no top directory"):
        bad = tmp_path / "b.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("other/file", "x")
        neo.extract(bad, "top2", dest)
    evil = tmp_path / "c.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape", "x")
    with pytest.raises(ValueError, match="outside the archive root"):
        neo.extract(evil, "top3", dest)
    assert sorted(p.name for p in dest.iterdir()) == ["top"]  # no partial or temporary directory stays


def _run(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_neo4j_load_algorithms_and_jvm(push_data_dir: Path, tmp_path: Path, shared_downloads: str) -> None:
    """One JVM per process with the shared caps; two loads in one process; counts, BFS, IC13 on one person.

    After `prepare_environment`, a load fetches, hashes, installs and starts nothing: `jvm.fetch`, `install` and
    `jvm.start_once` are replaced by counters for the loads.
    """
    script = """
        from pathlib import Path
        from graph_bench.databases import get_database, jvm
        from graph_bench.databases import neo4j as neo
        from graph_bench.databases.base import NotSupported
        from graph_bench.datasets import cache
        cls = get_database("neo4j")
        cls.prepare_environment()
        cls.prepare_environment()  # once per process: a second call does nothing
        assert jvm.owner() == "neo4j"
        assert jvm.effective_heap_flag() == jvm.HEAP_FLAG
        assert jvm.thread_caps() == {"ParallelGCThreads": "1", "ConcGCThreads": "1", "CICompilerCount": "2"}
        try:
            jvm.start_once("arcadedb", lambda: None)
            raise AssertionError("a second JVM database started in the same process")
        except RuntimeError as exc:
            assert "needs a process of its own" in str(exc)
        calls = []
        jvm.fetch = lambda *a, **k: calls.append("fetch")
        neo.install = lambda: calls.append("install")
        jvm.start_once = lambda *a, **k: calls.append("start_once")
        data = cache.prepare("push")
        ga, snb = data[("graphalytics", "small")], data[("snb", "small")]
        for i in range(2):  # the GDS projection is dropped with its instance, so a second load can project again
            Path(f"ga{i}").mkdir()
            db = cls(Path(f"ga{i}"), 1)
            db.load(ga)
            assert db.counts() == (ga.node_count, ga.edge_count)
            bfs = db.algorithm("bfs", {"source": 0})
            assert bfs[0] == 0 and isinstance(bfs[1], int)
            assert isinstance(db.algorithm("pr", {})[0], float)
            try:
                db.algorithm("cdlp", {})
                raise AssertionError("cdlp is not implemented")
            except NotSupported:
                pass
            db.close()
        Path("snb").mkdir()
        db = cls(Path("snb"), 1)
        db.load(snb)
        assert calls == [], calls  # three loads: no fetch, hash, install or JVM start inside a load
        assert db.counts() == (snb.node_count, snb.edge_count)
        same = "MATCH (p1:Person {id: $a}), (p2:Person {id: $b}) " \\
            "OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2)) " \\
            "RETURN CASE WHEN p1.id = p2.id THEN 0 WHEN path IS NULL THEN -1 ELSE length(path) END AS d"
        assert db.query(same, {"a": "person_1", "b": "person_1"}) == [(0,)]
        try:
            db.algorithm("wcc", {})
            raise AssertionError("no projection on the SNB dataset")
        except RuntimeError as exc:
            assert "graphalytics" in str(exc)
        db.close()
        print("ok")
    """
    env = {"GRAPH_BENCH_DATA_DIR": str(push_data_dir), **get_database("neo4j").process_env(1)}
    done = _run(script, tmp_path, env)
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    assert done.stdout.strip().endswith("ok")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ga0", "ga1", "snb"]


def test_neo4j_load_refuses_a_jvm_without_thread_options(
    push_data_dir: Path, tmp_path: Path, shared_downloads: str
) -> None:
    """threads=1 with a JVM started without `process_env(1)` fails the load instead of running wider."""
    script = """
        from pathlib import Path
        from graph_bench.databases import get_database
        from graph_bench.datasets import cache
        db = get_database("neo4j")(Path("."), 1)
        try:
            db.load(cache.prepare("push")[("graphalytics", "small")])
        except RuntimeError as exc:
            print(exc)
    """
    env = {"GRAPH_BENCH_DATA_DIR": str(push_data_dir), jvm.THREADS_ENV: ""}
    done = _run(script, tmp_path, env)
    assert done.returncode == 0, done.stderr[-3000:]
    assert "the JVM's thread options are" in done.stdout


def test_prepare_environment_returns_at_once_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a JVM: once the process is prepared, the call (also made by `load`) does no download or JVM work."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("called after the process was prepared")

    monkeypatch.setattr(neo, "_STATE", {"ready": True})
    monkeypatch.setattr(neo, "install", forbidden)
    monkeypatch.setattr(jvm, "fetch", forbidden)
    monkeypatch.setattr(jvm, "start_once", forbidden)
    neo.Neo4jDatabase.prepare_environment()


def test_admin_env_drops_the_callers_jvm_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "JAVA_OPTS", "CLASSPATH", "NEO4J_HOME"):
        monkeypatch.setenv(name, "-Xmx64m")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = neo.admin_env(tmp_path / "jre", tmp_path / "conf")
    assert env["PATH"] == "/usr/bin"
    assert env["JAVA_HOME"] == str(tmp_path / "jre")
    assert env["NEO4J_CONF"] == str(tmp_path / "conf")
    assert env["HEAP_SIZE"] == jvm.MAX_HEAP
    assert not set(env) - {*neo._ADMIN_ENV_KEEP, "JAVA_HOME", "NEO4J_CONF", "HEAP_SIZE"}


def test_run_admin_kills_the_whole_tree_on_a_timeout(tmp_path: Path) -> None:
    """A hung command is killed with the process it started (the launcher JVM starts the import JVM)."""
    psutil = pytest.importorskip("psutil")
    pid_file = tmp_path / "child.pid"
    parent = textwrap.dedent(
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        time.sleep(120)
        """
    )
    with pytest.raises(RuntimeError, match="did not finish in 5 s"):
        neo.run_admin([sys.executable, "-c", parent], tmp_path, dict(os.environ), timeout_s=5)
    child = int(pid_file.read_text())
    gone = False
    for _ in range(50):
        try:
            gone = psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            gone = True
        if gone:
            break
        psutil.wait_procs([], timeout=0.1)
    assert gone, f"child {child} outlived the timeout"


def test_run_admin_reports_a_failure(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('bad input'); raise SystemExit(3)"]
    with pytest.raises(RuntimeError, match=r"failed \(exit 3\): bad input"):
        neo.run_admin(command, tmp_path, dict(os.environ))
    assert neo.run_admin([sys.executable, "-c", "print('done')"], tmp_path, dict(os.environ)).strip() == "done"


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("linux", "x86_64", ("linux", "x86_64")),
        ("linux", "aarch64", ("linux", "aarch64")),
        ("linux", "arm64", ("linux", "aarch64")),
        ("win32", "AMD64", ("win32", "amd64")),
    ],
)
def test_jre_pin_follows_the_machine(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, expected: tuple[str, str]
) -> None:
    monkeypatch.setattr(neo.sys, "platform", system)
    monkeypatch.setattr(neo.platform, "machine", lambda: machine)
    assert neo._platform_key() == expected
    assert expected in neo.JRES
