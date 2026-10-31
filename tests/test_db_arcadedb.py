"""ArcadeDB adapter: the push selection agrees except the recorded known disagreements; JVM settings and load.

ArcadeDB starts a JVM, and a process hosts one JVM only, so everything that opens a database runs in a
subprocess (the bench workers, or `_run` below) and the pytest process stays free for other JVM databases.
"""

import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("arcadedb_embedded")

from graph_bench.databases import get_database
from graph_bench.results import Results
from graph_bench.types import Model
from tests.conftest import (
    assert_loads_agree,
    assert_resources_ok,
    assert_startup_agrees,
    not_yet_implemented,
    run_push,
    warn_if_fixed,
)

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.arcadedb


@pytest.fixture(scope="module")
def push_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The push run's working directory (its `out` and `work` are inside)."""
    return tmp_path_factory.mktemp("push-arcadedb")


@pytest.fixture(scope="module")
def push(push_data_dir: Path, push_root: Path) -> Results:
    """One push run of arcadedb, shared by the push, startup and load tests."""
    return run_push("arcadedb", push_root, push_data_dir, 120)


def test_arcadedb_push_agrees(push: Results, push_root: Path, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("arcadedb")
    entries = [e for e in results.entries if e.database == "arcadedb"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate; any other ⊘ (a missing capability) must be allowlisted in
        # docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)
    # The engine log goes to a per-process temporary directory, not to ./log in the working directory.
    assert not (push_root / "log").exists()
    info = results.databases["arcadedb"]
    assert info.java is not None
    assert info.storage == "on-disk"


def test_arcadedb_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "arcadedb")


def test_arcadedb_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "arcadedb")


def test_arcadedb_resources_ok(push: Results) -> None:
    """Memory and Disk: one ok, agreeing record per load with ordered readings; on disk."""
    assert_resources_ok(push, "arcadedb", on_disk=True)


def test_arcadedb_info_and_env(tmp_path: Path) -> None:
    """Without starting the JVM: the info probe, and the single-thread JVM options for the worker environment."""
    cls = get_database("arcadedb")
    info = cls(tmp_path, 1).info()
    assert info.process == "in-process"
    assert info.storage == "on-disk"
    assert info.threads_configurable
    assert "queryParallelismPoolThreads=1" in info.threads_note
    assert "-XX:ParallelGCThreads=1, -XX:ConcGCThreads=1, -XX:CICompilerCount=2" in info.threads_note
    assert "not capped" in info.threads_note
    assert re.fullmatch(r"\d+(\.\d+)*", info.java or "")
    assert cls.process_env(None) == {"GRAPH_BENCH_JVM_THREADS": "", "ARCADEDB_JVM_ARGS": ""}
    env = cls.process_env(1)
    assert env["GRAPH_BENCH_JVM_THREADS"] == "1"
    args = env["ARCADEDB_JVM_ARGS"].split()
    assert "-Darcadedb.queryParallelismPoolThreads=1" in args
    assert "-Darcadedb.queryParallelScan=false" in args
    assert "-Djava.util.concurrent.ForkJoinPool.common.parallelism=1" in args
    assert not [a for a in args if a.startswith("-Xmx")]


def _run(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_arcadedb_load_algorithms_and_jvm(push_data_dir: Path, tmp_path: Path) -> None:
    """One JVM per process with the shared cap and the thread options; load, counts and the BFS source row."""
    cls = get_database("arcadedb")
    script = """
        from pathlib import Path
        from graph_bench.databases import get_database, jvm
        from graph_bench.databases.base import NotSupported
        from graph_bench.datasets import cache
        cls = get_database("arcadedb")
        cls.prepare_environment()
        cls.prepare_environment()  # once per process: a second call does nothing
        assert jvm.owner() == "arcadedb"
        assert jvm.effective_heap_flag() == jvm.HEAP_FLAG
        assert jvm.system_property("arcadedb.queryParallelismPoolThreads") == "1"
        assert jvm.thread_caps() == {"ParallelGCThreads": "1", "ConcGCThreads": "1", "CICompilerCount": "2"}
        assert jvm._vm_option("TieredCompilation") == "true"  # tiered compilation stays on
        try:
            jvm.start_once("neo4j", lambda: None)
            raise AssertionError("a second JVM database started in the same process")
        except RuntimeError as exc:
            assert "needs a process of its own" in str(exc)
        dataset = cache.prepare("push")[("graphalytics", "small")]
        db = cls(Path("work"), 1)
        Path("work").mkdir()
        db.load(dataset)
        assert db.counts() == (dataset.node_count, dataset.edge_count)
        bfs = db.algorithm("bfs", {"source": 0})
        assert bfs[0] == 0 and min(bfs.values()) == 0 and isinstance(bfs[1], int)
        assert isinstance(db.algorithm("pr", {})[0], float)
        try:
            db.algorithm("cdlp", {})
            raise AssertionError("cdlp is not implemented")
        except NotSupported:
            pass
        db.close()
        print("ok")
    """
    env = {"GRAPH_BENCH_DATA_DIR": str(push_data_dir), **cls.process_env(1)}
    done = _run(script, tmp_path, env)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
    assert not (tmp_path / "log").exists()


def test_arcadedb_load_refuses_a_jvm_without_thread_options(push_data_dir: Path, tmp_path: Path) -> None:
    """threads=1 with a JVM started without the options of `process_env` fails the load instead of running wider.

    Here neither the JVM thread caps (GRAPH_BENCH_JVM_THREADS) nor the engine options (ARCADEDB_JVM_ARGS) are set.
    """
    script = """
        from pathlib import Path
        from graph_bench.databases import get_database
        from graph_bench.datasets import cache
        db = get_database("arcadedb")(Path("."), 1)
        try:
            db.load(cache.prepare("push")[("graphalytics", "small")])
        except RuntimeError as exc:
            print(exc)
    """
    env = {"GRAPH_BENCH_DATA_DIR": str(push_data_dir), "ARCADEDB_JVM_ARGS": "", "GRAPH_BENCH_JVM_THREADS": ""}
    done = _run(script, tmp_path, env)
    assert done.returncode == 0, done.stderr[-3000:]
    assert "the JVM's thread options are" in done.stdout
