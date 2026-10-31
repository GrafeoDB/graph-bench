"""FalkorDB adapter (falkordblite, Linux only): the push selection agrees except the recorded known disagreements."""

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("redislite")

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.datasets import cache
from graph_bench.results import Results
from graph_bench.runner.startup import run_startups
from graph_bench.types import Model
from tests.conftest import (
    assert_loads_agree,
    assert_startup_agrees,
    not_yet_implemented,
    run_push,
    warn_if_fixed,
)
from tests.startup_hangs import MARKER_ENV, alive, hang_until_killed, processes_named

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.falkordb


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of falkordb, shared by the push, startup and load tests."""
    return run_push("falkordb", tmp_path_factory.mktemp("push-falkordb"), push_data_dir, 60)


def test_falkordb_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("falkordb")
    entries = [e for e in results.entries if e.database == "falkordb"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate; any other ⊘ (a missing capability) must be allowlisted in
        # docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        # A load must agree too: its node and edge counts (`counts()`) match the dataset's.
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_falkordb_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "falkordb")


def test_falkordb_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "falkordb")


def test_falkordb_info_and_algorithms(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("falkordb")
    info = cls(tmp_path, 1).info()
    assert info.process == "local process"
    assert info.storage == "in-memory"
    assert not info.threads_configurable
    assert "falkordblite" in info.threads_note
    db = cls(tmp_path, 1)
    try:
        db.load(cache.prepare("push")[("graphalytics", "small")])
        assert db.counts()[0] > 0
        levels = db.algorithm("bfs", {"source": 0})
        assert levels[0] == 0
        assert all(depth >= 0 for depth in levels.values())
        with pytest.raises(NotSupported):
            db.algorithm("cdlp", {})
    finally:
        db.close()


@pytest.mark.skipif(sys.platform != "linux", reason="falkordblite is Linux only; the subreaper is Linux only")
def test_falkordb_startup_leaves_no_redis_server(tmp_path: Path) -> None:
    """Startup children that answer: the redis-server each one started (it daemonizes, so it leaves the child's
    tree) is killed with the child (`kill_tree`, through the subreaper); none outlives the run."""
    before = processes_named("redis-server")
    outcome = run_startups("falkordb", 1, 2, tmp_path, 60)
    assert (outcome.status, outcome.answer) == ("ok", "agree"), (outcome.error, outcome.detail)
    assert len(outcome.times_ns) == 2
    assert not alive(processes_named("redis-server") - before)


@pytest.mark.skipif(sys.platform != "linux", reason="falkordblite is Linux only; the subreaper is Linux only")
def test_falkordb_startup_timeout_kills_the_redis_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A child that hangs with its redis-server up is killed at the timeout, the redis-server with it."""
    marker = tmp_path / "marker"
    monkeypatch.setenv("GRAPH_BENCH_EXTRA_DATABASES", "falkordb_hang=tests.startup_hangs:HangingFalkor")
    monkeypatch.setenv(MARKER_ENV, str(marker))
    child, pids, names = hang_until_killed("falkordb_hang", tmp_path / "child", marker, 20)
    assert child.status == "timeout", child.error
    assert "redis-server" in names, names
    assert not alive(pids), names
