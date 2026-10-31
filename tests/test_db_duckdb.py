"""DuckDB + DuckPGQ adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest

pytest.importorskip("duckdb")

import duckdb

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.datasets import cache
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import not_yet_implemented, resources_off, warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.duckdb


def test_duckdb_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    resources_off(monkeypatch)  # Task 4 turns them on with test_<db>_resources_ok
    results = run_bench("push", ["duckdb"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=120)
    known = known_disagreements("duckdb")
    entries = [e for e in results.entries if e.database == "duckdb"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate ("model <model> not supported"): the database lacks the data
        # model. Any other ⊘ (a missing capability) must be allowlisted in docs/known-disagreements.md.
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


def test_duckdb_known_disagreements_are_the_ones_duckpgq_causes(
    known_disagreements: Callable[[str], set[str]],
) -> None:
    assert known_disagreements("duckdb") == {"snb_sql_is4", "snb_sql_is6", "graphalytics_pr"}


def test_duckdb_threads_settings(tmp_path: Path) -> None:
    cls = get_database("duckdb")
    assert cls.process_env(1) == {}
    assert cls.models == {"sql_pgq"}
    one = cls(tmp_path, 1).info()
    assert one.threads_note == "SET threads=1"
    assert one.threads_configurable
    assert one.storage == "in-memory"
    assert one.process == "in-process"
    assert cls(tmp_path, None).info().threads_note == "default threads"


def test_duckdb_installs_in_prepare_and_activates_the_grammar_in_load(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INSTALL runs in prepare_environment only; LOAD, SET threads, the grammar and the graph run in load()."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("duckdb")
    seen: list[str] = []
    real = duckdb.connect

    class Recording:
        def __init__(self, con: Any) -> None:
            self.con = con

        def execute(self, text: str, *args: Any) -> Any:
            seen.append(text)
            return self.con.execute(text, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.con, name)

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: Recording(real(*a, **k)))
    cls.prepare_environment()
    assert seen == ["INSTALL duckpgq FROM community"]
    seen.clear()
    db = cls(tmp_path, 1)
    try:
        db.load(cache.prepare("push")[("snb", "small")])
        assert not [text for text in seen if text.startswith("INSTALL")]
        assert seen[:3] == ["SET threads=1", "LOAD duckpgq", "SET active_grammar_extensions=['duckpgq']"]
        assert seen[-1].startswith("CREATE PROPERTY GRAPH snb ")
        # No derived union tables: the tables are the dataset's node tables and one table per edge table.
        # (DuckPGQ keeps its own bookkeeping table, `__duckpgq_internal`.)
        tables = {row[0] for row in db.con.execute("SHOW TABLES").fetchall()} - {"__duckpgq_internal"}
        dataset = cache.prepare("push")[("snb", "small")]
        assert tables == {t.label for t in dataset.nodes} | {f"{e.type}__{e.src}__{e.dst}" for e in dataset.edges}
        assert db.counts() == (dataset.node_count, dataset.edge_count)
    finally:
        db.close()


def test_duckdb_extension_failure_fails_the_load(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `LOAD duckpgq` (missing extension) raises in load(): the cases show ✗ with DuckDB's error."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("duckdb")
    db = cls(tmp_path, 1)
    real = duckdb.connect

    class Failing:
        def __init__(self, con: Any) -> None:
            self.con = con

        def execute(self, text: str, *args: Any) -> Any:
            if text == "LOAD duckpgq":
                raise duckdb.IOException("Extension duckpgq is not installed")
            return self.con.execute(text, *args)

        def close(self) -> None:
            self.con.close()

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: Failing(real(*a, **k)))
    try:
        with pytest.raises(duckdb.IOException, match="not installed"):
            db.load(cache.prepare("push")[("snb", "small")])
    finally:
        db.close()


def test_duckdb_algorithms(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("duckdb")
    cls.prepare_environment()
    data = cache.prepare("push")
    db = cls(tmp_path, 1)
    try:
        db.load(data[("graphalytics", "small")])
        assert db.algorithm("wcc", {})
        assert len(db.algorithm("pr", {})) == data[("graphalytics", "small")].node_count
        levels = db.algorithm("bfs", {"source": 0})
        assert levels[0] == 0
        with pytest.raises(NotSupported):
            db.algorithm("cdlp", {})
    finally:
        db.close()
    snb = cls(tmp_path, 1)
    try:
        snb.load(data[("snb", "small")])
        with pytest.raises(RuntimeError, match="graphalytics"):
            snb.algorithm("wcc", {})
    finally:
        snb.close()
