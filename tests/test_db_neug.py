"""NeuG adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest

pytest.importorskip("neug")

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.datasets import cache
from graph_bench.results import Results
from graph_bench.types import Model
from tests.conftest import (
    assert_loads_agree,
    assert_startup_agrees,
    not_yet_implemented,
    run_push,
    warn_if_fixed,
)

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.neug


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of neug, shared by the push, startup and load tests."""
    return run_push("neug", tmp_path_factory.mktemp("push-neug"), push_data_dir, 60)


def test_neug_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("neug")
    entries = [e for e in results.entries if e.database == "neug"]
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


def test_neug_startup_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected.

    NeuG 0.3.0 answers `MATCH (n) RETURN count(n)` with no row on a database without node tables (recorded in
    docs/known-disagreements.md as `neug/startup`), so the entry is ≠ at the first child. Once NeuG answers
    `[[0]]` the entry must agree like every other database's.
    """
    [entry] = [e for e in push.entries if e.database == "neug" and e.category == "startup"]
    if "startup" in known_disagreements("neug") and entry.answer == "disagree":
        assert entry.status == "ok"
        assert entry.detail == "child 1: probe returned [], expected [[0]]"
        assert entry.timing is None  # a ≠ startup has no timing
        assert entry.preflight_ns is not None and entry.preflight_ns > 0
        return
    assert_startup_agrees(push, "neug")


def test_neug_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "neug")


def test_neug_threads_settings(tmp_path: Path) -> None:
    cls = get_database("neug")
    assert cls.process_env(1) == {}
    one = cls(tmp_path, 1).info()
    assert one.threads_note == "max_thread_num=1; embedded queries are single-threaded"
    assert one.threads_configurable
    assert one.storage == "in-memory"
    assert one.process == "in-process"
    assert cls(tmp_path, None).info().threads_note.startswith("max_thread_num=0")


def test_neug_gds_load_failure_fails_the_graphalytics_load(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `LOAD gds` fails the graphalytics load (a failure, not not-available); SNB loads without GDS."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("neug")
    run = cls._run

    def failing(self: Any, text: str) -> None:
        if text == "LOAD gds":
            raise RuntimeError("extension gds is not installed")
        run(self, text)

    monkeypatch.setattr(cls, "_run", failing)
    data = cache.prepare("push")
    db = cls(tmp_path, 1)
    try:
        with pytest.raises(RuntimeError, match="not installed"):
            db.load(data[("graphalytics", "small")])
    finally:
        db.close()
    snb = cls(tmp_path, 1)
    try:
        snb.load(data[("snb", "small")])
        assert snb.counts()[0] > 0
    finally:
        snb.close()


def test_neug_projects_graph_in_load(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """INSTALL gds runs in prepare_environment; LOAD gds and the projection run in load(), so they are timed."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("neug")
    cls.prepare_environment()
    seen: list[str] = []
    run = cls._run

    def recording(self: Any, text: str) -> None:
        seen.append(text)
        run(self, text)

    monkeypatch.setattr(cls, "_run", recording)
    db = cls(tmp_path, 1)
    try:
        db.load(cache.prepare("push")[("graphalytics", "small")])
        assert not [text for text in seen if text.startswith("INSTALL")]
        assert seen[-2:] == ["LOAD gds", "CALL project_graph('g', ['V'], ['[V, E, V]'])"]
        assert db.algorithm("wcc", {})
        assert db.algorithm("bfs", {"source": 0})
        with pytest.raises(NotSupported):
            db.algorithm("cdlp", {})
    finally:
        db.close()
