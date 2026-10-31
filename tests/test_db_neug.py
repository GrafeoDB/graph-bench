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
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import not_yet_implemented, resources_off, warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.neug


def test_neug_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    resources_off(monkeypatch)  # Task 4 turns them on with test_<db>_resources_ok
    results = run_bench("push", ["neug"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
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
