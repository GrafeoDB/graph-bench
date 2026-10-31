"""LadybugDB adapter: the push selection agrees with the references, except the recorded known disagreements."""

import contextlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest

pytest.importorskip("ladybug")

from graph_bench.databases import get_database
from graph_bench.datasets import cache
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.ladybug


def test_ladybug_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["ladybug"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    known = known_disagreements("ladybug")
    entries = [e for e in results.entries if e.database == "ladybug"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate ("model <model> not supported"): the database lacks the data
        # model. Any other ⊘ (a missing capability) must be allowlisted in docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model:
            continue
        # A load must agree too: its node and edge counts (`counts()`) match the dataset's.
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_ladybug_threads_settings(tmp_path: Path) -> None:
    cls = get_database("ladybug")
    assert cls.process_env(1) == {}
    assert cls.process_env(None) == {}
    one = cls(tmp_path, 1).info()
    assert one.threads_note == "max_num_threads=1"
    assert one.threads_configurable
    assert one.storage == "in-memory"
    assert one.process == "in-process"
    assert cls(tmp_path, None).info().threads_note == "max_num_threads=0"


def test_ladybug_algo_load_failure_fails_only_the_algorithm(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `LOAD EXTENSION ALGO` raises from the algorithm (✗, not ⊘); the load and BFS still work."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("ladybug")
    run = cls._run

    def failing(self: Any, text: str) -> None:
        if text == "LOAD EXTENSION ALGO":
            raise RuntimeError("extension ALGO is not installed")
        run(self, text)

    monkeypatch.setattr(cls, "_run", failing)
    db = cls(tmp_path, 1)
    try:
        db.load(cache.prepare("push")[("graphalytics", "small")])
        assert db.algorithm("bfs", {"source": 0})
        with pytest.raises(RuntimeError, match="not installed"):
            db.algorithm("pr", {})
    finally:
        db.close()


def test_ladybug_installs_algo_outside_load(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INSTALL ALGO runs in prepare_environment and LOAD EXTENSION ALGO in the first algorithm: neither in load()."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("ladybug")
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
        assert "CALL project_graph('g', ['V'], ['E'])" in seen
        assert not [text for text in seen if text.startswith(("INSTALL", "LOAD EXTENSION"))]
        # The extension itself may fail to load (see docs/known-disagreements.md); only the attempt is checked.
        with contextlib.suppress(RuntimeError):
            db.algorithm("wcc", {})
    finally:
        db.close()
    assert "LOAD EXTENSION ALGO" in seen
