"""FalkorDB adapter (falkordblite, Linux only): the push selection agrees except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("redislite")

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.datasets import cache
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import not_yet_implemented, resources_off, warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.falkordb


def test_falkordb_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    resources_off(monkeypatch)  # Task 4 turns them on with test_<db>_resources_ok
    results = run_bench("push", ["falkordb"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
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
