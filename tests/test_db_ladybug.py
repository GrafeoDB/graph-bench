"""LadybugDB adapter: the push selection agrees with the references, except the recorded known disagreements."""

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("real_ladybug")

from graph_bench.databases import get_database
from graph_bench.runner.supervisor import run_bench

pytestmark = pytest.mark.ladybug


def test_ladybug_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["ladybug"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    known = known_disagreements("ladybug")
    entries = [e for e in results.entries if e.database == "ladybug"]
    assert entries
    failing = []
    for entry in entries:
        if entry.benchmark in known or entry.status == "not_supported":
            continue
        good = entry.status == "ok" and (entry.category == "load" or entry.answer == "agree")
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
