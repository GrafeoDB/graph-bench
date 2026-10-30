"""Grafeo adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("grafeo")

from graph_bench.databases import get_database
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.grafeo


def test_grafeo_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["grafeo"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    known = known_disagreements("grafeo")
    entries = [e for e in results.entries if e.database == "grafeo"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate ("model <model> not supported"): the database lacks the data
        # model. Any other ⊘ (a missing capability) must be allowlisted in docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        if entry.benchmark in known or lacks_model:
            continue
        # A load must agree too: its node and edge counts (`counts()`) match the dataset's.
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_grafeo_single_thread_env() -> None:
    cls = get_database("grafeo")
    assert cls.process_env(1) == {"RAYON_NUM_THREADS": "1"}
    assert cls.process_env(None) == {}
