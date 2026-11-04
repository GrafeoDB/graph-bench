"""Traverse adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("traverse")

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

pytestmark = pytest.mark.traverse


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of traverse, shared by the push, startup and load tests."""
    return run_push("traverse", tmp_path_factory.mktemp("push-traverse"), push_data_dir, 60)


def test_traverse_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    known = known_disagreements("traverse")
    entries = [e for e in push.entries if e.database == "traverse"]
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
        if not (entry.status == "ok" and entry.answer == "agree"):
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_traverse_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "traverse")


def test_traverse_load_unchanged(push: Results) -> None:
    """The loads load the whole dataset: their node and edge counts agree."""
    assert_loads_agree(push, "traverse")


def test_traverse_resources_ok(push: Results) -> None:
    """Memory and Disk: one ok, agreeing record per load, with the size of the files `open()` keeps on disk."""
    assert_resources_ok(push, "traverse", on_disk=True)


def test_traverse_threads_settings(tmp_path: Path) -> None:
    cls = get_database("traverse")
    assert cls.process_env(1) == {"RAYON_NUM_THREADS": "1"}
    assert cls.process_env(None) == {}
    one = cls(tmp_path, 1).info()
    assert one.threads_note == "RAYON_NUM_THREADS=1"
    assert one.threads_configurable
    assert one.storage == "on-disk"
    assert one.process == "in-process"
    assert cls(tmp_path, None).info().threads_note == "rayon pool = all cores"


def test_traverse_keeps_its_files_in_the_workdir(tmp_path: Path) -> None:
    """The Disk record reads the instance's workdir, which holds the `.tvdb` file and its redo log."""
    db = get_database("traverse")(tmp_path, 1)
    assert db.store_paths() == [tmp_path]
