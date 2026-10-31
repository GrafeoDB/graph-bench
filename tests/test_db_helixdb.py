"""HelixDB adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("helixdb")
pytest.importorskip("helixdb_uniffi")

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.databases.helix_queries import BUILDERS, snb_ic13
from graph_bench.runner.supervisor import run_bench
from graph_bench.suites import snb
from graph_bench.types import Model
from tests.conftest import not_yet_implemented, warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.helixdb


def test_helixdb_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["helixdb"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=120)
    known = known_disagreements("helixdb")
    entries = [e for e in results.entries if e.database == "helixdb"]
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


def test_helixdb_settings(tmp_path: Path) -> None:
    cls = get_database("helixdb")
    assert cls.process_env(1) == {}
    info = cls(tmp_path, 1).info()
    assert not info.threads_configurable
    assert info.storage == "in-memory"
    assert info.process == "in-process"


def test_every_snb_query_names_a_builder_or_a_reason() -> None:
    """The suite's `helix` text is the query id, and each id has a builder or a reason (never a KeyError)."""
    assert {q.text["helix"] for q in snb.QUERIES} == set(BUILDERS)
    for name, builder in BUILDERS.items():
        assert callable(builder) or (isinstance(builder, str) and builder.startswith(name.removeprefix("snb_").upper()))


def test_unsupported_queries_raise_not_supported(tmp_path: Path) -> None:
    db = get_database("helixdb")(tmp_path, 1)
    for name in ("snb_is3", "snb_is7", "snb_ic6"):
        with pytest.raises(NotSupported, match=r"ORDER BY|ranking"):
            db.query(name, {})
    for algo in ("bfs", "wcc", "pr"):
        with pytest.raises(NotSupported):
            db.algorithm(algo, {})


def test_ic13_rows_count_the_edges_of_the_returned_path() -> None:
    """IC13 reports hops: the nodes of HelixDB's path minus one, 0 for the single-node path, -1 for no path."""
    request, rows = snb_ic13({"person1Id": "a", "person2Id": "b"})
    assert request is not None
    assert rows({"p": []}) == [(-1,)]
    assert rows({"p": [7]}) == [(0,)]
    assert rows({"p": [7, 8, 9]}) == [(2,)]
