"""Grafeo on a closed graph type: its DDL, settings, and the push selection against the references."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("grafeo")

from graph_bench.databases import get_database
from graph_bench.databases.grafeo_typed import graph_type_ddl, index_ddl
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.results import Results
from graph_bench.types import Model
from tests.conftest import (
    GRAFEO_SKIP_REASON,
    assert_loads_agree,
    assert_resources_ok,
    assert_startup_agrees,
    grafeo_is_measured,
    not_yet_implemented,
    run_push,
    warn_if_fixed,
)

MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.grafeo

measured = pytest.mark.skipif(not grafeo_is_measured(), reason=GRAFEO_SKIP_REASON)

_KEY = Column("id", "STRING")
_FIXTURE = Dataset(
    "snb",
    "small",
    Path("."),
    (
        NodeTable("Person", (_KEY, Column("age", "INT64"), Column("score", "DOUBLE")), "p.csv"),
        NodeTable("City", (Column("id", "INT64"),), "c.csv"),
    ),
    (
        EdgeTable("LIVES_IN", "Person", "City", (), "l.csv"),
        EdgeTable("LIVES_IN", "City", "City", (), "m.csv"),
        EdgeTable("KNOWS", "Person", "Person", (Column("since", "INT64"),), "k.csv"),
    ),
    2,
    3,
)


def test_graph_type_declares_every_label_and_edge_type_once() -> None:
    assert graph_type_ddl(_FIXTURE) == (
        "CREATE GRAPH TYPE snb_type ("
        "NODE TYPE Person (id STRING NOT NULL, age INT64, score FLOAT64), "
        "NODE TYPE City (id INT64 NOT NULL), "
        "EDGE TYPE LIVES_IN (), "
        "EDGE TYPE KNOWS (since INT64))"
    )


def test_graph_type_refuses_one_edge_type_with_two_property_lists() -> None:
    edges = (
        EdgeTable("E", "City", "City", (), "a.csv"),
        EdgeTable("E", "Person", "City", (Column("x", "INT64"),), "b.csv"),
    )
    with pytest.raises(ValueError, match="edge type E"):
        graph_type_ddl(Dataset("d", "small", Path("."), _FIXTURE.nodes, edges, 0, 0))


def test_one_index_per_declared_label() -> None:
    assert index_ddl(_FIXTURE) == [
        "CREATE INDEX Person_id FOR (n:Person) ON (n.id)",
        "CREATE INDEX City_id FOR (n:City) ON (n.id)",
    ]


def test_settings(tmp_path: Path) -> None:
    cls = get_database("grafeo_typed")
    assert cls.display_name == "Grafeo (typed)"
    assert cls.models == frozenset({"property_graph"})
    assert cls.process_env(1) == {"RAYON_NUM_THREADS": "1"}
    info = cls(tmp_path, 1).info()
    assert info.name == "grafeo_typed"
    assert info.storage == "in-memory"
    assert "closed graph type" in info.threads_note


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of the typed row, shared by the push, startup, load and resources tests."""
    return run_push("grafeo_typed", tmp_path_factory.mktemp("push-grafeo-typed"), push_data_dir, 60)


@measured
def test_grafeo_typed_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    known = known_disagreements("grafeo_typed")
    entries = [e for e in push.entries if e.database == "grafeo_typed"]
    assert entries
    failing = []
    for entry in entries:
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        if not (entry.status == "ok" and entry.answer == "agree"):
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
    assert not failing, "\n".join(failing)


@measured
def test_grafeo_typed_startup_agrees(push: Results) -> None:
    assert_startup_agrees(push, "grafeo_typed")


@measured
def test_grafeo_typed_load_unchanged(push: Results) -> None:
    assert_loads_agree(push, "grafeo_typed")


@measured
def test_grafeo_typed_resources_ok(push: Results) -> None:
    assert_resources_ok(push, "grafeo_typed", on_disk=False)
