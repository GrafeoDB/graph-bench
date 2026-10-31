"""Oxigraph adapter: the push selection agrees with the references on the RDF category, the rest is not available."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("pyoxigraph")

from graph_bench.databases import get_database
from graph_bench.datasets import cache, rdf
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

pytestmark = pytest.mark.oxigraph


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of oxigraph, shared by the push, startup and load tests."""
    return run_push("oxigraph", tmp_path_factory.mktemp("push-oxigraph"), push_data_dir, 60)


def test_oxigraph_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("oxigraph")
    entries = [e for e in results.entries if e.database == "oxigraph"]
    assert entries
    assert any(e.status == "ok" and e.answer == "agree" and "rdf" in e.benchmark for e in entries)
    failing = []
    for entry in entries:
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    assert not failing, "\n".join(failing)


def test_oxigraph_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "oxigraph")


def test_oxigraph_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "oxigraph")


def test_oxigraph_load_reads_the_prepared_file(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load is Oxigraph's file loader on the prepared N-Triples file; counts and typed terms follow the mapping."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    ds = cache.prepare("push")[("snb", "small")]
    cls = get_database("oxigraph")
    assert cls.models == {"rdf"} and cls.dialect_for("rdf") == "sparql"
    db = cls(tmp_path, 1)
    try:
        assert db.info().threads_configurable is False
        db.load_store("rdf", ds)
        assert db.counts() == (ds.node_count, ds.edge_count)
        text = rdf.PREFIXES + "SELECT ?n ?b WHERE { VALUES ?id { $id } ?n v:id ?id ; v:birthday ?b }"
        [(node, birthday)] = db.query_model("rdf", text, {"id": "person_0"})
        assert node == rdf.node_iri("Person", "person_0")
        assert isinstance(birthday, int)
    finally:
        db.close()
