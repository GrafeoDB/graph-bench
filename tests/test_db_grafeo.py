"""Grafeo adapter: the push selection agrees with the references, except the recorded known disagreements."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest

pytest.importorskip("grafeo")

from graph_bench.databases import get_database
from graph_bench.datasets import cache, rdf
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

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.grafeo

# The database-agreement tests (the push run: push, startup, load, resources) run only on a grafeo graph-bench
# measures (`grafeo_is_measured`); the unit tests below them run on any.
measured = pytest.mark.skipif(not grafeo_is_measured(), reason=GRAFEO_SKIP_REASON)


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of grafeo, shared by the push, startup and load tests."""
    return run_push("grafeo", tmp_path_factory.mktemp("push-grafeo"), push_data_dir, 60)


@measured
def test_grafeo_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("grafeo")
    entries = [e for e in results.entries if e.database == "grafeo"]
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


@measured
def test_grafeo_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "grafeo")


@measured
def test_grafeo_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "grafeo")


@measured
def test_grafeo_resources_ok(push: Results) -> None:
    """Memory and Disk: one ok, agreeing record per load with ordered readings; in memory."""
    assert_resources_ok(push, "grafeo", on_disk=False)


class _Dist:
    """A stand-in for an installed distribution: its version and, for a direct-url install, its `direct_url.json`
    (an archive by default, or a VCS checkout with `vcs`)."""

    def __init__(self, version: str, url: str | None = None, *, vcs: bool = False) -> None:
        self.version = version
        self.url = url
        self.vcs = vcs

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        if self.url is None:
            return None
        info = {"vcs_info": {"vcs": "git", "commit_id": "e8b04ed2"}} if self.vcs else {"archive_info": {}}
        return json.dumps({"url": self.url, **info})


def test_grafeo_is_measured_by_its_distribution() -> None:
    """A file or VCS install is a 0.6.1-branch build (it reports 0.6.0): measured. A registry install or a plain
    archive url follows the version: 0.6.0 is not measured, 0.6.1 is."""
    branch = _Dist("0.6.0", "file:///D:/gb-wheels-e8b04ed/grafeo-0.6.0-cp312-abi3-win_amd64.whl")
    assert grafeo_is_measured(branch)  # type: ignore[arg-type]
    assert not grafeo_is_measured(_Dist("0.6.0"))  # type: ignore[arg-type]
    assert grafeo_is_measured(_Dist("0.6.1"))  # type: ignore[arg-type]
    assert not grafeo_is_measured(_Dist("0.6.0", "https://example.org/grafeo-0.6.0.whl"))  # type: ignore[arg-type]
    assert grafeo_is_measured(_Dist("0.6.1", "https://example.org/grafeo-0.6.1.whl"))  # type: ignore[arg-type]
    vcs = _Dist("0.6.0", "https://github.com/GrafeoDB/grafeo", vcs=True)
    assert grafeo_is_measured(vcs)  # type: ignore[arg-type]


def test_grafeo_single_thread_env() -> None:
    cls = get_database("grafeo")
    assert cls.process_env(1) == {"RAYON_NUM_THREADS": "1"}
    assert cls.process_env(None) == {}


def test_grafeo_graph_store_is_the_named_graph_sql_pgq_reads(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph load names its graph after the dataset, so `GRAPH_TABLE (snb ...)` reads the loaded data.

    The query is a minimal one (a single node, a graph pattern WHERE as in the benchmark texts) so that it checks
    the routing and the graph name alone, whatever SQL/PGQ forms Grafeo lacks (see docs/known-disagreements.md).
    """
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    db = get_database("grafeo")(tmp_path, 1)
    try:
        db.load_store("graph", cache.prepare("push")[("snb", "small")])
        text = "SELECT * FROM GRAPH_TABLE (snb MATCH (n:Person) WHERE n.id = $id COLUMNS (n.firstName AS f))"
        assert db.query_model("sql_pgq", text, {"id": "person_0"}) == [("Emily",)]
        assert db.query_model(
            "property_graph", "MATCH (n:Person {id: $id}) RETURN n.firstName", {"id": "person_0"}
        ) == [("Emily",)]
    finally:
        db.close()


def test_grafeo_rdf_store_is_separate(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The RDF load fills the RDF store of a fresh instance; SPARQL reads it, typed by the mapping."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    ds = cache.prepare("push")[("snb", "small")]
    db = get_database("grafeo")(tmp_path, 1)
    try:
        db.load_store("rdf", ds)
        assert db.counts() == (ds.node_count, ds.edge_count)
        text = rdf.PREFIXES + "SELECT ?b WHERE { VALUES ?id { $id } ?n v:id ?id ; v:birthday ?b }"
        [(birthday,)] = db.query_model("rdf", text, {"id": "person_0"})
        assert isinstance(birthday, int)
        assert db.query("MATCH (n) RETURN count(n)", {}) == [(0,)]  # the property graph stays empty
    finally:
        db.close()


@pytest.mark.parametrize("has_load_rdf", [True, False], ids=["load_rdf", "insert-data-fallback"])
def test_grafeo_rdf_load_reads_the_prepared_file(
    has_load_rdf: bool, push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RDF load reads the N-Triples file written at prepare time: `load_rdf(path, format="ntriples")` when the
    installed grafeo has it (0.6.1), else the transitional `INSERT DATA` chunks of the same file's lines."""
    from graph_bench.databases import grafeo as adapter

    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    ds = cache.prepare("push")[("snb", "small")]
    real_db, real_insert = adapter.grafeo.GrafeoDB, adapter._insert_data
    calls: list[tuple[str, ...]] = []

    def insert(db: Any, path: Path) -> None:
        calls.append(("insert", str(path)))
        real_insert(db, path)

    class Db:
        """A real GrafeoDB, with or without `load_rdf` whatever the installed version has."""

        def __init__(self) -> None:
            self._db = real_db()

        def __getattr__(self, name: str) -> Any:
            if name == "load_rdf":
                if not has_load_rdf:
                    raise AttributeError(name)
                return self._load_rdf
            return getattr(self._db, name)

        def _load_rdf(self, path: str, format: str | None = None) -> None:
            calls.append(("load_rdf", path, str(format)))
            real_insert(self._db, Path(path))

    monkeypatch.setattr(adapter.grafeo, "GrafeoDB", Db)
    monkeypatch.setattr(adapter, "_insert_data", insert)
    db = adapter.GrafeoDatabase(tmp_path, 1)
    try:
        db.load_store("rdf", ds)
        path = str(rdf.ntriples_path(ds))
        expected = ("load_rdf", path, "ntriples") if has_load_rdf else ("insert", path)
        assert calls == [expected]
        assert db.counts() == (ds.node_count, ds.edge_count)
    finally:
        db.close()


@pytest.mark.parametrize("public", [True, False], ids=["query_timeout", "no-setting"])
def test_grafeo_runs_without_a_query_timeout(public: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the harness's rules stop a case: a grafeo whose constructor takes `query_timeout` (0.6.1 on) gets None,
    no limit; one without the setting (0.6.0) is opened as before."""
    from graph_bench.databases import grafeo as adapter

    calls: list[dict[str, Any]] = []

    def with_timeout(*, query_timeout: Any = ...) -> str:
        calls.append({"query_timeout": query_timeout})
        return "db"

    def without(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "db"

    monkeypatch.setattr(adapter.grafeo, "GrafeoDB", with_timeout if public else without)
    db = get_database("grafeo")(tmp_path, 1)
    db.open_empty()
    assert calls == ([{"query_timeout": None}] if public else [{}])
