import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case

ROOT = Path(__file__).resolve().parent.parent
EXTRA = (
    "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2,fake_rdf=tests.fakes:FakeRdfDatabase,"
    "fake_sql=tests.fakes:FakeSqlDatabase"
)

Lines = list[dict[str, Any]]


def _id(item_id: str, db: str = "fake") -> str:
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db, "small", 1).bench_id


def run_worker(
    out: Path,
    data_dir: Path,
    *args: str,
    db: str = "fake",
    fake: str = "",
    env: Mapping[str, str] | None = None,
) -> tuple[Lines, subprocess.CompletedProcess[str]]:
    """Run the worker on the push selection; return its JSON lines and the finished process."""
    env = {
        **os.environ,
        "GRAPH_BENCH_EXTRA_DATABASES": EXTRA,
        "GRAPH_BENCH_DATA_DIR": str(data_dir),
        "GRAPH_BENCH_FAKE": fake,
        **(env or {}),
    }
    argv = [sys.executable, "-m", "graph_bench.runner.worker", "--db", db, "--selection", "push"]
    argv += ["--threads", "1", "--workdir", str(out.parent / "work"), "--out", str(out), *args]
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False)
    lines = []
    if out.exists():
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return lines, proc


@pytest.fixture(scope="module", autouse=True)
def _extra_databases() -> Iterator[None]:
    """Let the catalog in this process resolve the fake database names too."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
        yield


@pytest.fixture(scope="module")
def standard(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Lines, Path]:
    """One worker run on the fake database with a wrong, a too-slow, an unsupported and two skipped cases."""
    out = tmp_path_factory.mktemp("worker") / "fake.jsonl"
    fake = "wrong:snb_is5,slow:snb_is2,unsupported:graphalytics_pr"
    args = ["--budget", f"{_id('snb_is2')}=0.05", "--skip", _id("snb_is1"), _id("snb_is3")]
    lines, proc = run_worker(out, push_data_dir, *args, fake=fake)
    assert proc.returncode == 0, proc.stderr
    return lines, out


def _entries(lines: Lines) -> dict[str, dict[str, Any]]:
    return {line["id"]: line for line in lines if line["kind"] == "entry"}


def test_worker_emits_database_line_first(standard: tuple[Lines, Path]) -> None:
    lines, _ = standard
    first = lines[0]
    assert first["kind"] == "database"
    assert first["name"] == "fake"
    assert first["storage"] == "in-memory"
    assert first["process"] == "in-process"
    assert all(line["kind"] != "database" for line in lines[1:])


def test_worker_emits_start_before_entry(standard: tuple[Lines, Path]) -> None:
    lines, _ = standard
    started: set[str] = set()
    for line in lines[1:]:
        if line["kind"] == "start":
            started.add(line["id"])
        else:
            assert line["id"] in started, line
    entries = _entries(lines)
    expected = {c.bench_id for c in catalog.cases("push", ["fake"])} - {_id("snb_is1"), _id("snb_is3")}
    assert set(entries) == expected | set(catalog.load_ids("push", ["fake"]))


def test_worker_times_agreeing_cases(standard: tuple[Lines, Path]) -> None:
    entries = _entries(standard[0])
    query = entries[_id("snb_is4")]
    assert query["status"] == "ok"
    assert query["answer"] == "agree"
    assert query["reference"] == "python"
    assert query["error"] is None
    assert query["timing"]["iterations"] == catalog.selection("push").iterations
    assert query["preflight_ns"] > 0
    for algo in ("graphalytics_bfs", "graphalytics_wcc"):
        assert entries[_id(algo)]["answer"] == "agree"
        assert entries[_id(algo)]["reference"] == "networkx"
        assert entries[_id(algo)]["timing"] is not None
    for load_id in catalog.load_ids("push", ["fake"]):
        assert entries[load_id]["status"] == "ok"
        assert entries[load_id]["load_ns"] > 0
        assert entries[load_id]["answer"] == "agree"  # the node and edge counts match the dataset's


def test_worker_marks_not_supported(standard: tuple[Lines, Path]) -> None:
    entry = _entries(standard[0])[_id("graphalytics_pr")]
    assert entry["status"] == "not_supported"
    assert entry["timing"] is None
    assert entry["answer"] == "unchecked"
    assert entry["reference"] == "networkx"


def test_worker_marks_unsupported_model(push_data_dir: Path, tmp_path: Path) -> None:
    """An RDF-only database: its RDF cases run on its own RDF load, every other case is ⊘ without a start line."""
    lines, proc = run_worker(tmp_path / "rdf.jsonl", push_data_dir, db="fake_rdf")
    assert proc.returncode == 0, proc.stderr
    assert lines[0]["kind"] == "database"
    entries = _entries(lines)
    rdf_load = catalog.load_id("snb", "fake_rdf", "small", 1, "rdf")
    assert set(entries) == {c.bench_id for c in catalog.cases("push", ["fake_rdf"])} | set(
        catalog.load_ids("push", ["fake_rdf"])
    )
    models = {c.bench_id: c.item.model for c in catalog.cases("push", ["fake_rdf"])}
    # The graph loads its models do not reach are ⊘ like the cases, with the model gate's error.
    for dataset in ("snb", "graphalytics"):
        models[catalog.load_id(dataset, "fake_rdf", "small", 1)] = "property_graph"
    started = {line["id"] for line in lines if line["kind"] == "start"}
    assert started == {i for i, model in models.items() if model == "rdf"} | {rdf_load}
    for bench_id, entry in entries.items():
        if bench_id in started:
            assert (entry["status"], entry["answer"]) == ("ok", "agree"), entry
        else:
            assert (entry["status"], entry["error"]) == ("not_supported", f"model {models[bench_id]} not supported")


def test_worker_routes_each_model_to_its_store(push_data_dir: Path, tmp_path: Path) -> None:
    """The multi-model fake answers only when a case runs in its model's dialect on the store its model reads;
    each (dataset, store) is loaded once, the RDF store separately."""
    trace = tmp_path / "trace.txt"
    env_trace = {"GRAPH_BENCH_FAKE_TRACE": str(trace)}
    lines, proc = run_worker(tmp_path / "multi.jsonl", push_data_dir, env=env_trace)
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    by_model = {c.item.model for c in catalog.cases("push", ["fake"])}
    assert by_model == {"property_graph", "sql_pgq", "rdf"}
    for case in catalog.cases("push", ["fake"]):
        assert (entries[case.bench_id]["status"], entries[case.bench_id]["answer"]) == ("ok", "agree"), case.bench_id
    rdf_load = entries[catalog.load_id("snb", "fake", "small", 1, "rdf")]
    assert (rdf_load["status"], rdf_load["answer"]) == ("ok", "agree")
    loads = trace.read_text(encoding="utf-8").splitlines().count("load")
    assert loads == catalog.selection("push").load_iterations * 3  # snb graph, snb RDF, graphalytics graph


def test_worker_skips_ids(standard: tuple[Lines, Path]) -> None:
    ids = {line["id"] for line in standard[0] if "id" in line}
    assert _id("snb_is1") not in ids
    assert _id("snb_is3") not in ids
    assert _id("snb_is4") in ids


def test_worker_writes_answers(standard: tuple[Lines, Path]) -> None:
    lines, out = standard
    entry = _entries(lines)[_id("snb_is4")]
    files = list((out.parent / out.stem / "answers").glob("*snb_is4*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(doc["answers"]) == len(doc["bindings"]) > 0
    assert entry["answer_digest"]


def test_worker_wrong_answer_not_timed(standard: tuple[Lines, Path]) -> None:
    entry = _entries(standard[0])[_id("snb_is5")]
    assert entry["status"] == "ok"
    assert entry["answer"] == "disagree"
    assert entry["timing"] is None
    assert entry["detail"]


def test_worker_budget_marks_too_slow(standard: tuple[Lines, Path]) -> None:
    entry = _entries(standard[0])[_id("snb_is2")]
    assert entry["status"] == "timeout"
    assert entry["error"].startswith("too slow")
    assert entry["timing"] is None


def test_worker_emits_progress(standard: tuple[Lines, Path]) -> None:
    lines = standard[0]
    target = _id("snb_is4")
    start = next(i for i, line in enumerate(lines) if line["kind"] == "start" and line["id"] == target)
    end = next(i for i, line in enumerate(lines) if line["kind"] == "entry" and line["id"] == target)
    between = lines[start + 1 : end]
    sel = catalog.selection("push")
    # `run` right before the preflight's database call, `progress` right after it, then one per further run.
    assert [line["kind"] for line in between] == ["run"] + ["progress"] * (1 + sel.warmup - 1 + sel.iterations)
    assert all(line["id"] == target for line in between)
    load_id = catalog.load_id("snb", "fake", "small", 1)
    load_progress = [line for line in lines if line["kind"] == "progress" and line["id"] == load_id]
    assert len(load_progress) == sel.load_iterations


def test_worker_brackets_every_preflight_call(standard: tuple[Lines, Path]) -> None:
    """Every case that reaches its preflight gets `run` then `progress`, also when the call raises."""
    lines = standard[0]
    for target in (_id("snb_is4"), _id("snb_is2"), _id("graphalytics_pr")):  # agree, too slow, not supported
        start = next(i for i, line in enumerate(lines) if line["kind"] == "start" and line["id"] == target)
        assert [line["kind"] for line in lines[start + 1 : start + 3]] == ["run", "progress"], target


def _expected_counts(data_dir: Path, dataset: str) -> tuple[int, int]:
    ds = Dataset.load(data_dir / cache.dataset_key("push") / f"{dataset}-small")
    return ds.node_count, ds.edge_count


def test_worker_checks_load_counts(push_data_dir: Path, tmp_path: Path) -> None:
    """A load that lost an edge is ≠ with both counts; a database that cannot count leaves the load unchecked."""
    fake = "short:count_snb,unsupported:count_graphalytics"
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake=fake)
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    snb = entries[catalog.load_id("snb", "fake", "small", 1)]
    nodes, edges = _expected_counts(push_data_dir, "snb")
    assert (snb["status"], snb["answer"]) == ("ok", "disagree")
    assert snb["detail"] == f"loaded {nodes} nodes, {edges - 1} edges; expected {nodes} nodes, {edges} edges"
    assert snb["load_ns"] > 0
    graphalytics = entries[catalog.load_id("graphalytics", "fake", "small", 1)]
    assert (graphalytics["status"], graphalytics["answer"], graphalytics["detail"]) == ("ok", "unchecked", None)
    assert entries[_id("snb_is4")]["answer"] == "agree"  # the cases still run on the loaded instance


def test_worker_count_error_fails_only_the_load(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="crash_soft:count_snb")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    snb = entries[catalog.load_id("snb", "fake", "small", 1)]
    assert (snb["status"], snb["error"]) == ("failed", "counting the loaded data: RuntimeError: soft crash")
    assert (entries[_id("snb_is4")]["status"], entries[_id("snb_is4")]["answer"]) == ("ok", "agree")


def test_counts_default_is_not_supported(tmp_path: Path) -> None:
    class Plain(Database):
        name = display_name = "plain"
        models = frozenset()

        def info(self) -> DatabaseInfo:
            raise AssertionError

        def load(self, dataset: Dataset) -> None:
            raise AssertionError

        def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
            raise AssertionError

    with pytest.raises(NotSupported):
        Plain(tmp_path, 1).counts()


def test_worker_prepares_environment_once_before_loads(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.txt"
    monkeypatch.setenv("GRAPH_BENCH_FAKE_TRACE", str(trace))
    _, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir)
    assert proc.returncode == 0, proc.stderr
    calls = trace.read_text(encoding="utf-8").split()
    loads = catalog.selection("push").load_iterations * len(catalog.load_ids("push", ["fake"]))
    assert calls == ["environment"] + ["load"] * loads


def test_worker_environment_failure_fails_the_loads(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="crash_soft:environment")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    error = "preparing the environment: RuntimeError: soft crash"
    for load_id in catalog.load_ids("push", ["fake"]):
        assert (entries[load_id]["status"], entries[load_id]["error"]) == ("failed", error)
    cases = [c.bench_id for c in catalog.cases("push", ["fake"])]
    assert all((entries[i]["status"], entries[i]["error"]) == ("failed", f"load failed: {error}") for i in cases)


def test_worker_runs_algorithms_on_an_sql_pgq_database(push_data_dir: Path, tmp_path: Path) -> None:
    """An SQL/PGQ-only database (DuckDB's shape) loads the graph store of both datasets and runs the algorithms;
    property graph and RDF cases and the RDF load are ⊘."""
    lines, proc = run_worker(tmp_path / "sql.jsonl", push_data_dir, db="fake_sql")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    assert set(entries) == {c.bench_id for c in catalog.cases("push", ["fake_sql"])} | set(
        catalog.load_ids("push", ["fake_sql"])
    )
    for case in catalog.cases("push", ["fake_sql"]):
        entry = entries[case.bench_id]
        if case.item.category in ("sql_pgq", "algorithms"):
            assert (entry["status"], entry["answer"]) == ("ok", "agree"), entry
        else:
            assert (entry["status"], entry["error"]) == ("not_supported", f"model {case.item.model} not supported")
    for dataset in ("snb", "graphalytics"):
        load = entries[catalog.load_id(dataset, "fake_sql", "small", 1)]
        assert (load["status"], load["answer"]) == ("ok", "agree")
    rdf_load = entries[catalog.load_id("snb", "fake_sql", "small", 1, "rdf")]
    assert (rdf_load["status"], rdf_load["error"]) == ("not_supported", "model rdf not supported")
