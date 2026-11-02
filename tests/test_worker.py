import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from graph_bench.databases import get_database
from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.runner import worker
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case

ROOT = Path(__file__).resolve().parent.parent
EXTRA = (
    "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2,fake_rdf=tests.fakes:FakeRdfDatabase,"
    "fake_sql=tests.fakes:FakeSqlDatabase,fake_nostart=tests.fakes:FakeNoStartup"
)
WRITES = {"GRAPH_BENCH_EXTRA_SUITES": "tests.fake_writes"}

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
        elif line == {"kind": "progress", "id": None} or line["kind"] == "load":
            continue  # a load's own line, and the progress of an unreported load (the multigraph fixture's)
        else:
            assert line["id"] in started, line
    entries = _entries(lines)
    assert set(entries) == set(catalog.all_ids("push", ["fake"])) - {_id("snb_is1"), _id("snb_is3")}


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
        # The fake's loads are fast, so all of the selection's load iterations ran.
        assert entries[load_id]["load_iterations"] == catalog.selection("push").load_iterations
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
    startup = catalog.startup_id("fake_rdf", 1)
    assert set(entries) == set(catalog.all_ids("push", ["fake_rdf"]))
    models = {c.bench_id: c.item.model for c in catalog.cases("push", ["fake_rdf"])}
    # The graph loads its models do not reach are ⊘ like the cases, with the model gate's error; so are both
    # query-language loads (graph store only).
    for dataset in ("snb", "graphalytics"):
        models[catalog.load_id(dataset, "fake_rdf", "small", 1)] = "property_graph"
        models[catalog.load_ql_id(dataset, "fake_rdf", "small", 1)] = "property_graph"
    started = {line["id"] for line in lines if line["kind"] == "start"}
    assert started == {i for i, model in models.items() if model == "rdf"} | {rdf_load, startup}
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
    loads = _calls(trace).count("load")
    # snb graph, snb RDF and graphalytics graph, each reported; and the multigraph fixture's unreported load, once.
    assert loads == catalog.selection("push").load_iterations * 3 + 1


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


def _calls(trace: Path) -> list[str]:
    """The calls of a fake trace file, without their process ids."""
    return [line.split()[0] for line in trace.read_text(encoding="utf-8").splitlines()]


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
    sel = catalog.selection("push")
    # The reported loads, and the multigraph fixture's unreported one (answer-only, no Load entry), loaded once.
    loads = sel.load_iterations * len(catalog.load_ids("push", ["fake"])) + 1
    loads_ql = sel.load_iterations * len(catalog.load_ql_ids("push", ["fake"]))
    calls = [call for call in _calls(trace) if call != "open_empty"]
    # The startup's provision, then the environment (which provisions first), then the loads; the query-language
    # loads follow their dataset's cases (graphalytics, then snb, before the snb RDF store).
    assert calls[:3] == ["provision", "provision", "environment"]
    assert sorted(calls[3:]) == sorted(["load"] * loads + ["load_ql"] * loads_ql)


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
    assert set(entries) == set(catalog.all_ids("push", ["fake_sql"]))
    for case in catalog.cases("push", ["fake_sql"]):
        entry = entries[case.bench_id]
        # The SQL/PGQ cases (the SQL/PGQ category and the Scans twins) and the algorithms run.
        if case.item.model == "sql_pgq" or case.item.category == "algorithms":
            assert (entry["status"], entry["answer"]) == ("ok", "agree"), entry
        else:
            assert (entry["status"], entry["error"]) == ("not_supported", f"model {case.item.model} not supported")
    for dataset in ("snb", "graphalytics"):
        load = entries[catalog.load_id(dataset, "fake_sql", "small", 1)]
        assert (load["status"], load["answer"]) == ("ok", "agree")
    rdf_load = entries[catalog.load_id("snb", "fake_sql", "small", 1, "rdf")]
    assert (rdf_load["status"], rdf_load["error"]) == ("not_supported", "model rdf not supported")


def test_database_defaults_for_the_new_rows(tmp_path: Path) -> None:
    """provision is a no-op the default prepare_environment calls; open_empty and load_ql are ⊘ by default;
    write runs through query_model; an in-memory database has no store paths."""
    calls: list[str] = []

    class Plain(Database):
        name = display_name = "plain"
        models = frozenset({"property_graph"})

        @classmethod
        def provision(cls) -> None:
            calls.append("provision")

        def info(self) -> DatabaseInfo:
            raise AssertionError

        def load(self, dataset: Dataset) -> None:
            raise AssertionError

        def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
            calls.append(f"query {text} {dict(params)}")
            return [(1,)]

    Plain.prepare_environment()
    assert calls == ["provision"]
    assert Database.provision() is None
    db = Plain(tmp_path, 1)
    with pytest.raises(NotSupported, match=r"^startup$"):
        db.open_empty()
    with pytest.raises(NotSupported, match=r"^no query-language bulk loader$"):
        db.load_ql(None)  # type: ignore[arg-type]
    assert db.write("property_graph", "W", {"a": 1}) == [(1,)]
    assert calls[-1] == "query W {'a': 1}"
    assert db.store_paths() == []


# Startup ------------------------------------------------------------------------------------------------------


def test_startup_never_provisions_in_the_clock(push_data_dir: Path, tmp_path: Path) -> None:
    """The worker provisions, untimed, before the startup children; no child provisions, and every child has
    ended before the first load."""
    trace = tmp_path / "trace.txt"
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, env={"GRAPH_BENCH_FAKE_TRACE": str(trace)})
    assert proc.returncode == 0, proc.stderr
    calls = [line.split() for line in trace.read_text(encoding="utf-8").splitlines()]
    worker = {pid for call, pid, *_ in calls if call in ("provision", "environment", "load")}
    children = {pid for call, pid, *_ in calls if call == "open_empty"}
    assert len(worker) == 1
    assert len(children) == catalog.selection("push").startup_iterations
    assert worker.isdisjoint(children)
    names = [call for call, *_ in calls]
    assert names[0] == "provision"
    assert max(i for i, name in enumerate(names) if name == "open_empty") < names.index("load")
    startup = catalog.startup_id("fake", 1)
    entry = _entries(lines)[startup]
    assert (entry["status"], entry["answer"], entry["reference"]) == ("ok", "agree", "python")
    assert entry["timing"]["iterations"] == catalog.selection("push").startup_iterations
    assert entry["preflight_ns"] > 0
    first_start = next(line["id"] for line in lines if line["kind"] == "start")
    assert first_start == startup
    progress = [line for line in lines if line["kind"] == "progress" and line["id"] == startup]
    assert len(progress) == catalog.selection("push").startup_iterations


def test_startup_provisioning_error_fails_the_entry(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="crash_soft:provision")
    assert proc.returncode == 0, proc.stderr
    entry = _entries(lines)[catalog.startup_id("fake", 1)]
    assert (entry["status"], entry["error"]) == ("failed", "provisioning: RuntimeError: soft crash")


def test_worker_startup_without_open_empty_is_not_available(push_data_dir: Path, tmp_path: Path) -> None:
    skip = [i for i in catalog.all_ids("push", ["fake_nostart"]) if "test_startup" not in i]
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, "--skip", *skip, db="fake_nostart")
    assert proc.returncode == 0, proc.stderr
    entry = _entries(lines)[catalog.startup_id("fake_nostart", 1)]
    assert (entry["status"], entry["error"], entry["timing"]) == ("not_supported", "startup", None)


# Write cases ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def writes_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory for the push selection with the fake write suite (its own cache key)."""
    return tmp_path_factory.mktemp("writes-data")


def _write_id(db: str = "fake", item: str = "fake_write") -> str:
    return f"benches/test_batch_writes.py::test_{item}[{db}-small]"


def _run_writes(
    writes_dir: Path, tmp_path: Path, fake: str = "", trace: Path | None = None, item: str = "fake_write"
) -> Lines:
    """The worker on the snb graph store only, with one write case of the fake write suite; the other loads and
    cases skipped."""
    env = {**WRITES, **({"GRAPH_BENCH_FAKE_TRACE": str(trace)} if trace else {})}
    ids = _ids_with_writes()
    keep = {i for i in ids if i.endswith("[fake-small]") and ("snb_is" in i or "test_load_snb[" in i)}
    keep.add(_write_id(item=item))
    skip = sorted(set(ids) - keep)
    lines, proc = run_worker(tmp_path / "w.jsonl", writes_dir, "--skip", *skip, fake=fake, env=env)
    assert proc.returncode == 0, proc.stderr
    return lines


def _ids_with_writes() -> list[str]:
    """`graph-bench list` for fake with the fake write suite, from a process that has it."""
    code = "from graph_bench.suites import catalog; print('\\n'.join(catalog.all_ids('push', ['fake'])))"
    env = {**os.environ, "GRAPH_BENCH_EXTRA_DATABASES": EXTRA, **WRITES}
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return proc.stdout.split()


def test_write_case_phase_order(writes_dir: Path, tmp_path: Path) -> None:
    """Before the preflight the check reads the loaded state; then every run is setup, run, check, teardown,
    check (the trace's consecutive repeats are the per-binding calls)."""
    trace = tmp_path / "trace.txt"
    lines = _run_writes(writes_dir, tmp_path, trace=trace)
    entry = _entries(lines)[_write_id()]
    assert (entry["status"], entry["answer"], entry["error"]) == ("ok", "agree", None), entry
    sel = catalog.selection("push")
    runs = 1 + (sel.warmup - 1) + sel.iterations
    assert entry["timing"]["iterations"] == sel.iterations
    phases = [c for c in _calls(trace) if c in ("setup", "run", "check", "teardown")]
    collapsed = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
    assert collapsed == ["check"] + ["setup", "run", "check", "teardown", "check"] * runs
    assert phases.count("run") == 3 * runs  # three bindings
    assert phases.count("teardown") == 2 * 3 * runs  # two statements per binding


def test_write_cases_run_after_reads(writes_dir: Path, tmp_path: Path) -> None:
    lines = _run_writes(writes_dir, tmp_path)
    starts = [line["id"] for line in lines if line["kind"] == "start"]
    reads = [i for i, bench_id in enumerate(starts) if "snb_is" in bench_id]
    assert reads
    assert starts.index(_write_id()) > max(reads)  # catalog order puts it first in bench_id order (batch_writes)


@pytest.mark.parametrize(
    ("fake", "detail"),
    [
        ("wrong:fake_write", "binding 0: 2 rows != 1 rows"),
        ("sticky:fake_write", "read-back after the teardown differs: binding 0, row 0, col 1: "),
        ("wrong_later:fake_write", "read-back after timed run 1 differs: after the run: binding 0: 2 rows"),
    ],
)
def test_write_wrong_readback_is_disagree(writes_dir: Path, tmp_path: Path, fake: str, detail: str) -> None:
    """A read-back that misses the write, or a teardown that leaves it, is ≠ at the preflight; a later run's
    read-back that differs makes the entry ≠ with the run's number."""
    entry = _entries(_run_writes(writes_dir, tmp_path, fake=fake))[_write_id()]
    assert (entry["status"], entry["answer"], entry["timing"]) == ("ok", "disagree", None), entry
    assert entry["detail"].startswith(detail), entry["detail"]


def test_write_case_must_start_from_the_loaded_state(writes_dir: Path, tmp_path: Path) -> None:
    entry = _entries(_run_writes(writes_dir, tmp_path, fake="dirty:fake_write"))[_write_id()]
    assert entry["status"] == "failed"
    assert entry["error"].startswith("the graph is not in its loaded state before the case: binding 0: ")


def test_write_case_leaving_state_fails_the_guard(writes_dir: Path, tmp_path: Path) -> None:
    """A write that changes what no read-back sees is caught by the counts guard after the case."""
    lines = _run_writes(writes_dir, tmp_path, fake="leak:fake_write")
    entries = _entries(lines)
    entry = entries[_write_id()]
    assert (entry["status"], entry["answer"]) == ("failed", "agree"), entry
    snb = Dataset.load(next(writes_dir.glob("*/snb-small")))  # the write suite's own cache key
    nodes, edges = snb.node_count, snb.edge_count
    runs = 1 + (catalog.selection("push").warmup - 1) + catalog.selection("push").iterations
    leaked = f"{nodes + 3 * runs} nodes, {edges} edges; expected {nodes} nodes, {edges} edges"
    assert entry["error"] == f"the case left the graph changed: {leaked}"
    assert entries[_id("snb_is4")]["answer"] == "agree"


def test_write_setup_state_is_checked(writes_dir: Path, tmp_path: Path) -> None:
    """A write case with a `ready` reference: a setup that reaches it lets the case run and agree."""
    lines = _run_writes(writes_dir, tmp_path, item="fake_setup_write")
    entry = _entries(lines)[_write_id(item="fake_setup_write")]
    assert (entry["status"], entry["answer"], entry["error"]) == ("ok", "agree", None), entry
    assert entry["timing"]["iterations"] == catalog.selection("push").iterations


@pytest.mark.parametrize("fake", ["nosetup:fake_setup_write", "nosetup_later:fake_setup_write"])
def test_write_setup_that_misses_its_state_fails(writes_dir: Path, tmp_path: Path, fake: str) -> None:
    """A setup that inserts nothing fails the case, at the preflight or at a later run, and it is not timed."""
    lines = _run_writes(writes_dir, tmp_path, fake=fake, item="fake_setup_write")
    entry = _entries(lines)[_write_id(item="fake_setup_write")]
    assert (entry["status"], entry["timing"]) == ("failed", None), entry
    assert entry["error"] == "setup did not reach its state: binding 0, row 0, col 1: 'loaded' != 'ready'"
    assert entry["answer"] == ("unchecked" if fake.startswith("nosetup:") else "agree")


def test_write_with_ready_needs_ready_rows_in_its_reference(tmp_path: Path) -> None:
    """A write with a setup state whose reference file has no `ready` rows fails with a clear error, before any
    phase runs: the setup check is never skipped silently."""
    from tests.fake_writes import WRITES as FAKE_WRITES

    item = next(w for w in FAKE_WRITES if w.ready)
    reference = tmp_path / f"{item.id}-small.json"
    rows = [[[1, "loaded"]]]
    reference.write_text(json.dumps({"bindings": [{"batch": 0}], "answers": rows, "before": rows}), encoding="utf-8")

    class Untouched:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"the database was used: {name}")

    pf = worker.write_preflight(Case(item, "fake", "small", 1), Untouched(), reference, budget_s=60)  # type: ignore[arg-type]
    assert pf.status == "failed"
    assert pf.error == (
        f"ValueError: the reference file {reference.name} has no 'ready' rows, but {item.id} has a setup state;"
        " regenerate the references"
    )


def test_write_case_run_error_fails(writes_dir: Path, tmp_path: Path) -> None:
    entry = _entries(_run_writes(writes_dir, tmp_path, fake="crash_soft:fake_write"))[_write_id()]
    assert (entry["status"], entry["error"]) == ("failed", "RuntimeError: soft crash")


def test_only_queries_and_algorithms_seed_the_first_run_with_the_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight's duration predicts a query's or algorithm's first timed run. A write's full run adds its
    setup, read-back and teardown to what the preflight timed, so it is not seeded: its first run is predicted
    from nothing and later runs from the full run before them."""
    from types import SimpleNamespace

    from graph_bench.types import Timing
    from tests.fake_writes import WRITES as FAKE_WRITES

    seeds: dict[str, Any] = {}
    query = next(i for i in catalog.ITEMS if i.id == "snb_is2")

    def fake_measure(fn: Any, **kwargs: Any) -> tuple[Timing, None]:
        seeds[current[0]] = kwargs["first_s"]
        return Timing(1.0, 1.0, 1, 3), None

    def fake_start(case: Case, *args: Any) -> tuple[worker.Preflight, dict[str, Any]]:
        write = SimpleNamespace(run=list, prepare=lambda: None, readback=lambda: None, baseline=None)
        pf = worker.Preflight(run=list, bindings=[], preflight_ns=2_000_000_000, answer="agree")
        if case.item is not query:
            pf.write = write  # type: ignore[assignment]
        return pf, {"kind": "entry", "id": case.bench_id}

    monkeypatch.setattr(worker, "measure", fake_measure)
    monkeypatch.setattr(worker, "start_case", fake_start)
    settings = worker._Settings(catalog.selection("push"), tmp_path / "w.jsonl", {}, 60.0)
    current = ["query"]
    worker._run_case(Case(query, "fake", "small", 1), None, tmp_path, settings, lambda line: None)  # type: ignore[arg-type]
    current[0] = "write"
    worker._run_case(Case(FAKE_WRITES[0], "fake", "small", 1), None, tmp_path, settings, lambda line: None)  # type: ignore[arg-type]
    assert seeds == {"query": 2.0, "write": None}


def _deadline_worker(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep: set[str],
    deadline_at: float,
    passes_at: str = "test_snb_is",
) -> tuple[Lines, list[float]]:
    """`worker.run` in this process on the fake database, every id but `keep` skipped, with `worker._now` a clock
    the test moves: it reads 1000 until the first entry whose id contains `passes_at` is written (by default the
    first case's), then 2000. Returns the lines in order."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "")
    clock = [1000.0]
    monkeypatch.setattr(worker, "_now", lambda: clock[0])
    lines: Lines = []
    real = worker.emitter

    def emitter(fh: Any) -> worker.Emit:
        emit = real(fh)

        def hooked(line: dict[str, Any]) -> None:
            emit(line)
            lines.append(line)
            if line["kind"] == "entry" and passes_at in line["id"]:
                clock[0] = 2000.0  # the deadline passes as the case or load writes its entry

        return hooked

    monkeypatch.setattr(worker, "emitter", emitter)
    ids = catalog.all_ids("push", ["fake"])
    assert keep <= set(ids)
    out = tmp_path / "w.jsonl"
    worker.run(
        "fake", "push", 1, workdir=tmp_path / "work", out=out, skip=frozenset(set(ids) - keep), deadline_at=deadline_at
    )
    assert lines == [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return lines, clock


def test_worker_runs_a_pinned_row_on_its_own_scale(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row pinned to small in a medium selection loads small without a Load entry and checks small's reference."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "")
    data = cache.prepare("push")
    key = cache.dataset_key("push")
    monkeypatch.setattr(cache, "prepare", lambda _: data)
    monkeypatch.setattr(cache, "dataset_key", lambda _: key)
    medium = dataclasses.replace(catalog.SELECTIONS["push"], scales=("medium",))
    monkeypatch.setitem(catalog.SELECTIONS, "push", medium)
    ids = catalog.all_ids("push", ["fake"])
    pinned = next(i for i in ids if "pattern_join_rdf" in i)
    assert "fake-small" in pinned
    out = tmp_path / "w.jsonl"
    worker.run("fake", "push", 1, workdir=tmp_path / "work", out=out, skip=frozenset(set(ids) - {pinned}))
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = [line for line in lines if line["kind"] == "entry"]
    assert [e["id"] for e in entries] == [pinned]
    assert entries[0]["status"] == "ok" and entries[0]["answer"] == "agree", entries[0]


def test_worker_starts_nothing_past_the_deadline(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadline passes as a case writes its entry, the moment the supervisor may read that entry and the next
    `start` in one batch. The worker starts nothing more (no case, no load of the dataset or of another one, no
    Load (query language) entry): it writes a `deadline` line and returns, and the supervisor marks the rest."""
    load_snb = catalog.load_id("snb", "fake", "small", 1)
    keep = {_id("snb_is2"), _id("snb_is3"), load_snb, catalog.load_id("snb", "fake", "small", 1, "rdf")}
    keep.add(catalog.load_ql_id("snb", "fake", "small", 1))
    lines, _ = _deadline_worker(push_data_dir, tmp_path, monkeypatch, keep, deadline_at=1500.0)
    starts = [line["id"] for line in lines if line["kind"] == "start"]
    assert starts[0] == load_snb and len(starts) == 2 and starts[1] in (_id("snb_is2"), _id("snb_is3")), starts
    ended = next(i for i, line in enumerate(lines) if line["kind"] == "entry" and line["id"] == starts[1])
    assert lines[ended + 1 :] == [{"kind": "deadline"}]


def test_worker_starts_no_query_language_load_past_the_deadline(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadline passes as the graph load writes its entry: the dataset's Load (query language) entry, which
    runs next on instances of its own, is not started."""
    load_snb = catalog.load_id("snb", "fake", "small", 1)
    keep = {load_snb, catalog.load_ql_id("snb", "fake", "small", 1)}
    lines, _ = _deadline_worker(push_data_dir, tmp_path, monkeypatch, keep, 1500.0, passes_at=load_snb)
    assert [line["id"] for line in lines if line["kind"] == "start"] == [load_snb]
    ended = next(i for i, line in enumerate(lines) if line["kind"] == "entry" and line["id"] == load_snb)
    assert lines[ended + 1 :] == [{"kind": "deadline"}]


def test_worker_starts_no_next_load_past_the_deadline(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadline passes as the graph load writes its entry: the RDF load of the same dataset, next in order, is
    not started."""
    load_snb = catalog.load_id("snb", "fake", "small", 1)
    keep = {load_snb, catalog.load_id("snb", "fake", "small", 1, "rdf")}
    lines, _ = _deadline_worker(push_data_dir, tmp_path, monkeypatch, keep, 1500.0, passes_at=load_snb)
    assert [line["id"] for line in lines if line["kind"] == "start"] == [load_snb]
    assert [line["kind"] for line in lines if line["kind"] == "load"] == ["load"]
    ended = next(i for i, line in enumerate(lines) if line["kind"] == "entry" and line["id"] == load_snb)
    assert lines[ended + 1 :] == [{"kind": "deadline"}]


def test_worker_past_the_deadline_starts_no_startup_row(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep = {catalog.startup_id("fake", 1), _id("snb_is2"), catalog.load_id("snb", "fake", "small", 1)}
    lines, _ = _deadline_worker(push_data_dir, tmp_path, monkeypatch, keep, deadline_at=500.0)
    assert [line["kind"] for line in lines if line["kind"] in ("start", "load", "deadline")] == ["deadline"]
    assert lines[0]["kind"] == "database"


# Load (query language) ------------------------------------------------------------------------------------------


def test_load_ql_failure_does_not_fail_cases(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="crash_soft:load_ql_snb")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    snb = entries[catalog.load_ql_id("snb", "fake", "small", 1)]
    assert (snb["status"], snb["error"]) == ("failed", "RuntimeError: soft crash")
    ga = entries[catalog.load_ql_id("graphalytics", "fake", "small", 1)]
    assert (ga["status"], ga["answer"]) == ("ok", "agree") and ga["load_ns"] > 0
    for case in catalog.cases("push", ["fake"]):
        assert (entries[case.bench_id]["status"], entries[case.bench_id]["answer"]) == ("ok", "agree")
    assert entries[catalog.load_id("snb", "fake", "small", 1)]["answer"] == "agree"
    # It runs after the dataset's cases, on instances of its own.
    starts = [line["id"] for line in lines if line["kind"] == "start"]
    snb_cases = [starts.index(_id(q)) for q in ("snb_is1", "snb_ic13", "snb_sql_is1")]
    assert starts.index(catalog.load_ql_id("snb", "fake", "small", 1)) > max(snb_cases)


def test_load_ql_counts_and_not_supported(push_data_dir: Path, tmp_path: Path) -> None:
    fake = "short:count_ql_snb,unsupported:load_ql_graphalytics"
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake=fake)
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    snb = entries[catalog.load_ql_id("snb", "fake", "small", 1)]
    nodes, edges = _expected_counts(push_data_dir, "snb")
    assert (snb["status"], snb["answer"]) == ("ok", "disagree")
    assert snb["detail"] == f"loaded {nodes} nodes, {edges - 1} edges; expected {nodes} nodes, {edges} edges"
    ga = entries[catalog.load_ql_id("graphalytics", "fake", "small", 1)]
    assert (ga["status"], ga["error"]) == ("not_supported", "fake has no LOAD CSV")
    assert entries[catalog.load_id("snb", "fake", "small", 1)]["answer"] == "agree"


def test_load_ql_probe_mismatch_is_disagree(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a content probe for (snb, property graph), a query-language load whose content answers it wrongly is
    ≠ with the probe's name; the right content agrees."""
    from tests.fakes import FakeDatabase

    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setitem(catalog.LOAD_QL_PROBES, ("snb", "property_graph"), "snb_is4")
    data = cache.prepare("push")[("snb", "small")]
    base = cache.data_dir() / cache.dataset_key("push")
    for fake, answer in (("", "agree"), ("wrong:snb_is4", "disagree")):
        monkeypatch.setenv("GRAPH_BENCH_FAKE", fake)
        lines: Lines = []
        load_id = catalog.load_ql_id("snb", "fake", "small", 1)
        worker.run_load_ql(FakeDatabase, data, tmp_path / "work", 1, 2, base, load_id, lines.append)
        entry = _entries(lines)[load_id]
        assert (entry["status"], entry["answer"]) == ("ok", answer), entry
        if answer == "disagree":
            assert entry["detail"].startswith("content probe snb_is4: binding 0: ")
        assert [line["kind"] for line in lines] == ["start", "progress", "progress", "entry"]
    assert not any((tmp_path / "work").iterdir())  # every instance closed and removed


def test_load_ql_probe_error_fails_the_entry(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakes import FakeDatabase

    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setitem(catalog.LOAD_QL_PROBES, ("snb", "property_graph"), "snb_is4")
    data = cache.prepare("push")[("snb", "small")]
    base = cache.data_dir() / cache.dataset_key("push")
    lines: Lines = []
    load_id = catalog.load_ql_id("snb", "fake", "small", 1)
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "crash_soft:snb_is4")
    worker.run_load_ql(FakeDatabase, data, tmp_path / "work", 1, 1, base, load_id, lines.append)
    entry = _entries(lines)[load_id]
    assert (entry["status"], entry["error"]) == ("failed", "content probe snb_is4: RuntimeError: soft crash")
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "unsupported:snb_is4")
    lines.clear()
    worker.run_load_ql(FakeDatabase, data, tmp_path / "work", 1, 2, base, load_id, lines.append)
    entry = _entries(lines)[load_id]
    # A probe the database cannot run leaves the content unchecked, and says so; the counts agreed.
    assert (entry["status"], entry["answer"]) == ("ok", "unchecked")
    assert entry["detail"] == "content probe snb_is4 not run: fake does not run snb_is4"
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "unsupported:snb_is4,short:count_ql_snb")
    lines.clear()
    worker.run_load_ql(FakeDatabase, data, tmp_path / "work", 1, 1, base, load_id, lines.append)
    assert _entries(lines)[load_id]["answer"] == "disagree"  # a count mismatch still wins
    lines.clear()
    worker.run_load_ql(FakeDatabase, data, tmp_path / "work", 1, 1, base, load_id, lines.append, "setup broke")
    assert _entries(lines)[load_id] == {"kind": "entry", "id": load_id, "status": "failed", "error": "setup broke"}


PANIC = "PanicException: assertion failed: index <= Self::MAX_CAPACITY"


@pytest.mark.parametrize(
    ("flag", "failed"),
    [("panic:snb_is4", "snb_is4"), ("panic:graphalytics_bfs", "graphalytics_bfs")],
)
def test_worker_database_panic_fails_the_case_only(push_data_dir: Path, tmp_path: Path, flag: str, failed: str) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake=flag)
    assert proc.returncode == 0, proc.stderr
    assert len([line for line in lines if line["kind"] == "database"]) == 1  # one launch, no relaunch
    entries = _entries(lines)
    assert (entries[_id(failed)]["status"], entries[_id(failed)]["error"]) == ("failed", PANIC)
    assert (entries[_id("snb_is3")]["status"], entries[_id("snb_is3")]["answer"]) == ("ok", "agree")


def test_worker_database_panic_in_a_load_fails_the_load(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="panic:load_snb")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    load = entries[catalog.load_id("snb", "fake", "small", 1)]
    assert (load["status"], load["error"]) == ("failed", PANIC)
    assert entries[_id("snb_is4")]["error"] == f"load failed: {PANIC}"
    assert entries[_id("graphalytics_bfs")]["status"] == "ok"  # the other dataset still loads


def test_worker_database_panic_in_open_empty_fails_the_startup(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="panic:startup")
    assert proc.returncode == 0, proc.stderr
    entry = _entries(lines)[catalog.startup_id("fake", 1)]
    assert entry["status"] == "failed"
    assert "exited with code" not in entry["error"]  # the child reported the panic itself, it did not die
    assert PANIC in entry["error"]
    assert _entries(lines)[_id("snb_is4")]["status"] == "ok"


def test_worker_keyboard_interrupt_from_the_database_stops_the_worker(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="interrupt:snb_is4")
    assert proc.returncode != 0
    assert _id("snb_is4") not in _entries(lines)


def test_worker_database_panic_in_close_does_not_kill_the_worker(push_data_dir: Path, tmp_path: Path) -> None:
    """Every close panics (after the cases, between loads, on the info probe): the worker still finishes all."""
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, fake="panic:close")
    assert proc.returncode == 0, proc.stderr
    assert "PanicException: assertion failed" in proc.stderr
    assert len([line for line in lines if line["kind"] == "database"]) == 1
    entries = _entries(lines)
    for item in ("snb_is4", "graphalytics_bfs"):  # both datasets' cases ran, so the first close did not end it
        assert (entries[_id(item)]["status"], entries[_id(item)]["answer"]) == ("ok", "agree")
    assert entries[catalog.load_id("snb", "fake", "small", 1)]["status"] == "ok"


def test_worker_load_over_the_ceiling_skips_its_cases(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cooperative load ceiling: `slower:load_snb` takes 2 s, past a 1 s ceiling that the other fake loads (a few
    ms each, more on a busy host) stay under. The worker records the Load entry ⏱ "load over 1 s" and every case of
    that dataset and store, and the dataset's Load (query language) entry, ⏱ with the same error, each after its
    `start` line and without running it; the RDF store and the other datasets load as usual. Each load's `load`
    line names its dataset, store and scale for the supervisor's hard path."""
    trace = tmp_path / "trace.txt"
    monkeypatch.setenv("GRAPH_BENCH_FAKE_TRACE", str(trace))
    lines, proc = run_worker(tmp_path / "fake.jsonl", push_data_dir, "--load-ceiling", "1", fake="slower:load_snb")
    assert proc.returncode == 0, proc.stderr
    entries = _entries(lines)
    over = ("timeout", "load over 1 s")
    load_id = catalog.load_id("snb", "fake", "small", 1)
    ql_id = catalog.load_ql_id("snb", "fake", "small", 1)
    assert (entries[load_id]["status"], entries[load_id]["error"]) == over
    assert (entries[ql_id]["status"], entries[ql_id]["error"]) == over
    graph_cases = [_id(i.id) for i in catalog.ITEMS if (i.dataset, i.store) == ("snb", "graph")]
    started = {line["id"] for line in lines if line["kind"] == "start"}
    for bench_id in graph_cases:
        assert (entries[bench_id]["status"], entries[bench_id]["error"]) == over, bench_id
        assert bench_id in started
    assert not [line for line in lines if line["kind"] == "run" and line["id"] in graph_cases]
    rdf = entries[catalog.load_id("snb", "fake", "small", 1, "rdf")]
    assert (rdf["status"], rdf["answer"]) == ("ok", "agree")
    # snb's query-language load never starts; the other datasets' do (graphalytics in the push selection).
    others = [i for i in entries if "test_load_ql.py" in i and i != ql_id]
    assert all((entries[i]["status"], entries[i]["answer"]) == ("ok", "agree") for i in others), others
    assert _calls(trace).count("load_ql") == len(others)
    loads = [(line["dataset"], line["store"], line["scale"]) for line in lines if line["kind"] == "load"]
    assert ("snb", "graph", "small") in loads and ("multigraph", "graph", "small") in loads


def test_worker_time_boxes_the_load_iterations(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loads follow the time box with a minimum of one: of three load iterations of 0.2 s each only the first runs
    once the box is 0.1 s; the median is over the load that ran."""
    monkeypatch.setattr(worker, "TIME_BOX_S", 0.1)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "slow:load_snb")
    data = cache.prepare("push")[("snb", "small")]
    loaded = worker.load(get_database("fake"), data, tmp_path / "work", 1, 3, check=True)
    try:
        assert loaded.iterations == 1
        assert loaded.load_ns >= 0.2e9
        assert worker.load_entry("x", loaded.load_ns, loaded.check, loaded.iterations)["load_iterations"] == 1
    finally:
        worker.close_quietly(loaded.db)
