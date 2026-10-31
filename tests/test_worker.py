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
    loads = sel.load_iterations * len(catalog.load_ids("push", ["fake"]))
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
        if case.item.category in ("sql_pgq", "algorithms"):
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


def test_write_case_run_error_fails(writes_dir: Path, tmp_path: Path) -> None:
    entry = _entries(_run_writes(writes_dir, tmp_path, fake="crash_soft:fake_write"))[_write_id()]
    assert (entry["status"], entry["error"]) == ("failed", "RuntimeError: soft crash")


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
