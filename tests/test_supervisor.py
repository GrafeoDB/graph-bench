import dataclasses
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from graph_bench import databases
from graph_bench.cli import app
from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import Entry, Results
from graph_bench.runner import supervisor
from graph_bench.runner.supervisor import apply_budget_rule, run_bench
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case
from graph_bench.types import Timing

EXTRA = (
    "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2,"
    "fake3=tests.fakes:FakeDatabase3,fake4=tests.fakes:FakeDatabase4,"
    "ghost=tests.no_such_module:Ghost,ghost_rdf=tests.no_such_module:Ghost"
)


def _id(item_id: str, db: str = "fake") -> str:
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db, "small", 1).bench_id


def _by_id(results: Results) -> dict[str, Entry]:
    return {e.id: e for e in results.entries}


def _expected_ids(db: str) -> set[str]:
    return {c.bench_id for c in catalog.cases("push", [db])} | set(catalog.load_ids("push", [db]))


def _set_env(mp: pytest.MonkeyPatch, data_dir: Path, fake: str = "", fake2: str = "") -> None:
    mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
    mp.setenv("GRAPH_BENCH_DATA_DIR", str(data_dir))
    mp.setenv("GRAPH_BENCH_FAKE", fake)
    mp.setenv("GRAPH_BENCH_FAKE2", fake2)


@pytest.fixture(scope="module", autouse=True)
def _extra_databases() -> Iterator[None]:
    """Let the catalog in this process resolve the fake database names too."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
        yield


@pytest.fixture
def bench_env(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch, push_data_dir)
    return monkeypatch


@pytest.fixture(scope="module")
def two_dbs(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Results, list[list[str]], Path]:
    """fake then fake2 (which answers snb_is5 wrongly), recording every worker argv.

    The final 10x rule is switched off, so a scheduling hiccup in one fake cannot mark it; the rule has its own
    deterministic tests below.
    """
    root = tmp_path_factory.mktemp("two-dbs")
    argvs: list[list[str]] = []
    launch = supervisor._launch

    def recording(argv: list[str], *args: Any) -> Any:
        argvs.append(list(argv))
        return launch(argv, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(supervisor, "BUDGET_FACTOR", 1e9)
        _set_env(mp, push_data_dir, fake2="wrong:snb_is5")
        mp.chdir(root)  # run_bench reads .graph-bench/grafeo.json from the cwd; keep the real one out of it
        mp.setattr(supervisor, "_launch", recording)
        results = run_bench("push", ["fake", "fake2"], root / "out", workdir=root / "work", case_timeout_s=60)
    return results, argvs, root / "out"


def test_supervisor_all_agree(two_dbs: tuple[Results, list[list[str]], Path]) -> None:
    results, _, out = two_dbs
    entries = [e for e in results.entries if e.database == "fake"]
    assert {e.id for e in entries} == _expected_ids("fake")
    for entry in entries:
        assert entry.status == "ok", entry
        if entry.category == "load":
            assert entry.load_ns is not None and entry.load_ns > 0
        else:
            assert entry.answer == "agree", entry
            assert entry.timing is not None
            assert entry.reference == ("networkx" if entry.category == "algorithms" else "python")
    assert Results.read(out / "results.json") == results
    assert results.selection == "push"
    assert results.mode == "bench"
    assert set(results.databases) == {"fake", "fake2"}
    assert results.databases["fake"].name == "fake"


def test_supervisor_writes_raw_output(two_dbs: tuple[Results, list[list[str]], Path]) -> None:
    _, _, out = two_dbs
    raw = out / "raw" / "fake-1"
    assert (raw / "worker.jsonl").is_file()
    assert list(raw.rglob("*snb_is4*.json"))


def test_supervisor_metadata(two_dbs: tuple[Results, list[list[str]], Path]) -> None:
    results, _, _ = two_dbs
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", results.graph_bench_commit)
    assert results.machine["cores"] > 0
    assert results.machine["cpu"]
    assert results.machine["os"]
    assert results.grafeo["source"] == "pypi"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", results.started_at)


def _grafeo_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: str) -> dict[str, str]:
    """Write `.graph-bench/grafeo.json` for version 0.6.1.dev0 in the cwd and fake the installed grafeo version."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor.importlib.metadata, "version", lambda name: installed)
    (tmp_path / ".graph-bench").mkdir()
    doc = {"version": "0.6.1.dev0", "source": "git", "sha": "abc123", "git_sha": "abc123"}
    (tmp_path / ".graph-bench" / "grafeo.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_grafeo_source_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _grafeo_file(tmp_path, monkeypatch, installed="0.6.1.dev0")
    assert supervisor.grafeo_source() == doc


def test_grafeo_source_ignores_a_stale_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A later `uv sync` replaced the recorded build: the installed version wins, from PyPI."""
    _grafeo_file(tmp_path, monkeypatch, installed="0.6.0")
    assert supervisor.grafeo_source() == {"version": "0.6.0", "source": "pypi", "sha": None, "git_sha": None}


def test_supervisor_marks_wrong_answer(two_dbs: tuple[Results, list[list[str]], Path]) -> None:
    entry = _by_id(two_dbs[0])[_id("snb_is5", "fake2")]
    assert entry.status == "ok"
    assert entry.answer == "disagree"
    assert entry.detail
    assert entry.timing is None


def test_supervisor_passes_budgets(two_dbs: tuple[Results, list[list[str]], Path]) -> None:
    """With fewer than 3 earlier databases no in-run budget applies; the worker gets only --case-timeout."""
    results, argvs, _ = two_dbs
    assert len(argvs) == 2
    assert [argv[argv.index("--db") + 1] for argv in argvs] == ["fake", "fake2"]
    assert all("--budget" not in argv for argv in argvs)
    assert all(argv[argv.index("--case-timeout") + 1] == "60" for argv in argvs)
    assert _by_id(results)[_id("snb_is4", "fake2")].status == "ok"


def _entry(db: str, ms: float, *, item: str = "snb_ic13", status: str = "ok", answer: str = "agree") -> Entry:
    return Entry(
        id=_id(item, db),
        category="hops",
        model="property_graph",
        benchmark=item,
        database=db,
        scale="small",
        threads=1,
        status=status,  # type: ignore[arg-type]
        timing=Timing(ms * 1e6, ms * 1e6, int(ms * 1e6), 5),
        preflight_ns=int(ms * 1e6),
        answer=answer,  # type: ignore[arg-type]
        reference="python",
    )


def test_budgets_need_three_earlier_databases() -> None:
    meta = supervisor._expected(catalog.selection("push"), "fake", 1)
    target = _id("snb_ic13")
    two = [_entry("a", 1), _entry("b", 2)]
    assert supervisor._budgets(two, "fake", meta) == {}
    four = [
        *two,
        _entry("c", 3),
        _entry("d", 100),
        _entry("e", 0.5, answer="disagree"),
        _entry("f", 0.5, status="failed"),
    ]
    # 30x (a runaway guard on cold first runs), not the final rule's 10x, and never below the 10 s floor
    assert supervisor._budgets(four, "fake", meta) == {target: pytest.approx(10.0)}
    assert supervisor._budgets([*four, _entry("fake", 0.1)], "fake", meta) == {target: pytest.approx(10.0)}
    slow = [_entry("a", 1000), _entry("b", 2000), _entry("c", 3000)]
    assert supervisor._budgets(slow, "fake", meta) == {target: pytest.approx(30 * 2.0)}


def test_in_run_floor_keeps_cold_first_runs() -> None:
    """Fast earlier databases (2 ms preflights) no longer give a 60 ms cutoff: a cold JVM's first run survives."""
    meta = supervisor._expected(catalog.selection("push"), "fake", 1)
    fast = [_entry("a", 1), _entry("b", 2), _entry("c", 3)]
    assert set(supervisor._budgets(fast, "fake", meta).values()) == {supervisor.IN_RUN_FLOOR_S}


def test_in_run_cutoff_kills_a_slow_preflight(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """3 fast earlier databases give a 4th an in-run budget; its slow preflight is killed at that budget.

    `slow:snb_is2` sleeps 0.2 s per binding (2 s in all), far beyond 30x the fakes' few-ms preflights, so only
    the supervisor's cutoff can produce this error; `case_timeout_s` (60 s) never comes into play.
    `slow_check:snb_is3` answers at once but makes the worker's answer check take 2 s: with a 0.5 s budget it
    is not cut off, because the cutoff times only the database call. Every other case gets a generous budget
    and the final 10x rule is switched off, so a scheduling hiccup in a fast fake cannot mark it.
    """
    bench_env.setenv("GRAPH_BENCH_FAKE3", "")
    bench_env.setenv("GRAPH_BENCH_FAKE4", "slow:snb_is2,slow_check:snb_is3")
    bench_env.setattr(supervisor, "BUDGET_FACTOR", 1e9)
    tight, harness = _id("snb_is2", "fake4"), _id("snb_is3", "fake4")
    computed = supervisor._budgets

    def budgets(entries: Any, db: str, meta: Any) -> dict[str, float]:
        found = computed(entries, db, meta)
        return {i: 0.5 if i in (tight, harness) else 60.0 for i in found}

    bench_env.setattr(supervisor, "_budgets", budgets)
    launches: list[list[str]] = []
    launch = supervisor._launch

    def recording(argv: list[str], *args: Any) -> Any:
        launches.append(list(argv))
        return launch(argv, *args)

    bench_env.setattr(supervisor, "_launch", recording)
    dbs = ["fake", "fake2", "fake3", "fake4"]
    started = time.monotonic()
    results = run_bench("push", dbs, tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    assert time.monotonic() - started < 60

    entry = _by_id(results)[tight]
    assert entry.status == "timeout", entry
    pattern = r"too slow: cut off at (\S+) s \(30x the top-3 average of earlier databases, at least 10 s\)"
    match = re.fullmatch(pattern, entry.error or "")
    assert match, entry.error
    assert float(match.group(1)) < 2.0
    assert entry.timing is None
    slow_check = _by_id(results)[harness]
    assert (slow_check.status, slow_check.answer) == ("ok", "agree"), slow_check
    fake4 = [argv for argv in launches if argv[argv.index("--db") + 1] == "fake4"]
    assert len(fake4) == 2  # killed during the preflight, then relaunched for the rest
    assert any(arg.startswith(tight + "=") for arg in fake4[0])  # its in-run budget
    assert harness + "=0.5" in fake4[0]
    assert tight in fake4[1][fake4[1].index("--skip") + 1 :]
    rest = [e for e in results.entries if e.database == "fake4" and e.id != entry.id]
    assert all(e.status == "ok" for e in rest), [e for e in rest if e.status != "ok"]


# A stand-in worker: argv is out file, scenario, bench id. "slow_harness" spends 1 s before and after a fast
# database call; "slow_run" stays silent for 5 s after its `run` line.
_SCRIPTED_WORKER = """\
import json, sys, time
out, scenario, bench_id = sys.argv[1:4]
def emit(**line):
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\\n")
emit(kind="database", name="fake", version="1", process="in-process", threads_configurable=False,
     threads_note="", storage="in-memory")
emit(kind="start", id=bench_id)
if scenario == "slow_harness":
    time.sleep(1.0)
    emit(kind="run", id=bench_id)
    emit(kind="progress", id=bench_id)
    time.sleep(1.0)
else:
    emit(kind="run", id=bench_id)
    time.sleep(5.0)
emit(kind="entry", id=bench_id, status="ok", answer="agree", preflight_ns=1000)
"""


@pytest.mark.parametrize("scenario", ["slow_harness", "slow_run"])
def test_in_run_cutoff_times_only_the_database_call(scenario: str, tmp_path: Path) -> None:
    """The cutoff is armed by the `run` line and disarmed by the next line, so harness work never counts."""
    sel = catalog.selection("push")
    results = _results([])
    run = supervisor._Run(sel, tmp_path, tmp_path, 30.0, results)
    meta = supervisor._expected(sel, "fake", 1)
    bench_id = _id("snb_is2")
    out = tmp_path / "worker.jsonl"
    argv = [sys.executable, "-c", _SCRIPTED_WORKER, str(out), scenario, bench_id]
    attempt = run._attempt(argv, dict(os.environ), out, tmp_path / "worker.log", "fake", meta, {bench_id: 0.3})
    if scenario == "slow_harness":
        assert (attempt.hung, attempt.cutoff_s, attempt.code) == (False, None, 0)
        assert (run.recorded[bench_id].status, run.recorded[bench_id].answer) == ("ok", "agree")
    else:
        assert (attempt.hung, attempt.cutoff_s) == (True, 0.3)
        run._settle(attempt, meta)
        expected = "too slow: cut off at 0.3 s (30x the top-3 average of earlier databases, at least 10 s)"
        assert (run.recorded[bench_id].status, run.recorded[bench_id].error) == ("timeout", expected)


def test_supervisor_resumes_after_crash(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setenv("GRAPH_BENCH_FAKE", "crash:snb_is3")
    results = run_bench("push", ["fake"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    entries = _by_id(results)
    assert set(entries) == _expected_ids("fake")
    assert len(results.entries) == len(entries)
    crashed = entries.pop(_id("snb_is3"))
    assert crashed.status == "failed"
    assert crashed.error is not None and "exited with code 3" in crashed.error
    assert all(e.status == "ok" for e in entries.values()), [e for e in entries.values() if e.status != "ok"]


def test_supervisor_marks_hang_as_timeout(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setenv("GRAPH_BENCH_FAKE", "hang:snb_is4")
    results = run_bench("push", ["fake"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=2)
    entries = _by_id(results)
    assert set(entries) == _expected_ids("fake")
    hung = entries.pop(_id("snb_is4"))
    assert hung.status == "timeout"
    assert hung.timing is None
    assert all(e.status == "ok" for e in entries.values()), [e for e in entries.values() if e.status != "ok"]


def test_supervisor_kills_worker_on_harness_error(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setenv("GRAPH_BENCH_FAKE", "hang:snb_is4")
    procs: list[Any] = []
    launch = supervisor._launch

    def recording(*args: Any) -> Any:
        procs.append(launch(*args))
        return procs[-1]

    def broken(*args: Any) -> str | None:
        raise RuntimeError("harness bug")

    bench_env.setattr(supervisor, "_launch", recording)
    bench_env.setattr(supervisor._Run, "_line", broken)
    with pytest.raises(RuntimeError, match="harness bug"):
        run_bench("push", ["fake"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    assert len(procs) == 1
    assert procs[0].poll() is not None


def test_supervisor_load_crash_is_not_retried(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setenv("GRAPH_BENCH_FAKE", "crash:load_snb")
    launches: list[list[str]] = []
    launch = supervisor._launch

    def recording(argv: list[str], *args: Any) -> Any:
        launches.append(list(argv))
        return launch(argv, *args)

    bench_env.setattr(supervisor, "_launch", recording)
    results = run_bench("push", ["fake"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    entries = _by_id(results)
    assert set(entries) == _expected_ids("fake")
    load_id = catalog.load_id("snb", "fake", "small", 1)
    assert entries[load_id].status == "failed"
    assert "exited with code 3" in (entries[load_id].error or "")
    # The RDF store is a load of its own: the graph load's crash leaves it and its cases alone.
    assert entries[catalog.load_id("snb", "fake", "small", 1, "rdf")].status == "ok"
    for entry in entries.values():
        if entry.id == load_id:
            continue
        if entry.benchmark.startswith("snb_") and entry.model != "rdf":
            assert entry.status == "failed", entry
            assert entry.error == f"load failed: {load_id}"
        else:
            assert entry.status == "ok", entry
    assert len(launches) <= 2


def _results(entries: list[Entry], mode: str = "bench") -> Results:
    info = DatabaseInfo("x", "1", "in-process", False, "", "in-memory")
    return Results(
        graph_bench_commit="x",
        grafeo={"version": None, "source": "pypi", "sha": None},
        machine={"cpu": "x", "cores": 1, "os": "x"},
        selection="push",
        mode=mode,  # type: ignore[arg-type]
        started_at="2026-10-30T00:00:00Z",
        databases={e.database: info for e in entries},
        entries=entries,
    )


def _markers(order: list[tuple[str, float]], mode: str = "bench") -> dict[str, tuple[str, str | None]]:
    results = _results([_entry(db, ms) for db, ms in order], mode)
    apply_budget_rule(results)
    return {e.database: (e.status, e.error) for e in results.entries}


def test_budget_rule_is_order_independent() -> None:
    slow_first = _markers([("slow", 50), ("fast", 1)])
    slow_last = _markers([("fast", 1), ("slow", 50)])
    assert slow_first == slow_last
    assert slow_first["slow"] == ("timeout", "too slow: 50.0x the top-3 average")
    assert slow_first["fast"] == ("ok", None)
    four = _markers([("d", 50), ("a", 1), ("b", 1), ("c", 1)], mode="codspeed")
    assert four == _markers([("a", 1), ("b", 1), ("c", 1), ("d", 50)], mode="codspeed")
    assert four["d"] == ("timeout", "too slow: 50.0x the top-3 average")


def test_budget_rule_compares_with_the_others() -> None:
    markers = _markers([("a", 1), ("b", 15), ("c", 2), ("d", 3)])
    assert all(marker == ("ok", None) for marker in markers.values()), markers  # 15 vs others' mean 2: budget 20


def test_budget_rule_exempts_loads() -> None:
    entries = [dataclasses.replace(_entry(db, ms), category="load") for db, ms in (("a", 1), ("b", 1), ("c", 500))]
    results = _results(entries)
    apply_budget_rule(results)
    assert [(e.status, e.error) for e in results.entries] == [("ok", None)] * 3
    mixed = [*entries, dataclasses.replace(_entry("d", 1000), category="load")]
    assert all(e.status == "ok" for e in _apply(mixed))


def _apply(entries: list[Entry]) -> list[Entry]:
    results = _results(entries)
    apply_budget_rule(results)
    return results.entries


def test_budget_rule_needs_another_database() -> None:
    results = _results(
        [_entry("a", 1e6), _entry("b", 1, item="snb_is1", answer="disagree"), _entry("c", 1e9, status="failed")]
    )
    apply_budget_rule(results)
    assert [(e.status, e.error) for e in results.entries] == [("ok", None), ("ok", None), ("failed", None)]


def test_missing_driver_marks_failed(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setitem(databases.MODELS, "ghost", frozenset({"property_graph"}))
    bench_env.setitem(databases.MODELS, "ghost_rdf", frozenset({"rdf"}))
    results = run_bench("push", ["ghost", "ghost_rdf"], tmp_path / "out", workdir=tmp_path / "work")
    entries = _by_id(results)
    assert set(entries) == _expected_ids("ghost") | _expected_ids("ghost_rdf")
    for entry in entries.values():
        if entry.database == "ghost" and entry.model != "property_graph":
            assert (entry.status, entry.error) == ("not_supported", f"model {entry.model} not supported"), entry
        elif entry.database == "ghost" or entry.model == "rdf":
            assert entry.status == "failed", entry
            assert entry.error is not None and entry.error.startswith("not installed: ")
            assert "no_such_module" in entry.error
            assert "no build for" in entry.error
        else:
            assert entry.status == "not_supported", entry
    assert entries[catalog.load_id("snb", "ghost_rdf", "small", 1, "rdf")].status == "failed"
    assert results.databases == {}


def test_bench_exit_zero_on_disagree(bench_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_env.setenv("GRAPH_BENCH_FAKE", "wrong:snb_is5")
    out = tmp_path / "out"
    args = ["bench", "--selection", "push", "--db", "fake", "--out", str(out), "--case-timeout", "60"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    path = out / "results.json"
    assert str(path) in result.output
    entry = _by_id(Results.read(path))[_id("snb_is5")]
    assert entry.answer == "disagree"


def test_bench_rejects_bad_grafeo_source(tmp_path: Path) -> None:
    args = ["bench", "--selection", "push", "--grafeo", "x", "--out", str(tmp_path / "out")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "expected a .whl file or a grafeo checkout" in result.output


def test_bench_rejects_unknown_db(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["bench", "--selection", "push", "--db", "nope", "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert "nope" in result.output


def test_cpu_name_falls_back_to_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "_native_cpu_name", lambda: None)
    monkeypatch.setattr(supervisor.platform, "processor", lambda: " Generic CPU ")
    assert supervisor.cpu_name() == "Generic CPU"
    monkeypatch.setattr(supervisor.platform, "processor", lambda: "")
    monkeypatch.setattr(supervisor.platform, "machine", lambda: "x86_64")
    assert supervisor.cpu_name() == "x86_64"


def test_cpu_name_strips_native_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "_native_cpu_name", lambda: "  Test CPU 9000 \n")
    assert supervisor.cpu_name() == "Test CPU 9000"
    monkeypatch.setattr(supervisor, "_native_cpu_name", lambda: "   ")
    monkeypatch.setattr(supervisor.platform, "processor", lambda: "Generic CPU")
    assert supervisor.cpu_name() == "Generic CPU"


def test_native_cpu_name_unknown_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor.sys, "platform", "plan9")
    assert supervisor._native_cpu_name() is None
