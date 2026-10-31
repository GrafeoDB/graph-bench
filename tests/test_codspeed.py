"""The pytest-codspeed entry point (`benches/`) and the `codspeed` command."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from graph_bench.cli import app
from graph_bench.results import Entry, Results
from graph_bench.runner import supervisor
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case
from tests.conftest import HANG_TIMEOUT_S, resources_off

ROOT = Path(__file__).resolve().parents[1]
EXTRA = (
    "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2,fake_rdf=tests.fakes:FakeRdfDatabase,"
    "fake_nostart=tests.fakes:FakeNoStartup"
)
WRITES = {"GRAPH_BENCH_EXTRA_SUITES": "tests.fake_writes"}
WRITE_ID = "benches/test_batch_writes.py::test_fake_write[fake-small]"


def _id(item_id: str, db: str = "fake") -> str:
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db, "small", 1).bench_id


def _expected_ids(db: str) -> set[str]:
    return set(catalog.all_ids("push", [db]))


def _env(data_dir: Path, fake: str = "") -> dict[str, str]:
    return {
        **os.environ,
        "GRAPH_BENCH_EXTRA_DATABASES": EXTRA,
        "GRAPH_BENCH_DATA_DIR": str(data_dir),
        "GRAPH_BENCH_FAKE": fake,
    }


def _pytest(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run `pytest benches` the way the `codspeed` command does, without `--codspeed`."""
    argv = [sys.executable, "-m", "pytest", "benches", "-o", "addopts=", "-p", "no:cacheprovider", *args]
    return subprocess.run(
        argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600, check=False
    )


@pytest.fixture(scope="module", autouse=True)
def _extra_databases() -> Iterator[None]:
    """Let the catalog in this process resolve the fake database names too."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
        yield


def _without_codspeed(launches: list[tuple[list[str], dict[str, str]]]) -> Any:
    """A `_launch` that records argv and env, and drops `--codspeed` so no CodSpeed measurement runs."""
    launch = supervisor._launch

    def recording(argv: list[str], env: dict[str, str], *args: Any, **kwargs: Any) -> Any:
        launches.append((list(argv), dict(env)))
        return launch([a for a in argv if a != "--codspeed"], env, *args, **kwargs)

    return recording


@pytest.fixture(scope="module")
def codspeed_run(
    push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Results, list[tuple[list[str], dict[str, str]]], Path]:
    """`graph-bench codspeed --selection push --db fake,fake2`, with CODSPEED_PROFILE_FOLDER set.

    The final 10x rule is switched off: here it would compare single cold preflights of two fakes, which can
    differ by 10x at random. `test_budget_rule_is_order_independent` (test_supervisor.py) covers it in codspeed
    mode.
    """
    root = tmp_path_factory.mktemp("codspeed")
    launches: list[tuple[list[str], dict[str, str]]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(supervisor, "BUDGET_FACTOR", 1e9)
        resources_off(mp)  # tests/test_resources.py covers resources in codspeed mode
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
        mp.setenv("GRAPH_BENCH_FAKE", "")
        mp.setenv("GRAPH_BENCH_FAKE2", "wrong:snb_is5")
        mp.setenv("CODSPEED_PROFILE_FOLDER", "/x")
        mp.chdir(root)
        mp.setattr(supervisor, "_launch", _without_codspeed(launches))
        result = CliRunner().invoke(
            app, ["codspeed", "--selection", "push", "--db", "fake,fake2", "--out", str(root / "out")]
        )
    assert result.exit_code == 0, result.output
    return Results.read(root / "out" / "results.json"), launches, root / "out"


def test_codspeed_names_match_list(monkeypatch: pytest.MonkeyPatch, push_data_dir: Path) -> None:
    proc = _pytest(["--collect-only", "-q", "--db", "fake", "--selection", "push"], _env(push_data_dir))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    collected = sorted(line for line in proc.stdout.splitlines() if "::" in line)
    listed = CliRunner().invoke(app, ["list", "--selection", "push", "--db", "fake"])
    assert listed.exit_code == 0, listed.output
    assert collected == listed.output.split()


def test_codspeed_passes_profile_folder(
    codspeed_run: tuple[Results, list[tuple[list[str], dict[str, str]]], Path],
) -> None:
    _, launches, _ = codspeed_run
    assert [argv[argv.index("--db") + 1] for argv, _ in launches] == ["fake", "fake2"]
    for argv, env in launches:
        assert argv[1:5] == ["-m", "pytest", "benches", "--codspeed"]
        assert env["CODSPEED_PROFILE_FOLDER"] == "/x"


def test_codspeed_writes_results_without_timings(
    codspeed_run: tuple[Results, list[tuple[list[str], dict[str, str]]], Path],
) -> None:
    results, _, out = codspeed_run
    assert results.mode == "codspeed"
    assert results.selection == "push"
    assert set(results.databases) == {"fake", "fake2"}
    entries = {e.id: e for e in results.entries}
    assert set(entries) == _expected_ids("fake") | _expected_ids("fake2")
    for entry in entries.values():
        if entry.category == "startup":  # the children's in-process times, in both modes
            assert (entry.status, entry.answer) == ("ok", "agree"), entry
            assert entry.timing is not None and entry.timing.iterations >= 2, entry
            continue
        assert entry.timing is None, entry
        if entry.category in ("load", "load_ql"):
            assert entry.status == "ok", entry
            continue
        if entry.id == _id("snb_is5", "fake2"):
            assert (entry.status, entry.answer) == ("ok", "disagree"), entry
        else:
            assert (entry.status, entry.answer) == ("ok", "agree"), entry
        assert entry.preflight_ns is not None and entry.preflight_ns > 0
    assert (out / "raw" / "fake-1" / "pytest.jsonl").is_file()
    assert list((out / "raw" / "fake-1").rglob("*snb_is4*.json"))


def test_codspeed_marker_outcomes(push_data_dir: Path, tmp_path: Path) -> None:
    fake = "unsupported:snb_is1,crash_soft:snb_is2,wrong:snb_is3,slow:snb_is4"
    env = {**_env(push_data_dir, fake), "GRAPH_BENCH_SPY": str(tmp_path / "spy.txt")}
    junit = tmp_path / "junit.xml"
    out = tmp_path / "gb.jsonl"
    args = ["--db", "fake", "--selection", "push", "-p", "tests.benchmark_spy", f"--junitxml={junit}"]
    args += ["--gb-out", str(out), "--gb-budget", f"{_id('snb_is4')}=0.05"]
    args += ["-k", "test_snb_is1 or test_snb_is2 or test_snb_is3 or test_snb_is4 or test_snb_is5"]
    proc = _pytest(args, env)
    assert proc.returncode == 1, proc.stdout + proc.stderr

    outcomes: dict[str, tuple[str, str]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "")
        for kind in ("skipped", "failure", "error"):
            node = case.find(kind)
            if node is not None:
                # junit prefixes a failure's reason with pytest's own "Failed: "
                outcomes[name] = (kind, node.get("message", "").removeprefix("Failed: "))
                break
        else:
            outcomes[name] = ("passed", "")
    assert outcomes["test_snb_is1[fake-small]"][0] == "skipped"
    assert outcomes["test_snb_is1[fake-small]"][1].startswith("not available: ")
    assert outcomes["test_snb_is2[fake-small]"][0] == "failure"
    assert outcomes["test_snb_is2[fake-small]"][1].startswith("error: RuntimeError")
    assert outcomes["test_snb_is3[fake-small]"][0] == "failure"
    assert outcomes["test_snb_is3[fake-small]"][1].startswith("wrong answer: ")
    assert outcomes["test_snb_is4[fake-small]"][0] == "failure"
    assert outcomes["test_snb_is4[fake-small]"][1].startswith("too slow: ")
    assert outcomes["test_snb_is5[fake-small]"] == ("passed", "")

    benchmarked = (tmp_path / "spy.txt").read_text(encoding="utf-8").split()
    assert benchmarked == [_id("snb_is5")]

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["kind"] == "database"
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    assert entries[_id("snb_is1")]["status"] == "not_supported"
    assert entries[_id("snb_is2")]["status"] == "failed"
    assert (entries[_id("snb_is3")]["status"], entries[_id("snb_is3")]["answer"]) == ("ok", "disagree")
    assert entries[_id("snb_is4")]["status"] == "timeout"
    assert entries[_id("snb_is5")]["status"] == "ok"
    assert all(e["timing"] is None for e in entries.values())


PANIC = "PanicException: assertion failed: index <= Self::MAX_CAPACITY"


def test_codspeed_database_panic_is_the_tests_failure(push_data_dir: Path, tmp_path: Path) -> None:
    """A panic from a query, an algorithm, a load and open_empty fails that test with its message; the rest run."""
    fake = "panic:snb_is2,panic:graphalytics_bfs,panic:load_graphalytics,panic:startup"
    junit = tmp_path / "junit.xml"
    out = tmp_path / "gb.jsonl"
    args = ["--db", "fake", "--selection", "push", f"--junitxml={junit}", "--gb-out", str(out)]
    args += ["-k", "test_snb_is2 or test_snb_is3 or test_graphalytics_bfs or test_load_graphalytics or test_startup"]
    proc = _pytest(args, _env(push_data_dir, fake))
    assert proc.returncode == 1, proc.stdout + proc.stderr  # failures, not an interrupted or crashed session
    outcomes = _junit_outcomes(junit)
    assert outcomes["test_snb_is2[fake-small]"] == ("failure", f"error: {PANIC}")
    assert outcomes["test_snb_is3[fake-small]"] == ("passed", "")
    assert outcomes["test_graphalytics_bfs[fake-small]"][0] == "failure"
    assert outcomes["test_graphalytics_bfs[fake-small]"][1] == f"error: load failed: {PANIC}"
    load = next(name for name in outcomes if name.startswith("test_load_graphalytics"))
    assert outcomes[load] == ("failure", f"error: {PANIC}")
    startup = next(name for name in outcomes if name.startswith("test_startup"))
    assert outcomes[startup][0] == "failure"
    assert PANIC in outcomes[startup][1]


def test_codspeed_database_panic_in_an_algorithm_fails_that_test(push_data_dir: Path, tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    args = ["--db", "fake", "--selection", "push", f"--junitxml={junit}", "--gb-out", str(tmp_path / "gb.jsonl")]
    args += ["-k", "test_graphalytics_bfs or test_graphalytics_wcc"]
    proc = _pytest(args, _env(push_data_dir, "panic:graphalytics_bfs"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    outcomes = _junit_outcomes(junit)
    assert outcomes["test_graphalytics_bfs[fake-small]"] == ("failure", f"error: {PANIC}")
    assert outcomes["test_graphalytics_wcc[fake-small]"] == ("passed", "")


def test_codspeed_keyboard_interrupt_from_the_database_stops_the_session(push_data_dir: Path, tmp_path: Path) -> None:
    args = ["--db", "fake", "--selection", "push", "--gb-out", str(tmp_path / "gb.jsonl")]
    args += ["-k", "test_snb_is2 or test_snb_is3"]
    proc = _pytest(args, _env(push_data_dir, "interrupt:snb_is2"))
    assert proc.returncode == 2, proc.stdout + proc.stderr  # pytest: interrupted


def test_codspeed_prepares_environment_once_before_loads(push_data_dir: Path, tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    env = {**_env(push_data_dir), "GRAPH_BENCH_FAKE_TRACE": str(trace)}
    proc = _pytest(["--db", "fake", "--selection", "push", "--gb-out", str(tmp_path / "gb.jsonl")], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    sel = catalog.selection("push")
    loads = sel.load_iterations * len(catalog.load_ids("push", ["fake"]))
    loads_ql = sel.load_iterations * len(catalog.load_ql_ids("push", ["fake"]))
    calls = [line.split() for line in trace.read_text(encoding="utf-8").splitlines()]
    names = [call for call, *_ in calls if call != "open_empty"]
    assert names[:3] == ["provision", "provision", "environment"]
    assert sorted(names[3:]) == sorted(["load"] * loads + ["load_ql"] * loads_ql)
    # The startup test's children (a preflight child, then one round without --codspeed) ran before any load.
    opens = [i for i, (call, *_) in enumerate(calls) if call == "open_empty"]
    assert len(opens) == 2
    assert max(opens) < [call for call, *_ in calls].index("load")


def test_codspeed_environment_failure_fails_the_loads(push_data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "gb.jsonl"
    args = ["--db", "fake", "--selection", "push", "--gb-out", str(out)]
    args += ["-k", "test_load_snb or test_snb_is1 or test_graphalytics_bfs"]
    proc = _pytest(args, _env(push_data_dir, "crash_soft:environment"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    error = "preparing the environment: RuntimeError: soft crash"
    load_id = catalog.load_id("snb", "fake", "small", 1)
    assert (entries[load_id]["status"], entries[load_id]["error"]) == ("failed", error)
    for item in ("snb_is1", "graphalytics_bfs"):
        assert (entries[_id(item)]["status"], entries[_id(item)]["error"]) == ("failed", f"load failed: {error}")


def _junit_outcomes(junit: Path) -> dict[str, tuple[str, str]]:
    outcomes: dict[str, tuple[str, str]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "")
        for kind in ("skipped", "failure", "error"):
            node = case.find(kind)
            if node is not None:
                outcomes[name] = (kind, node.get("message", "").removeprefix("Failed: "))
                break
        else:
            outcomes[name] = ("passed", "")
    return outcomes


def test_codspeed_load_count_mismatch_is_a_wrong_answer(push_data_dir: Path, tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    out = tmp_path / "gb.jsonl"
    args = ["--db", "fake", "--selection", "push", "--gb-out", str(out), f"--junitxml={junit}"]
    args += ["-k", "test_load_snb or test_snb_is4"]
    proc = _pytest(args, _env(push_data_dir, "short:count_snb"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    outcomes = _junit_outcomes(junit)
    kind, reason = outcomes["test_load_snb[fake-small]"]
    assert kind == "failure"
    assert reason.startswith("wrong answer: loaded "), reason
    assert outcomes["test_snb_is4[fake-small]"] == ("passed", "")  # it runs on the instance the load test kept

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    load = entries[catalog.load_id("snb", "fake", "small", 1)]
    assert (load["status"], load["answer"]) == ("ok", "disagree")
    assert load["detail"].startswith("loaded ") and load["load_ns"] > 0


def _codspeed_run(tmp_path: Path) -> tuple[supervisor._Run, dict[str, supervisor._Meta]]:
    sel = catalog.selection("push")
    results = Results(
        graph_bench_commit="x",
        grafeo={"version": None, "source": "pypi", "sha": None},
        machine={"cpu": "c", "cores": 1, "os": "o"},
        selection="push",
        mode="codspeed",
        started_at="t",
        databases={},
    )
    return supervisor._Run(sel, tmp_path, tmp_path, 5.0, results, supervisor.PYTEST), supervisor._expected(
        sel, "fake", 1
    )


def test_codspeed_rerun_skips_a_load_with_wrong_counts(tmp_path: Path) -> None:
    run, meta = _codspeed_run(tmp_path)
    good, bad = catalog.load_id("graphalytics", "fake", "small", 1), catalog.load_id("snb", "fake", "small", 1)
    run.record(meta[good].entry(good, status="ok", answer="agree", load_ns=1))
    run.record(meta[bad].entry(bad, status="ok", answer="disagree", load_ns=1))
    assert run._skip(meta) == [bad]


@pytest.mark.parametrize(("code", "relaunch"), [(1, False), (3, True)])
def test_codspeed_rerun_only_after_a_lost_session(tmp_path: Path, code: int, relaunch: bool) -> None:
    """Exit code 1 with an open id is a harness bug in a session that ended normally: CodSpeed kept its
    measurements, so nothing is relaunched. Another code (a crash) lost them: the passed ids run again."""
    run, meta = _codspeed_run(tmp_path)
    pending = _id("snb_is4")
    for bench_id, m in meta.items():
        if bench_id != pending:
            run.record(m.entry(bench_id, status="ok", answer="agree"))
    attempt = supervisor._Attempt(code=code, pending=pending)
    assert run._settle(attempt, meta)
    assert (run.recorded[pending].status, run.recorded[pending].error) == ("failed", f"pytest exited with code {code}")
    assert run._again(attempt, meta) is relaunch


def test_codspeed_rerun_remeasures_passed_cases(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a hang kills pytest (losing its CodSpeed results), the relaunch runs the passed cases again."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "wrong:snb_is3,hang:snb_is4")
    launches: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(supervisor, "_launch", _without_codspeed(launches))
    resources_off(monkeypatch)
    out = tmp_path / "out"
    results = supervisor.run_codspeed("push", ["fake"], out, workdir=tmp_path / "work", case_timeout_s=HANG_TIMEOUT_S)

    ids = [e.id for e in results.entries]
    assert len(ids) == len(set(ids))
    entries: dict[str, Entry] = {e.id: e for e in results.entries}
    assert set(entries) == _expected_ids("fake")
    hung = entries.pop(_id("snb_is4"))
    assert hung.status == "timeout"
    assert f"no output for {HANG_TIMEOUT_S:g} s" in (hung.error or "")
    wrong = entries.pop(_id("snb_is3"))
    assert (wrong.status, wrong.answer) == ("ok", "disagree")
    assert all(e.status == "ok" for e in entries.values()), [e for e in entries.values() if e.status != "ok"]

    assert len(launches) == 2
    first = launches[0][0]
    assert "--deselect" not in first
    rerun = launches[1][0]
    deselected = {rerun[i + 1] for i, a in enumerate(rerun) if a == "--deselect"}
    assert deselected == {_id("snb_is3"), _id("snb_is4")}  # passed cases and loads run again

    lines = [json.loads(line) for line in (out / "raw" / "fake-1" / "pytest.jsonl").read_text("utf-8").splitlines()]
    runs = [line["id"] for line in lines if line["kind"] == "entry"]
    assert runs.count(_id("snb_is1")) == 2
    assert runs.count(catalog.load_id("snb", "fake", "small", 1)) == 2
    assert runs.count(_id("snb_is3")) == 1
    assert runs.count(_id("snb_is5")) == 1


def test_codspeed_rerun_when_the_last_case_hangs(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last case hangs, so nothing is left unrecorded; the passed cases still get a relaunch to be measured."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "hang:snb_rdf_is6")
    launches: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(supervisor, "_launch", _without_codspeed(launches))
    resources_off(monkeypatch)
    out = tmp_path / "out"
    results = supervisor.run_codspeed("push", ["fake"], out, workdir=tmp_path / "work", case_timeout_s=HANG_TIMEOUT_S)

    lines = [json.loads(line) for line in (out / "raw" / "fake-1" / "pytest.jsonl").read_text("utf-8").splitlines()]
    hung = _id("snb_rdf_is6")  # the last case: the RDF store is loaded and run last
    second = next(i for i, line in enumerate(lines) if i > 0 and line["kind"] == "database")
    first_starts = [line["id"] for line in lines[:second] if line["kind"] == "start"]
    assert first_starts[-1] == hung  # the last case of the first process
    assert set(first_starts) == _expected_ids("fake")

    entries: dict[str, Entry] = {e.id: e for e in results.entries}
    assert set(entries) == _expected_ids("fake")
    assert entries.pop(hung).status == "timeout"
    assert all(e.status == "ok" for e in entries.values()), [e for e in entries.values() if e.status != "ok"]

    assert len(launches) == 2
    rerun = launches[1][0]
    deselected = {rerun[i + 1] for i, a in enumerate(rerun) if a == "--deselect"}
    assert deselected == {hung}  # the passed cases are not deselected: they run again
    runs = [line["id"] for line in lines if line["kind"] == "entry"]
    assert runs.count(_id("snb_is1")) == 2
    assert runs.count(hung) == 0


def test_codspeed_rerun_retires_a_passed_case_that_hangs(tmp_path: Path) -> None:
    """A passed id that hangs when run again keeps its first entry and is deselected from then on."""
    sel = catalog.selection("push")
    results = Results(
        graph_bench_commit="x",
        grafeo={"version": None, "source": "pypi", "sha": None},
        machine={"cpu": "c", "cores": 1, "os": "o"},
        selection="push",
        mode="codspeed",
        started_at="t",
        databases={},
    )
    run = supervisor._Run(sel, tmp_path, tmp_path, 5.0, results, supervisor.PYTEST)
    meta = supervisor._expected(sel, "fake", 1)
    passed, marked = _id("snb_is1"), _id("snb_is2")
    run.record(meta[passed].entry(passed, status="ok", answer="agree"))
    run.record(meta[marked].entry(marked, status="failed", error="boom"))
    assert run._skip(meta) == [marked]

    assert run._settle(supervisor._Attempt(hung=True, silence_s=5.0, pending=passed), meta)
    assert run.recorded[passed].status == "ok"
    assert set(run._skip(meta)) == {passed, marked}
    assert len(results.entries) == 2

    worker_run = supervisor._Run(sel, tmp_path, tmp_path, 5.0, results)
    worker_run.record(meta[passed].entry(passed, status="ok", answer="agree"))
    assert worker_run._skip(meta) == [passed]


def test_pytest_argv_budgets_and_deselect() -> None:
    request = supervisor.LaunchRequest(
        db="fake",
        selection="push",
        threads=1,
        workdir=Path("w"),
        out=Path("o.jsonl"),
        case_timeout_s=7.0,
        skip=["a"],
        budgets={"b": 0.5},
    )
    argv = supervisor.pytest_argv(request)
    assert argv[:5] == [sys.executable, "-m", "pytest", "benches", "--codspeed"]

    def after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert after("--deselect") == "a"
    assert after("--gb-budget") == "b=0.5"
    assert after("--case-timeout") == "7.0"
    assert after("--gb-out") == "o.jsonl"
    assert after("--threads") == "1"


def test_codspeed_rejects_unknown_db_before_install(monkeypatch: pytest.MonkeyPatch) -> None:
    def install(path: Path) -> dict[str, str]:
        raise AssertionError("installed before validating")

    monkeypatch.setattr("graph_bench.cli.install_grafeo", install)
    result = CliRunner().invoke(
        app, ["codspeed", "--selection", "push", "--db", "nope", "--grafeo", "x.whl", "--out", "o"]
    )
    assert result.exit_code == 2
    assert "nope" in result.output


def test_codspeed_reports_unreached_loads_as_not_available(push_data_dir: Path, tmp_path: Path) -> None:
    """An RDF-only database: the graph loads are skipped as ⊘ with the model gate's error, its RDF load and RDF
    cases pass, and the ids equal `graph-bench list`."""
    out = tmp_path / "rdf.jsonl"
    proc = _pytest(["--db", "fake_rdf", "--selection", "push", "--gb-out", str(out), "-rs"], _env(push_data_dir))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    assert set(entries) == _expected_ids("fake_rdf")
    for dataset in ("snb", "graphalytics"):
        load = entries[catalog.load_id(dataset, "fake_rdf", "small", 1)]
        assert (load["status"], load["error"]) == ("not_supported", "model property_graph not supported")
        load_ql = entries[catalog.load_ql_id(dataset, "fake_rdf", "small", 1)]
        assert (load_ql["status"], load_ql["error"]) == ("not_supported", "model property_graph not supported")
    startup = entries[catalog.startup_id("fake_rdf", 1)]
    assert (startup["status"], startup["answer"]) == ("ok", "agree")  # an RDF probe on an empty store
    assert "not available: model property_graph not supported" in proc.stdout
    rdf_load = entries[catalog.load_id("snb", "fake_rdf", "small", 1, "rdf")]
    assert (rdf_load["status"], rdf_load["answer"]) == ("ok", "agree")
    for case in catalog.cases("push", ["fake_rdf"]):
        if case.item.model == "rdf":
            assert (entries[case.bench_id]["status"], entries[case.bench_id]["answer"]) == ("ok", "agree")


# Startup, Load (query language) and write cases ---------------------------------------------------------------


@pytest.fixture(scope="module")
def writes_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory for the push selection with the fake write suite (its own cache key)."""
    return tmp_path_factory.mktemp("codspeed-writes")


def test_codspeed_order_startup_first_writes_after_reads(writes_dir: Path) -> None:
    """With a write case in the catalog, the collected order is the startup, then per dataset its load, its reads,
    its writes and its query-language load; the ids still equal `graph-bench list`."""
    env = {**_env(writes_dir), **WRITES}
    proc = _pytest(["--collect-only", "-q", "--db", "fake", "--selection", "push"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    collected = [line for line in proc.stdout.splitlines() if "::" in line]
    code = "from graph_bench.suites import catalog; print(chr(10).join(catalog.all_ids('push', ['fake'])))"
    listed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert sorted(collected) == listed
    assert WRITE_ID in collected
    assert collected[0] == catalog.startup_id("fake", 1)
    snb = [i for i, bench_id in enumerate(collected) if "test_snb_is" in bench_id or "test_snb_ic" in bench_id]
    write = collected.index(WRITE_ID)
    assert collected.index(catalog.load_id("snb", "fake", "small", 1)) < min(snb)
    assert max(snb) < write < collected.index(catalog.load_ql_id("snb", "fake", "small", 1))
    assert collected.index(catalog.load_ql_id("snb", "fake", "small", 1)) < collected.index(
        catalog.load_id("snb", "fake", "small", 1, "rdf")
    )


def _write_session(
    writes_dir: Path, tmp_path: Path, fake: str = "", item: str = "fake_write"
) -> tuple[dict[str, Any], list[str], list[str]]:
    """`pytest benches` on the snb load, IS4 and one fake write case: the entries, the spy's calls and the traced
    phases."""
    trace, calls = tmp_path / "trace.txt", tmp_path / "calls.txt"
    env = {
        **_env(writes_dir, fake),
        **WRITES,
        "GRAPH_BENCH_SPY": str(tmp_path / "spy.txt"),
        "GRAPH_BENCH_SPY_CALLS": str(calls),
        "GRAPH_BENCH_FAKE_TRACE": str(trace),
    }
    out = tmp_path / "gb.jsonl"
    args = ["--db", "fake", "--selection", "push", "-p", "tests.benchmark_spy", "--gb-out", str(out)]
    args += ["-k", f"test_load_snb or test_snb_is4 or test_{item}["]
    proc = _pytest(args, env)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    phases = [line.split()[0] for line in trace.read_text(encoding="utf-8").splitlines()]
    phases = [p for p in phases if p in ("setup", "run", "check", "teardown")]
    collapsed = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
    entries["returncode"] = proc.returncode  # type: ignore[assignment]
    return entries, calls.read_text(encoding="utf-8").splitlines(), collapsed


def test_codspeed_write_case_uses_pedantic_setup(writes_dir: Path, tmp_path: Path) -> None:
    """A write case reaches CodSpeed through `pedantic` with a setup; its read-back and teardown run outside the
    timed target (one round without --codspeed: setup, run, then the read-back after `pedantic` returns)."""
    entries, calls, phases = _write_session(writes_dir, tmp_path)
    assert entries.pop("returncode") == 0
    entry = entries[WRITE_ID]
    assert (entry["status"], entry["answer"], entry["error"]) == ("ok", "agree", None), entry
    rounds = catalog.selection("push").iterations
    assert f"{WRITE_ID} pedantic setup=True rounds={rounds}" in calls
    assert f"{_id('snb_is4')} call" in calls
    assert phases == ["check"] + ["setup", "run", "check", "teardown", "check"] * 2  # preflight, one round


@pytest.mark.parametrize(
    ("fake", "status", "answer", "start"),
    [
        ("wrong_later:fake_write", "ok", "disagree", "read-back after timed run 1 differs: after the run: "),
        ("leak:fake_write", "failed", "agree", "the case left the graph changed: "),
        ("wrong:fake_write", "ok", "disagree", "binding 0: 2 rows != 1 rows"),
    ],
)
def test_codspeed_write_case_markers(
    writes_dir: Path, tmp_path: Path, fake: str, status: str, answer: str, start: str
) -> None:
    entries, calls, _ = _write_session(writes_dir, tmp_path, fake)
    assert entries.pop("returncode") == 1
    entry = entries[WRITE_ID]
    assert (entry["status"], entry["answer"]) == (status, answer), entry
    assert (entry["detail"] if answer == "disagree" else entry["error"]).startswith(start)
    assert entries[_id("snb_is4")]["answer"] == "agree"
    pedantic = [call for call in calls if call.startswith(WRITE_ID)]
    assert len(pedantic) == (0 if fake.startswith("wrong:") else 1)  # a wrong preflight never reaches CodSpeed


def test_codspeed_startup_and_load_ql_outcomes(push_data_dir: Path, tmp_path: Path) -> None:
    """A database without `open_empty` skips its startup as not available; a failing query-language load fails
    its own test only; a wrong probe answer is a wrong answer, never benchmarked."""
    junit = tmp_path / "junit.xml"
    out = tmp_path / "gb.jsonl"
    env = {**_env(push_data_dir), "GRAPH_BENCH_FAKE_NOSTART": "crash_soft:load_ql_snb"}
    env["GRAPH_BENCH_SPY"] = str(tmp_path / "spy.txt")
    args = ["--db", "fake_nostart", "--selection", "push", "--gb-out", str(out), f"--junitxml={junit}"]
    args += ["-p", "tests.benchmark_spy", "-k", "test_startup or test_load_ql or test_snb_is1"]
    proc = _pytest(args, env)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    outcomes = _junit_outcomes(junit)
    assert outcomes["test_startup[fake_nostart-empty]"] == ("skipped", "not available: startup")
    assert outcomes["test_load_ql_snb[fake_nostart-small]"] == ("failure", "error: RuntimeError: soft crash")
    assert outcomes["test_load_ql_graphalytics[fake_nostart-small]"] == ("passed", "")
    assert outcomes["test_snb_is1[fake_nostart-small]"] == ("passed", "")
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = {line["id"]: line for line in lines if line["kind"] == "entry"}
    ql = entries[catalog.load_ql_id("graphalytics", "fake_nostart", "small", 1)]
    assert (ql["status"], ql["answer"]) == ("ok", "agree") and ql["load_ns"] > 0
    spied = (tmp_path / "spy.txt").read_text(encoding="utf-8").split()
    assert catalog.startup_id("fake_nostart", 1) not in spied

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    env = {**_env(push_data_dir, "wrong:startup"), "GRAPH_BENCH_SPY": str(wrong / "spy.txt")}
    args = ["--db", "fake", "--selection", "push", "--gb-out", str(wrong / "gb.jsonl"), f"--junitxml={wrong / 'j.xml'}"]
    args += ["-p", "tests.benchmark_spy", "-k", "test_startup"]
    assert _pytest(args, env).returncode == 1
    kind, reason = _junit_outcomes(wrong / "j.xml")["test_startup[fake-empty]"]
    assert (kind, reason) == ("failure", "wrong answer: child 1: probe returned [[1]], expected [[0]]")
    assert not (wrong / "spy.txt").exists()


def test_codspeed_write_case_checks_the_setup_state(writes_dir: Path, tmp_path: Path) -> None:
    """CodSpeed's pedantic setup checks the `ready` reference too: a setup that misses it after the preflight
    fails the case."""
    write_id = "benches/test_batch_writes.py::test_fake_setup_write[fake-small]"
    entries, calls, _ = _write_session(writes_dir, tmp_path, item="fake_setup_write")
    assert entries.pop("returncode") == 0
    assert (entries[write_id]["status"], entries[write_id]["answer"]) == ("ok", "agree")
    later = tmp_path / "later"
    later.mkdir()
    entries, calls, _ = _write_session(writes_dir, later, "nosetup_later:fake_setup_write", item="fake_setup_write")
    assert entries.pop("returncode") == 1
    entry = entries[write_id]
    assert entry["status"] == "failed"
    assert entry["error"] == "setup did not reach its state: binding 0, row 0, col 1: 'loaded' != 'ready'"
    assert any(call.startswith(write_id + " pedantic setup=True") for call in calls)


class _Rounds:
    """A stand-in for the benchmark fixture whose `pedantic` runs every round, as CodSpeed does."""

    def __init__(self) -> None:
        self.targets = 0

    def pedantic(self, target: Any, setup: Any = None, rounds: int = 1, **_: Any) -> None:
        for _ in range(rounds):
            args, kwargs = setup() if setup is not None else ((), {})
            self.targets += 1
            target(*args, **kwargs)


def test_codspeed_startup_series_stops_at_the_first_failing_child(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With CodSpeed's rounds, a failing child after the preflight ends the series, as in bench mode."""
    from benches._cases import Bench, StartupCase, run_startup

    monkeypatch.setenv("GRAPH_BENCH_FAKE", "crash_soft:startup_later")
    out = tmp_path / "gb.jsonl"
    bench = Bench("fake", catalog.selection("push"), (1,), out, tmp_path / "work", {}, 60.0)
    benchmark = _Rounds()
    with pytest.raises(pytest.fail.Exception, match="error: RuntimeError: soft crash"):
        run_startup(benchmark, bench, StartupCase("fake", 1))
    bench.close()
    assert benchmark.targets == 1  # the second round never starts
    entry = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()][-1]
    assert (entry["status"], entry["error"], entry["timing"]) == ("failed", "RuntimeError: soft crash", None)
    assert entry["preflight_ns"] > 0
