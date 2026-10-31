"""The Startup row: fresh child processes open an empty database and answer the probe (`runner.startup`)."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from graph_bench.runner.startup import STARTUP_TIMEOUT_S, run_startups, startup_entry
from graph_bench.suites import startup
from graph_bench.types import STANDARD_DIALECT

ROOT = Path(__file__).resolve().parent.parent
EXTRA = (
    "fake=tests.fakes:FakeDatabase,fake_rdf=tests.fakes:FakeRdfDatabase,fake_sql=tests.fakes:FakeSqlDatabase,"
    "fake_nostart=tests.fakes:FakeNoStartup"
)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The children inherit this environment: the fake database names and no behaviour flags."""
    monkeypatch.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "")
    monkeypatch.delenv("GRAPH_BENCH_FAKE_TRACE", raising=False)
    yield


def _trace(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines()]


def test_startup_probe_texts_and_answers() -> None:
    assert startup.PROBE["cypher"] == "MATCH (n) RETURN count(n) AS nodes"
    assert startup.PROBE["sql_pgq"] == "SELECT 1 AS one"
    assert startup.PROBE["sparql"] == "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"
    assert startup.PROBE["helix"] == "startup_probe"
    assert startup.EXPECTED == {"property_graph": [[0]], "sql_pgq": [[1]], "rdf": [[0]]}
    assert startup.primary_model(frozenset({"rdf", "sql_pgq", "property_graph"})) == "property_graph"
    assert startup.primary_model(frozenset({"rdf", "sql_pgq"})) == "sql_pgq"
    assert startup.primary_model(frozenset({"rdf"})) == "rdf"
    with pytest.raises(ValueError):
        startup.primary_model(frozenset())
    assert startup.probe_text("ladybug", "property_graph") == startup.PROBE["cypher"]
    assert startup.probe_text("duckpgq", "sql_pgq") == startup.PROBE["sql_pgq"]
    assert set(STANDARD_DIALECT.values()) <= set(startup.PROBE)


def test_startup_uses_fresh_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each run is its own process in its own fresh directory, which is removed after it; none provisions."""
    trace = tmp_path / "trace.txt"
    monkeypatch.setenv("GRAPH_BENCH_FAKE_TRACE", str(trace))
    work = tmp_path / "work"
    progress: list[int] = []
    outcome = run_startups("fake", 1, 3, work, STARTUP_TIMEOUT_S, progress=lambda: progress.append(1))
    assert (outcome.status, outcome.answer, outcome.error, outcome.detail) == ("ok", "agree", None, None)
    assert len(outcome.times_ns) == 3 and all(t > 0 for t in outcome.times_ns)
    assert progress == [1, 1, 1]
    calls = _trace(trace)
    opens = [c for c in calls if c[0] == "open_empty"]
    pids = {c[1] for c in opens}
    assert len(pids) == 3
    assert str(os.getpid()) not in pids
    assert [c[2] for c in opens] == [str(work / f"startup-{i}") for i in range(3)]
    assert [c[0] for c in calls] == ["open_empty"] * 3  # the children never provision nor prepare
    assert not any(work.glob("startup-*"))


def test_startup_entry_has_the_children_timing(tmp_path: Path) -> None:
    outcome = run_startups("fake", 1, 2, tmp_path, STARTUP_TIMEOUT_S)
    entry = startup_entry("benches/test_startup.py::test_startup[fake-empty]", outcome)
    assert entry["kind"] == "entry"
    assert (entry["status"], entry["answer"], entry["reference"]) == ("ok", "agree", "python")
    assert entry["timing"]["iterations"] == 2
    assert entry["timing"]["min_ns"] == min(outcome.times_ns)
    assert entry["preflight_ns"] == outcome.times_ns[0]


@pytest.mark.parametrize("db", ["fake_sql", "fake_rdf"])
def test_startup_probes_the_primary_model(db: str, tmp_path: Path) -> None:
    """An SQL/PGQ-only database answers `SELECT 1`, an RDF-only one counts triples."""
    outcome = run_startups(db, 1, 1, tmp_path, STARTUP_TIMEOUT_S)
    assert (outcome.status, outcome.answer) == ("ok", "agree"), outcome


def test_startup_wrong_probe_is_disagree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong probe answer in the first child is ≠, and no further child runs (it is the preflight)."""
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "wrong:startup")
    outcome = run_startups("fake", 1, 3, tmp_path, STARTUP_TIMEOUT_S)
    assert (outcome.status, outcome.answer) == ("ok", "disagree")
    assert outcome.detail == "child 1: probe returned [[1]], expected [[0]]"
    assert len(outcome.times_ns) == 1
    entry = startup_entry("x", outcome)
    assert entry["timing"] is None
    assert entry["preflight_ns"] == outcome.times_ns[0]


def test_startup_without_open_empty_is_not_available(tmp_path: Path) -> None:
    outcome = run_startups("fake_nostart", 1, 3, tmp_path, STARTUP_TIMEOUT_S)
    assert (outcome.status, outcome.answer, outcome.error) == ("not_supported", "unchecked", "startup")
    assert outcome.times_ns == []


@pytest.mark.parametrize(
    ("flags", "status", "error"),
    [
        ("crash_soft:startup", "failed", "RuntimeError: soft crash"),
        ("crash:startup", "failed", "startup child 1 exited with code 3"),
    ],
)
def test_startup_child_failures(
    flags: str, status: str, error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRAPH_BENCH_FAKE", flags)
    outcome = run_startups("fake", 1, 3, tmp_path, STARTUP_TIMEOUT_S)
    assert (outcome.status, outcome.answer) == (status, "unchecked")
    assert (outcome.error or "").startswith(error), outcome.error


def test_startup_child_silent_too_long_is_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "hang:startup")
    outcome = run_startups("fake", 1, 3, tmp_path, 2.0)
    assert (outcome.status, outcome.error) == ("timeout", "startup child 1 gave no answer within 2 s")


def test_startup_child_prints_one_json_line(tmp_path: Path) -> None:
    argv = [sys.executable, "-m", "graph_bench.runner.startup", "--db", "fake", "--threads", "all"]
    argv += ["--workdir", str(tmp_path)]
    proc = subprocess.run(argv, cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=120, check=True)
    doc = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(doc) == {"elapsed_ns", "rows"}
    assert doc["rows"] == [[0]]
    assert doc["elapsed_ns"] > 0
