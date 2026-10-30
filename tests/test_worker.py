import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case

ROOT = Path(__file__).resolve().parent.parent
EXTRA = "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2,fake_rdf=tests.fakes:FakeRdfDatabase"

Lines = list[dict[str, Any]]


def _id(item_id: str, db: str = "fake") -> str:
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db, "small", 1).bench_id


def run_worker(
    out: Path, data_dir: Path, *args: str, db: str = "fake", fake: str = ""
) -> tuple[Lines, subprocess.CompletedProcess[str]]:
    """Run the worker on the push selection; return its JSON lines and the finished process."""
    env = {
        **os.environ,
        "GRAPH_BENCH_EXTRA_DATABASES": EXTRA,
        "GRAPH_BENCH_DATA_DIR": str(data_dir),
        "GRAPH_BENCH_FAKE": fake,
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


def test_worker_marks_not_supported(standard: tuple[Lines, Path]) -> None:
    entry = _entries(standard[0])[_id("graphalytics_pr")]
    assert entry["status"] == "not_supported"
    assert entry["timing"] is None
    assert entry["answer"] == "unchecked"
    assert entry["reference"] == "networkx"


def test_worker_marks_unsupported_model(push_data_dir: Path, tmp_path: Path) -> None:
    lines, proc = run_worker(tmp_path / "rdf.jsonl", push_data_dir, db="fake_rdf")
    assert proc.returncode == 0, proc.stderr
    assert lines[0]["kind"] == "database"
    entries = _entries(lines)
    assert set(entries) == {c.bench_id for c in catalog.cases("push", ["fake_rdf"])}
    assert all(e["status"] == "not_supported" for e in entries.values())
    assert not [line for line in lines if line["kind"] == "start"]


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
