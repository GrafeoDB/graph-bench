"""The Memory and Disk records: the resources child, the supervisor's sampling and the records it writes."""

import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_bench import databases
from graph_bench.results import Resource, Results
from graph_bench.runner import resources, supervisor
from graph_bench.runner.resources import run_resource, window
from graph_bench.runner.sampler import Sample
from graph_bench.suites import catalog

MIB = 1 << 20
EXTRA = "fake=tests.fakes:FakeDatabase,fake_rdf=tests.fakes:FakeRdfDatabase,ghost=tests.no_such_module:Ghost"


@pytest.fixture(scope="module", autouse=True)
def _extra_databases() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
        yield


def _env(data_dir: Path, fake: str = "") -> dict[str, str]:
    return {**os.environ, "GRAPH_BENCH_DATA_DIR": str(data_dir), "GRAPH_BENCH_FAKE": fake}


def _measure(
    data_dir: Path,
    tmp_path: Path,
    fake: str = "",
    *,
    dataset: str = "snb",
    store: Any = "graph",
    timeout_s: float = 120,
) -> Resource:
    return run_resource(
        "fake",
        "push",
        dataset,
        store,
        "small",
        1,
        workdir=tmp_path / "work",
        raw=tmp_path / "raw",
        env=_env(data_dir, fake),
        timeout_s=timeout_s,
    )


def test_in_memory_disk_is_none(push_data_dir: Path, tmp_path: Path) -> None:
    """A load of an in-memory database: ok, its count check agrees, consistent bytes, no disk size."""
    r = _measure(push_data_dir, tmp_path)
    assert (r.id, r.database, r.dataset, r.store, r.scale, r.threads) == (
        "resources::load_snb[fake-small]",
        "fake",
        "snb",
        "graph",
        "small",
        1,
    )
    assert (r.status, r.error, r.answer, r.detail) == ("ok", None, "agree", None), r
    assert r.storage == "in-memory"
    assert r.disk_bytes is None
    assert r.baseline_bytes is not None and r.after_bytes is not None and r.peak_bytes is not None
    assert r.baseline_bytes > 0
    assert r.footprint_bytes == r.after_bytes - r.baseline_bytes
    assert r.peak_bytes >= max(r.baseline_bytes, r.after_bytes)
    assert r.samples is not None and r.samples >= 0
    assert r.sample_interval_ms == 10.0
    assert r.processes is not None and r.processes >= 1
    assert not (tmp_path / "work").exists()  # the instance's directory is removed


def test_raw_lines_of_the_child(push_data_dir: Path, tmp_path: Path) -> None:
    """The child's lines stay under `raw`: `before` and `after` readings (`after` with the tree's process names),
    then the resource line."""
    _measure(push_data_dir, tmp_path)
    lines = [json.loads(line) for line in (tmp_path / "raw" / "load_snb-small.jsonl").read_text().splitlines()]
    assert [(line["kind"], line.get("phase")) for line in lines] == [
        ("memory", "before"),
        ("memory", "after"),
        ("resource", None),
    ]
    before, after, _ = lines
    assert before["t_ns"] <= after["t_ns"]
    assert after["processes"] == len(after["names"]) >= 1


def test_fake_allocation_shows_in_footprint(push_data_dir: Path, tmp_path: Path) -> None:
    """A load that keeps 200 MiB: the footprint and the peak above the baseline both show it."""
    r = _measure(push_data_dir, tmp_path, "memory:load_snb=200")
    assert (r.status, r.answer) == ("ok", "agree"), r
    assert r.footprint_bytes is not None and r.footprint_bytes >= 190 * MIB, r
    assert r.peak_bytes is not None and r.baseline_bytes is not None
    assert r.peak_bytes - r.baseline_bytes >= 190 * MIB


def test_disk_size_of_store_paths(push_data_dir: Path, tmp_path: Path) -> None:
    """The store's files after `close()`, to the byte."""
    r = _measure(push_data_dir, tmp_path, "disk:load_snb=123457")
    assert (r.status, r.answer, r.storage) == ("ok", "agree", "on-disk"), r
    assert r.disk_bytes == 123457


def test_resource_count_mismatch_is_disagree(push_data_dir: Path, tmp_path: Path) -> None:
    r = _measure(push_data_dir, tmp_path, "short:count_snb")
    assert (r.status, r.answer) == ("ok", "disagree"), r
    assert r.detail is not None and r.detail.startswith("loaded ")
    assert r.footprint_bytes is not None  # measured all the same; the report shows ≠


def test_rdf_store_is_its_own_record(push_data_dir: Path, tmp_path: Path) -> None:
    r = _measure(push_data_dir, tmp_path, dataset="snb", store="rdf")
    assert (r.id, r.status, r.answer) == ("resources::load_snb_rdf[fake-small]", "ok", "agree"), r


@pytest.mark.parametrize(
    ("fake", "status", "error"),
    [
        ("crash_soft:load_snb", "failed", "RuntimeError: soft crash"),
        ("crash_soft:environment", "failed", "preparing the environment: RuntimeError: soft crash"),
        ("crash_soft:count_snb", "failed", "counting the loaded data: RuntimeError: soft crash"),
        ("crash:load_snb", "failed", "resources child exited with code 3"),
        ("hang:load_snb", "timeout", "no output for 6 s"),
    ],
)
def test_resource_failures(push_data_dir: Path, tmp_path: Path, fake: str, status: str, error: str) -> None:
    r = _measure(push_data_dir, tmp_path, fake, timeout_s=6)
    assert (r.status, r.error) == (status, error), r
    assert r.footprint_bytes is None and r.peak_bytes is None and r.disk_bytes is None


def test_unsupported_count_leaves_the_record_unchecked(push_data_dir: Path, tmp_path: Path) -> None:
    r = _measure(push_data_dir, tmp_path, "unsupported:count_snb")
    assert (r.status, r.answer) == ("ok", "unchecked"), r


def _s(t: int, rss: int, n: int = 1) -> Sample:
    return Sample(t, rss, n)


def test_peak_window_uses_the_shared_clock() -> None:
    """Only samples from `before` to `after` count; the peak is never below `before` or `after`."""
    before, after = _s(100, 50), _s(200, 60)
    w = window(before, after, [_s(50, 999, 5), _s(100, 70, 2), _s(150, 80, 3), _s(200, 65), _s(201, 999, 9)])
    assert (w.peak_bytes, w.samples, w.processes) == (80, 3, 3)
    assert window(before, after, []) == resources.Window(60, 0, 1)
    assert window(_s(100, 90, 2), after, [_s(150, 10)]).peak_bytes == 90


# The supervisor ---------------------------------------------------------------------------------------------


def _all_records(db: str, scale: str = "small") -> set[str]:
    return {catalog.resource_id(d, db, scale, 1, st) for d, st in catalog.all_load_keys("push")}


@pytest.fixture(scope="module")
def supervised(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Results, list[tuple[str, str]]]:
    """run_bench over fake, fake_rdf and ghost (not installed), the workers replaced by a recorder: only the
    resource stage runs."""
    root = tmp_path_factory.mktemp("resources")
    calls: list[tuple[str, str]] = []
    real = supervisor.run_resource

    def worker(self: Any, db: str, *args: Any) -> None:
        calls.append(("worker", db))

    def resource(db: str, *args: Any, **kwargs: Any) -> Resource:
        calls.append(("resource", db))
        return real(db, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
        mp.setenv("GRAPH_BENCH_FAKE", "")
        mp.chdir(root)
        mp.setitem(databases.MODELS, "ghost", frozenset({"property_graph"}))
        mp.setattr(supervisor._Run, "_worker", worker)
        mp.setattr(supervisor, "run_resource", resource)
        results = supervisor.run_bench(
            "push", ["fake", "fake_rdf", "ghost"], root / "out", workdir=root / "work", case_timeout_s=60
        )
    return results, calls


def test_resource_record_per_load(supervised: tuple[Results, list[tuple[str, str]]]) -> None:
    """One record per (dataset, store, scale) of the selection and database: measured where the database's
    models reach the load, ⊘ elsewhere, ✗ for a database that is not installed."""
    results, _ = supervised
    by_db: dict[str, dict[str, Resource]] = {}
    for r in results.resources:
        by_db.setdefault(r.database, {})[r.id] = r
    for db in ("fake", "fake_rdf", "ghost"):
        assert set(by_db[db]) == _all_records(db)
    for r in by_db["fake"].values():
        assert (r.status, r.answer) == ("ok", "agree"), r
    for r in by_db["fake_rdf"].values():
        if r.store == "rdf":
            assert (r.status, r.answer) == ("ok", "agree"), r
        else:
            assert (r.status, r.error) == ("not_supported", "model property_graph not supported"), r
    for r in by_db["ghost"].values():
        if r.store == "rdf":
            assert (r.status, r.error) == ("not_supported", "model rdf not supported"), r
        else:
            assert r.status == "failed" and r.error is not None and r.error.startswith("not installed: "), r
    assert [r.id for r in results.resources] == sorted(r.id for r in results.resources)


def test_resources_run_after_the_worker(supervised: tuple[Results, list[tuple[str, str]]]) -> None:
    _, calls = supervised
    fake = [c for c in calls if c[1] == "fake"]
    assert fake[0] == ("worker", "fake")
    assert fake[1:] == [("resource", "fake")] * len(_all_records("fake"))
    assert ("worker", "ghost") not in calls and ("resource", "ghost") not in calls


def test_machine_records_memory(supervised: tuple[Results, list[tuple[str, str]]]) -> None:
    results, _ = supervised
    assert results.machine.get("memory_bytes", 0) > 0


def test_resources_off_when_selection_disables(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off = dataclasses.replace(catalog.SELECTIONS["push"], resources=False)
    monkeypatch.setitem(catalog.SELECTIONS, "push", off)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(databases.MODELS, "ghost", frozenset({"property_graph"}))
    monkeypatch.setattr(supervisor._Run, "_worker", lambda self, *args: None)
    results = supervisor.run_bench("push", ["fake", "ghost"], tmp_path / "out", workdir=tmp_path / "work")
    assert results.resources == []
    assert Results.read(tmp_path / "out" / "results.json").resources == []


def test_codspeed_mode_records_resources(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resource stage runs after the pytest process too; CodSpeed never sees it."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor._Run, "_worker", lambda self, *args: None)
    results = supervisor.run_codspeed("push", ["fake_rdf"], tmp_path / "out", workdir=tmp_path / "work")
    assert results.mode == "codspeed"
    assert {r.id for r in results.resources} == _all_records("fake_rdf")
    assert {r.status for r in results.resources} == {"ok", "not_supported"}
