"""The benchmark tests: one test function per catalog item, plus one load test per dataset.

Each test writes the worker's JSON lines (see `graph_bench.runner.worker`) through `Bench.emit` and maps the
preflight's outcome to pytest, always after appending the entry line:

- not available (model not supported, `NotSupported`): `pytest.skip("not available: ...")`;
- an exception: `pytest.fail("error: ...")`;
- over the time budget: `pytest.fail("too slow: ...")`;
- a wrong answer: `pytest.fail("wrong answer: ...")`;
- otherwise `benchmark(run)`, the only path that reaches CodSpeed.
"""

import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NoReturn

import pytest

from graph_bench.databases import get_database
from graph_bench.databases.base import Database
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.runner.worker import (
    Emit,
    LoadCheck,
    Loaded,
    Preflight,
    case_entry,
    database_line,
    emitter,
    fresh_instance,
    load,
    load_entry,
    message,
    prepare_environment,
    remove,
    start_case,
)
from graph_bench.suites import Algorithm, Query, catalog
from graph_bench.suites.catalog import Case, Selection

# (dataset, scale, threads)
DataKey = tuple[str, str, int | None]


@dataclass(frozen=True)
class LoadCase:
    """One dataset loaded by one database at one scale and thread setting."""

    dataset: str
    db: str
    scale: str
    threads: int | None

    @property
    def bench_id(self) -> str:
        return catalog.load_id(self.dataset, self.db, self.scale, self.threads)


class LoadFailed(Exception):
    """The dataset a case needs could not be loaded."""


class EnvironmentFailed(Exception):
    """The database's `prepare_environment` failed; the message is the worker's load error text."""


def _load_error(exc: Exception) -> str:
    """A failed load's error text, as the worker writes it."""
    return str(exc) if isinstance(exc, EnvironmentFailed) else message(exc)


class Bench:
    """The session state: options, the JSON lines file and the one loaded dataset instance."""

    def __init__(
        self,
        db: str,
        selection: Selection,
        threads: tuple[int | None, ...],
        out: Path | None,
        workdir: Path | None,
        budgets: dict[str, float],
        case_timeout_s: float,
    ) -> None:
        self.db = db
        self.selection = selection
        self.threads = threads
        self.out = out
        self.budgets = budgets
        self.case_timeout_s = case_timeout_s
        self._workdir = workdir
        self._own_workdir: Path | None = None
        self._fh: IO[str] | None = None
        self._emit: Emit | None = None
        self._db_cls: type[Database] | None = None
        self._datasets: dict[tuple[str, str], Dataset] | None = None
        self._loaded: tuple[DataKey, Loaded] | None = None
        self._failed: dict[DataKey, str] = {}
        self._setup: tuple[str | None] | None = None  # (error,) once `prepare_environment` ran

    # The plan, as `graph-bench list` prints it.

    def cases(self, item: Query | Algorithm | None = None) -> list[Case]:
        found = catalog.cases(self.selection.name, [self.db])
        return [c for c in found if c.threads in self.threads and (item is None or c.item.id == item.id)]

    def loads(self, dataset: str | None = None) -> list[LoadCase]:
        ids = set(catalog.load_ids(self.selection.name, [self.db]))
        datasets = sorted({i.dataset for i in catalog.ITEMS}) if dataset is None else [dataset]
        found = [
            LoadCase(name, self.db, scale, threads)
            for name in datasets
            for scale in self.selection.scales
            for threads in self.threads
        ]
        return [lc for lc in found if lc.bench_id in ids]

    def expected_ids(self) -> set[str]:
        return {c.bench_id for c in self.cases()} | {lc.bench_id for lc in self.loads()}

    # Output.

    def emit(self, line: dict[str, Any]) -> None:
        """Append one JSON line to `--gb-out`; without it, do nothing."""
        if self.out is None:
            return
        if self._emit is None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.out.open("a", encoding="utf-8", newline="\n")
            self._emit = emitter(self._fh)
        self._emit(line)

    def emit_database(self) -> None:
        self.emit(database_line(self.db_cls, self.workdir, self.threads[0] if self.threads else 1))

    # Databases and datasets.

    @property
    def db_cls(self) -> type[Database]:
        if self._db_cls is None:
            self._db_cls = get_database(self.db)
        return self._db_cls

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            self._own_workdir = self._workdir = Path(tempfile.mkdtemp(prefix="graph-bench-"))
        self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    @property
    def base(self) -> Path:
        """The selection's cache directory, which holds the reference answers."""
        return cache.data_dir() / cache.dataset_key(self.selection.name)

    def dataset(self, key: DataKey) -> Dataset:
        if self._datasets is None:
            self._datasets = cache.prepare(self.selection.name)
        return self._datasets[(key[0], key[1])]

    def prepare(self) -> None:
        """Run the database's `prepare_environment` once per session, before the first load; re-raise its error."""
        if self._setup is None:
            self._setup = (prepare_environment(self.db_cls),)
        if self._setup[0] is not None:
            raise EnvironmentFailed(self._setup[0])

    def budget(self, bench_id: str) -> float:
        return self.budgets.get(bench_id, self.case_timeout_s)

    def database(self, key: DataKey) -> Database:
        """The instance holding dataset `key`, loaded once (untimed) unless the load test left it here."""
        if key in self._failed:
            raise LoadFailed(self._failed[key])
        if self._loaded is not None and self._loaded[0] == key:
            return self._loaded[1].db
        self.release()
        progress = {"kind": "progress", "id": None}
        try:
            self.prepare()
            loaded = load(self.db_cls, self.dataset(key), self.workdir, key[2], 1, lambda: self.emit(progress))
        except Exception as exc:
            self._failed[key] = _load_error(exc)
            raise LoadFailed(self._failed[key]) from exc
        self._loaded = (key, loaded)
        return loaded.db

    def adopt(self, key: DataKey, db: Database, db_dir: Path) -> None:
        """Keep the instance the load test loaded last for the cases of its dataset."""
        self.release()
        self._loaded = (key, Loaded(db, db_dir, 0, LoadCheck()))

    def load_failed(self, key: DataKey, error: str) -> None:
        self._failed[key] = error

    def release(self) -> None:
        if self._loaded is not None:
            _, loaded = self._loaded
            self._loaded = None
            loaded.db.close()
            remove(loaded.workdir)

    def close(self) -> None:
        self.release()
        if self._fh is not None:
            self._fh.close()
            self._fh = self._emit = None
        remove(self._own_workdir)


def _fail(prefix: str, text: str | None) -> NoReturn:
    text = text or ""
    pytest.fail(text if text.startswith(prefix) else prefix + text, pytrace=False)


def _outcome(pf: Preflight) -> NoReturn:
    """Raise the pytest outcome of a preflight that may not be timed."""
    if pf.status == "not_supported":
        pytest.skip(f"not available: {pf.error}")
    if pf.status == "timeout":
        _fail("too slow: ", pf.error)
    if pf.status == "failed":
        _fail("error: ", pf.error)
    if pf.answer == "disagree":
        _fail("wrong answer: ", pf.detail)
    _fail("error: ", f"answer {pf.answer}")


def run_case(benchmark: Any, bench: Bench, case: Case) -> None:
    """Preflight the case; benchmark it only when it completed in budget with a correct answer."""
    model = case.item.model
    if model not in bench.db_cls.models:
        error = f"model {model} not supported"
        bench.emit(case_entry(case, status="not_supported", error=error))
        pytest.skip(f"not available: {error}")
    try:
        db = bench.database((case.item.dataset, case.scale, case.threads))
    except LoadFailed as exc:
        bench.emit({"kind": "start", "id": case.bench_id})
        bench.emit(case_entry(case, status="failed", error=f"load failed: {exc}"))
        _fail("error: ", f"load failed: {exc}")
    pf, entry = start_case(case, db, bench.base, bench.out, bench.budget(case.bench_id), bench.emit)
    if not pf.passed:
        bench.emit(entry)
        _outcome(pf)
    try:
        benchmark(pf.run)
    except Exception as exc:
        entry["status"], entry["error"] = "failed", message(exc)
        bench.emit(entry)
        _fail("error: ", entry["error"])
    bench.emit(entry)


def run_load(benchmark: Any, bench: Bench, lc: LoadCase) -> None:
    """Load the dataset `load_iterations` times, each into a fresh instance, and keep the last instance.

    `load_ns` is the median of the loads this test ran, as the worker reports it; CodSpeed times them too. Each
    instance's node and edge counts are checked outside the timed target (in the next round's setup, and after the
    last round), so a ≠ or a counting error is known only after CodSpeed measured the load.
    """
    key = (lc.dataset, lc.scale, lc.threads)
    bench.emit({"kind": "start", "id": lc.bench_id})
    bench.release()
    progress = {"kind": "progress", "id": lc.bench_id}
    instances: list[tuple[Database, Path]] = []
    times: list[int] = []
    datasets: list[Dataset] = []
    check = LoadCheck()

    def drop() -> None:
        while instances:
            db, db_dir = instances.pop()
            db.close()
            remove(db_dir)

    def setup() -> tuple[tuple[Database], dict[str, Any]]:
        if instances:
            check.add(instances[-1][0], datasets[0])
            bench.emit(progress)
            drop()
        instances.append(fresh_instance(bench.db_cls, datasets[0], bench.workdir, lc.threads, len(times)))
        return (instances[-1][0],), {}

    def target(db: Database) -> None:
        start = time.perf_counter_ns()
        db.load(datasets[0])
        times.append(time.perf_counter_ns() - start)

    try:
        datasets.append(bench.dataset(key))
        bench.prepare()
        benchmark.pedantic(target, setup=setup, rounds=bench.selection.load_iterations)
    except Exception as exc:
        drop()
        error = _load_error(exc)
        bench.load_failed(key, error)
        bench.emit({"kind": "entry", "id": lc.bench_id, "status": "failed", "error": error})
        _fail("error: ", error)
    check.add(instances[-1][0], datasets[0])
    bench.emit(progress)
    db, db_dir = instances.pop()
    bench.adopt(key, db, db_dir)
    entry = load_entry(lc.bench_id, int(statistics.median(times)), check)
    bench.emit(entry)
    if entry["status"] == "failed":
        _fail("error: ", entry["error"])
    if entry["answer"] == "disagree":
        _fail("wrong answer: ", entry["detail"])


def make_test(item: Query | Algorithm) -> Callable[..., None]:
    """The test function of one catalog item, parametrized over its cases by `conftest.pytest_generate_tests`."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: Case) -> None:
        run_case(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = f"test_{item.id}"
    test.gb_item = item  # type: ignore[attr-defined]
    return test


def make_load_test(dataset: str) -> Callable[..., None]:
    """The load test of one dataset, parametrized over its scales and thread settings."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: LoadCase) -> None:
        run_load(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = f"test_load_{dataset}"
    test.gb_dataset = dataset  # type: ignore[attr-defined]
    return test
