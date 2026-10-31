"""The benchmark tests: one test function per catalog item, one load test per dataset and store, one Load (query
language) test per graph-store dataset, and the startup test.

Each test writes the worker's JSON lines (see `graph_bench.runner.worker`) through `Bench.emit` and maps the
preflight's outcome to pytest, always after appending the entry line:

- not available (model not supported, `NotSupported`): `pytest.skip("not available: ...")`;
- an exception: `pytest.fail("error: ...")`;
- over the time budget: `pytest.fail("too slow: ...")`;
- a wrong answer: `pytest.fail("wrong answer: ...")`;
- otherwise `benchmark(run)` (a read case) or `benchmark.pedantic(...)` (a write case, a load, the startup), the
  only paths that reach CodSpeed.

A write case's setup, read-back and teardown are untimed: `benchmark.pedantic(run, setup=...)`, where the setup of
round k+1 first runs the read-back and teardown of round k (and the last round's run after `pedantic` returns).
The startup's first child is its untimed preflight; CodSpeed then times `startup_iterations - 1` further children
(at least one), each the whole child process, interpreter start included, while results.json records the
children's in-process times (`graph_bench.runner.startup`).
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
from graph_bench.databases.base import Database, NotSupported
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.runner.startup import Child, fresh_dir, passed, run_child, spawn_child, startup_entry, summarize
from graph_bench.runner.worker import (
    Emit,
    LoadCheck,
    Loaded,
    ReadbackDiffers,
    WriteRun,
    case_entry,
    database_line,
    emitter,
    fresh_instance,
    load,
    load_entry,
    load_ql_instance,
    message,
    prepare_environment,
    probe_case,
    remove,
    start_case,
    startup_timeout,
    unsupported_load_entry,
    write_guard,
)
from graph_bench.suites import Item, catalog
from graph_bench.suites.catalog import Case, Selection
from graph_bench.suites.startup import primary_model
from graph_bench.types import Store

# (dataset, store, scale, threads)
DataKey = tuple[str, Store, str, int | None]


@dataclass(frozen=True)
class LoadCase:
    """One dataset loaded into one store by one database at one scale and thread setting."""

    dataset: str
    db: str
    scale: str
    threads: int | None
    store: Store = "graph"

    @property
    def bench_id(self) -> str:
        return catalog.load_id(self.dataset, self.db, self.scale, self.threads, self.store)

    @property
    def key(self) -> DataKey:
        return (self.dataset, self.store, self.scale, self.threads)


@dataclass(frozen=True)
class LoadQlCase:
    """One dataset loaded through the database's query language, at one scale and thread setting."""

    dataset: str
    db: str
    scale: str
    threads: int | None

    @property
    def bench_id(self) -> str:
        return catalog.load_ql_id(self.dataset, self.db, self.scale, self.threads)

    @property
    def key(self) -> DataKey:
        return (self.dataset, "graph", self.scale, self.threads)


@dataclass(frozen=True)
class StartupCase:
    """The startup of one database at one thread setting."""

    db: str
    threads: int | None
    scale: str = catalog.STARTUP_SCALE

    @property
    def bench_id(self) -> str:
        return catalog.startup_id(self.db, self.threads)


def case_key(case: Case) -> DataKey:
    """The loaded instance a case runs on: its dataset in the store its model reads."""
    return (case.item.dataset, case.item.store, case.scale, case.threads)


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

    def cases(self, item: Item | None = None) -> list[Case]:
        found = catalog.cases(self.selection.name, [self.db])
        return [c for c in found if c.threads in self.threads and (item is None or c.item.id == item.id)]

    def loads(self, load: tuple[str, Store] | None = None) -> list[LoadCase]:
        """The planned loads, or those of one (dataset, store)."""
        ids = set(catalog.load_ids(self.selection.name, [self.db]))
        keys = load_keys() if load is None else [load]
        found = [
            LoadCase(name, self.db, scale, threads, store)
            for name, store in keys
            for scale in self.selection.scales
            for threads in self.threads
        ]
        return [lc for lc in found if lc.bench_id in ids]

    def load_qls(self, dataset: str | None = None) -> list[LoadQlCase]:
        """The planned query-language loads, or those of one dataset."""
        ids = set(catalog.load_ql_ids(self.selection.name, [self.db]))
        names = load_ql_datasets() if dataset is None else [dataset]
        found = [
            LoadQlCase(name, self.db, scale, threads)
            for name in names
            for scale in self.selection.scales
            for threads in self.threads
        ]
        return [lq for lq in found if lq.bench_id in ids]

    def startups(self) -> list[StartupCase]:
        ids = set(catalog.startup_ids(self.selection.name, [self.db]))
        return [sc for sc in (StartupCase(self.db, threads) for threads in self.threads) if sc.bench_id in ids]

    def expected_ids(self) -> set[str]:
        ids = {c.bench_id for c in self.cases()} | {lc.bench_id for lc in self.loads()}
        return ids | {lq.bench_id for lq in self.load_qls()} | {sc.bench_id for sc in self.startups()}

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
        return self._datasets[(key[0], key[2])]

    def prepare(self) -> None:
        """Run the database's `prepare_environment` once per session, before the first load; re-raise its error."""
        if self._setup is None:
            self._setup = (prepare_environment(self.db_cls),)
        if self._setup[0] is not None:
            raise EnvironmentFailed(self._setup[0])

    def budget(self, bench_id: str) -> float:
        return self.budgets.get(bench_id, self.case_timeout_s)

    def database(self, key: DataKey) -> Database:
        """The instance holding `key` (a dataset in a store), loaded once (untimed) unless the load test left it."""
        if key in self._failed:
            raise LoadFailed(self._failed[key])
        if self._loaded is not None and self._loaded[0] == key:
            return self._loaded[1].db
        self.release()
        progress = {"kind": "progress", "id": None}
        try:
            self.prepare()

            def on_load() -> None:
                self.emit(progress)

            loaded = load(self.db_cls, self.dataset(key), self.workdir, key[3], 1, on_load, store=key[1])
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


def _outcome(entry: dict[str, Any]) -> NoReturn:
    """Raise the pytest outcome of an entry that may not be (or was not) timed."""
    status, error = entry.get("status"), entry.get("error")
    if status == "not_supported":
        pytest.skip(f"not available: {error}")
    if status == "timeout":
        _fail("too slow: ", error)
    if status == "failed":
        _fail("error: ", error)
    if entry.get("answer") == "disagree":
        _fail("wrong answer: ", entry.get("detail"))
    _fail("error: ", f"answer {entry.get('answer')}")


def _ok(entry: dict[str, Any]) -> bool:
    return entry.get("status") == "ok" and entry.get("answer") != "disagree"


def run_case(benchmark: Any, bench: Bench, case: Case) -> None:
    """Preflight the case; benchmark it only when it completed in budget with a correct answer."""
    if not case.item.supported_by(bench.db_cls.models):
        error = f"model {case.item.model} not supported"
        bench.emit(case_entry(case, status="not_supported", error=error))
        pytest.skip(f"not available: {error}")
    try:
        db = bench.database(case_key(case))
    except LoadFailed as exc:
        bench.emit({"kind": "start", "id": case.bench_id})
        bench.emit(case_entry(case, status="failed", error=f"load failed: {exc}"))
        _fail("error: ", f"load failed: {exc}")
    pf, entry = start_case(case, db, bench.base, bench.out, bench.budget(case.bench_id), bench.emit)
    if pf.passed:
        if pf.write is not None:
            _bench_write(benchmark, bench, pf.write, entry)
        else:
            try:
                benchmark(pf.run)
            except Exception as exc:
                entry["status"], entry["error"] = "failed", message(exc)
    write_guard(entry, pf.write)
    bench.emit(entry)
    if not (pf.passed and _ok(entry)):
        _outcome(entry)


def _bench_write(benchmark: Any, bench: Bench, write: WriteRun, entry: dict[str, Any]) -> None:
    """CodSpeed times the write runs; each round's setup is untimed, and so are its read-back and teardown, which
    run in the next round's setup and, for the last round, after `pedantic` returns."""
    pending = [False]
    runs = [0]

    def settle() -> None:
        if pending[0]:
            pending[0] = False
            runs[0] += 1
            problem = write.readback()
            if problem is not None:
                raise ReadbackDiffers(f"read-back after timed run {runs[0]} differs: {problem}")

    def setup() -> tuple[tuple[()], dict[str, Any]]:
        settle()
        write.setup()
        return (), {}

    def target() -> None:
        write.run()
        pending[0] = True

    try:
        benchmark.pedantic(target, setup=setup, rounds=bench.selection.iterations)
        settle()
    except ReadbackDiffers as exc:
        entry["answer"], entry["detail"] = "disagree", str(exc)
    except Exception as exc:
        write.undo()
        entry["status"], entry["error"] = "failed", message(exc)


def run_load(benchmark: Any, bench: Bench, lc: LoadCase) -> None:
    """Load the dataset into the load's store `load_iterations` times, each into a fresh instance; keep the last.

    `load_ns` is the median of the loads this test ran, as the worker reports it; CodSpeed times them too. Each
    instance's node and edge counts are checked outside the timed target (in the next round's setup, and after the
    last round), so a ≠ or a counting error is known only after CodSpeed measured the load.
    """
    if (lc.dataset, lc.store) not in catalog.load_keys(bench.selection.name, bench.db_cls.models):
        model = catalog.all_load_keys(bench.selection.name)[(lc.dataset, lc.store)]
        entry = unsupported_load_entry(lc.bench_id, model)
        bench.emit(entry)
        pytest.skip(f"not available: {entry['error']}")
    key = lc.key
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
        instances.append(fresh_instance(bench.db_cls, datasets[0], bench.workdir, lc.threads, len(times), lc.store))
        return (instances[-1][0],), {}

    def target(db: Database) -> None:
        start = time.perf_counter_ns()
        db.load_store(lc.store, datasets[0])
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
    if not _ok(entry):
        _outcome(entry)


def run_load_ql(benchmark: Any, bench: Bench, lq: LoadQlCase) -> None:
    """Load the dataset through the query language `load_iterations` times, each into a fresh instance that is
    count-checked, content-probed and closed outside the timed target; no case uses these instances.

    It releases the session's loaded instance first, so it runs once its dataset's cases are done (`conftest`
    orders it after them). A failure fails this test alone: the dataset's cases keep their own load.
    """
    if (lq.dataset, "graph") not in catalog.load_keys(bench.selection.name, bench.db_cls.models):
        model = catalog.load_ql_keys(bench.selection.name)[lq.dataset]
        entry = unsupported_load_entry(lq.bench_id, model)
        bench.emit(entry)
        pytest.skip(f"not available: {entry['error']}")
    bench.emit({"kind": "start", "id": lq.bench_id})
    bench.release()
    progress = {"kind": "progress", "id": lq.bench_id}
    instances: list[tuple[Database, Path]] = []  # the current instance, until its round is finished
    times: list[int] = []
    datasets: list[Dataset] = []
    check = LoadCheck()
    probes: list[tuple[Case, Path]] = []

    def finish() -> None:
        """Count, probe and close the last round's instance (its load completed: `target` returned)."""
        while instances:
            db, db_dir = instances.pop()
            try:
                check.add(db, datasets[0])
                for case, reference_file in probes:
                    check.probe(case, db, reference_file)
            finally:
                db.close()
                remove(db_dir)
            bench.emit(progress)

    def setup() -> tuple[tuple[Database], dict[str, Any]]:
        finish()
        instances.append(load_ql_instance(bench.db_cls, datasets[0], bench.workdir, lq.threads, len(times)))
        return (instances[-1][0],), {}

    def target(db: Database) -> None:
        start = time.perf_counter_ns()
        db.load_ql(datasets[0])
        times.append(time.perf_counter_ns() - start)

    try:
        datasets.append(bench.dataset(lq.key))
        bench.prepare()
        probe = probe_case(bench.db_cls, datasets[0], lq.threads)
        if probe is not None:
            probes.append((probe, catalog.reference_path(probe.item, lq.scale, base=bench.base)))
        benchmark.pedantic(target, setup=setup, rounds=bench.selection.load_iterations)
        finish()
    except Exception as exc:
        for db, db_dir in instances:
            db.close()
            remove(db_dir)
        status = "not_supported" if isinstance(exc, NotSupported) else "failed"
        entry = {"kind": "entry", "id": lq.bench_id, "status": status, "error": _load_error(exc)}
        bench.emit(entry)
        _outcome(entry)
    entry = load_entry(lq.bench_id, int(statistics.median(times)), check)
    bench.emit(entry)
    if not _ok(entry):
        _outcome(entry)


def run_startup(benchmark: Any, bench: Bench, sc: StartupCase) -> None:
    """Provision (untimed), run the preflight child, then let CodSpeed time further children, whole processes."""
    bench.emit({"kind": "start", "id": sc.bench_id})
    try:
        bench.db_cls.provision()
    except Exception as exc:
        entry = {"kind": "entry", "id": sc.bench_id, "status": "failed", "error": f"provisioning: {message(exc)}"}
        bench.emit(entry)
        _outcome(entry)
    model = primary_model(bench.db_cls.models)
    timeout_s = startup_timeout(bench.case_timeout_s)
    progress = {"kind": "progress", "id": sc.bench_id}
    children: list[Child] = [run_child(bench.db, sc.threads, bench.workdir / "startup-0", timeout_s)]
    bench.emit(progress)
    if passed(children[0], model):
        dirs: list[Path] = []

        def done() -> None:
            while dirs:
                remove(dirs.pop())
                bench.emit(progress)

        def setup() -> tuple[tuple[Path, int], dict[str, Any]]:
            done()
            dirs.append(fresh_dir(bench.workdir / f"startup-{len(children)}"))
            return (dirs[-1], len(children) + 1), {}

        def target(workdir: Path, index: int) -> None:
            children.append(spawn_child(bench.db, sc.threads, workdir, timeout_s, index=index))

        try:
            benchmark.pedantic(target, setup=setup, rounds=max(bench.selection.startup_iterations - 1, 1))
        finally:
            done()
    entry = startup_entry(sc.bench_id, summarize(model, children))
    bench.emit(entry)
    if entry["status"] != "ok" or entry["answer"] != "agree":
        _outcome(entry)


def make_test(item: Item) -> Callable[..., None]:
    """The test function of one catalog item, parametrized over its cases by `conftest.pytest_generate_tests`."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: Case) -> None:
        run_case(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = f"test_{item.id}"
    test.gb_item = item  # type: ignore[attr-defined]
    return test


def load_keys() -> list[tuple[str, Store]]:
    """Every (dataset, store) a catalog item reads: one load test each."""
    return sorted({(item.dataset, item.store) for item in catalog.ITEMS})


def load_ql_datasets() -> list[str]:
    """Every dataset a catalog item reads as a graph: one Load (query language) test each."""
    return sorted({dataset for dataset, store in load_keys() if store == "graph"})


def make_load_test(dataset: str, store: Store = "graph") -> Callable[..., None]:
    """The load test of one dataset into one store, parametrized over its scales and thread settings."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: LoadCase) -> None:
        run_load(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = f"test_{catalog.load_benchmark(dataset, store)}"
    test.gb_load = (dataset, store)  # type: ignore[attr-defined]
    return test


def make_load_ql_test(dataset: str) -> Callable[..., None]:
    """The Load (query language) test of one dataset, parametrized over its scales and thread settings."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: LoadQlCase) -> None:
        run_load_ql(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = f"test_{catalog.load_ql_benchmark(dataset)}"
    test.gb_load_ql = dataset  # type: ignore[attr-defined]
    return test


def make_startup_test() -> Callable[..., None]:
    """The startup test, parametrized over the thread settings (`[<db>-empty]`)."""

    def test(benchmark: Any, graph_bench: Bench, gb_case: StartupCase) -> None:
        run_startup(benchmark, graph_bench, gb_case)

    test.__name__ = test.__qualname__ = "test_startup"
    test.gb_startup = True  # type: ignore[attr-defined]
    return test
