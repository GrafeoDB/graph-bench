"""Fake databases for the runner tests. They answer from the reference files, so they agree.

FakeDatabase (and FakeDatabase2 to FakeDatabase4) serve all three models, one dialect each (the standard ones), in
two stores, like Grafeo: `load_store` remembers the store, and `query_model` fails unless the instance holds the
store the model reads and the text is the model's dialect text, so a wrong route shows up as ✗. FakeRdfDatabase
serves RDF only and FakeSqlDatabase SQL/PGQ only, through the single-model defaults (`load` and `query`).

Behaviour flags, a comma list in `GRAPH_BENCH_FAKE` (FakeDatabase), or `GRAPH_BENCH_FAKE2` to `GRAPH_BENCH_FAKE4`
(FakeDatabase2 to FakeDatabase4):
`crash:<item_id>` exits the process with code 3, `crash_soft:<item_id>` raises RuntimeError,
`panic:<item_id>` raises `PanicException` (a `BaseException`, like pyo3's), `interrupt:<item_id>` raises
KeyboardInterrupt, `hang:<item_id>` sleeps an hour, `slow:<item_id>` sleeps 0.2 s per call,
`slow_check:<item_id>` answers at once but returns rows that take 0.2 s to iterate (so the worker's answer check, not
the database call, is slow), `wrong:<item_id>` returns a perturbed answer and
`unsupported:<item_id>` raises NotSupported.
`crash:load_<dataset>` and `hang:load_<dataset>` do the same while loading that dataset into the graph store
(`load_<dataset>_rdf` into the RDF store: the load's benchmark name), and `crash_soft:environment` makes
`prepare_environment` raise.
`counts()` counts the rows of the dataset's CSVs; with `short:count_<dataset>` it reports one edge too few (a load
that lost an edge), `unsupported:count_<dataset>` raises NotSupported and `crash_soft:count_<dataset>` raises
RuntimeError. After a query-language load the flags are `short:count_ql_<dataset>` and so on.

Startup: `open_empty` makes the instance answer the startup probe of its primary model (`suites.startup`), checking
that the text is the probe of the model's dialect. Its flags target `startup` (`wrong:startup` answers one more,
`unsupported:startup` raises NotSupported, `slow:startup` sleeps 0.2 s, `orphan:startup` starts a grandchild
process and hangs, `linger:startup` starts a grandchild and answers, leaving it running; the grandchild's pid goes
to the file `GRAPH_BENCH_FAKE_GRANDCHILD` names; `crash_soft:startup_later` fails every child after the
first), and `crash_soft:provision` makes `provision` raise. FakeNoStartup keeps the base class's `open_empty`, and
FakeBadClose also fails in `close()`. `load_ql` loads like `load`; its flags target `load_ql_<dataset>`.

Writes (the items of `tests.fake_writes`, added with GRAPH_BENCH_EXTRA_SUITES): the instance keeps which bindings
a setup prepared and a run wrote, so the check read answers the reference's `before` rows, then its `ready` rows
after a setup (items with `ready`), then its `answers` rows until the teardown. Flags on the item id: `wrong`
perturbs every read-back after a run, `wrong_later` only after the second run, `sticky` makes the teardown undo
nothing, `dirty` adds a row to the loaded state, `leak` adds a node per run that no read-back sees (only
`counts()`), `nosetup` makes the setup prepare nothing and `nosetup_later` only after the first run, and `crash`,
`crash_soft`, `hang` and `unsupported` apply to the run.

Resources (the Memory and Disk records): `memory:<load benchmark>=<MiB>` makes that load allocate and keep that
many MiB, touched so they are resident (e.g. `memory:load_snb=200`); `disk:<load benchmark>=<bytes>` makes it write
exactly that many bytes in two files under `<workdir>/store`, which `store_paths` then returns, and makes `info()`
report on-disk storage. `crash_soft:instance` makes the constructor raise, `crash_soft:store_paths` makes
`store_paths` raise, `slow:environment` makes `prepare_environment` sleep 3 s and `hang:exit` makes the process hang
at exit (an atexit handler registered by `close()`), like a runtime whose shutdown never ends.

With `GRAPH_BENCH_FAKE_TRACE` set to a file, every call of `provision`, `prepare_environment` ("environment"),
`load`, `load_ql`, `open_empty` and every write phase ("setup", "run", "check", "teardown") appends a line to it:
the call, the process id and, for `open_empty`, the instance's directory.
"""

import atexit
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.databases.base import Database as _Base
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import Algorithm, Query, Write, catalog
from graph_bench.suites.startup import EXPECTED, primary_model, probe_text
from graph_bench.types import MODEL_STORE, STANDARD_DIALECT, Model, Store


class PanicException(BaseException):
    """What pyo3 raises for a Rust panic: a `BaseException`, not an `Exception`."""


class SlowRows(list[tuple[Any, ...]]):
    """Rows that sleep 0.2 s each time they are iterated: harness work after the database call returns."""

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        time.sleep(0.2)
        return super().__iter__()


class FakeDatabase(Database):
    name: ClassVar[str] = "fake"
    display_name: ClassVar[str] = "Fake"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph", "sql_pgq", "rdf"})
    dialects: ClassVar[Mapping[Model, str]] = STANDARD_DIALECT
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE"

    dataset: Dataset | None = None
    store: Store = "graph"
    empty: bool = False
    via: str = ""  # "ql" after a query-language load

    def __init__(self, workdir: Path, threads: int | None) -> None:
        super().__init__(workdir, threads)
        self._misbehave(self._flags("instance"))
        self.written: dict[str, set[int]] = {}
        self.prepared: dict[str, set[int]] = {}
        self.runs: dict[str, int] = {}
        self.leaked = 0
        self.ballast: list[bytes] = []
        self.stored: list[Path] = []

    @classmethod
    def provision(cls) -> None:
        cls._trace("provision")
        cls._misbehave(cls._flags("provision"))

    @classmethod
    def prepare_environment(cls) -> None:
        cls.provision()
        cls._trace("environment")
        flags = cls._flags("environment")
        cls._misbehave(flags)
        if "slow" in flags:
            time.sleep(3)

    def info(self) -> DatabaseInfo:
        storage = "on-disk" if "disk:" in os.environ.get(self.env_var, "") else "in-memory"
        return DatabaseInfo(self.name, "1.0", "in-process", False, "fake, no threads", storage)

    def open_empty(self) -> None:
        self._trace("open_empty", str(self.workdir))
        flags = self._flags("startup")
        self._misbehave(flags)
        if self.workdir.name != "startup-0":  # a child after the preflight
            self._misbehave(self._flags("startup_later"))
        if "slow" in flags:
            time.sleep(0.2)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} does not start empty")
        if "orphan" in flags or "linger" in flags:
            grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
            Path(os.environ["GRAPH_BENCH_FAKE_GRANDCHILD"]).write_text(str(grandchild.pid), encoding="utf-8")
            time.sleep(3600 if "orphan" in flags else 0.5)  # long enough for the parent to see it running
        self.empty = True

    def load(self, dataset: Dataset) -> None:
        self._trace("load")
        benchmark = catalog.load_benchmark(dataset.name, self.store)
        self._misbehave(self._flags(benchmark))
        for mib in self._values("memory", benchmark):
            self.ballast.append(b"\x01" * (int(mib) << 20))
        for size in self._values("disk", benchmark):
            store = self.workdir / "store"
            (store / "sub").mkdir(parents=True, exist_ok=True)
            half = int(size) // 2
            (store / "data.bin").write_bytes(b"d" * half)
            (store / "sub" / "wal.bin").write_bytes(b"w" * (int(size) - half))
            self.stored = [store]
        self.dataset = dataset

    def store_paths(self) -> list[Path]:
        self._misbehave(self._flags("store_paths"))
        return list(self.stored)

    def close(self) -> None:
        if "hang" in self._flags("exit"):
            atexit.register(time.sleep, 3600)

    def load_store(self, store: Store, dataset: Dataset) -> None:
        self.store = store
        self.load(dataset)

    def load_ql(self, dataset: Dataset) -> None:
        self._trace("load_ql")
        flags = self._flags(catalog.load_ql_benchmark(dataset.name))
        self._misbehave(flags)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} has no LOAD CSV")
        self.store, self.dataset, self.via = "graph", dataset, "ql"

    def query_model(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        if self.dataset is None and self.empty:
            return self._probe(model, text)
        if MODEL_STORE[model] != self.store:
            raise RuntimeError(f"{model} query on the {self.store} store")
        dialect = self.dialect_for(model)
        write = self._write_item(model, lambda w: w.check_for(dialect) == text)
        if write is not None:
            return self._check(write, params)
        item = self._item(lambda i: isinstance(i, Query) and i.model == model and i.text.get(dialect) == text)
        return self._rows(item, params)

    def write(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        dialect = self.dialect_for(model)
        for phase in ("setup", "run", "teardown"):
            item = self._write_item(model, lambda w, p=phase: text in self._statements(w, p, dialect))
            if item is not None:
                return self._write_step(item, phase, params)
        raise RuntimeError(f"unknown write statement {text!r}")

    def counts(self) -> tuple[int, int]:
        assert self.dataset is not None, "counts before load"
        ds = self.dataset
        flags = self._flags(f"count_{self.via}_{ds.name}" if self.via else f"count_{ds.name}")
        self._misbehave(flags)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} does not count")
        nodes = sum(1 for table in ds.nodes for _ in ds.rows(table))
        edges = sum(1 for table in ds.edges for _ in ds.rows(table))
        return nodes + self.leaked, edges - ("short" in flags)

    @staticmethod
    def _trace(call: str, *detail: str) -> None:
        path = os.environ.get("GRAPH_BENCH_FAKE_TRACE")
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(" ".join([call, str(os.getpid()), *detail]) + "\n")

    @staticmethod
    def _misbehave(flags: set[str]) -> None:
        if "crash" in flags:
            os._exit(3)
        if "crash_soft" in flags:
            raise RuntimeError("soft crash")
        if "panic" in flags:
            raise PanicException("assertion failed: index <= Self::MAX_CAPACITY")
        if "interrupt" in flags:
            raise KeyboardInterrupt
        if "hang" in flags:
            time.sleep(3600)

    @classmethod
    def _flags(cls, item_id: str) -> set[str]:
        flags = set()
        for part in os.environ.get(cls.env_var, "").split(","):
            flag, _, target = part.strip().partition(":")
            if target == item_id:
                flags.add(flag)
        return flags

    @classmethod
    def _values(cls, flag: str, target: str) -> list[str]:
        """The values of `<flag>:<target>=<value>` flags."""
        values = []
        for part in os.environ.get(cls.env_var, "").split(","):
            name, _, rest = part.strip().partition(":")
            key, sep, value = rest.partition("=")
            if name == flag and sep and key == target:
                values.append(value)
        return values

    def _probe(self, model: Model, text: str) -> list[tuple[Any, ...]]:
        """The startup probe on the empty instance: `EXPECTED[model]`, or one more with `wrong:startup`."""
        if text != probe_text(self.dialect_for(model), model):
            raise RuntimeError(f"not the {model} startup probe: {text!r}")
        bump = 1 if "wrong" in self._flags("startup") else 0
        return [tuple(value + bump for value in row) for row in EXPECTED[model]]

    def _answer(self, item: Query | Algorithm, params: Mapping[str, Any]) -> Any:
        assert self.dataset is not None, "query before load"
        flags = self._flags(item.id)
        self._misbehave(flags)
        if "slow" in flags:
            time.sleep(0.2)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} does not run {item.id}")
        path = catalog.reference_path(item, self.dataset.scale, base=self.dataset.root.parent)
        bindings, answers = catalog.load_reference(path)
        answer = answers[bindings.index(dict(params))]
        if "wrong" in flags:
            if isinstance(item, Algorithm):
                return {**answer, -1: 0}
            return [*answer, ["wrong"]]
        return answer

    def _item(self, match: Any) -> Query | Algorithm:
        assert self.dataset is not None, "query before load"
        name = self.dataset.name
        return next(i for i in catalog.ITEMS if i.dataset == name and not isinstance(i, Write) and match(i))

    @staticmethod
    def _statements(item: Write, phase: str, dialect: str) -> tuple[str, ...]:
        if phase == "run":
            return (item.run_for(dialect),)
        return item.setup_for(dialect) if phase == "setup" else item.teardown_for(dialect)

    def _write_item(self, model: Model, match: Callable[[Write], bool]) -> Write | None:
        if self.dataset is None:
            return None
        name = self.dataset.name
        found = [i for i in catalog.ITEMS if isinstance(i, Write) and i.dataset == name and i.model == model]
        return next((i for i in found if match(i)), None)

    def _write_reference(self, item: Write, params: Mapping[str, Any]) -> tuple[int, list[Any], list[Any]]:
        assert self.dataset is not None
        path = catalog.reference_path(item, self.dataset.scale, base=self.dataset.root.parent)
        bindings, after, before = catalog.load_write_reference(path)
        index = bindings.index(dict(params))
        return index, after[index], before[index]

    def _write_step(self, item: Write, phase: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        self._trace(phase)
        index, _, _ = self._write_reference(item, params)
        written = self.written.setdefault(item.id, set())
        prepared = self.prepared.setdefault(item.id, set())
        flags = self._flags(item.id)
        if phase == "setup":
            if "nosetup" not in flags and not ("nosetup_later" in flags and self.runs.get(item.id, 0) >= 1):
                prepared.add(index)
        elif phase == "run":
            self._misbehave(flags)
            if "unsupported" in flags:
                raise NotSupported(f"{self.name} does not run {item.id}")
            if index == 0:
                self.runs[item.id] = self.runs.get(item.id, 0) + 1
            written.add(index)
            prepared.discard(index)
            self.leaked += "leak" in flags
        elif "sticky" not in flags:
            written.discard(index)
            prepared.discard(index)
        return []

    def _check(self, item: Write, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        self._trace("check")
        index, after, before = self._write_reference(item, params)
        flags = self._flags(item.id)
        if index not in self.written.get(item.id, set()):
            if item.ready and index in self.prepared.get(item.id, set()):
                assert self.dataset is not None
                path = catalog.reference_path(item, self.dataset.scale, base=self.dataset.root.parent)
                return [tuple(row) for row in (catalog.load_write_ready(path) or [])[index]]
            return [tuple(row) for row in before] + ([("dirty",)] if "dirty" in flags else [])
        wrong = "wrong" in flags or ("wrong_later" in flags and self.runs.get(item.id, 0) > 1)
        return [tuple(row) for row in after] + ([("wrong",)] if wrong else [])

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        if self.dataset is None and self.empty:
            return self._probe(primary_model(self.models), text)
        item = self._item(lambda i: isinstance(i, Query) and i.text_for(self.dialect_for(i.model)) == text)
        return self._rows(item, params)

    def _rows(self, item: Query | Algorithm, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        rows = [tuple(row) for row in self._answer(item, params)]
        return SlowRows(rows) if "slow_check" in self._flags(item.id) else rows

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        item = self._item(lambda i: isinstance(i, Algorithm) and i.name == name)
        return dict(self._answer(item, params))


class FakeDatabase2(FakeDatabase):
    name: ClassVar[str] = "fake2"
    display_name: ClassVar[str] = "Fake 2"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE2"


class FakeRdfDatabase(FakeDatabase):
    """Supports only RDF (dialect "sparql") through the single-model defaults; other cases are not available."""

    name: ClassVar[str] = "fake_rdf"
    display_name: ClassVar[str] = "Fake RDF"
    models: ClassVar[frozenset[Model]] = frozenset({"rdf"})
    dialect: ClassVar[str] = "sparql"
    dialects: ClassVar[Mapping[Model, str]] = {}
    store: Store = "rdf"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE_RDF"

    load_store = _Base.load_store
    query_model = _Base.query_model


class FakeSqlDatabase(FakeDatabase):
    """Supports only SQL/PGQ (dialect "duckpgq", falling back to the standard texts) through the single-model
    defaults, like DuckDB: its algorithms run too, since an SQL/PGQ database reads a property graph."""

    name: ClassVar[str] = "fake_sql"
    display_name: ClassVar[str] = "Fake SQL"
    models: ClassVar[frozenset[Model]] = frozenset({"sql_pgq"})
    dialect: ClassVar[str] = "duckpgq"
    dialects: ClassVar[Mapping[Model, str]] = {}
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE_SQL"

    load_store = _Base.load_store
    query_model = _Base.query_model


class FakeDatabase3(FakeDatabase):
    name: ClassVar[str] = "fake3"
    display_name: ClassVar[str] = "Fake 3"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE3"


class FakeDatabase4(FakeDatabase):
    name: ClassVar[str] = "fake4"
    display_name: ClassVar[str] = "Fake 4"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE4"


class FakeNoStartup(FakeDatabase):
    """The multi-model fake without `open_empty`: the base class's default makes its Startup entry ⊘."""

    name: ClassVar[str] = "fake_nostart"
    display_name: ClassVar[str] = "Fake no start"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE_NOSTART"

    open_empty = _Base.open_empty


class FakeBadClose(FakeNoStartup):
    """Without `open_empty`, and its `close()` raises: the startup child must report the `open_empty` error."""

    name: ClassVar[str] = "fake_badclose"

    def close(self) -> None:
        raise RuntimeError("close failed")
