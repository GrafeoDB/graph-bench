"""Fake databases for the runner tests. They answer from the reference files, so they agree.

FakeDatabase (and FakeDatabase2 to FakeDatabase4) serve all three models, one dialect each (the standard ones), in
two stores, like Grafeo: `load_store` remembers the store, and `query_model` fails unless the instance holds the
store the model reads and the text is the model's dialect text, so a wrong route shows up as ✗. FakeRdfDatabase
serves RDF only and FakeSqlDatabase SQL/PGQ only, through the single-model defaults (`load` and `query`).

Behaviour flags, a comma list in `GRAPH_BENCH_FAKE` (FakeDatabase), or `GRAPH_BENCH_FAKE2` to `GRAPH_BENCH_FAKE4`
(FakeDatabase2 to FakeDatabase4):
`crash:<item_id>` exits the process with code 3, `crash_soft:<item_id>` raises RuntimeError, `hang:<item_id>`
sleeps an hour, `slow:<item_id>` sleeps 0.2 s per call, `slow_check:<item_id>` answers at once but returns rows
that take 0.2 s to iterate (so the worker's answer check, not the database call, is slow), `wrong:<item_id>`
returns a perturbed answer and `unsupported:<item_id>` raises NotSupported.
`crash:load_<dataset>` and `hang:load_<dataset>` do the same while loading that dataset into the graph store
(`load_<dataset>_rdf` into the RDF store: the load's benchmark name), and `crash_soft:environment` makes
`prepare_environment` raise.
`counts()` counts the rows of the dataset's CSVs; with `short:count_<dataset>` it reports one edge too few (a load
that lost an edge), `unsupported:count_<dataset>` raises NotSupported and `crash_soft:count_<dataset>` raises
RuntimeError.
With `GRAPH_BENCH_FAKE_TRACE` set to a file, `prepare_environment` and `load` append "environment" and "load" to it.
"""

import os
import time
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.databases.base import Database as _Base
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import Algorithm, Query, catalog
from graph_bench.types import MODEL_STORE, STANDARD_DIALECT, Model, Store


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

    @classmethod
    def prepare_environment(cls) -> None:
        cls._trace("environment")
        cls._misbehave(cls._flags("environment"))

    def info(self) -> DatabaseInfo:
        return DatabaseInfo(self.name, "1.0", "in-process", False, "fake, no threads", "in-memory")

    def load(self, dataset: Dataset) -> None:
        self._trace("load")
        self._misbehave(self._flags(catalog.load_benchmark(dataset.name, self.store)))
        self.dataset = dataset

    def load_store(self, store: Store, dataset: Dataset) -> None:
        self.store = store
        self.load(dataset)

    def query_model(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        if MODEL_STORE[model] != self.store:
            raise RuntimeError(f"{model} query on the {self.store} store")
        dialect = self.dialect_for(model)
        item = self._item(lambda i: isinstance(i, Query) and i.model == model and i.text.get(dialect) == text)
        return self._rows(item, params)

    def counts(self) -> tuple[int, int]:
        assert self.dataset is not None, "counts before load"
        ds = self.dataset
        flags = self._flags(f"count_{ds.name}")
        self._misbehave(flags)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} does not count")
        nodes = sum(1 for table in ds.nodes for _ in ds.rows(table))
        edges = sum(1 for table in ds.edges for _ in ds.rows(table))
        return nodes, edges - ("short" in flags)

    @staticmethod
    def _trace(call: str) -> None:
        path = os.environ.get("GRAPH_BENCH_FAKE_TRACE")
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(call + "\n")

    @staticmethod
    def _misbehave(flags: set[str]) -> None:
        if "crash" in flags:
            os._exit(3)
        if "crash_soft" in flags:
            raise RuntimeError("soft crash")
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
        return next(i for i in catalog.ITEMS if i.dataset == name and match(i))

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
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
