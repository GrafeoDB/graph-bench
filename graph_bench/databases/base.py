"""The contract every benchmarked database implements."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model, Store


class NotSupported(Exception):
    """The database lacks the capability; the case is reported as not available."""


_PROVISIONED: set[str] = set()


def provision_once(key: str, install: Callable[[], None]) -> None:
    """Run `install` once per process for `key`; later calls return at once.

    The worker calls `provision` before the Startup children and again through `prepare_environment`, and an
    install that opens a throwaway database is not free. A failed `install` is not recorded, so the next call
    tries again.
    """
    if key in _PROVISIONED:
        return
    install()
    _PROVISIONED.add(key)


@dataclass(frozen=True)
class DatabaseInfo:
    """What results.json records about a database run."""

    name: str
    version: str
    process: Literal["in-process", "local process"]
    threads_configurable: bool
    threads_note: str
    storage: Literal["in-memory", "on-disk"]
    java: str | None = None


class Database(ABC):
    """One database instance, holding one loaded dataset in one store.

    `workdir` is a fresh, empty directory the instance may use for its files; `threads` is 1 or None (all cores).

    Models and dialects: `models` are the data models the database serves. Each model's queries come in one
    dialect, `dialect_for(model)`: `dialects[model]` when given, else `dialect`. A single-model database sets only
    `dialect` (e.g. "duckpgq" for an SQL/PGQ database, "sparql" for an RDF one); a query without a text for that
    dialect falls back to its model's standard text (`types.STANDARD_DIALECT`).

    Stores: a model's queries read the store `types.MODEL_STORE[model]` ("graph" for property graph and SQL/PGQ,
    "rdf" for RDF). The worker loads each (dataset, store) a database's items need into its own fresh instance,
    with `load_store(store, dataset)`, and runs that store's cases on it with `query_model(model, text, params)`.
    A database with one store implements `load` and `query` only: the defaults of `load_store` and `query_model`
    call them. A database with several stores or query interfaces (Grafeo) overrides the two.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    models: ClassVar[frozenset[Model]]
    dialect: ClassVar[str] = "cypher"
    dialects: ClassVar[Mapping[Model, str]] = {}
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    @classmethod
    def dialect_for(cls, model: Model) -> str:
        """The dialect of `model`'s queries: `dialects[model]`, else `dialect`."""
        return cls.dialects.get(model, cls.dialect)

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        """Environment variables the worker process needs before the driver is imported."""
        return {}

    @classmethod
    def provision(cls) -> None:
        """What a database needs on disk before any start: downloads, extraction, sha256 checks, extension installs.

        It never starts an engine (no JVM, no server process, no database opened). The worker (and the CodSpeed
        session) calls it untimed before the Startup row's child processes, which never call it, so no download or
        install lands in a startup time; `prepare_environment` calls it too. It must be idempotent. The default
        does nothing.
        """
        return None

    @classmethod
    def prepare_environment(cls) -> None:
        """One-time environment setup: `provision()`, then any untimed warm start; never loads or opens data.

        The worker (and the CodSpeed session) calls it once per process, before any timed load, so its cost never
        lands in a load time. An exception fails every load of the process, and so their cases, with its error.
        The default calls `provision()`; an adapter that overrides it calls `provision()` first.
        """
        cls.provision()

    def __init__(self, workdir: Path, threads: int | None) -> None:
        """Only store the settings: adapters create their database instance in `load()`, inside the load clock."""
        self.workdir = workdir
        self.threads = threads

    @abstractmethod
    def info(self) -> DatabaseInfo: ...

    @abstractmethod
    def load(self, dataset: Dataset) -> None: ...

    @abstractmethod
    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]: ...

    def open_empty(self) -> None:
        """Start what this process lacks (a JVM, an engine, a server process) and open an empty, usable database in
        `self.workdir`: the Startup row's work, timed with the probe query that follows it.

        Run in a fresh process (`graph_bench.runner.startup`) that never called `provision` or
        `prepare_environment`; it never fetches, hashes or installs anything. The default raises `NotSupported`.
        """
        raise NotSupported("startup")

    def load_ql(self, dataset: Dataset) -> None:
        """Load `dataset` into this fresh instance's graph store through the database's query language (Load
        (query language)); timed by the caller, which then counts the result and runs its content probe.

        The instance is closed afterwards and never queried by cases. The default raises `NotSupported`.
        """
        raise NotSupported("no query-language bulk loader")

    def write(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """Run the write statement `text` (in `dialect_for(model)`) and commit it; rows of scalars, if any.

        A write case calls it once per binding inside the clock, and for its untimed setup and teardown. The
        default calls `query_model`; a database whose `query_model` does not commit overrides it.
        """
        return self.query_model(model, text, params)

    def store_paths(self) -> list[Path]:
        """The files or directories holding this instance's stored data, for the Disk record; `[]` when in memory."""
        return []

    def load_store(self, store: Store, dataset: Dataset) -> None:
        """Load `dataset` into this (fresh) instance as `store`; timed by the caller. The default calls `load`."""
        self.load(dataset)

    def query_model(self, model: Model, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """Run `text`, in `dialect_for(model)`, on the instance's store; rows of scalars. The default calls `query`."""
        return self.query(text, params)

    def counts(self) -> tuple[int, int]:
        """The number of nodes and edges loaded, counted through the database's own query interface; untimed.

        For an RDF store, the nodes and edges of the RDF mapping (`datasets.rdf.COUNT_NODES` and `COUNT_EDGES`).

        The worker calls it right after each timed load, outside the clock, and compares it with the dataset's
        `node_count` and `edge_count`: a mismatch makes the load ≠. The default raises `NotSupported`, which
        leaves the load unchecked.
        """
        raise NotSupported("counts")

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        """Run a native graph algorithm ("bfs", "wcc" or "pr"); returns vertex id -> value."""
        raise NotSupported(name)

    def close(self) -> None:
        """Release the instance; the default has nothing to release."""
        return None
