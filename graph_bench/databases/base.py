"""The contract every benchmarked database implements."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model


class NotSupported(Exception):
    """The database lacks the capability; the case is reported as not available."""


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
    """One database instance, holding one loaded dataset.

    `workdir` is a fresh, empty directory the instance may use for its files; `threads` is 1 or None (all cores).
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    models: ClassVar[frozenset[Model]]
    dialect: ClassVar[str] = "cypher"
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        """Environment variables the worker process needs before the driver is imported."""
        return {}

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

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        """Run a native graph algorithm ("bfs", "wcc" or "pr"); returns vertex id -> value."""
        raise NotSupported(name)

    def close(self) -> None:
        """Release the instance; the default has nothing to release."""
        return None
