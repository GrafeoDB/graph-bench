"""results.json: one entry per benchmark result, with the run's metadata."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from graph_bench.databases.base import DatabaseInfo
from graph_bench.types import AnswerVerdict, Category, Model, Status, Timing

SCHEMA_VERSION = 1

Reference = Literal["python", "networkx"]


class GrafeoSource(TypedDict):
    version: str
    source: str
    sha: str | None


class Machine(TypedDict):
    cpu: str
    cores: int
    os: str


@dataclass
class Entry:
    """One result: a benchmark on one database at one scale and thread setting.

    `id` is the benchmark id (`Case.bench_id`, or the load id); `benchmark` is the item id or `load_<dataset>`.
    """

    id: str
    category: Category
    model: Model
    benchmark: str
    database: str
    scale: str
    threads: int | None
    status: Status
    error: str | None = None
    timing: Timing | None = None
    preflight_ns: int | None = None
    load_ns: int | None = None
    answer: AnswerVerdict = "unchecked"
    reference: Reference | None = None
    answer_digest: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "Entry":
        timing = doc.get("timing")
        return cls(**{**doc, "timing": Timing(**timing) if timing is not None else None})


@dataclass(kw_only=True)
class Results:
    """The whole results.json document."""

    schema_version: int = SCHEMA_VERSION
    graph_bench_commit: str
    grafeo: GrafeoSource
    machine: Machine
    selection: str
    mode: Literal["bench", "codspeed"]
    started_at: str
    databases: dict[str, DatabaseInfo]
    entries: list[Entry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_bench_commit": self.graph_bench_commit,
            "grafeo": dict(self.grafeo),
            "machine": dict(self.machine),
            "selection": self.selection,
            "mode": self.mode,
            "started_at": self.started_at,
            "databases": {name: asdict(info) for name, info in self.databases.items()},
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "Results":
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"results schema_version {version!r} is not supported (expected {SCHEMA_VERSION})")
        return cls(
            schema_version=version,
            graph_bench_commit=doc["graph_bench_commit"],
            grafeo=doc["grafeo"],
            machine=doc["machine"],
            selection=doc["selection"],
            mode=doc["mode"],
            started_at=doc["started_at"],
            databases={name: DatabaseInfo(**info) for name, info in doc["databases"].items()},
            entries=[Entry.from_dict(e) for e in doc["entries"]],
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: Path) -> "Results":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
