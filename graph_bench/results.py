"""results.json: one entry per benchmark result, one resource record per measured load, and the run's metadata.

Schema 2 added `resources` and `machine.memory_bytes`; a schema 1 document still reads (no resources).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from graph_bench.databases.base import DatabaseInfo
from graph_bench.types import AnswerVerdict, Category, Model, Status, Store, Timing

SCHEMA_VERSION = 2
# The older schemas `from_dict` still reads.
READABLE_VERSIONS = (1, 2)

Reference = Literal["python", "networkx"]


class GrafeoSource(TypedDict):
    version: str | None
    source: str
    sha: str | None
    # The grafeo commit: a checkout's sha, or `setup --grafeo-sha` for a wheel; None when unknown.
    git_sha: NotRequired[str | None]


class Machine(TypedDict):
    cpu: str
    cores: int
    os: str
    # Physical memory, `psutil.virtual_memory().total`; absent in schema 1.
    memory_bytes: NotRequired[int]


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


@dataclass
class Resource:
    """The Memory and Disk record of one load: a database loading one dataset into one store, at one scale and
    thread setting, in a fresh process the supervisor sampled (`graph_bench.runner.resources`).

    `id` is `resources::<load benchmark>[<db>-<scale>]` (`catalog.resource_id`). Bytes: `baseline_bytes` is the
    process tree's RSS right before the load (engine started and warmed), `after_bytes` right after it,
    `footprint_bytes` the difference (may be negative) and `peak_bytes` the highest reading from before to after,
    the supervisor's `samples` included. `sample_interval_ms` is the sampler's nominal interval; a reading can
    take longer (on Windows each one lists every process), so `samples` over the load time gives the effective
    rate. `processes` is the largest process count seen during the load (in `before`, `after` and every sample of
    the window), not the count at the peak reading. `disk_bytes` is the size of the store's files after `close()`,
    None when the database keeps its store in memory. `answer` is the load's node and edge count check.
    """

    id: str
    database: str
    dataset: str
    store: Store
    scale: str
    threads: int | None
    status: Status
    error: str | None = None
    answer: AnswerVerdict = "unchecked"
    detail: str | None = None
    storage: Literal["in-memory", "on-disk"] | None = None
    baseline_bytes: int | None = None
    after_bytes: int | None = None
    footprint_bytes: int | None = None
    peak_bytes: int | None = None
    samples: int | None = None
    sample_interval_ms: float | None = None
    processes: int | None = None
    disk_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "Resource":
        return cls(**doc)


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
    resources: list[Resource] = field(default_factory=list)

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
            "resources": [resource.to_dict() for resource in self.resources],
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "Results":
        version = doc.get("schema_version")
        if version not in READABLE_VERSIONS:
            raise ValueError(
                f"results schema_version {version!r} is not supported (expected one of {READABLE_VERSIONS})"
            )
        return cls(
            schema_version=SCHEMA_VERSION,  # an older document reads as the current schema, which it satisfies
            graph_bench_commit=doc["graph_bench_commit"],
            grafeo=doc["grafeo"],
            machine=doc["machine"],
            selection=doc["selection"],
            mode=doc["mode"],
            started_at=doc["started_at"],
            databases={name: DatabaseInfo(**info) for name, info in doc["databases"].items()},
            entries=[Entry.from_dict(e) for e in doc["entries"]],
            resources=[Resource.from_dict(r) for r in doc.get("resources", [])],
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: Path) -> "Results":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
