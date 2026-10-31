"""The benchmark catalog: items, selections, and their expansion into cases and benchmark ids.

Besides the items' cases and the loads, a selection plans two rows that are not items: Startup (one entry per
database and thread setting, `startup_id`) and Load (query language) (one per graph-store dataset, `load_ql_id`).

Suite modules: each module in `SUITES` exports its items (`QUERIES`, `ALGORITHMS` or `WRITES`), and
`bindings(item_id, graph, n)` and `reference(item_id, graph, binding)`, where `graph` is the dataset's reference
graph (`datasets.snb.SnbGraph` for SNB, the `Dataset` itself for Graphalytics). A query's or algorithm's
`reference` returns its answer; a write's returns `(after, before)`, the read-back rows after the write and in the
loaded state, or `(after, before, ready)` for a write with `ready` (the rows after its setup).
`reference_source(item_id)` names the module that holds an item's references.

`GRAPH_BENCH_EXTRA_SUITES` (a comma list of module names, read at import) adds suite modules after the built-in
ones, as `GRAPH_BENCH_EXTRA_DATABASES` adds databases; the test suite uses it for fake write cases.
"""

import importlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from graph_bench.databases import model_support
from graph_bench.suites import Algorithm, Item, Query, Write, graphalytics, hops, scans, snb, snb_rdf, snb_sql
from graph_bench.types import CATEGORY_ORDER, Category, Model, Store

__all__ = [
    "EXTRA_SUITES",
    "EXTRA_SUITES_ENV",
    "ITEMS",
    "LOAD_QL_PROBES",
    "SELECTIONS",
    "STARTUP_SCALE",
    "SUITES",
    "Algorithm",
    "Case",
    "Item",
    "Query",
    "Selection",
    "Write",
    "all_ids",
    "all_load_keys",
    "cases",
    "load_benchmark",
    "load_id",
    "load_ids",
    "load_keys",
    "load_ql_benchmark",
    "load_ql_id",
    "load_ql_ids",
    "load_ql_keys",
    "load_reference",
    "load_write_ready",
    "load_write_reference",
    "param_id",
    "reference_items",
    "reference_path",
    "reference_source",
    "resource_id",
    "selection",
    "startup_id",
    "startup_ids",
]

EXTRA_SUITES_ENV = "GRAPH_BENCH_EXTRA_SUITES"


def _module_items(module: ModuleType) -> tuple[Item, ...]:
    return tuple(item for name in ("QUERIES", "ALGORITHMS", "WRITES") for item in getattr(module, name, ()))


def _extra_suites() -> tuple[ModuleType, ...]:
    names = [name.strip() for name in os.environ.get(EXTRA_SUITES_ENV, "").split(",") if name.strip()]
    return tuple(importlib.import_module(name) for name in names)


# The modules GRAPH_BENCH_EXTRA_SUITES adds, and all suite modules in catalog order; their items, in that order, are
# the catalog.
EXTRA_SUITES: tuple[ModuleType, ...] = _extra_suites()
SUITES: tuple[ModuleType, ...] = (scans, hops, snb, graphalytics, snb_sql, snb_rdf, *EXTRA_SUITES)

ITEMS: tuple[Item, ...] = tuple(item for module in SUITES for item in _module_items(module))

_BY_ID = {item.id: item for item in ITEMS}
_SOURCE = {item.id: module for module in SUITES for item in _module_items(module)}

# The Load (query language) content probe per (dataset, primary model): the item whose preflight call runs,
# untimed, on each instance a query-language load filled, against that item's references. The order plus limit scan
# returns comment contents, strings with commas and apostrophes, so a loader that mangles quoting disagrees.
LOAD_QL_PROBES: dict[tuple[str, Model], str] = {
    ("snb", "property_graph"): "scan_order_limit",
    ("snb", "sql_pgq"): "scan_order_limit_sql",
}

# The scale of the Startup row: a database opened empty.
STARTUP_SCALE = "empty"


@dataclass(frozen=True)
class Selection:
    """A named set of categories and scales with its run settings.

    `warmup` counts the preflight as its first warm-up run. `startup_iterations` is the number of fresh processes
    the Startup row starts; `resources` turns the Memory and Disk records on.
    """

    name: str
    categories: frozenset[Category]
    scales: tuple[str, ...]
    threads: tuple[int | None, ...]
    warmup: int
    iterations: int
    load_iterations: int
    bindings: int
    startup_iterations: int
    resources: bool


SELECTIONS: dict[str, Selection] = {
    "push": Selection(
        "push",
        frozenset(
            {
                "startup",
                "load",
                "load_ql",
                "batch_writes",
                "scans",
                "hops",
                "pattern_matching",
                "complex_queries",
                "algorithms",
                "sql_pgq",
                "rdf",
            }
        ),
        ("small",),
        (1,),
        warmup=1,
        iterations=5,
        load_iterations=1,
        bindings=10,
        startup_iterations=3,
        resources=True,
    ),
    "full": Selection(
        "full",
        frozenset(CATEGORY_ORDER),
        ("small", "medium"),
        (1,),
        warmup=2,
        iterations=10,
        load_iterations=3,
        bindings=20,
        startup_iterations=5,
        resources=True,
    ),
}


def selection(name: str) -> Selection:
    """Look up a selection by name, with a clear error for unknown names."""
    try:
        return SELECTIONS[name]
    except KeyError:
        raise KeyError(f"unknown selection {name!r} (known: {', '.join(SELECTIONS)})") from None


def param_id(db: str, scale: str, threads: int | None) -> str:
    """The pytest parameter id of a benchmark: `<db>-<scale>`, with `-allcores` for the all-cores setting."""
    return f"{db}-{scale}" + ("-allcores" if threads is None else "")


@dataclass(frozen=True)
class Case:
    """One item run on one database at one scale and thread setting."""

    item: Item
    db: str
    scale: str
    threads: int | None

    @property
    def test_name(self) -> str:
        return f"test_{self.item.id}"

    @property
    def bench_id(self) -> str:
        params = param_id(self.db, self.scale, self.threads)
        return f"benches/test_{self.item.category}.py::{self.test_name}[{params}]"


def items(selection_name: str) -> list[Item]:
    """The catalog items in a selection's categories, in catalog order."""
    categories = selection(selection_name).categories
    return [item for item in ITEMS if item.category in categories]


def cases(selection_name: str, dbs: Sequence[str]) -> list[Case]:
    """Every item of the selection for every database, scale and thread setting, sorted by `bench_id`.

    Items a database's models do not support (`item.supported_by`) are included; they become not-available
    entries.
    """
    sel = selection(selection_name)
    for db in dbs:
        model_support(db)  # unknown names fail here, with a clear message
    result = [
        Case(item, db, scale, threads)
        for item in items(selection_name)
        for db in dbs
        for scale in sel.scales
        for threads in sel.threads
    ]
    return sorted(result, key=lambda c: c.bench_id)


def all_load_keys(selection_name: str) -> dict[tuple[str, Store], Model]:
    """Every (dataset, store) the selection's items read, sorted, with the model its load entry carries.

    An item reads `item.store` of its dataset: property graph and SQL/PGQ items and the algorithms share the graph
    store, RDF items read the RDF store, a separate load. The model is the first reading item's (catalog order):
    `property_graph` for a graph load, `rdf` for an RDF load.
    """
    found: dict[tuple[str, Store], Model] = {}
    for item in items(selection_name):
        found.setdefault((item.dataset, item.store), item.model)
    return dict(sorted(found.items()))


def load_keys(selection_name: str, models: frozenset[Model]) -> list[tuple[str, Store]]:
    """The (dataset, store) loads of a database serving `models`, sorted: one per store its supported items read."""
    return sorted({(item.dataset, item.store) for item in items(selection_name) if item.supported_by(models)})


def load_ids(selection_name: str, dbs: Sequence[str]) -> list[str]:
    """The load benchmark ids, `benches/test_load.py::test_load_<dataset>[...]` (`..._<dataset>_rdf` for RDF), sorted.

    Every database gets every load of the selection (`all_load_keys`), as it gets every case: a load its models do
    not reach (`load_keys`) is reported `not_supported` (⊘, "model <m> not supported").
    """
    sel = selection(selection_name)
    if "load" not in sel.categories:
        return []
    result = []
    for db in dbs:
        model_support(db)  # unknown names fail here, with a clear message
        for dataset, store in all_load_keys(selection_name):
            for scale in sel.scales:
                for threads in sel.threads:
                    result.append(load_id(dataset, db, scale, threads, store))
    return sorted(result)


def load_benchmark(dataset: str, store: Store = "graph") -> str:
    """The benchmark name of a load: `load_<dataset>` for the graph store, `load_<dataset>_rdf` for RDF triples."""
    return f"load_{dataset}" if store == "graph" else f"load_{dataset}_{store}"


def load_id(dataset: str, db: str, scale: str, threads: int | None, store: Store = "graph") -> str:
    """The load benchmark id of one dataset and store: `benches/test_load.py::test_<load_benchmark>[<params>]`."""
    return f"benches/test_load.py::test_{load_benchmark(dataset, store)}[{param_id(db, scale, threads)}]"


def resource_id(dataset: str, db: str, scale: str, threads: int | None, store: Store = "graph") -> str:
    """The id of a load's Memory and Disk record (results.json `resources`): `resources::<load_benchmark>[<params>]`.

    Not a benchmark id: resources never reach pytest or CodSpeed.
    """
    return f"resources::{load_benchmark(dataset, store)}[{param_id(db, scale, threads)}]"


def startup_id(db: str, threads: int | None) -> str:
    """The Startup id of a database: `benches/test_startup.py::test_startup[<db>-empty]` (`-allcores` suffix)."""
    return f"benches/test_startup.py::test_startup[{param_id(db, STARTUP_SCALE, threads)}]"


def startup_ids(selection_name: str, dbs: Sequence[str]) -> list[str]:
    """The Startup ids of a selection, one per database and thread setting, sorted; none without "startup"."""
    sel = selection(selection_name)
    if "startup" not in sel.categories:
        return []
    for db in dbs:
        model_support(db)  # unknown names fail here, with a clear message
    return sorted(startup_id(db, threads) for db in dbs for threads in sel.threads)


def load_ql_benchmark(dataset: str) -> str:
    """The benchmark name of a dataset's query-language load: `load_ql_<dataset>`."""
    return f"load_ql_{dataset}"


def load_ql_id(dataset: str, db: str, scale: str, threads: int | None) -> str:
    """The Load (query language) id: `benches/test_load_ql.py::test_load_ql_<dataset>[<params>]`."""
    return f"benches/test_load_ql.py::test_{load_ql_benchmark(dataset)}[{param_id(db, scale, threads)}]"


def load_ql_keys(selection_name: str) -> dict[str, Model]:
    """The datasets of the selection's graph-store loads, sorted, with the model their entries carry.

    Load (query language) fills the graph store only: an RDF store has no row of its own here.
    """
    return {dataset: model for (dataset, store), model in all_load_keys(selection_name).items() if store == "graph"}


def load_ql_ids(selection_name: str, dbs: Sequence[str]) -> list[str]:
    """The Load (query language) ids of a selection, sorted; none without "load_ql".

    Every database gets every one, as with `load_ids`: one its models do not reach is ⊘ ("model <m> not
    supported").
    """
    sel = selection(selection_name)
    if "load_ql" not in sel.categories:
        return []
    result = []
    for db in dbs:
        model_support(db)  # unknown names fail here, with a clear message
        for dataset in load_ql_keys(selection_name):
            for scale in sel.scales:
                for threads in sel.threads:
                    result.append(load_ql_id(dataset, db, scale, threads))
    return sorted(result)


def all_ids(selection_name: str, dbs: Sequence[str]) -> list[str]:
    """Every benchmark id of a selection, sorted: what `graph-bench list` prints and pytest collects."""
    ids = [c.bench_id for c in cases(selection_name, dbs)] + load_ids(selection_name, dbs)
    return sorted(ids + startup_ids(selection_name, dbs) + load_ql_ids(selection_name, dbs))


def reference_path(item: Item, scale: str, *, base: Path) -> Path:
    """Where the reference answers of an item live: `<base>/<dataset>-<scale>/references/<ref_id>-<scale>.json`.

    `ref_id` is the item's `reference_id`: its own id, or the id of the item whose references it shares.
    `base` is the selection's cache directory, `data_dir() / dataset_key(selection)`.
    """
    return base / f"{item.dataset}-{scale}" / "references" / f"{item.reference_id}-{scale}.json"


def reference_items(selection_name: str) -> list[Item]:
    """The items whose reference files a selection needs: each item's `reference_id` item, once, in catalog order."""
    wanted = {item.reference_id for item in items(selection_name)}
    return [item for item in ITEMS if item.id in wanted]


def reference_source(item_id: str) -> ModuleType:
    """The suite module whose `bindings` and `reference` give the references `item_id` is checked against.

    That is the module of the item's `reference_id` item: a twin (`Query.same_as`) uses the module of the item it
    shares references with.
    """
    return _SOURCE[_BY_ID[item_id].reference_id]


def reference_document(
    item: Item,
    bindings: list[dict[str, Any]],
    answers: list[Any],
    *,
    before: list[Any] | None = None,
    ready: list[Any] | None = None,
) -> dict[str, Any]:
    """The JSON document of a reference file. Algorithm maps become `[vertex, value]` pairs sorted by vertex.

    A write's file also holds `before`, the read-back rows of the loaded state per binding (`answers` holds the
    rows after the write), and for a write with `ready` the rows after its setup, `ready` (absent otherwise).
    """
    if isinstance(item, Algorithm):
        answers = [sorted([int(k), v] for k, v in answer.items()) for answer in answers]
    doc: dict[str, Any] = {"bindings": bindings, "answers": answers}
    if before is not None:
        doc["before"] = before
    if ready is not None:
        doc["ready"] = ready
    return doc


def load_reference(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    """Read a reference file; algorithm answers come back as `dict[int, Any]`.

    The item is identified by the file name, `<item_id>-<scale>.json` (the `reference_id` of its users). For a
    write, the answers are the rows after the write (`load_write_reference` also gives the rows before it).
    """
    item = _BY_ID[path.stem.rsplit("-", 1)[0]]
    doc = json.loads(path.read_text(encoding="utf-8"))
    answers = doc["answers"]
    if isinstance(item, Algorithm):
        answers = [{int(k): v for k, v in pairs} for pairs in answers]
    return doc["bindings"], answers


def load_write_reference(path: Path) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    """Read a write's reference file: its bindings, the rows after the write and the rows before it."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["bindings"], doc["answers"], doc["before"]


def load_write_ready(path: Path) -> list[Any] | None:
    """The rows after the setup, per binding, of a write's reference file; None when the write has no `ready`."""
    ready: list[Any] | None = json.loads(path.read_text(encoding="utf-8")).get("ready")
    return ready
