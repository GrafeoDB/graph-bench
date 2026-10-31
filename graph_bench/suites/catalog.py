"""The benchmark catalog: items, selections, and their expansion into cases and benchmark ids."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_bench.databases import model_support
from graph_bench.suites import Algorithm, Query, graphalytics, snb, snb_rdf, snb_sql
from graph_bench.types import CATEGORY_ORDER, Category, Model, Store

__all__ = [
    "ITEMS",
    "SELECTIONS",
    "Algorithm",
    "Case",
    "Query",
    "Selection",
    "all_load_keys",
    "cases",
    "load_benchmark",
    "load_id",
    "load_ids",
    "load_keys",
    "load_reference",
    "param_id",
    "reference_items",
    "reference_path",
    "selection",
]

ITEMS: tuple[Query | Algorithm, ...] = (*snb.QUERIES, *graphalytics.ALGORITHMS, *snb_sql.QUERIES, *snb_rdf.QUERIES)

_BY_ID = {item.id: item for item in ITEMS}


@dataclass(frozen=True)
class Selection:
    """A named set of categories and scales with its run settings.

    `warmup` counts the preflight as its first warm-up run.
    """

    name: str
    categories: frozenset[Category]
    scales: tuple[str, ...]
    threads: tuple[int | None, ...]
    warmup: int
    iterations: int
    load_iterations: int
    bindings: int


SELECTIONS: dict[str, Selection] = {
    "push": Selection(
        "push",
        frozenset({"load", "complex_queries", "hops", "algorithms", "sql_pgq", "rdf"}),
        ("small",),
        (1,),
        warmup=1,
        iterations=5,
        load_iterations=1,
        bindings=10,
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

    item: Query | Algorithm
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


def items(selection_name: str) -> list[Query | Algorithm]:
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


def reference_path(item: Query | Algorithm, scale: str, *, base: Path) -> Path:
    """Where the reference answers of an item live: `<base>/<dataset>-<scale>/references/<ref_id>-<scale>.json`.

    `ref_id` is the item's `reference_id`: its own id, or the id of the item whose references it shares.
    `base` is the selection's cache directory, `data_dir() / dataset_key(selection)`.
    """
    return base / f"{item.dataset}-{scale}" / "references" / f"{item.reference_id}-{scale}.json"


def reference_items(selection_name: str) -> list[Query | Algorithm]:
    """The items whose reference files a selection needs: each item's `reference_id` item, once, in catalog order."""
    wanted = {item.reference_id for item in items(selection_name)}
    return [item for item in ITEMS if item.id in wanted]


def reference_document(item: Query | Algorithm, bindings: list[dict[str, Any]], answers: list[Any]) -> dict[str, Any]:
    """The JSON document of a reference file. Algorithm maps become `[vertex, value]` pairs sorted by vertex."""
    if isinstance(item, Algorithm):
        answers = [sorted([int(k), v] for k, v in answer.items()) for answer in answers]
    return {"bindings": bindings, "answers": answers}


def load_reference(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    """Read a reference file; algorithm answers come back as `dict[int, Any]`.

    The item is identified by the file name, `<item_id>-<scale>.json` (the `reference_id` of its users).
    """
    item = _BY_ID[path.stem.rsplit("-", 1)[0]]
    doc = json.loads(path.read_text(encoding="utf-8"))
    answers = doc["answers"]
    if isinstance(item, Algorithm):
        answers = [{int(k): v for k, v in pairs} for pairs in answers]
    return doc["bindings"], answers
