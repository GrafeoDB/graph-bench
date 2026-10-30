"""The benchmark catalog: items, selections, and their expansion into cases and benchmark ids."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_bench.databases import model_support
from graph_bench.suites import Algorithm, Query, graphalytics, snb
from graph_bench.types import CATEGORY_ORDER, Category

__all__ = [
    "ITEMS",
    "SELECTIONS",
    "Algorithm",
    "Case",
    "Query",
    "Selection",
    "cases",
    "load_ids",
    "load_reference",
    "reference_path",
    "selection",
]

ITEMS: tuple[Query | Algorithm, ...] = (*snb.QUERIES, *graphalytics.ALGORITHMS)

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
        frozenset({"load", "complex_queries", "hops", "algorithms"}),
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


def _params(db: str, scale: str, threads: int | None) -> str:
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
        params = _params(self.db, self.scale, self.threads)
        return f"benches/test_{self.item.category}.py::{self.test_name}[{params}]"


def items(selection_name: str) -> list[Query | Algorithm]:
    """The catalog items in a selection's categories, in catalog order."""
    categories = selection(selection_name).categories
    return [item for item in ITEMS if item.category in categories]


def cases(selection_name: str, dbs: Sequence[str]) -> list[Case]:
    """Every item of the selection for every database, scale and thread setting, sorted by `bench_id`.

    Items whose model a database does not support are included; they become not-available entries.
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


def load_ids(selection_name: str, dbs: Sequence[str]) -> list[str]:
    """The load benchmark ids, `benches/test_load.py::test_load_<dataset>[...]`, sorted.

    A database gets a load id for each dataset that at least one of its supported items uses.
    """
    sel = selection(selection_name)
    if "load" not in sel.categories:
        return []
    result = []
    for db in dbs:
        models = model_support(db)
        datasets = sorted({item.dataset for item in items(selection_name) if item.model in models})
        for dataset in datasets:
            for scale in sel.scales:
                for threads in sel.threads:
                    params = _params(db, scale, threads)
                    result.append(f"benches/test_load.py::test_load_{dataset}[{params}]")
    return sorted(result)


def reference_path(item: Query | Algorithm, scale: str, *, base: Path) -> Path:
    """Where the reference answers of an item live: `<base>/<dataset>-<scale>/references/<item_id>-<scale>.json`.

    `base` is the selection's cache directory, `data_dir() / dataset_key(selection)`.
    """
    return base / f"{item.dataset}-{scale}" / "references" / f"{item.id}-{scale}.json"


def reference_document(item: Query | Algorithm, bindings: list[dict[str, Any]], answers: list[Any]) -> dict[str, Any]:
    """The JSON document of a reference file. Algorithm maps become `[vertex, value]` pairs sorted by vertex."""
    if isinstance(item, Algorithm):
        answers = [sorted([int(k), v] for k, v in answer.items()) for answer in answers]
    return {"bindings": bindings, "answers": answers}


def load_reference(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    """Read a reference file; algorithm answers come back as `dict[int, Any]`.

    The item is identified by the file name, `<item_id>-<scale>.json`.
    """
    item = _BY_ID[path.stem.rsplit("-", 1)[0]]
    doc = json.loads(path.read_text(encoding="utf-8"))
    answers = doc["answers"]
    if isinstance(item, Algorithm):
        answers = [{int(k): v for k, v in pairs} for pairs in answers]
    return doc["bindings"], answers
