"""Ranking of results: per benchmark, the databases with a correct, in-budget result, fastest first.

A result is ranked when its status is `ok` and its answer `agree`. A load's answer (Load and Load (query
language)) is its node and edge count check (`unchecked` when the database has no `counts()`), so a load is ranked
when its status is `ok` and its answer is not `disagree`, by `load_ns`. Startup ranks by `timing.median_ns` in
both modes (its children's in-process times, which a CodSpeed run records too). Other results rank by
`timing.median_ns` in bench mode and by `preflight_ns` in codspeed mode, where timings are null.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from graph_bench.databases import display_name
from graph_bench.results import Entry, Results
from graph_bench.suites.catalog import ITEMS, load_benchmark, load_ql_benchmark
from graph_bench.types import CATEGORY_ORDER, LOAD_CATEGORIES, Category

SCALE_ORDER: tuple[str, ...] = ("small", "medium", "large")

DATASET_NAMES: dict[str, str] = {"snb": "LDBC SNB", "graphalytics": "Graphalytics"}

# How a store other than the property graph is named after its dataset in the Load table.
STORE_NAMES: dict[str, str] = {"rdf": "RDF"}

_ITEM_ORDER = {item.id: i for i, item in enumerate(ITEMS)}
_TITLES = {item.id: item.title for item in ITEMS}
# Load benchmark -> (dataset, store), in catalog order: a dataset's graph load, then its RDF load after them all.
_LOADS = {
    load_benchmark(dataset, store): (dataset, store)
    for dataset, store in sorted(
        dict.fromkeys((item.dataset, item.store) for item in ITEMS),
        key=lambda key: key[1] != "graph",
    )
}
_LOAD_ORDER = list(_LOADS)
# Load (query language) benchmark -> dataset, in the order of the graph loads.
_LOADS_QL = {load_ql_benchmark(dataset): dataset for dataset, store in _LOADS.values() if store == "graph"}
_LOAD_QL_ORDER = list(_LOADS_QL)

LOAD_PREFIX = "load_"
LOAD_QL_PREFIX = "load_ql_"
STARTUP_TITLE = "Empty database"

_CATEGORY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CATEGORY_ORDER)}


@dataclass
class Ranked:
    """One benchmark at one scale and thread setting: `order` holds (database, speed in ns), fastest first."""

    category: Category
    benchmark: str
    title: str
    scale: str
    threads: int | None
    order: list[tuple[str, float]]

    def position(self, db: str) -> int | None:
        """1 + the number of databases strictly faster than `db`; None when `db` is not ranked."""
        speeds = dict(self.order)
        if db not in speeds:
            return None
        return 1 + sum(1 for _, s in self.order if s < speeds[db])

    def fastest(self) -> set[str]:
        """The databases with the lowest speed (several on a tie, none when nothing is ranked)."""
        if not self.order:
            return set()
        best = self.order[0][1]
        return {db for db, s in self.order if s == best}


def is_load(entry: Entry) -> bool:
    """Either load row: Load or Load (query language)."""
    return entry.category in LOAD_CATEGORIES


def is_ranked(entry: Entry) -> bool:
    """Correct and within the time budget."""
    if entry.status != "ok":
        return False
    return entry.answer == "agree" or (is_load(entry) and entry.answer == "unchecked")


def speed(entry: Entry, mode: str) -> float | None:
    """The value a result ranks by, in ns: `load_ns` for loads, the median for Startup, else the median (bench)
    or preflight (codspeed)."""
    if is_load(entry):
        return float(entry.load_ns) if entry.load_ns is not None else None
    if mode == "codspeed" and entry.category != "startup":
        return float(entry.preflight_ns) if entry.preflight_ns is not None else None
    return entry.timing.median_ns if entry.timing is not None else None


def title(benchmark: str) -> str:
    """The row label of a benchmark: the catalog title; for a load the dataset name, with the store if not the
    property graph ("LDBC SNB (RDF)"); for a query-language load the dataset name; "Empty database" for Startup;
    else the id."""
    if benchmark == "startup":
        return STARTUP_TITLE
    if benchmark.startswith(LOAD_QL_PREFIX):
        dataset = benchmark.removeprefix(LOAD_QL_PREFIX)
        return DATASET_NAMES.get(dataset, dataset)
    if benchmark in _LOADS:
        dataset, store = _LOADS[benchmark]
        name = DATASET_NAMES.get(dataset, dataset)
        return name if store == "graph" else f"{name} ({STORE_NAMES.get(store, store)})"
    if benchmark.startswith(LOAD_PREFIX):
        dataset = benchmark.removeprefix(LOAD_PREFIX)
        return DATASET_NAMES.get(dataset, dataset)
    return _TITLES.get(benchmark, benchmark)


def scale_key(scale: str) -> tuple[int, str]:
    return (SCALE_ORDER.index(scale) if scale in SCALE_ORDER else len(SCALE_ORDER), scale)


def _category_key(category: str) -> int:
    return _CATEGORY_INDEX.get(category, len(_CATEGORY_INDEX))


def _benchmark_key(benchmark: str) -> tuple[int, str]:
    if benchmark.startswith(LOAD_QL_PREFIX):
        known = benchmark in _LOADS_QL
        return (_LOAD_QL_ORDER.index(benchmark) if known else len(_LOAD_QL_ORDER), benchmark)
    if benchmark.startswith(LOAD_PREFIX):
        return (_LOAD_ORDER.index(benchmark) if benchmark in _LOADS else len(_LOAD_ORDER), benchmark)
    return (_ITEM_ORDER.get(benchmark, len(_ITEM_ORDER)), benchmark)


def group_key(entry: Entry) -> tuple[int, tuple[int, str], int, tuple[int, str]]:
    """Sort key of an entry's benchmark: category order, scale, threads (1 before all cores), catalog order."""
    return (
        _category_key(entry.category),
        scale_key(entry.scale),
        0 if entry.threads is not None else 1,
        _benchmark_key(entry.benchmark),
    )


def rank(results: Results) -> list[Ranked]:
    """One `Ranked` per (benchmark, scale, threads) in the results, in matrix order; `order` may be empty."""
    groups: dict[tuple[str, str, int | None], list[Entry]] = {}
    for e in sorted(results.entries, key=group_key):
        groups.setdefault((e.benchmark, e.scale, e.threads), []).append(e)
    ranked = []
    for (benchmark, scale, threads), members in groups.items():
        order = []
        for e in members:
            value = speed(e, results.mode)
            if is_ranked(e) and value is not None:
                order.append((e.database, value))
        order.sort(key=lambda pair: (pair[1], pair[0]))
        ranked.append(Ranked(members[0].category, benchmark, title(benchmark), scale, threads, order))
    return ranked


def safe_display_name(name: str) -> str:
    """`display_name`, or the plain name for a database this install does not know."""
    try:
        return display_name(name)
    except (KeyError, ImportError, AttributeError):
        return name


def ordered_databases(names: Iterable[str]) -> list[str]:
    """Grafeo first, then the others alphabetically by display name."""
    unique = list(dict.fromkeys(names))
    others = sorted((n for n in unique if n != "grafeo"), key=lambda n: (safe_display_name(n).casefold(), n))
    return (["grafeo"] if "grafeo" in unique else []) + others


def format_sig3(value: float) -> str:
    """At least 3 significant digits: 0.395 -> "0.395", 45.151 -> "45.2", 497.03 -> "497", 1234.5 -> "1235"."""
    if value == 0:
        return "0"
    if abs(value) >= 999.5:
        return f"{value:.0f}"
    rounded = float(f"{value:.3g}")
    exponent = math.floor(math.log10(abs(rounded)))
    return f"{rounded:.{max(0, 2 - exponent)}f}"


def all_databases(results: Results) -> list[str]:
    """Every database in the results, from `databases` and the entries, in column order."""
    names: Sequence[str] = [*results.databases, *(e.database for e in results.entries)]
    return ordered_databases(names)
