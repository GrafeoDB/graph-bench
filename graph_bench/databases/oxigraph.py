"""Oxigraph (pyoxigraph), embedded and in memory: the RDF store, queried with SPARQL 1.1."""

from collections.abc import Mapping
from typing import Any, ClassVar

import pyoxigraph as ox

from graph_bench.databases.base import Database, DatabaseInfo
from graph_bench.datasets import rdf
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model, Store


def _term(value: Any) -> Any:
    """A SPARQL result term as a Python value: a literal by its datatype (`rdf.python_value`), an IRI as its text."""
    if value is None:
        return None
    if isinstance(value, ox.Literal):
        return rdf.python_value(value.value, value.datatype.value)
    if isinstance(value, ox.NamedNode):
        return value.value
    return str(value)


class OxigraphDatabase(Database):
    """An in-memory `pyoxigraph.Store`, the RDF store only (no property graph, no SQL/PGQ).

    Load: the dataset's N-Triples file (`rdf.ntriples_path`, written untimed at prepare time by the shared mapping)
    with `Store.bulk_load(path=..., format=RdfFormat.N_TRIPLES)`, Oxigraph's documented loader for big files. Only
    that call is in the load clock; no triple is built from the CSVs here. Queries: `Store.query` with the shared
    SPARQL text, its parameters inlined as the same literals Grafeo gets (`rdf.bind`, an inline `VALUES` first in the
    group), so both engines run identical text; Oxigraph's `substitutions` argument is not used (it requires the
    variable in the SELECT projection, which the shared text does not do). Result literals are converted by
    datatype (`rdf.python_value`) and read by projection order. The load check counts the RDF mapping's nodes and
    edges with SPARQL (`rdf.COUNT_NODES`, `COUNT_EDGES`), as for Grafeo.
    """

    name: ClassVar[str] = "oxigraph"
    display_name: ClassVar[str] = "Oxigraph"
    models: ClassVar[frozenset[Model]] = frozenset({"rdf"})
    dialect: ClassVar[str] = "sparql"

    store: Any = None

    def info(self) -> DatabaseInfo:
        # bulk_load is multi-threaded internally and exposes no thread knob; queries run on the calling thread.
        note = "bulk_load uses several threads internally, not configurable; queries run on the calling thread"
        return DatabaseInfo(self.name, ox.__version__, "in-process", False, note, "in-memory")

    def open_empty(self) -> None:
        """An empty in-memory store (`Store()` without a path): what `load` does before the bulk load."""
        self.store = ox.Store()

    def load(self, dataset: Dataset) -> None:
        self.open_empty()
        self.store.bulk_load(path=str(rdf.ntriples_path(dataset)), format=ox.RdfFormat.N_TRIPLES)

    def load_store(self, store: Store, dataset: Dataset) -> None:
        self.load(dataset)

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        solutions = self.store.query(rdf.bind(text, params))
        variables = solutions.variables
        return [tuple(_term(solution[v]) for v in variables) for solution in solutions]

    def counts(self) -> tuple[int, int]:
        nodes = self.query(rdf.COUNT_NODES, {})[0][0]
        edges = self.query(rdf.COUNT_EDGES, {})[0][0]
        return int(nodes), int(edges)

    def close(self) -> None:
        self.store = None
