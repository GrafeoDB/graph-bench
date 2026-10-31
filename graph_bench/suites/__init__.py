"""Benchmark suites: what is measured, with which parameters, and the reference answers.

The item types live here so that the suite modules and the catalog can share them without an import cycle;
`graph_bench.suites.catalog` re-exports them.

Eligibility, in one place: an item runs on a database when `item.supported_by(database.models)`. A query needs
its own model. A Graphalytics algorithm needs a model that reads a property graph (`GRAPH_MODELS`: property graph
or SQL/PGQ), since it runs through the database's native implementation over its loaded graph, not through a
query text; an RDF-only database gets ⊘. Either way the item reads `item.store` of its dataset.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from graph_bench.types import MODEL_STORE, STANDARD_DIALECT, Category, Model, Store

# The models whose databases may run a Graphalytics algorithm: those that read a property graph.
GRAPH_MODELS: frozenset[Model] = frozenset(m for m, store in MODEL_STORE.items() if store == "graph")


class _Eligibility:
    """`models`, `store` and `supported_by`, shared by both item types."""

    model: Model

    @property
    def models(self) -> frozenset[Model]:
        """The models a database needs one of to run the item."""
        return frozenset({self.model})

    @property
    def store(self) -> Store:
        """The store of its dataset the item reads."""
        return MODEL_STORE[self.model]

    def supported_by(self, models: frozenset[Model]) -> bool:
        """Whether a database serving `models` runs the item (else the worker's model gate gives ⊘)."""
        return not self.models.isdisjoint(models)


@dataclass(frozen=True)
class Query(_Eligibility):
    """A query benchmark. `text` maps a dialect to its query text; the model's standard dialect is always present.

    `same_as` names the item whose parameter bindings and reference answers this query shares: the same question
    asked through another model (e.g. SNB IS1 in SQL/PGQ shares `snb_is1`'s references). Empty: its own.
    """

    id: str
    title: str
    category: Category
    model: Model
    dataset: str
    text: Mapping[str, str] = field(hash=False)
    ordered: bool
    rel_eps: float
    same_as: str = ""

    @property
    def reference_id(self) -> str:
        """The item id whose references this query is checked against."""
        return self.same_as or self.id

    def text_for(self, dialect: str) -> str:
        """The text for `dialect`, falling back to the standard text of the query's model."""
        return self.text.get(dialect, self.text[STANDARD_DIALECT[self.model]])


@dataclass(frozen=True)
class Algorithm(_Eligibility):
    """A graph algorithm benchmark run through the database's native implementation.

    `model` is the model its entries carry in results.json ("property_graph"); any of `GRAPH_MODELS` runs it.
    """

    id: str
    title: str
    category: Category
    model: Model
    dataset: str
    name: Literal["bfs", "wcc", "pr"]
    rel_eps: float

    @property
    def models(self) -> frozenset[Model]:
        return GRAPH_MODELS

    @property
    def reference_id(self) -> str:
        return self.id
