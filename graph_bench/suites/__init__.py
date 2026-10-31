"""Benchmark suites: what is measured, with which parameters, and the reference answers.

The item types live here so that the suite modules and the catalog can share them without an import cycle;
`graph_bench.suites.catalog` re-exports them.

Eligibility, in one place: an item runs on a database when `item.supported_by(database.models)`. A query needs
its own model. A Graphalytics algorithm needs a model that reads a property graph (`GRAPH_MODELS`: property graph
or SQL/PGQ), since it runs through the database's native implementation over its loaded graph, not through a
query text; an RDF-only database gets ⊘. A write case needs its own model, as a query. Every item reads
`item.store` of its dataset.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from graph_bench.types import MODEL_STORE, STANDARD_DIALECT, Category, Model, Store

# The models whose databases may run a Graphalytics algorithm: those that read a property graph.
GRAPH_MODELS: frozenset[Model] = frozenset(m for m, store in MODEL_STORE.items() if store == "graph")


class _Eligibility:
    """`models`, `store` and `supported_by`, shared by the item types."""

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


@dataclass(frozen=True)
class Write(_Eligibility):
    """A write case: a timed write, checked by reading it back, then undone, on the dataset's loaded instance.

    Every map takes a dialect and falls back to the model's standard dialect, as `Query.text_for`; `setup` and
    `teardown` may be empty (no statements). Each run, for every binding: the `setup` statements (untimed); with
    `ready`, the `check` read (untimed, compared with the reference's `ready` rows: a mismatch fails the case,
    "setup did not reach its state: ..."); the `run` statement (timed; one committed `Database.write` call per
    binding); the `check` read (untimed, compared with the reference's `answers`, the rows after the write); the
    `teardown` statements (untimed); and the `check` read again (compared with the reference's `before`, the rows
    of the loaded state, which the check must also return before the case). The worker runs write cases after
    every read case of their dataset's instance.

    `ready` is for a case whose setup builds the state its run changes, so that a setup that did nothing cannot
    make the run a timed no-op. Batch delete (`write_delete`) is the example: its setup inserts its own new ids
    and its run deletes them, so `before` is `[]` (the ids are not in the loaded data), `ready` is the 100 inserted
    rows per batch and `answers` is `[]` again; the teardown deletes the ids too, for a run that deleted nothing.

    Teardown statements must be idempotent: after a phase raised or a read-back differed, the harness runs the
    teardown again (best effort) to leave the loaded state for the cases after it.
    """

    id: str
    title: str
    category: Category
    model: Model
    dataset: str
    run: Mapping[str, str] = field(hash=False)
    check: Mapping[str, str] = field(hash=False)
    setup: Mapping[str, tuple[str, ...]] = field(hash=False)
    teardown: Mapping[str, tuple[str, ...]] = field(hash=False)
    ordered: bool
    rel_eps: float
    ready: bool = False

    @property
    def reference_id(self) -> str:
        return self.id

    def _text(self, texts: Mapping[str, str], dialect: str) -> str:
        return texts.get(dialect, texts[STANDARD_DIALECT[self.model]])

    def _for(self, texts: Mapping[str, tuple[str, ...]], dialect: str, default: tuple[str, ...]) -> tuple[str, ...]:
        return texts.get(dialect, texts.get(STANDARD_DIALECT[self.model], default))

    def run_for(self, dialect: str) -> str:
        """The timed write statement for `dialect`."""
        return self._text(self.run, dialect)

    def check_for(self, dialect: str) -> str:
        """The read-back query for `dialect`."""
        return self._text(self.check, dialect)

    def setup_for(self, dialect: str) -> tuple[str, ...]:
        """The untimed statements before each run, for `dialect` (empty: none)."""
        return self._for(self.setup, dialect, ())

    def teardown_for(self, dialect: str) -> tuple[str, ...]:
        """The untimed statements that undo a run, for `dialect` (empty: none)."""
        return self._for(self.teardown, dialect, ())


# Every catalog item: a query, a native algorithm or a write case.
Item = Query | Algorithm | Write
