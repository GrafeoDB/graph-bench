"""Benchmark suites: what is measured, with which parameters, and the reference answers.

The item types live here so that the suite modules and the catalog can share them without an import cycle;
`graph_bench.suites.catalog` re-exports them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from graph_bench.types import STANDARD_DIALECT, Category, Model


@dataclass(frozen=True)
class Query:
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
class Algorithm:
    """A graph algorithm benchmark run through the database's native implementation."""

    id: str
    title: str
    category: Category
    model: Model
    dataset: str
    name: Literal["bfs", "wcc", "pr"]
    rel_eps: float

    @property
    def reference_id(self) -> str:
        return self.id
