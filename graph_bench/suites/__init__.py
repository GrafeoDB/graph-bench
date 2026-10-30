"""Benchmark suites: what is measured, with which parameters, and the reference answers.

The item types live here so that the suite modules and the catalog can share them without an import cycle;
`graph_bench.suites.catalog` re-exports them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from graph_bench.types import Category, Model


@dataclass(frozen=True)
class Query:
    """A query benchmark. `text` maps a dialect to its query text; `"cypher"` is always present."""

    id: str
    title: str
    category: Category
    model: Model
    dataset: str
    text: Mapping[str, str] = field(hash=False)
    ordered: bool
    rel_eps: float

    def text_for(self, dialect: str) -> str:
        """The text for `dialect`, falling back to the openCypher text."""
        return self.text.get(dialect, self.text["cypher"])


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
