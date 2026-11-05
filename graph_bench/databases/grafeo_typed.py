"""Grafeo on a closed graph type, with an index per declared label: a row beside the schemaless Grafeo, never ranked.

It is the same engine and the same loader as `graph_bench.databases.grafeo`; only the schema and the indexes differ,
both documented settings (FAIRNESS.md, "Variants"). The graph type is declared, then the graph bound to it
(docs/user-guide/gql/schema.md, "Typed Graphs"), and `CREATE INDEX` makes one index per label, which holds that label
only (the same page, "Property Indexes"). Those are in the load clock, like the schemaless row's index.

Edge types are declared without their end labels. A pattern such as `(:Post)-[:HAS_CREATOR]->(:Person)` would state
them, but patterns of one edge type that differ on both sides "are not supported yet" (the same page, "Creating Graph
Types"), and SNB's IS_LOCATED_IN does (Person to City, Post to Country).

A query with no label scans here, by the documented meaning of a per-label index, where the schemaless row's key-wide
index serves it. That is what this row is meant to show, so it is measured as it is.
"""

from dataclasses import replace
from typing import ClassVar

from graph_bench.databases.base import DatabaseInfo
from graph_bench.databases.grafeo import GrafeoDatabase
from graph_bench.datasets.schema import Column, Dataset
from graph_bench.types import Model

# The dataset's column types as graph type property types (docs/user-guide/gql/schema.md, "Property Types").
PROPERTY_TYPES = {"INT64": "INT64", "DOUBLE": "FLOAT64", "STRING": "STRING"}


def _properties(columns: tuple[Column, ...], *, key: bool = False) -> str:
    """A property list. Only the key is `NOT NULL`: an empty CSV field sets no property, the dataset contract."""
    return ", ".join(
        f"{c.name} {PROPERTY_TYPES[c.type]}" + (" NOT NULL" if key and c.name == "id" else "") for c in columns
    )


def graph_type_ddl(dataset: Dataset) -> str:
    """The statement declaring `<dataset>_type`: a closed type holding every label and edge type of the dataset."""
    elements = [f"NODE TYPE {t.label} ({_properties(t.columns, key=True)})" for t in dataset.nodes]
    edge_columns: dict[str, tuple[Column, ...]] = {}
    for edge in dataset.edges:
        if edge_columns.setdefault(edge.type, edge.columns) != edge.columns:
            raise ValueError(f"edge type {edge.type} has different properties per node pair")
    elements += [f"EDGE TYPE {name} ({_properties(columns)})" for name, columns in edge_columns.items()]
    return f"CREATE GRAPH TYPE {dataset.name}_type ({', '.join(elements)})"


def index_ddl(dataset: Dataset) -> list[str]:
    """One `CREATE INDEX` on the key per declared label."""
    return [f"CREATE INDEX {t.label}_id FOR (n:{t.label}) ON (n.id)" for t in dataset.nodes]


class GrafeoTypedDatabase(GrafeoDatabase):
    """`GrafeoDatabase` on a closed graph type: the property graph model only, as the type constrains that store."""

    name: ClassVar[str] = "grafeo_typed"
    display_name: ClassVar[str] = "Grafeo (typed)"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialects: ClassVar[dict[Model, str]] = {"property_graph": "cypher"}

    def info(self) -> DatabaseInfo:
        base = super().info()
        return replace(base, threads_note=f"{base.threads_note}; closed graph type, an index per declared label")

    def _create_graph(self, dataset: Dataset) -> None:
        self.db.execute(graph_type_ddl(dataset))
        self.db.execute(f"CREATE GRAPH {dataset.name} TYPED {dataset.name}_type")

    def _index_keys(self, dataset: Dataset) -> None:
        for statement in index_ddl(dataset):
            self.db.execute(statement)
