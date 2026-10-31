"""Startup: the probe that ends a cold start, and its expected answer on an empty database.

The Startup row times a fresh process opening an empty, usable database and answering one query (see
`graph_bench.runner.startup`). The probe runs in the database's primary model, `primary_model(models)`, in that
model's dialect for the database (`Database.dialect_for`); a dialect without its own probe text uses the model's
standard one. HelixDB's text is the name of its DSL builder in `graph_bench.databases.helix_queries`.
"""

from collections.abc import Mapping
from typing import Any

from graph_bench.types import STANDARD_DIALECT, Model

PROBE: Mapping[str, str] = {
    "cypher": "MATCH (n) RETURN count(n) AS nodes",
    "sql_pgq": "SELECT 1 AS one",
    "sparql": "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }",
    "helix": "startup_probe",
}

# The canonical rows of the probe on an empty database: no nodes, one constant row, no triples.
EXPECTED: Mapping[Model, list[list[Any]]] = {
    "property_graph": [[0]],
    "sql_pgq": [[1]],
    "rdf": [[0]],
}

_PREFERENCE: tuple[Model, ...] = ("property_graph", "sql_pgq", "rdf")


def primary_model(models: frozenset[Model]) -> Model:
    """The model a database's startup probe (and Load (query language) content probe) uses: the property graph
    when it serves one, else SQL/PGQ, else RDF."""
    for model in _PREFERENCE:
        if model in models:
            return model
    raise ValueError(f"no known model in {sorted(models)}")


def probe_text(dialect: str, model: Model) -> str:
    """The probe for `dialect`, falling back to the standard dialect of `model`."""
    return PROBE.get(dialect, PROBE[STANDARD_DIALECT[model]])
