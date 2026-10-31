"""Shared vocabulary: models, categories, result statuses, answer verdicts and display markers."""

from dataclasses import dataclass
from typing import Literal

Model = Literal["property_graph", "sql_pgq", "rdf"]

# What a loaded instance holds: a property graph or RDF triples.
Store = Literal["graph", "rdf"]

# The store each model's queries read. Property graph and SQL/PGQ queries both read a property graph (SQL/PGQ is
# property graph matching inside SQL), so a database serving both loads a dataset once for them; RDF queries read
# triples, a separate load (and a separate Load benchmark, `load_<dataset>_rdf`).
MODEL_STORE: dict[Model, Store] = {"property_graph": "graph", "sql_pgq": "graph", "rdf": "rdf"}

# The dialect of each model's shared, standard query text: every query of the model has a text for it.
STANDARD_DIALECT: dict[Model, str] = {"property_graph": "cypher", "sql_pgq": "sql_pgq", "rdf": "sparql"}

Category = Literal[
    "startup",
    "load",
    "load_ql",
    "batch_writes",
    "scans",
    "hops",
    "pattern_matching",
    "complex_queries",
    "algorithms",
    "vector",
    "hybrid",
    "rdf",
    "sql_pgq",
    "crash_restore",
    "deriva",
    "project_suites",
]

CATEGORY_ORDER: tuple[Category, ...] = (
    "startup",
    "load",
    "load_ql",
    "batch_writes",
    "scans",
    "hops",
    "pattern_matching",
    "complex_queries",
    "algorithms",
    "vector",
    "hybrid",
    "rdf",
    "sql_pgq",
    "crash_restore",
    "deriva",
    "project_suites",
)

CATEGORY_NAMES: dict[Category, str] = {
    "startup": "Startup",
    "load": "Load",
    "load_ql": "Load (query language)",
    "batch_writes": "Batch writes",
    "scans": "Scans",
    "hops": "Hops",
    "pattern_matching": "Pattern matching",
    "complex_queries": "Complex queries",
    "algorithms": "Algorithms",
    "vector": "Vector search",
    "hybrid": "Hybrid",
    "rdf": "RDF/SPARQL",
    "sql_pgq": "SQL/PGQ",
    "crash_restore": "Crash and restore",
    "deriva": "Deriva workload",
    "project_suites": "Project suites",
}

# The load rows: Load (each database's fastest documented bulk path) and Load (query language). Their entries carry
# `load_ns`, rank by it, and count as passed when their node and edge count check did not disagree.
LOAD_CATEGORIES: frozenset[Category] = frozenset({"load", "load_ql"})

# The categories the 10x budget rule (and the in-run runaway guard) skips, as entries and as yardsticks.
BUDGET_EXEMPT: frozenset[Category] = frozenset({"startup", "load", "load_ql"})

Status = Literal["ok", "failed", "timeout", "not_supported"]
AnswerVerdict = Literal["agree", "disagree", "unchecked"]
Marker = Literal["", "⊘", "✗", "≠", "⏱"]


def marker(status: Status, answer: AnswerVerdict) -> Marker:
    """Return the display marker for a result cell."""
    if status == "not_supported":
        return "⊘"
    if status == "failed":
        return "✗"
    if status == "timeout":
        return "⏱"
    if answer == "disagree":
        return "≠"
    return ""


@dataclass(frozen=True)
class Timing:
    """Timing summary of one measured query, in nanoseconds."""

    median_ns: float
    p99_ns: float
    min_ns: int
    iterations: int
