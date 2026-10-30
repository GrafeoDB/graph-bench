"""Shared vocabulary: models, categories, result statuses, answer verdicts and display markers."""

from dataclasses import dataclass
from typing import Literal

Model = Literal["property_graph", "sql_pgq", "rdf"]

Category = Literal[
    "load",
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
    "load",
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
    "load": "Load",
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
