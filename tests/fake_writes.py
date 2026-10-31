"""A write suite for the harness tests: `GRAPH_BENCH_EXTRA_SUITES=tests.fake_writes` adds it to the catalog.

Its one write case runs on SNB, in the push selection's Batch writes category. The fake databases recognize its
statements (`tests.fakes`); a real database would not. The reference marks three persons as written: the rows
after the write differ from the loaded state's, so a read-back that missed the write, or a teardown that did not
undo it, disagrees.
"""

from typing import Any

from graph_bench.answers import SNB_EPS, canonical
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import Write

WRITES: tuple[Write, ...] = (
    Write(
        "fake_write",
        "Fake write",
        "batch_writes",
        "property_graph",
        "snb",
        run={"cypher": "FAKE WRITE RUN"},
        check={"cypher": "FAKE WRITE CHECK"},
        setup={"cypher": ("FAKE WRITE SETUP",)},
        teardown={"cypher": ("FAKE WRITE TEARDOWN 1", "FAKE WRITE TEARDOWN 2")},
        ordered=False,
        rel_eps=SNB_EPS,
    ),
)


def bindings(item_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]:
    """Three batches of one person each."""
    persons = sorted(graph.nodes["Person"])[:3]
    return [{"batch": i, "ids": [pid]} for i, pid in enumerate(persons)]


def reference(item_id: str, graph: SnbGraph, binding: dict[str, Any]) -> tuple[list[list[Any]], list[list[Any]]]:
    """(after, before): each person's state after the write and in the loaded data."""
    after = canonical([[pid, "written"] for pid in binding["ids"]], ordered=False)
    before = canonical([[pid, "loaded"] for pid in binding["ids"]], ordered=False)
    return after, before
