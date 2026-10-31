"""A write suite for the harness tests: `GRAPH_BENCH_EXTRA_SUITES=tests.fake_writes` adds it to the catalog.

Its write cases run on SNB, in the push selection's Batch writes category. The fake databases recognize their
statements (`tests.fakes`); a real database would not. The reference marks three persons as written: the rows
after the write differ from the loaded state's, so a read-back that missed the write, or a teardown that did not
undo it, disagrees. `fake_setup_write` also has a `ready` reference, the rows its setup must leave (like
`write_delete`, whose setup inserts the rows its run deletes).
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
    Write(
        "fake_setup_write",
        "Fake write with a setup state",
        "batch_writes",
        "property_graph",
        "snb",
        run={"cypher": "FAKE SETUP WRITE RUN"},
        check={"cypher": "FAKE SETUP WRITE CHECK"},
        setup={"cypher": ("FAKE SETUP WRITE SETUP",)},
        teardown={"cypher": ("FAKE SETUP WRITE TEARDOWN",)},
        ordered=False,
        rel_eps=SNB_EPS,
        ready=True,
    ),
)


def bindings(item_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]:
    """Three batches of one person each."""
    persons = sorted(graph.nodes["Person"])[:3]
    return [{"batch": i, "ids": [pid]} for i, pid in enumerate(persons)]


def reference(item_id: str, graph: SnbGraph, binding: dict[str, Any]) -> tuple[list[list[Any]], ...]:
    """(after, before), plus `ready` for the item with a setup state: each person's state after the write, in the
    loaded data, and after the setup."""
    after = canonical([[pid, "written"] for pid in binding["ids"]], ordered=False)
    before = canonical([[pid, "loaded"] for pid in binding["ids"]], ordered=False)
    if item_id == "fake_setup_write":
        return after, before, canonical([[pid, "ready"] for pid in binding["ids"]], ordered=False)
    return after, before
