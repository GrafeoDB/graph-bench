"""`graph-bench merge`: one results.json from several, as one run over all their databases would write it.

CI runs each database of `catalog.CI_DATABASES` in its own job, one after another on fresh macro-runner boxes, and
each job writes a results.json for its database. The merge checks that the inputs belong together (selection, mode,
graph-bench commit and Grafeo source agree; no database appears twice), puts the database infos in the catalog's
order and the entries and Memory and Disk records in id order, as one run writes them, and applies the final 10x
budget rule once over the merged entries. A database expected but without an input (its job died) is recorded
as a database that does not start: every id it would run ✗ with `NO_RESULTS`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from graph_bench.databases import KNOWN
from graph_bench.results import Results
from graph_bench.runner import supervisor
from graph_bench.suites import catalog

NO_RESULTS = "no results: its CI job left no results.json"
# The top-level fields every input must share: the same cases, measured the same way, by the same harness, with
# the same Grafeo. The machine and start time may differ (one box per job); they come from the first input.
AGREE = ("selection", "mode", "graph_bench_commit", "grafeo")


class MergeError(ValueError):
    """The inputs cannot be merged: they disagree on a field, or a database appears in two of them."""


def database_order(names: Iterable[str]) -> list[str]:
    """`names` in the catalog's order: `catalog.CI_DATABASES` (the order CI runs them), then the other built-in
    databases (`KNOWN`), then the rest by name; never the order of the command line."""
    order = list(dict.fromkeys([*catalog.CI_DATABASES, *KNOWN]))
    return sorted(set(names), key=lambda n: (order.index(n), "") if n in order else (len(order), n))


def _databases(doc: Results) -> set[str]:
    return {*doc.databases, *(e.database for e in doc.entries), *(r.database for r in doc.resources)}


def merge(inputs: Sequence[tuple[Path, Results]], expected: Sequence[str] = ()) -> tuple[Results, list[str]]:
    """The merged results of `inputs` (path, document), and the databases of `expected` that no input holds.

    Raises `MergeError` when the inputs disagree on a field of `AGREE` or share a database; `inputs` must not be
    empty.
    """
    if not inputs:
        raise MergeError("no inputs to merge")
    first_path, first = inputs[0]
    for path, doc in inputs[1:]:
        for name in AGREE:
            if getattr(doc, name) != getattr(first, name):
                raise MergeError(
                    f"the inputs disagree on {name}: {first_path} has {getattr(first, name)!r},"
                    f" {path} has {getattr(doc, name)!r}"
                )
    owner: dict[str, Path] = {}
    for path, doc in inputs:
        for db in sorted(_databases(doc)):
            if db in owner:
                raise MergeError(f"the database {db} is in two inputs: {owner[db]} and {path}")
            owner[db] = path

    docs = [doc for _, doc in inputs]
    infos = {name: info for doc in docs for name, info in doc.databases.items()}
    entries = [e for doc in docs for e in doc.entries]
    resources = [r for doc in docs for r in doc.resources]
    missing = [db for db in dict.fromkeys(expected) if db not in owner]
    sel = catalog.selection(first.selection)
    for db in missing:
        failed, unmeasured = supervisor.failed_database(sel, db, NO_RESULTS)
        entries += failed
        resources += unmeasured
    merged = replace(
        first,
        databases={name: infos[name] for name in database_order(infos)},
        entries=sorted(entries, key=lambda e: e.id),
        resources=sorted(resources, key=lambda r: r.id),
    )
    # Applied once, here. Each input came from a run of one database, whose own final rule had no other database
    # to compare with and so marked nothing; its in-run cutoffs (from `--yardstick`) are `timeout` entries that
    # this rule skips. So nothing is counted twice, and the markers are those of one run over all the databases.
    supervisor.apply_budget_rule(merged)
    return merged, missing


__all__ = ["AGREE", "NO_RESULTS", "MergeError", "database_order", "merge"]
