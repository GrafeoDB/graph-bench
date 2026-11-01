"""`graph-bench merge`: one results.json from several, as one run over all their databases would write it.

CI runs each database of `catalog.CI_DATABASES` in its own job, one after another on fresh macro-runner boxes, and
each job writes a results.json for its database. The merge checks that the inputs belong together (selection, mode,
graph-bench commit and Grafeo source agree; no database appears twice), puts the database infos in the catalog's
order and the entries and Memory and Disk records in id order, as one run writes them, and applies the final 10x
budget rule once over the merged CI entries. A database expected but without an input (its job died) is recorded
as a database that does not start: every id it would run ✗ with `NO_RESULTS`.

`--local` adds the local snapshot (`results/local/results.json`, all ten databases on the dev box,
`.github/scripts/local-bench.sh`): its databases that are neither in a CI input nor expected join the merge, each
entry, Memory and Disk record and database info tagged `local`, and `sources` records both runs. The snapshot may
come from another machine, day, graph-bench commit and Grafeo, but must share the selection and mode. Its other
databases are dropped, and so is every roster database (`catalog.CI_DATABASES`, `from_snapshot`), with or
without `--db`: CI measures them, and a local timing never stands in for a missing CI one. Budgets and
ranks never compare the two sources (`supervisor.apply_budget_rule`, `report.ranking`), and the snapshot's entries
keep the budget verdicts of their own full run.
"""

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from graph_bench.databases import KNOWN
from graph_bench.results import Results, Run
from graph_bench.runner import supervisor
from graph_bench.suites import catalog

NO_RESULTS = "no results: its CI job left no results.json"
# The top-level fields every input must share: the same cases, measured the same way, by the same harness, with
# the same Grafeo. The machine and start time may differ (one box per job); they come from the first input.
AGREE = ("selection", "mode", "graph_bench_commit", "grafeo")
# The fields the local snapshot shares with the CI inputs: the same cases, measured the same way.
LOCAL_AGREE = ("selection", "mode")


class MergeError(ValueError):
    """The inputs cannot be merged: they disagree on a field, or a database appears in two of them."""


def database_order(names: Iterable[str]) -> list[str]:
    """`names` in the catalog's order: `catalog.CI_DATABASES` (the order CI runs them), then the other built-in
    databases (`KNOWN`), then the rest by name; never the order of the command line."""
    order = list(dict.fromkeys([*catalog.CI_DATABASES, *KNOWN]))
    return sorted(set(names), key=lambda n: (order.index(n), "") if n in order else (len(order), n))


def _databases(doc: Results) -> set[str]:
    return {*doc.databases, *(e.database for e in doc.entries), *(r.database for r in doc.resources)}


def from_snapshot(db: str) -> bool:
    """Whether `merge --local` may take `db` from the local snapshot: any database outside the CI roster.
    `graph-bench local-check` checks the same databases."""
    return db not in catalog.CI_DATABASES


def _local_part(
    first_path: Path, first: Results, local_path: Path, local: Results, taken: set[str]
) -> tuple[list[str], Results]:
    """The snapshot's databases not in `taken`, in catalog order, and the snapshot cut down to them, every entry,
    record and info tagged `local`."""
    for name in LOCAL_AGREE:
        if getattr(local, name) != getattr(first, name):
            raise MergeError(
                f"the local snapshot disagrees on {name}: {first_path} has {getattr(first, name)!r},"
                f" {local_path} has {getattr(local, name)!r}"
            )
    added = [db for db in database_order(_databases(local)) if db not in taken and from_snapshot(db)]
    keep = set(added)
    part = replace(
        local,
        databases={n: replace(i, source="local") for n, i in local.databases.items() if n in keep},
        entries=[replace(e, source="local") for e in local.entries if e.database in keep],
        resources=[replace(r, source="local") for r in local.resources if r.database in keep],
    )
    return added, part


def merge(
    inputs: Sequence[tuple[Path, Results]],
    expected: Sequence[str] = (),
    *,
    local: tuple[Path, Results] | None = None,
) -> tuple[Results, list[str]]:
    """The merged results of `inputs` (path, document), and the databases of `expected` that no input holds.

    `local` (path, document) is the local snapshot: see the module docstring. Raises `MergeError` when the inputs
    disagree on a field of `AGREE` or share a database, or the snapshot disagrees on one of `LOCAL_AGREE`;
    `inputs` must not be empty.
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
    sources: dict[str, Run] = {}
    if local is not None:
        local_path, local_doc = local
        added, part = _local_part(first_path, first, local_path, local_doc, {*owner, *missing})
        if added:
            infos.update(part.databases)
            entries += part.entries
            resources += part.resources
            sources = {"ci": first.run("ci"), "local": local_doc.run("ci")}
    merged = replace(
        first,
        databases={name: infos[name] for name in database_order(infos)},
        entries=sorted(entries, key=lambda e: e.id),
        resources=sorted(resources, key=lambda r: r.id),
        sources=sources,
    )
    # Applied once, here. Each input came from a run of one database, whose own final rule had no other database
    # to compare with and so marked nothing; its in-run cutoffs (from `--yardstick`) are `timeout` entries that
    # this rule skips. So nothing is counted twice, and the markers are those of one run over all the databases.
    # It runs over the CI entries only. The snapshot's entries keep the verdicts of their own run's rule over all
    # ten databases: with its roster databases dropped, fewer than three peers can remain, so a slow peer no longer
    # lifts the mean and the rule here could mark an entry its own run kept. The view shares the entry objects, so
    # the marks land in `merged`.
    supervisor.apply_budget_rule(replace(merged, entries=[e for e in merged.entries if e.source == "ci"]))
    return merged, missing


def earlier_legs(db: str, paths: Iterable[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """The leg results among `paths` a CI leg of `db` takes as yardsticks, in roster order, and the unreadable ones
    with their error.

    A file is taken when every database it holds comes before `db` in `catalog.CI_DATABASES`: a re-run leg, or one
    GitHub started out of matrix order, never takes a later leg's results. Raises ValueError when `db` is not in the
    roster.
    """
    roster = list(catalog.CI_DATABASES)
    if db not in roster:
        raise ValueError(f"{db} is not in the CI roster ({', '.join(roster)})")
    taken: list[tuple[int, Path]] = []
    unreadable: list[tuple[Path, str]] = []
    for path in paths:
        try:
            dbs = _databases(Results.read(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unreadable.append((path, str(exc) or type(exc).__name__))
            continue
        if dbs and all(name in roster[: roster.index(db)] for name in dbs):
            taken.append((max(roster.index(name) for name in dbs), path))
    return [path for _, path in sorted(taken, key=lambda t: t[0])], unreadable


def local_databases(results: Results) -> list[str]:
    """The databases of `results` that come from the local snapshot, in its database order."""
    return [name for name, info in results.databases.items() if info.source == "local"]


__all__ = [
    "AGREE",
    "LOCAL_AGREE",
    "NO_RESULTS",
    "MergeError",
    "database_order",
    "earlier_legs",
    "from_snapshot",
    "local_databases",
    "merge",
]
