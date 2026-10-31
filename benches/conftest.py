"""pytest-codspeed entry point: `pytest benches --codspeed --db NAME [options]`, one process per database.

`graph-bench codspeed` starts these processes with the environment each database needs (`process_env`) and
reads the JSON lines they write to `--gb-out`. Test ids equal `graph-bench list` exactly:
`benches/test_<category>.py::test_<item>[<db>-<scale>]` and `benches/test_load.py::test_load_<dataset>[...]`
(`test_load_<dataset>_rdf` for the RDF store). Cases run grouped by dataset and store, the load test first, so each
(dataset, store) is loaded once per session.

Running it by hand: pass `-o addopts=` (or `--no-cov`), as `graph-bench codspeed` does. Otherwise pyproject's
test-suite addopts (`--cov ... --cov-fail-under=88`) apply, which fail the run and distort the timings. Also set
the database's `process_env` (e.g. `RAYON_NUM_THREADS=1` for Grafeo) yourself, e.g.
`RAYON_NUM_THREADS=1 python -m pytest benches --codspeed -o addopts= --db grafeo`.
"""

from pathlib import Path
from typing import Any

import pytest

from benches._cases import Bench, LoadCase, case_key
from graph_bench.runner.worker import DEFAULT_CASE_TIMEOUT_S, parse_budget
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case

_BENCH = pytest.StashKey[Bench]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("graph-bench")
    group.addoption("--db", required=True, help="Database name.")
    group.addoption("--selection", default="push", help="Selection name (default: push).")
    group.addoption(
        "--threads", choices=("1", "all"), default=None, help="1 or all cores; default: the selection's settings."
    )
    group.addoption("--gb-out", default=None, metavar="FILE.jsonl", help="JSON lines file to append to.")
    group.addoption("--workdir", default=None, help="Directory for database files; default: a temporary one.")
    group.addoption(
        "--gb-budget",
        action="append",
        default=[],
        type=parse_budget,
        metavar="ID=SECONDS",
        help="Preflight budget of a case (repeatable).",
    )
    group.addoption(
        "--case-timeout",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_S,
        metavar="SECONDS",
        help="Preflight budget of a case without --gb-budget.",
    )


def pytest_configure(config: pytest.Config) -> None:
    try:
        selection = catalog.selection(config.getoption("--selection"))
        catalog.cases(selection.name, [config.getoption("--db")])
    except KeyError as exc:
        raise pytest.UsageError(str(exc.args[0])) from None
    threads_opt = config.getoption("--threads")
    threads = selection.threads if threads_opt is None else (1 if threads_opt == "1" else None,)
    out, workdir = config.getoption("--gb-out"), config.getoption("--workdir")
    config.stash[_BENCH] = Bench(
        config.getoption("--db"),
        selection,
        threads,
        Path(out) if out else None,
        Path(workdir) if workdir else None,
        dict(config.getoption("--gb-budget")),
        config.getoption("--case-timeout"),
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    bench = config.stash.get(_BENCH, None)
    if bench is not None:
        bench.close()


def pytest_sessionstart(session: pytest.Session) -> None:
    bench = session.config.stash[_BENCH]
    if bench.out is not None and not session.config.option.collectonly:
        bench.emit_database()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    bench = metafunc.config.stash[_BENCH]
    item = getattr(metafunc.function, "gb_item", None)
    load = getattr(metafunc.function, "gb_load", None)
    params: list[Case] | list[LoadCase]
    if item is not None:
        params = bench.cases(item)
    elif load is not None:
        params = bench.loads(load)
    else:
        return
    ids = [catalog.param_id(p.db, p.scale, p.threads) for p in params]
    metafunc.parametrize("gb_case", params, ids=ids)


def _order(item: pytest.Item) -> tuple[Any, ...]:
    """Group by dataset and store, load first; keep pytest's order otherwise (the sort is stable)."""
    case = item.callspec.params["gb_case"]  # type: ignore[attr-defined]
    dataset, store, scale, threads = case.key if isinstance(case, LoadCase) else case_key(case)
    return (dataset, store, scale, threads is None, 0 if isinstance(case, LoadCase) else 1)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep exactly the planned ids (an item without cases would show up as `[NOTSET]`), grouped by dataset."""
    expected = config.stash[_BENCH].expected_ids()
    kept = [i for i in items if i.nodeid in expected]
    dropped = [i for i in items if i.nodeid not in expected]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = sorted(kept, key=_order)


@pytest.fixture(scope="session")
def graph_bench(pytestconfig: pytest.Config) -> Bench:
    """The session state; it loads each dataset once and keeps one loaded instance at a time."""
    return pytestconfig.stash[_BENCH]
