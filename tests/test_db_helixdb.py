"""HelixDB adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("helixdb")
pytest.importorskip("helixdb_uniffi")

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.databases.helix_queries import BUILDERS, scan_filter_avg, snb_ic13
from graph_bench.results import Results
from graph_bench.suites import Query, catalog
from graph_bench.suites.startup import PROBE
from graph_bench.types import Model
from tests.conftest import (
    assert_loads_agree,
    assert_resources_ok,
    assert_startup_agrees,
    not_yet_implemented,
    run_push,
    warn_if_fixed,
)

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

pytestmark = pytest.mark.helixdb


@pytest.fixture(scope="module")
def push(push_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Results:
    """One push run of helixdb, shared by the push, startup and load tests."""
    return run_push("helixdb", tmp_path_factory.mktemp("push-helixdb"), push_data_dir, 120)


def test_helixdb_push_agrees(push: Results, known_disagreements: Callable[[str], set[str]]) -> None:
    results = push
    known = known_disagreements("helixdb")
    entries = [e for e in results.entries if e.database == "helixdb"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate ("model <model> not supported"): the database lacks the data
        # model. Any other ⊘ (a missing capability) must be allowlisted in docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        # A load must agree too: its node and edge counts (`counts()`) match the dataset's.
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_helixdb_startup_agrees(push: Results) -> None:
    """The Startup row: every child opens an empty database and answers the probe as expected."""
    assert_startup_agrees(push, "helixdb")


def test_helixdb_load_unchanged(push: Results) -> None:
    """The loads still load the whole dataset (their counts agree) with `open_empty` split out."""
    assert_loads_agree(push, "helixdb")


def test_helixdb_resources_ok(push: Results) -> None:
    """Memory and Disk: one ok, agreeing record per load with ordered readings; in memory."""
    assert_resources_ok(push, "helixdb", on_disk=False)


def test_helixdb_settings(tmp_path: Path) -> None:
    cls = get_database("helixdb")
    assert cls.process_env(1) == {}
    info = cls(tmp_path, 1).info()
    assert not info.threads_configurable
    assert info.storage == "in-memory"
    assert info.process == "in-process"


def test_every_query_names_a_builder_or_a_reason() -> None:
    """Each property graph query's `helix` text is its id, and each id has a builder or a reason (never a KeyError)
    that starts with the query's name (IS3, ...; the title for a scan); the Startup probe's text names its builder
    too."""
    queries = [q for q in catalog.ITEMS if isinstance(q, Query) and q.model == "property_graph"]
    assert all(q.text["helix"] == q.id for q in queries)
    assert {q.id for q in queries} | {PROBE["helix"]} == set(BUILDERS)
    assert callable(BUILDERS[PROBE["helix"]])
    for q in queries:
        builder = BUILDERS[q.id]
        name = q.id.removeprefix("snb_").upper() if q.id.startswith("snb_") else q.title
        assert callable(builder) or (isinstance(builder, str) and builder.startswith(name)), q.id


def test_unsupported_queries_raise_not_supported(tmp_path: Path) -> None:
    db = get_database("helixdb")(tmp_path, 1)
    for name in ("snb_is3", "snb_is7", "snb_ic6"):
        with pytest.raises(NotSupported, match=r"ORDER BY|ranking"):
            db.query(name, {})
    with pytest.raises(NotSupported, match="grouping by two properties"):
        db.query("scan_aggregate", {})
    for algo in ("bfs", "wcc", "pr"):
        with pytest.raises(NotSupported):
            db.algorithm(algo, {})


def test_ic13_rows_count_the_edges_of_the_returned_path() -> None:
    """IC13 reports hops: the nodes of HelixDB's path minus one, 0 for the single-node path, -1 for no path."""
    request, rows = snb_ic13({"person1Id": "a", "person2Id": "b"})
    assert request is not None
    assert rows({"p": []}) == [(-1,)]
    assert rows({"p": [7]}) == [(0,)]
    assert rows({"p": [7, 8, 9]}) == [(2,)]


def test_scan_filter_avg_rows_read_the_count_and_the_mean() -> None:
    """The count entry is a number; the mean entry is `aggregate_by`'s one map with one value."""
    request, rows = scan_filter_avg({"min": 25})
    assert request is not None
    assert rows({"n": 3, "mean": [{"length_Mean": 27.5}]}) == [(3, 27.5)]
