import dataclasses
import warnings
from pathlib import Path

import pytest

from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import SCHEMA_VERSION, Entry, Results
from graph_bench.types import Timing


def test_results_round_trip(tmp_path: Path) -> None:
    info = DatabaseInfo("fake", "1.0", "in-process", False, "fake", "in-memory")
    entries = [
        Entry(
            id="benches/test_hops.py::test_snb_ic13[fake-small]",
            category="hops",
            model="property_graph",
            benchmark="snb_ic13",
            database="fake",
            scale="small",
            threads=1,
            status="ok",
            timing=Timing(10.0, 12.0, 9, 5),
            preflight_ns=11,
            answer="agree",
            reference="python",
            answer_digest="abc",
        ),
        Entry(
            id="benches/test_load.py::test_load_snb[fake-small]",
            category="load",
            model="property_graph",
            benchmark="load_snb",
            database="fake",
            scale="small",
            threads=None,
            status="ok",
            load_ns=100,
        ),
    ]
    results = Results(
        graph_bench_commit="abc",
        grafeo={"version": "0.6.0", "source": "pypi", "sha": None},
        machine={"cpu": "x", "cores": 4, "os": "y"},
        selection="push",
        mode="bench",
        started_at="2026-10-30T00:00:00Z",
        databases={"fake": info},
        entries=entries,
    )
    path = tmp_path / "out" / "results.json"
    results.write(path)
    back = Results.read(path)
    assert back == results
    assert back.schema_version == SCHEMA_VERSION


def test_warn_if_fixed_flags_an_allowlisted_entry_that_agrees() -> None:
    from tests.conftest import warn_if_fixed

    entry = Entry(
        id="x/snb_is1/small/1",
        category="hops",
        model="property_graph",
        benchmark="snb_is1",
        database="x",
        scale="small",
        threads=1,
        status="ok",
        answer="agree",
    )
    with pytest.warns(UserWarning, match=r"ALLOWLISTED BUT NOW AGREES: x/snb_is1"):
        warn_if_fixed(entry, {"snb_is1"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_fixed(entry, set())
        warn_if_fixed(dataclasses.replace(entry, answer="disagree"), {"snb_is1"})
