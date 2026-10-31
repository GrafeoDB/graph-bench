import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import SCHEMA_VERSION, Entry, Resource, Results
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


def test_results_round_trip_the_new_rows(tmp_path: Path) -> None:
    """A Startup entry keeps its timing in codspeed mode; a Load (query language) entry carries `load_ns`."""
    entries = [
        Entry(
            id="benches/test_startup.py::test_startup[fake-empty]",
            category="startup",
            model="property_graph",
            benchmark="startup",
            database="fake",
            scale="empty",
            threads=1,
            status="ok",
            timing=Timing(10.0, 12.0, 9, 3),
            preflight_ns=12,
            answer="agree",
            reference="python",
        ),
        Entry(
            id="benches/test_load_ql.py::test_load_ql_snb[fake-small]",
            category="load_ql",
            model="property_graph",
            benchmark="load_ql_snb",
            database="fake",
            scale="small",
            threads=1,
            status="ok",
            load_ns=100,
            answer="disagree",
            detail="content probe scan_order_limit: binding 0: 1 rows != 2 rows",
        ),
    ]
    results = Results(
        graph_bench_commit="abc",
        grafeo={"version": "0.6.0", "source": "pypi", "sha": None},
        machine={"cpu": "x", "cores": 4, "os": "y"},
        selection="push",
        mode="codspeed",
        started_at="2026-10-31T00:00:00Z",
        databases={},
        entries=entries,
    )
    path = tmp_path / "results.json"
    results.write(path)
    assert Results.read(path) == results


def _resource(**fields: Any) -> Resource:
    base: dict[str, Any] = {
        "id": "resources::load_snb[fake-small]",
        "database": "fake",
        "dataset": "snb",
        "store": "graph",
        "scale": "small",
        "threads": 1,
        "status": "ok",
        "answer": "agree",
        "storage": "in-memory",
        "baseline_bytes": 100,
        "after_bytes": 150,
        "footprint_bytes": 50,
        "peak_bytes": 170,
        "samples": 4,
        "sample_interval_ms": 10.0,
        "processes": 1,
    }
    return Resource(**{**base, **fields})


def test_results_round_trip_resources(tmp_path: Path) -> None:
    results = Results(
        graph_bench_commit="abc",
        grafeo={"version": "0.6.0", "source": "pypi", "sha": None},
        machine={"cpu": "x", "cores": 4, "os": "y", "memory_bytes": 16 << 30},
        selection="push",
        mode="bench",
        started_at="2026-10-31T00:00:00Z",
        databases={},
        resources=[
            _resource(),
            _resource(id="resources::load_snb_rdf[fake-small]", store="rdf", storage="on-disk", disk_bytes=4096),
            Resource(
                id="resources::load_graphalytics[fake-small]",
                database="fake",
                dataset="graphalytics",
                store="graph",
                scale="small",
                threads=1,
                status="timeout",
                error="no output for 600 s",
            ),
        ],
    )
    path = tmp_path / "results.json"
    results.write(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == SCHEMA_VERSION == 2
    assert doc["resources"][1]["disk_bytes"] == 4096
    assert set(doc["resources"][0]) == {f.name for f in dataclasses.fields(Resource)}
    assert Results.read(path) == results


def test_results_v1_still_reads(tmp_path: Path) -> None:
    """A schema 1 document (no `resources`, no `machine.memory_bytes`) reads with an empty resources list."""
    doc = {
        "schema_version": 1,
        "graph_bench_commit": "abc",
        "grafeo": {"version": "0.6.0", "source": "pypi", "sha": None},
        "machine": {"cpu": "x", "cores": 4, "os": "y"},
        "selection": "push",
        "mode": "bench",
        "started_at": "2026-10-30T00:00:00Z",
        "databases": {},
        "entries": [],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    results = Results.read(path)
    assert results.resources == []
    assert results.schema_version == SCHEMA_VERSION  # read as the current schema, which it satisfies
    assert "memory_bytes" not in results.machine
    assert results.to_dict()["resources"] == []


def test_results_unknown_schema_is_refused() -> None:
    with pytest.raises(ValueError, match="schema_version 3"):
        Results.from_dict({"schema_version": 3})
