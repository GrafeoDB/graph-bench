"""`graph-bench local-check`: the installed database versions against the local snapshot's."""

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph_bench import local_check
from graph_bench.cli import app
from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import Results

EXTRA = "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2"


@pytest.fixture(autouse=True)
def _extra_databases(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
    yield


def _snapshot(**versions: str) -> Results:
    info = DatabaseInfo("x", "1", "in-process", False, "", "in-memory")
    return Results(
        graph_bench_commit="10ca1",
        grafeo={"version": "0.6.3", "source": "pypi", "sha": None},
        machine={"cpu": "Ryzen 9", "cores": 16, "os": "Linux"},
        selection="full",
        mode="bench",
        started_at="2026-10-20T10:00:00Z",
        databases={name: dataclasses.replace(info, name=name, version=v) for name, v in versions.items()},
    )


def test_installed_version_is_what_a_run_records() -> None:
    """The version a run's database line would record (the fakes report 1.0); None when the database is unknown
    or its driver is not installed."""
    assert local_check.installed_version("fake") == "1.0"
    assert local_check.installed_version("no-such-database") is None


def test_changes_list_the_databases_that_differ() -> None:
    snapshot = _snapshot(fake="1.0", fake2="0.9", gone="2.0")
    changes = local_check.version_changes(snapshot)
    assert changes == [
        local_check.VersionChange("fake2", "0.9", "1.0"),
        local_check.VersionChange("gone", "2.0", None),
    ]
    assert [local_check.describe(c) for c in changes] == [
        "Fake 2: 0.9 in the local snapshot, 1.0 installed",
        "gone: 2.0 in the local snapshot, not installed",
    ]


def test_cli_prints_the_changes_and_exits_0(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _snapshot(fake="1.0", fake2="0.9").write(path)
    result = CliRunner().invoke(app, ["local-check", "--local", str(path)])
    assert result.exit_code == 0, result.output
    # stdout holds only the changes, one per line, for the workflow's notice; the summary goes to stderr.
    assert result.stdout.splitlines() == ["Fake 2: 0.9 in the local snapshot, 1.0 installed"]
    assert "1 of 2 databases differ from the local snapshot of 2026-10-20" in result.stderr


def test_cli_prints_nothing_on_stdout_when_every_version_matches(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _snapshot(fake="1.0", fake2="1.0").write(path)
    result = CliRunner().invoke(app, ["local-check", "--local", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "every database matches the local snapshot of 2026-10-20" in result.stderr


def test_cli_without_a_snapshot_exits_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["local-check", "--local", str(tmp_path / "none.json")])
    assert result.exit_code == 2
