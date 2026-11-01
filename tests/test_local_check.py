"""`graph-bench local-check`: the local snapshot's versions against the newest stable releases, outside the roster.

Every PyPI lookup is a fake: no test touches the network.
"""

import dataclasses
import io
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from graph_bench import local_check, release
from graph_bench.cli import app
from graph_bench.databases import PACKAGES, pins
from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import Results
from graph_bench.suites import catalog

# The newest stable release per PyPI package, as the fake lookup reports it.
LATEST = {"falkordblite": "0.12.0", "helix-db": "0.3.4", "arcadedb-embedded": "26.10.2.post1", "grafeo": "9.9.9"}


def _lookup(package: str) -> str:
    if package == "helix-db":
        raise OSError("network down")
    return LATEST.get(package, "1.0")


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


def test_every_non_roster_database_has_a_package_or_a_pin() -> None:
    """Neo4j is checked against its pin; every other database outside the roster has a PyPI package."""
    from graph_bench.databases import KNOWN

    outside = [db for db in KNOWN if db not in catalog.CI_DATABASES]
    assert sorted(outside) == ["arcadedb", "falkordb", "helixdb", "neo4j"]
    assert {db: PACKAGES[db] for db in outside if db != "neo4j"} == {
        "falkordb": "falkordblite",
        "helixdb": "helix-db",
        "arcadedb": "arcadedb-embedded",
    }
    assert "neo4j" not in PACKAGES


def test_checks_compare_the_snapshot_with_the_newest_stable_release() -> None:
    snapshot = _snapshot(falkordb="0.11.0", arcadedb="26.10.2.post1", helixdb="0.3.3", neo4j="2026.9.0")
    checks = local_check.checks(snapshot, _lookup)
    by_db = {c.database: c for c in checks}
    assert list(by_db) == ["falkordb", "arcadedb", "helixdb", "neo4j"]
    assert (by_db["falkordb"].latest, by_db["falkordb"].changed) == ("0.12.0", True)
    assert (by_db["arcadedb"].latest, by_db["arcadedb"].changed) == ("26.10.2.post1", False)  # a post release
    assert (by_db["helixdb"].latest, by_db["helixdb"].changed) == (None, False)  # a lookup error: unknown
    assert by_db["helixdb"].error == "network down"
    assert (by_db["neo4j"].latest, by_db["neo4j"].pinned, by_db["neo4j"].changed) == (pins.NEO4J_VERSION, True, True)
    assert [c.database for c in local_check.version_changes(snapshot, _lookup)] == ["falkordb", "neo4j"]
    assert [local_check.describe(c) for c in checks] == [
        "FalkorDB: 0.11.0 in the local snapshot, 0.12.0 is the newest stable release",
        "ArcadeDB: 26.10.2.post1 in the local snapshot, 26.10.2.post1 is the newest stable release",
        "HelixDB: 0.3.3 in the local snapshot, newest stable release unknown (network down)",
        f"Neo4j: 2026.9.0 in the local snapshot, {pins.NEO4J_VERSION} is the pinned release (Neo4j is not on PyPI;"
        " pins.yml keeps graph_bench/databases/pins.py at the newest stable)",
    ]


def test_a_changed_roster_or_grafeo_version_is_not_reported() -> None:
    """The roster's databases, Grafeo included, are measured by CI: the snapshot's versions of them never matter."""
    roster = {db: "0.0.1" for db in catalog.CI_DATABASES}
    snapshot = _snapshot(**roster, falkordb="0.12.0")
    looked_up: list[str] = []

    def lookup(package: str) -> str:
        looked_up.append(package)
        return _lookup(package)

    assert local_check.version_changes(snapshot, lookup) == []
    assert looked_up == ["falkordblite"]  # the roster's packages, grafeo's included, are never looked up
    assert [c.database for c in local_check.checks(snapshot, lookup)] == ["falkordb"]


def test_an_unknown_database_is_unknown() -> None:
    [check] = local_check.checks(_snapshot(custom="1.0"), _lookup)
    assert (check.latest, check.changed) == (None, False)
    assert check.error == "no PyPI package known"


def test_cli_prints_the_changes_and_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_check, "newest_release", _lookup)
    path = tmp_path / "results.json"
    _snapshot(grafeo="0.0.1", falkordb="0.11.0", arcadedb="26.10.2.post1", helixdb="0.3.3").write(path)
    result = CliRunner().invoke(app, ["local-check", "--local", str(path)])
    assert result.exit_code == 0, result.output
    # stdout holds only the changes, one per line, for the workflow's notice; the rest goes to stderr.
    assert result.stdout.splitlines() == ["FalkorDB: 0.11.0 in the local snapshot, 0.12.0 is the newest stable release"]
    assert "HelixDB: 0.3.3 in the local snapshot, newest stable release unknown (network down)" in result.stderr
    assert "1 of 3 databases outside the CI roster differ from the local snapshot of 2026-10-20" in result.stderr


def test_cli_prints_nothing_on_stdout_when_every_version_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_check, "newest_release", _lookup)
    path = tmp_path / "results.json"
    _snapshot(falkordb="0.12.0").write(path)
    result = CliRunner().invoke(app, ["local-check", "--local", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "every database outside the CI roster matches the local snapshot of 2026-10-20" in result.stderr


def test_cli_without_a_snapshot_exits_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["local-check", "--local", str(tmp_path / "none.json")])
    assert result.exit_code == 2


def test_fetch_latest_takes_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """One PyPI lookup for every package: the url and the error name it; post releases count only when asked."""
    urls: list[str] = []
    live = [{"yanked": False}]
    body: dict[str, Any] = {"releases": {"26.10.2": live, "26.10.2.post1": live, "27.0.0rc1": live}}

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
    assert release.fetch_latest("arcadedb-embedded") == "26.10.2"
    assert release.fetch_latest("arcadedb-embedded", post=True) == "26.10.2.post1"
    assert release.fetch_latest() == "26.10.2"
    assert urls == [
        "https://pypi.org/pypi/arcadedb-embedded/json",
        "https://pypi.org/pypi/arcadedb-embedded/json",
        "https://pypi.org/pypi/grafeo/json",
    ]
    body["releases"] = {"1.0.0a1": live}
    with pytest.raises(ValueError, match="no stable helix-db release"):
        release.fetch_latest("helix-db")
