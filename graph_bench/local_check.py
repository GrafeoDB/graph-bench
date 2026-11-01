"""`graph-bench local-check`: which databases of the local snapshot have a newer stable release than it measured.

The local snapshot (`results/local/results.json`, `.github/scripts/local-bench.sh`) is refreshed by hand. Only its
databases outside the CI roster reach the README (`merge.from_snapshot`, the same set `merge --local` takes), so
only those are checked. Each recorded version (`databases[*].version`) is compared with the newest stable release
of the database's PyPI package (`databases.PACKAGES`, `release.fetch_latest`): CI installs that release, so a
difference means the snapshot is due for a refresh. Neo4j is not on PyPI: it is compared with the pinned release,
`pins.NEO4J_VERSION`, which pins.yml keeps at the newest stable. A lookup that fails leaves that database
"unknown"; the check never fails.
"""

from collections.abc import Callable
from dataclasses import dataclass

from graph_bench import release
from graph_bench.databases import PACKAGES, pins
from graph_bench.merge import from_snapshot
from graph_bench.report.ranking import safe_display_name
from graph_bench.results import Results

NEO4J = "neo4j"
NO_PACKAGE = "no PyPI package known"
PINNED_NOTE = "Neo4j is not on PyPI; pins.yml keeps graph_bench/databases/pins.py at the newest stable"


@dataclass(frozen=True)
class VersionCheck:
    """One database: the snapshot's version and the newest stable release (None: unknown, see `error`).

    `pinned` marks Neo4j, whose newest release is the pin in pins.py, not a PyPI lookup.
    """

    database: str
    snapshot: str
    latest: str | None
    pinned: bool = False
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.latest is not None and not release.same_version(self.snapshot, self.latest)


def newest_release(package: str) -> str:
    """The newest stable release of `package` on PyPI, post releases included (as an install picks them)."""
    return release.fetch_latest(package, post=True)


def checks(snapshot: Results, lookup: Callable[[str], str] | None = None) -> list[VersionCheck]:
    """One check per snapshot database outside the CI roster, in the snapshot's order. `lookup` (package to its
    newest stable version) defaults to `newest_release`; any error it raises makes that database unknown."""
    lookup = lookup or newest_release
    found = []
    for name, info in snapshot.databases.items():
        if not from_snapshot(name):
            continue
        if name == NEO4J:
            found.append(VersionCheck(name, info.version, pins.NEO4J_VERSION, pinned=True))
            continue
        package = PACKAGES.get(name)
        if package is None:
            found.append(VersionCheck(name, info.version, None, error=NO_PACKAGE))
            continue
        try:
            found.append(VersionCheck(name, info.version, lookup(package)))
        except Exception as exc:  # a network, HTTP or PyPI format error: unknown, never a failed check
            found.append(VersionCheck(name, info.version, None, error=str(exc) or type(exc).__name__))
    return found


def version_changes(snapshot: Results, lookup: Callable[[str], str] | None = None) -> list[VersionCheck]:
    """The checks whose newest stable release is known and differs from the snapshot's version."""
    return [check for check in checks(snapshot, lookup) if check.changed]


def describe(check: VersionCheck) -> str:
    head = f"{safe_display_name(check.database)}: {check.snapshot} in the local snapshot"
    if check.latest is None:
        return f"{head}, newest stable release unknown ({check.error})"
    if check.pinned:
        return f"{head}, {check.latest} is the pinned release ({PINNED_NOTE})"
    return f"{head}, {check.latest} is the newest stable release"


__all__ = ["VersionCheck", "checks", "describe", "newest_release", "version_changes"]
