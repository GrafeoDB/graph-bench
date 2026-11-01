"""`graph-bench local-check`: which databases the local snapshot measured at another version than is installed now.

The local snapshot (`results/local/results.json`, `.github/scripts/local-bench.sh`) is refreshed by hand. CI
installs the newest stable release of every database, so when a database's installed version differs from the
snapshot's `databases[*].version`, the snapshot is due for a refresh. The installed version is read the way a run
records it (`worker.database_line`: a probe instance's `info()`), so the two strings compare like for like.
"""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graph_bench.databases import get_database
from graph_bench.report.ranking import safe_display_name
from graph_bench.results import Results
from graph_bench.runner.worker import database_line


@dataclass(frozen=True)
class VersionChange:
    """A database whose installed version (None: not installed) is not the snapshot's."""

    database: str
    snapshot: str
    installed: str | None


def installed_version(name: str) -> str | None:
    """The version a run would record for `name` now; None when it is unknown, not installed or its probe fails."""
    try:
        cls = get_database(name)
        with tempfile.TemporaryDirectory(prefix="graph-bench-check-") as work:
            return str(database_line(cls, Path(work), 1)["version"])
    except Exception:  # any driver error (import, JVM, native library) means the version cannot be read
        return None


def version_changes(
    snapshot: Results, installed: Callable[[str], str | None] = installed_version
) -> list[VersionChange]:
    """The snapshot's databases, in its order, whose installed version differs from the recorded one."""
    changes = []
    for name, info in snapshot.databases.items():
        now = installed(name)
        if now != info.version:
            changes.append(VersionChange(name, info.version, now))
    return changes


def describe(change: VersionChange) -> str:
    now = f"{change.installed} installed" if change.installed is not None else "not installed"
    return f"{safe_display_name(change.database)}: {change.snapshot} in the local snapshot, {now}"


__all__ = ["VersionChange", "describe", "installed_version", "version_changes"]
