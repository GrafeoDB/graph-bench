import dataclasses
import re
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from graph_bench.datasets import cache
from graph_bench.results import Entry
from graph_bench.suites import catalog

KNOWN_DISAGREEMENTS = Path(__file__).resolve().parent.parent / "docs" / "known-disagreements.md"


def resources_off(mp: pytest.MonkeyPatch, name: str = "push") -> None:
    """Switch selection `name`'s Memory and Disk records off for a test that does not check them.

    `Selection.resources` is read only by the supervisor, in the test's own process (CliRunner included), so
    patching `catalog.SELECTIONS` there is enough. Call it inside the fixture's or test's `MonkeyPatch` (module
    fixtures need their own `MonkeyPatch.context()`, which an autouse function fixture would not reach).
    """
    mp.setitem(catalog.SELECTIONS, name, dataclasses.replace(catalog.SELECTIONS[name], resources=False))


@pytest.fixture(scope="session")
def push_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A `GRAPH_BENCH_DATA_DIR` holding the push selection's datasets and references, generated once per session."""
    root = tmp_path_factory.mktemp("push-data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(root))
        cache.prepare("push")
    yield root


def _known(db: str) -> set[str]:
    """Benchmark ids recorded for `db` in docs/known-disagreements.md, from its `- id: <db>/<benchmark>` lines."""
    if not KNOWN_DISAGREEMENTS.exists():
        return set()
    text = KNOWN_DISAGREEMENTS.read_text(encoding="utf-8")
    return {m.group(2) for m in re.finditer(r"^- id: (\S+)/(\S+)\s*$", text, re.MULTILINE) if m.group(1) == db}


@pytest.fixture
def known_disagreements() -> Callable[[str], set[str]]:
    """The allowlist reader: database name -> benchmark ids that may fail in the adapter tests."""
    return _known


# The base class's errors for the rows later plan tasks implement per adapter: until an adapter has `open_empty`
# (Startup, Task 3) and `load_ql` (Load (query language), Task 8), those entries are ⊘ with exactly these errors.
# Those tasks remove this allowance (and `not_yet_implemented`) from the adapter tests.
NOT_YET_IMPLEMENTED: dict[str, str] = {"startup": "startup", "load_ql": "no query-language bulk loader"}


def not_yet_implemented(entry: Entry) -> bool:
    """Whether an adapter test's entry is a Startup or Load (query language) ⊘ from the base class default."""
    return entry.status == "not_supported" and NOT_YET_IMPLEMENTED.get(entry.category) == entry.error


def warn_if_fixed(entry: Entry, known: set[str]) -> None:
    """Warn when an allowlisted benchmark's entry now agrees, so a fixed upstream bug is noticed.

    The adapter test keeps passing (an allowlisted id may agree); pytest lists the warning in its summary.
    """
    if entry.benchmark in known and entry.status == "ok" and entry.answer == "agree":
        warnings.warn(f"ALLOWLISTED BUT NOW AGREES: {entry.database}/{entry.benchmark}", stacklevel=2)
