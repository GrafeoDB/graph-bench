import re
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from graph_bench.datasets import cache
from graph_bench.results import Entry

KNOWN_DISAGREEMENTS = Path(__file__).resolve().parent.parent / "docs" / "known-disagreements.md"


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


def warn_if_fixed(entry: Entry, known: set[str]) -> None:
    """Warn when an allowlisted benchmark's entry now agrees, so a fixed upstream bug is noticed.

    The adapter test keeps passing (an allowlisted id may agree); pytest lists the warning in its summary.
    """
    if entry.benchmark in known and entry.status == "ok" and entry.answer == "agree":
        warnings.warn(f"ALLOWLISTED BUT NOW AGREES: {entry.database}/{entry.benchmark}", stacklevel=2)
