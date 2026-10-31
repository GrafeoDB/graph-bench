import dataclasses
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import get_args

import pytest

from graph_bench.datasets import cache
from graph_bench.results import Entry, Results
from graph_bench.runner.supervisor import run_bench
from graph_bench.suites import catalog
from graph_bench.types import Model

# The silence limit of tests that hang a case on purpose. Startup children get half of it, and a fresh Python
# importing graph-bench can take several seconds on a loaded machine, so 6 s (3 s per child) was flaky.
HANG_TIMEOUT_S = 12.0

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


# The base class's errors for the rows later plan tasks implement per adapter: until an adapter has `load_ql`
# (Load (query language), Task 8), those entries are ⊘ with exactly this error. Task 8 removes this allowance (and
# `not_yet_implemented`) from the adapter tests. Every adapter has `open_empty` (Startup), so Startup has none.
NOT_YET_IMPLEMENTED: dict[str, str] = {"load_ql": "no query-language bulk loader"}

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")


def not_yet_implemented(entry: Entry) -> bool:
    """Whether an adapter test's entry is a Load (query language) ⊘ from the base class default."""
    return entry.status == "not_supported" and NOT_YET_IMPLEMENTED.get(entry.category) == entry.error


def run_push(
    db: str,
    root: Path,
    data_dir: Path,
    case_timeout_s: float,
    env: Mapping[str, str] | None = None,
) -> Results:
    """One push run of `db` (resources off) in `root` (the working directory, `out` and `work` go there), shared by
    an adapter module's push, startup and load tests. For module fixtures: it uses its own `MonkeyPatch`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(data_dir))
        for key, value in (env or {}).items():
            mp.setenv(key, value)
        resources_off(mp)  # Task 4 turns them on with test_<db>_resources_ok
        return run_bench("push", [db], root / "out", workdir=root / "work", case_timeout_s=case_timeout_s)


def assert_startup_agrees(results: Results, db: str) -> Entry:
    """The push run's Startup entry of `db`: ok, agree, every child timed (median and first child > 0)."""
    [entry] = [e for e in results.entries if e.database == db and e.category == "startup"]
    assert (entry.status, entry.answer) == ("ok", "agree"), f"{entry.error} {entry.detail}"
    assert entry.timing is not None
    assert entry.timing.median_ns > 0
    assert entry.timing.iterations == catalog.SELECTIONS["push"].startup_iterations
    assert entry.preflight_ns is not None and entry.preflight_ns > 0
    return entry


def assert_loads_agree(results: Results, db: str) -> None:
    """Every Load entry of the push run that the model gate lets run is ok and agrees (its counts match)."""
    loads = [e for e in results.entries if e.database == db and e.category == "load"]
    run = [e for e in loads if not (e.status == "not_supported" and MODEL_GATE.fullmatch(e.error or ""))]
    assert run
    bad = [
        f"{e.id}: {e.status} {e.answer} {e.error} {e.detail}" for e in run if (e.status, e.answer) != ("ok", "agree")
    ]
    assert not bad, "\n".join(bad)


def warn_if_fixed(entry: Entry, known: set[str]) -> None:
    """Warn when an allowlisted benchmark's entry now agrees, so a fixed upstream bug is noticed.

    The adapter test keeps passing (an allowlisted id may agree); pytest lists the warning in its summary.
    """
    if entry.benchmark in known and entry.status == "ok" and entry.answer == "agree":
        warnings.warn(f"ALLOWLISTED BUT NOW AGREES: {entry.database}/{entry.benchmark}", stacklevel=2)
