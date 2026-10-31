"""The Startup row: a database's cold start, timed in fresh child processes.

Child: `python -m graph_bench.runner.startup --db NAME --threads (1|all) --workdir DIR` prints one JSON line,
`{"elapsed_ns": int, "rows": [...]}` (the probe's canonical rows) or `{"error": str, "not_supported": bool}`, and
exits 0 either way.

The child's clock: graph-bench's own modules (the database contract, the JVM helper, the dataset schema and RDF
mapping, the probe suite) are imported first, then the clock starts. It covers `get_database(name)` (the
driver's import), `cls(workdir, threads)`, `open_empty()` (whatever the process lacks: a JVM, an engine, a server
process, and an empty database in `workdir`) and the probe, `query_model(primary model, probe, {})`, until its
rows return. `close()` is not timed, nor is the interpreter's own start, the same for every database. Nothing in
the child provisions: the parent ran `Database.provision()` before, untimed. The child inherits the parent's
environment, so the database's `process_env(threads)` applies.

Parent: `run_startups` runs the children one after another, each in a fresh `workdir/startup-<i>` removed after
it. The first child is the preflight: unless it completes with the expected probe answer, no other child runs.
This measures a process-cold start, not an OS-cache-cold one: the driver's files are in the OS cache after the
first child.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# graph-bench's own modules, imported before the child's clock starts (`time.perf_counter_ns` in `measure_start`):
# every adapter imports some of them, and they cost every database the same.
import graph_bench.databases.jvm
import graph_bench.datasets.rdf
import graph_bench.datasets.schema  # noqa: F401
from graph_bench.answers import SNB_EPS, canonical, verify
from graph_bench.databases import get_database, model_support
from graph_bench.databases.base import NotSupported
from graph_bench.runner import message
from graph_bench.runner.timing import summary
from graph_bench.suites.startup import EXPECTED, primary_model, probe_text
from graph_bench.types import AnswerVerdict, Model, Status, Timing

__all__ = [
    "STARTUP_TIMEOUT_S",
    "Child",
    "StartupOutcome",
    "fresh_dir",
    "main",
    "measure_start",
    "passed",
    "run_child",
    "run_startups",
    "spawn_child",
    "startup_entry",
    "summarize",
]

# A child that has not answered after this long is killed and the entry is `timeout`.
STARTUP_TIMEOUT_S = 120.0


def measure_start(db: str, threads: int | None, workdir: Path) -> tuple[int, list[list[Any]]]:
    """In the child: the timed cold start and probe; the elapsed ns and the probe's canonical rows."""
    start = time.perf_counter_ns()
    cls = get_database(db)
    model = primary_model(cls.models)
    text = probe_text(cls.dialect_for(model), model)
    instance = cls(workdir, threads)
    try:
        instance.open_empty()
        rows = [tuple(row) for row in instance.query_model(model, text, {})]
        elapsed = time.perf_counter_ns() - start
    finally:
        instance.close()
    return elapsed, canonical(rows, ordered=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graph_bench.runner.startup",
        description="Open an empty database in this fresh process, run the startup probe, print one JSON line.",
    )
    parser.add_argument("--db", required=True, help="Database name.")
    parser.add_argument("--threads", required=True, choices=("1", "all"), help="1, or all cores.")
    parser.add_argument("--workdir", required=True, help="A fresh directory for the database's files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    doc: dict[str, Any]
    try:
        elapsed, rows = measure_start(args.db, 1 if args.threads == "1" else None, Path(args.workdir))
        doc = {"elapsed_ns": elapsed, "rows": rows}
    except NotSupported as exc:
        doc = {"error": message(exc), "not_supported": True}
    except Exception as exc:
        doc = {"error": message(exc), "not_supported": False}
    sys.stdout.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
    return 0


@dataclass
class Child:
    """One child's result: `ok` with its time and rows, or the reason it gave none."""

    status: Status
    elapsed_ns: int | None = None
    rows: list[list[Any]] | None = None
    error: str | None = None


def _answer(stdout: str) -> dict[str, Any] | None:
    """The child's JSON line: the last stdout line that is one (a driver may print to stdout before it)."""
    for line in reversed(stdout.splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and ("elapsed_ns" in doc or "error" in doc):
            return doc
    return None


def fresh_dir(workdir: Path) -> Path:
    """Empty `workdir` for a child (created, or emptied when a killed run left it)."""
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    return workdir


def run_child(db: str, threads: int | None, workdir: Path, timeout_s: float, *, index: int = 1) -> Child:
    """Run one child in the fresh directory `workdir`, removed afterwards; `index` numbers it in errors."""
    fresh_dir(workdir)
    try:
        return spawn_child(db, threads, workdir, timeout_s, index=index)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def spawn_child(db: str, threads: int | None, workdir: Path, timeout_s: float, *, index: int = 1) -> Child:
    """Start one child in the existing directory `workdir` and read its answer: the process alone, which is what
    CodSpeed times (`benches`); `run_child` adds the directory's creation and removal."""
    argv = [sys.executable, "-m", "graph_bench.runner.startup", "--db", db]
    argv += ["--threads", "1" if threads == 1 else "all", "--workdir", str(workdir)]
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Child("timeout", error=f"startup child {index} gave no answer within {timeout_s:g} s")
    doc = _answer(proc.stdout)
    if doc is None:
        tail = proc.stderr.strip().splitlines()
        error = f"startup child {index} exited with code {proc.returncode}" + (f": {tail[-1]}" if tail else "")
        return Child("failed", error=error)
    if "error" in doc:
        return Child("not_supported" if doc.get("not_supported") else "failed", error=str(doc["error"]))
    return Child("ok", elapsed_ns=int(doc["elapsed_ns"]), rows=doc["rows"])


@dataclass
class StartupOutcome:
    """The Startup entry of one database: the children's times and the verdict over them.

    `answer` is `agree` only when every child's probe rows equal `EXPECTED[primary model]`; `detail` names the
    first that did not. `times_ns` holds the time of every child that answered.
    """

    times_ns: list[int]
    status: Status
    answer: AnswerVerdict
    error: str | None
    detail: str | None
    children: list[Child] = field(default_factory=list, repr=False)

    @property
    def timing(self) -> Timing | None:
        """The median, p99 and minimum over the children, when every child completed with the expected answer."""
        if self.status != "ok" or self.answer != "agree" or not self.times_ns:
            return None
        return summary(self.times_ns)

    @property
    def preflight_ns(self) -> int | None:
        """The first child's time."""
        return self.times_ns[0] if self.times_ns else None


def _verdict(child: Child, model: Model, index: int) -> str | None:
    expected = EXPECTED[model]
    if verify(child.rows, expected, rel_eps=SNB_EPS) is None:
        return None
    return f"child {index}: probe returned {child.rows}, expected {expected}"


def passed(child: Child, model: Model) -> bool:
    """Whether a child completed with the expected probe answer, so the next child may run."""
    return child.status == "ok" and _verdict(child, model, 0) is None


def summarize(model: Model, children: list[Child]) -> StartupOutcome:
    """The outcome of the children run so far: the first that failed or disagreed decides it."""
    times: list[int] = []
    for index, child in enumerate(children, 1):
        if child.status != "ok":
            return StartupOutcome(times, child.status, "unchecked", child.error, None, children)
        assert child.elapsed_ns is not None
        times.append(child.elapsed_ns)
        detail = _verdict(child, model, index)
        if detail is not None:
            return StartupOutcome(times, "ok", "disagree", None, detail, children)
    return StartupOutcome(times, "ok", "agree", None, None, children)


def run_startups(
    db: str,
    threads: int | None,
    iterations: int,
    workdir: Path,
    timeout_s: float,
    *,
    progress: Callable[[], None] | None = None,
) -> StartupOutcome:
    """Run up to `iterations` children, one after another, each in a fresh `workdir/startup-<i>`.

    The first child that fails or disagrees ends the series. `progress` is called after each child. The caller
    runs `Database.provision()` before, untimed.
    """
    model = primary_model(model_support(db))
    children: list[Child] = []
    for i in range(iterations):
        child = run_child(db, threads, workdir / f"startup-{i}", timeout_s, index=i + 1)
        children.append(child)
        if progress is not None:
            progress()
        if not passed(child, model):
            break
    return summarize(model, children)


def startup_entry(bench_id: str, outcome: StartupOutcome) -> dict[str, Any]:
    """The Startup entry line: the timing over the children in both modes, `preflight_ns` the first child's."""
    timing = outcome.timing
    return {
        "kind": "entry",
        "id": bench_id,
        "status": outcome.status,
        "error": outcome.error,
        "timing": asdict(timing) if timing is not None else None,
        "preflight_ns": outcome.preflight_ns,
        "answer": outcome.answer,
        "reference": "python",
        "answer_digest": None,
        "detail": outcome.detail,
    }


if __name__ == "__main__":
    sys.exit(main())
