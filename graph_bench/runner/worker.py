"""The worker: runs one database over a selection in its own process and streams JSON lines.

`python -m graph_bench.runner.worker --db NAME --selection S --threads (1|all) --workdir DIR --out FILE.jsonl
[--skip ID ...] [--budget ID=SECONDS ...] [--case-timeout SECONDS]`

Lines, appended to `--out` and flushed one by one:

- `{"kind": "database", <DatabaseInfo fields>}`, always first;
- `{"kind": "start", "id": ...}` before each case, each reported load and the startup, so a supervisor knows what
  hung or crashed;
- `{"kind": "run", "id": ...}` immediately before a case's preflight database call, and a `progress` line
  immediately after it returns (or raises), before any harness work: the window between the two is the database
  call alone, which is what a supervisor's in-run cutoff times;
- `{"kind": "progress", "id": ...}` after a case's preflight call, after each of its warm-up and timed runs, after
  each load iteration (`id` is null for an unreported load) and after each startup child, so a supervisor can tell
  a long case from a hang;
- `{"kind": "load", "dataset": ..., "store": ..., "scale": ...}` immediately before the loads of a dataset its
  cases run on (after the Load row's `start` line, if reported), so a supervisor can stop a load at the load ceiling
  and knows which cases it takes with it;
- `{"kind": "entry", "id": ..., "status": ..., ...}` with a case's result, a load's `load_ns` and its count
  check (`answer` and `detail`, see `load_entry`), or the startup's (see `runner.startup.startup_entry`).

The order: the startup first (before any load, after `Database.provision()`, untimed), then per (dataset, store,
scale) its load, its read cases, its write cases and, once that instance is released, its Load (query language)
entry on fresh instances of its own. The answer-only fixtures (`catalog.ANSWER_ONLY_DATASETS`, loaded without a Load
entry) come last.

The canonical answers of each case go to `<out dir>/<out stem>/answers/<sanitized id>.json`.

Limits (`graph_bench.runner`, Task T): a case stops at its ceiling (`--case-ceiling`, from its `start` line): no
warm-up or timed run starts past it, and with fewer than `MIN_ITERATIONS` timed runs the case is ⏱ "too slow: over
60 s". Timed runs stop once they total `TIME_BOX_S` (at least `MIN_ITERATIONS` of them; loads at least
`MIN_LOAD_ITERATIONS`). A load that ends past the load ceiling (`--load-ceiling`) is ⏱ "load over 5 min", and so
is every case of its dataset and store, and its Load (query language) entry (graph store), without running them.
"""

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

from graph_bench.answers import canonical, canonical_map, check_case, digest, verify
from graph_bench.databases import get_database
from graph_bench.databases.base import Database, NotSupported
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.results import Reference
from graph_bench.runner import (
    CASE_CEILING_S,
    CEILING_GRACE_S,
    CONTROL_FLOW,
    DEFAULT_CASE_TIMEOUT_S,
    LOAD_CEILING_S,
    MIN_ITERATIONS,
    MIN_LOAD_ITERATIONS,
    TIME_BOX_S,
    close_quietly,
    load_over,
    message,
    over_ceiling,
    reraise_control,
)
from graph_bench.runner.startup import STARTUP_TIMEOUT_S, run_startups, startup_entry
from graph_bench.runner.timing import OverCeiling, measure
from graph_bench.suites import Algorithm, Item, Query, Write, catalog
from graph_bench.suites.catalog import Case, Selection
from graph_bench.suites.graphalytics import min_member
from graph_bench.suites.startup import primary_model
from graph_bench.types import AnswerVerdict, Status, Store

__all__ = [
    "DEFAULT_CASE_TIMEOUT_S",
    "Emit",
    "LoadCheck",
    "LoadOverCeiling",
    "Loaded",
    "Preflight",
    "ReadbackDiffers",
    "SetupDiffers",
    "WriteRun",
    "case_entry",
    "database_line",
    "emitter",
    "fresh_instance",
    "load",
    "load_entry",
    "main",
    "message",
    "prepare_environment",
    "reads_first",
    "remove",
    "run",
    "run_load_ql",
    "run_startup",
    "start_case",
    "startup_timeout",
    "time_write",
    "unsupported_load_entry",
    "write_guard",
]

Emit = Callable[[dict[str, Any]], None]


def reference_kind(item: Item) -> Reference:
    """Where an item's reference answers come from: NetworkX for algorithms, plain Python for queries and writes."""
    return "networkx" if isinstance(item, Algorithm) else "python"


def make_run(db: Database, item: Item, bindings: list[dict[str, Any]]) -> Callable[[], list[Any]]:
    """The timed function: run the item once per binding and return the raw per-binding results.

    A query runs in its model's dialect (`Database.dialect_for`) through `Database.query_model`, a write's `run`
    statement through `Database.write` (each call committed).
    """
    if isinstance(item, Algorithm):
        name = item.name
        return lambda: [db.algorithm(name, binding) for binding in bindings]
    model = item.model
    if isinstance(item, Write):
        statement = item.run_for(db.dialect_for(model))
        return lambda: [db.write(model, statement, binding) for binding in bindings]
    text = item.text_for(db.dialect_for(model))
    return lambda: [db.query_model(model, text, binding) for binding in bindings]


def canonical_answers(item: Item, raw: list[Any]) -> list[Any]:
    """Canonicalize raw per-binding results so they compare and hash like the reference answers."""
    if isinstance(item, (Query, Write)):
        return [canonical(rows, ordered=item.ordered) for rows in raw]
    answers = [canonical_map(result, drop_unreachable=item.name == "bfs") for result in raw]
    if item.name == "wcc":
        answers = [min_member(answer) for answer in answers]
    return answers


class ReadbackDiffers(Exception):
    """A write case's read-back after a run (past the preflight) did not match its references."""


class SetupDiffers(Exception):
    """A write case's setup did not leave the state of its `ready` reference: the case fails (✗)."""


@dataclass
class WriteRun:
    """A write case's phases on one loaded instance, in the database's dialect; each runs once per binding.

    `after` and `before` are the reference rows per binding after the write and in the loaded state; `ready` the
    rows after the setup, for a write with `Write.ready` (else None). `baseline` is the instance's `counts()` before
    the case (None when it cannot count), which `write_guard` compares with.
    """

    db: Database
    item: Write
    bindings: list[dict[str, Any]]
    after: list[Any]
    before: list[Any]
    baseline: tuple[int, int] | None = None
    ready: list[Any] | None = None

    @property
    def dialect(self) -> str:
        return self.db.dialect_for(self.item.model)

    def _statements(self, statements: tuple[str, ...]) -> None:
        for binding in self.bindings:
            for statement in statements:
                self.db.write(self.item.model, statement, binding)

    def setup(self) -> None:
        """The untimed statements before a run."""
        self._statements(self.item.setup_for(self.dialect))

    def prepare(self) -> None:
        """Before every run, untimed: `setup`, then the check against `ready`; `SetupDiffers` when it misses."""
        self.setup()
        if self.ready is not None:
            problem = self.differs(self.ready)
            if problem is not None:
                raise SetupDiffers(f"setup did not reach its state: {problem}")

    def run(self) -> list[Any]:
        """The timed write: one committed `write` call per binding."""
        return make_run(self.db, self.item, self.bindings)()

    def teardown(self) -> None:
        """The untimed statements that undo a run."""
        self._statements(self.item.teardown_for(self.dialect))

    def undo(self) -> None:
        """Best-effort teardown after a run or setup that raised; the counts guard reports what it leaves."""
        try:
            self.teardown()
        except BaseException as exc:
            reraise_control(exc)

    def check(self) -> list[Any]:
        """The read-back, untimed: the check query per binding, canonicalized."""
        text = self.item.check_for(self.dialect)
        rows = [self.db.query_model(self.item.model, text, binding) for binding in self.bindings]
        return canonical_answers(self.item, rows)

    def differs(self, reference: list[Any]) -> str | None:
        """The first difference between a read-back now and `reference`, or None."""
        return verify(self.check(), reference, rel_eps=self.item.rel_eps)

    def readback(self) -> str | None:
        """After a run: the check against `after`, the teardown, the check against `before`; the first difference."""
        problem = self.differs(self.after)
        self.teardown()
        if problem is not None:
            return f"after the run: {problem}"
        problem = self.differs(self.before)
        return None if problem is None else f"after the teardown: {problem}"


@dataclass
class Preflight:
    """The untimed first run of a case: its answers, verdict and whether it may be timed.

    For a write case, `write` holds its phases and `answers` its read-back after the run.
    """

    run: Callable[[], list[Any]]
    bindings: list[dict[str, Any]]
    status: Status = "ok"
    error: str | None = None
    preflight_ns: int | None = None
    answer: AnswerVerdict = "unchecked"
    answers: list[Any] | None = None
    detail: str | None = None
    write: WriteRun | None = field(default=None, repr=False)

    @property
    def passed(self) -> bool:
        """Completed within budget with a correct answer: only then is the case timed."""
        return self.status == "ok" and self.answer == "agree"

    def fail(self, exc: BaseException, prefix: str = "") -> "Preflight":
        """`not_supported` for `NotSupported`, else `failed`, with the exception's message."""
        self.status = "not_supported" if isinstance(exc, NotSupported) else "failed"
        self.error = prefix + message(exc)
        return self


def _no_emit(line: dict[str, Any]) -> None:
    return None


def _budget(result: Preflight, budget_s: float) -> Preflight:
    assert result.preflight_ns is not None
    seconds = result.preflight_ns / 1e9
    if seconds > budget_s:
        result.status, result.error = "timeout", f"too slow: {seconds:.3g} s > budget {budget_s:.3g} s"
    return result


def preflight(case: Case, db: Database, reference_file: Path, *, budget_s: float, emit: Emit = _no_emit) -> Preflight:
    """Run a case once, wall-clocked, check its answers and apply the time budget.

    `NotSupported` gives `not_supported`, any other exception `failed`; a run slower than `budget_s` gives
    `timeout` (its answers are still checked); a wrong answer keeps status `ok` with answer `disagree`.
    The database call alone is bracketed by a `run` line before it and a `progress` line after it (both outside
    the clock), so the reference loading and the answer check never count towards a supervisor's cutoff.
    A write case runs `write_preflight` instead.
    """
    if isinstance(case.item, Write):
        return write_preflight(case, db, reference_file, budget_s=budget_s, emit=emit)
    bindings, _ = catalog.load_reference(reference_file)
    result = Preflight(make_run(db, case.item, bindings), bindings)
    emit({"kind": "run", "id": case.bench_id})
    try:
        start = time.perf_counter_ns()
        raw = result.run()
        result.preflight_ns = time.perf_counter_ns() - start
    except BaseException as exc:
        reraise_control(exc)
        return result.fail(exc)
    finally:
        emit({"kind": "progress", "id": case.bench_id})
    try:
        result.answers = canonical_answers(case.item, raw)
        result.answer, result.detail = check_case(case, result.answers, reference_file)
    except Exception as exc:
        return result.fail(exc, "checking the answer: ")
    return _budget(result, budget_s)


def _counts(db: Database) -> tuple[int, int] | None:
    try:
        return db.counts()
    except NotSupported:
        return None


def write_preflight(
    case: Case, db: Database, reference_file: Path, *, budget_s: float, emit: Emit = _no_emit
) -> Preflight:
    """The first run of a write case: every phase once, the run wall-clocked; the verdict and the budget.

    Before it, untimed: the instance's counts (the guard's baseline) and the check read, which must return the
    reference's `before` rows, else ✗ "the graph is not in its loaded state before the case: ...". Then `setup`
    and, with `ready`, its check (else ✗ "setup did not reach its state: ..."), the timed `run` (bracketed by the
    `run` and `progress` lines like a query's call), the check against `after` (the verdict), the `teardown` and
    the check against `before` again (a teardown that does not restore the loaded state is ≠ too). A phase that
    raises, or a setup that misses its state, gets a best-effort teardown.
    """
    item = case.item
    assert isinstance(item, Write)
    bindings, after, before = catalog.load_write_reference(reference_file)
    ready = catalog.load_write_ready(reference_file) if item.ready else None
    write = WriteRun(db, item, bindings, after, before, ready=ready)
    result = Preflight(write.run, bindings, write=write)
    if item.ready and ready is None:  # never skip the setup check silently
        problem = (
            f"the reference file {reference_file.name} has no 'ready' rows, but {item.id} has a setup state;"
            " regenerate the references"
        )
        return result.fail(ValueError(problem))
    try:
        write.baseline = _counts(db)
    except BaseException as exc:
        reraise_control(exc)
        return result.fail(exc, "counting the graph before the case: ")
    try:
        state = write.differs(before)
    except BaseException as exc:
        reraise_control(exc)
        return result.fail(exc)
    if state is not None:
        result.status, result.error = "failed", f"the graph is not in its loaded state before the case: {state}"
        return result
    try:
        write.prepare()
    except SetupDiffers as exc:
        write.undo()
        result.status, result.error = "failed", str(exc)
        return result
    except BaseException as exc:
        reraise_control(exc)
        write.undo()
        return result.fail(exc, "setup: ")
    emit({"kind": "run", "id": case.bench_id})
    try:
        start = time.perf_counter_ns()
        write.run()
        result.preflight_ns = time.perf_counter_ns() - start
    except BaseException as exc:
        reraise_control(exc)
        write.undo()
        return result.fail(exc)
    finally:
        emit({"kind": "progress", "id": case.bench_id})
    try:
        result.answers = write.check()
        problem = verify(result.answers, after, rel_eps=item.rel_eps)
        write.teardown()
        restored = write.differs(before)
    except BaseException as exc:
        reraise_control(exc)
        write.undo()
        return result.fail(exc, "checking the answer: ")
    result.answer = "agree"
    if problem is not None:
        result.answer, result.detail = "disagree", problem
    elif restored is not None:
        result.answer, result.detail = "disagree", f"read-back after the teardown differs: {restored}"
    return _budget(result, budget_s)


def time_write(
    write: WriteRun,
    *,
    warmup: int,
    iterations: int,
    on_run: Callable[[], None] | None = None,
    deadline: float | None = None,
    ceiling_s: float = CASE_CEILING_S,
    preflight_ns: int | None = None,
) -> dict[str, Any]:
    """Time a write case's runs after its preflight: setup (and its `ready` check) before each, read-back and
    teardown after each, all untimed.

    Returns the entry fields: `timing`, or (answer `disagree`, detail "read-back after timed run <k> differs:
    ...", k counting every run after the preflight, warm-ups included) at the first read-back that differs, or
    (status `failed`, error) when a phase raised or a setup missed its state, or (status `timeout`, "too slow:
    over ...") when the case reached its ceiling (`deadline`, `ceiling_s` after its start) with too few timed runs.
    The runs are time-boxed like a query's (`_limits`), the first predicted from `preflight_ns`.
    """
    runs = [0]

    def after(_: Any) -> None:
        runs[0] += 1
        problem = write.readback()
        if problem is not None:
            raise ReadbackDiffers(f"read-back after timed run {runs[0]} differs: {problem}")

    try:
        timing, _ = measure(
            write.run,
            warmup=warmup,
            iterations=iterations,
            on_run=on_run,
            before=write.prepare,
            after=after,
            **_limits(deadline, preflight_ns),
        )
    except OverCeiling:
        return {"status": "timeout", "error": over_ceiling(ceiling_s)}
    except ReadbackDiffers as exc:
        return {"answer": "disagree", "detail": str(exc)}
    except SetupDiffers as exc:
        write.undo()
        return {"status": "failed", "error": str(exc)}
    except BaseException as exc:
        reraise_control(exc)
        write.undo()
        return {"status": "failed", "error": message(exc)}
    return {"timing": asdict(timing)}


def _limits(deadline: float | None, preflight_ns: int | None = None) -> dict[str, Any]:
    """`measure`'s limits for a case: the time box, at least `MIN_ITERATIONS`, the ceiling's deadline (if any), the
    supervisor's kill `CEILING_GRACE_S` after it, and the preflight's duration as the first run's prediction."""
    kill_at = None if deadline is None else deadline + CEILING_GRACE_S
    first_s = None if preflight_ns is None else preflight_ns / 1e9
    limits = {"time_box_s": TIME_BOX_S, "min_iterations": MIN_ITERATIONS, "deadline": deadline, "kill_at": kill_at}
    return {**limits, "first_s": first_s}


def write_guard(entry: dict[str, Any], write: WriteRun | None) -> None:
    """After a write case, untimed: the instance's counts must equal those before it, else the entry fails.

    "the case left the graph changed: <n> nodes, <m> edges; expected ..." (after any earlier error). Nothing to
    compare when the database cannot count.
    """
    if write is None or write.baseline is None:
        return
    try:
        counted = write.db.counts()
        problem = None
        if counted != write.baseline:
            nodes, edges = counted
            expected = f"expected {write.baseline[0]} nodes, {write.baseline[1]} edges"
            problem = f"the case left the graph changed: {nodes} nodes, {edges} edges; {expected}"
    except BaseException as exc:
        reraise_control(exc)
        problem = f"counting the graph after the case: {message(exc)}"
    if problem is not None:
        entry["status"] = "failed"
        entry["error"] = f"{entry['error']}; {problem}" if entry.get("error") else problem


def answers_path(out: Path, bench_id: str) -> Path:
    """Where the worker writes a case's canonical answers."""
    return out.parent / out.stem / "answers" / (re.sub(r"[^A-Za-z0-9._-]+", "_", bench_id) + ".json")


def emitter(fh: IO[str]) -> Emit:
    def emit(obj: dict[str, Any]) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()

    return emit


def remove(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def fresh_instance(
    db_cls: type[Database],
    dataset: Dataset,
    workdir: Path,
    threads: int | None,
    i: int,
    store: Store = "graph",
    *,
    name: str | None = None,
) -> tuple[Database, Path]:
    """A new instance in an empty directory, for the i-th load of a dataset into `store`.

    The directory is `workdir/<dataset>-<scale>-<i>` for the graph store, `workdir/<dataset>-rdf-<scale>-<i>` for
    the RDF store; `name` replaces the part before the scale (`<dataset>-ql` for Load (query language)).
    """
    if name is None:
        name = dataset.name if store == "graph" else f"{dataset.name}-{store}"
    db_dir = workdir / f"{name}-{dataset.scale}-{i}"
    remove(db_dir)
    db_dir.mkdir(parents=True)
    return db_cls(db_dir, threads), db_dir


def prepare_environment(db_cls: type[Database]) -> str | None:
    """Run the database's one-time environment setup (outside any clock); its error message, or None."""
    try:
        db_cls.prepare_environment()
    except BaseException as exc:
        reraise_control(exc)
        return f"preparing the environment: {message(exc)}"
    return None


def count_check(db: Database, dataset: Dataset) -> tuple[AnswerVerdict, str | None]:
    """Compare the loaded node and edge counts (`Database.counts`) with the dataset's; the verdict and a detail.

    `NotSupported` leaves the load `unchecked`; another exception propagates.
    """
    try:
        nodes, edges = db.counts()
    except NotSupported:
        return "unchecked", None
    if (nodes, edges) == (dataset.node_count, dataset.edge_count):
        return "agree", None
    expected = f"expected {dataset.node_count} nodes, {dataset.edge_count} edges"
    return "disagree", f"loaded {nodes} nodes, {edges} edges; {expected}"


@dataclass
class LoadCheck:
    """The count check over a load's iterations: the first disagreement or counting error is kept.

    Load (query language) adds its content probe (`probe`) to the same verdict. A probe that could not run
    (`skipped`) leaves the answer `unchecked`, with its reason as the detail, unless a count disagrees.
    """

    answer: AnswerVerdict = "unchecked"
    detail: str | None = None
    error: str | None = None
    skipped: str | None = None

    @property
    def settled(self) -> bool:
        return self.error is not None or self.answer == "disagree"

    def add(self, db: Database, dataset: Dataset) -> None:
        """Check one loaded instance (untimed), unless an earlier one already disagreed or failed to count."""
        if self.settled:
            return
        try:
            answer, detail = count_check(db, dataset)
        except BaseException as exc:
            reraise_control(exc)
            self.error = f"counting the loaded data: {message(exc)}"
            return
        if answer == "disagree" or self.skipped is None:
            self.answer, self.detail = answer, detail

    def probe(self, case: Case, db: Database, reference_file: Path) -> None:
        """Run `case`'s preflight call on a query-language load (untimed) against its references.

        A mismatch makes the load ≠ ("content probe <item>: ..."), an error fails it; a database that does not run
        the probe item (`NotSupported`) leaves the load `unchecked`, "content probe <item> not run: <reason>".
        """
        if self.settled or self.skipped is not None:
            return
        label = f"content probe {case.item.id}: "
        try:
            bindings, _ = catalog.load_reference(reference_file)
            answers = canonical_answers(case.item, make_run(db, case.item, bindings)())
            verdict, detail = check_case(case, answers, reference_file)
        except NotSupported as exc:
            self.skipped = f"content probe {case.item.id} not run: {message(exc)}"
            self.answer, self.detail = "unchecked", self.skipped
            return
        except BaseException as exc:
            reraise_control(exc)
            self.error = label + message(exc)
            return
        if verdict == "disagree":
            self.answer, self.detail = "disagree", label + (detail or "")


def unsupported_load_entry(load_id: str, model: str) -> dict[str, Any]:
    """The entry of a load the database's models do not reach: ⊘, with the error of the cases' model gate."""
    return {"kind": "entry", "id": load_id, "status": "not_supported", "error": f"model {model} not supported"}


def load_entry(load_id: str, load_ns: int, check: LoadCheck, iterations: int) -> dict[str, Any]:
    """A load's entry line: `ok` with the count verdict, or `failed` when counting raised; `load_ns` is the median
    of the `iterations` loads that ran (`load_iterations`: fewer than the selection's when the time box stopped them).

    A counting error fails only the load entry: the load itself worked, so the dataset's cases still run on it.
    """
    entry: dict[str, Any] = {
        "kind": "entry",
        "id": load_id,
        "status": "ok",
        "load_ns": load_ns,
        "load_iterations": iterations,
        "answer": check.answer,
        "detail": check.detail,
    }
    if check.error is not None:
        entry["status"], entry["error"] = "failed", check.error
    return entry


@dataclass
class Loaded:
    """A loaded instance, its directory, the median load time, the count check of the loads (if run) and the
    number of loads the time box let run."""

    db: Database
    workdir: Path
    load_ns: int
    check: LoadCheck
    iterations: int = 1


class LoadOverCeiling(Exception):
    """A load (with its count check) ended past its deadline: the load ceiling (Task T)."""


def _boxed(times: list[int]) -> bool:
    """Whether a load's time box is full: at least `MIN_LOAD_ITERATIONS` loads, together `TIME_BOX_S` or more."""
    return len(times) >= MIN_LOAD_ITERATIONS and sum(times) >= TIME_BOX_S * 1e9


def load(
    db_cls: type[Database],
    dataset: Dataset,
    workdir: Path,
    threads: int | None,
    iterations: int,
    progress: Callable[[], None] = lambda: None,
    *,
    check: bool = False,
    store: Store = "graph",
    deadline: float | None = None,
) -> Loaded:
    """Load `dataset` into `store` up to `iterations` times, each into a fresh instance in a fresh directory; keep
    the last.

    The clock covers `Database.load_store` alone. `progress` is called after each load, outside its clock. With
    `check`, each loaded instance's node and edge counts are checked right after its load, outside the clock too
    (`LoadCheck`). No further load starts once the time box is full (`_boxed`). A load (with its count check) that
    ends at or past `deadline` (a `time.monotonic()` value) raises `LoadOverCeiling`, its instance removed.
    """
    times: list[int] = []
    counted = LoadCheck()
    previous: tuple[Database, Path] | None = None
    for i in range(iterations):
        if previous is not None:
            close_quietly(previous[0])
            remove(previous[1])
        db, db_dir = fresh_instance(db_cls, dataset, workdir, threads, i, store)
        previous = (db, db_dir)
        start = time.perf_counter_ns()
        try:
            db.load_store(store, dataset)
        except BaseException as exc:
            close_quietly(db)
            remove(db_dir)
            if deadline is not None and time.monotonic() >= deadline and not isinstance(exc, CONTROL_FLOW):
                # The adapter's own timeout stopped a load that had not finished by the ceiling: over it.
                raise LoadOverCeiling from exc
            raise
        times.append(time.perf_counter_ns() - start)
        if check:
            counted.add(db, dataset)
        progress()
        if deadline is not None and time.monotonic() >= deadline:
            close_quietly(db)
            remove(db_dir)
            raise LoadOverCeiling
        if _boxed(times):
            break
    assert previous is not None
    return Loaded(previous[0], previous[1], int(statistics.median(times)), counted, len(times))


def probe_case(db_cls: type[Database], dataset: Dataset, threads: int | None) -> Case | None:
    """The Load (query language) content probe for a database and dataset: `catalog.LOAD_QL_PROBES` of the
    database's primary model, as a case, or None."""
    item_id = catalog.LOAD_QL_PROBES.get((dataset.name, primary_model(db_cls.models)))
    if item_id is None:
        return None
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db_cls.name, dataset.scale, threads)


def load_ql_instance(
    db_cls: type[Database], dataset: Dataset, workdir: Path, threads: int | None, i: int
) -> tuple[Database, Path]:
    """A fresh instance for the i-th query-language load, in `workdir/<dataset>-ql-<scale>-<i>`."""
    return fresh_instance(db_cls, dataset, workdir, threads, i, name=f"{dataset.name}-ql")


def run_load_ql(
    db_cls: type[Database],
    dataset: Dataset,
    workdir: Path,
    threads: int | None,
    iterations: int,
    base: Path,
    load_id: str,
    emit: Emit,
    setup_error: str | None = None,
    *,
    ceiling_s: float = LOAD_CEILING_S,
) -> None:
    """The Load (query language) entry of a dataset: up to `iterations` fresh instances, each timed around
    `Database.load_ql` alone, count-checked like Load and content-probed (`LoadCheck.probe`), then closed.

    No case ever runs on these instances. Any failure fails this entry only. With `setup_error` (from
    `prepare_environment`) the entry fails with it. The loads are time-boxed like Load's (`_boxed`), and one that
    ends `ceiling_s` or more after the `start` line makes the entry ⏱ "load over ...".
    """
    emit({"kind": "start", "id": load_id})
    deadline = time.monotonic() + ceiling_s
    if setup_error is not None:
        emit({"kind": "entry", "id": load_id, "status": "failed", "error": setup_error})
        return
    probe = probe_case(db_cls, dataset, threads)
    reference_file = catalog.reference_path(probe.item, dataset.scale, base=base) if probe else None
    check = LoadCheck()
    times: list[int] = []
    try:
        for i in range(iterations):
            db, db_dir = load_ql_instance(db_cls, dataset, workdir, threads, i)
            try:
                start = time.perf_counter_ns()
                db.load_ql(dataset)
                times.append(time.perf_counter_ns() - start)
                check.add(db, dataset)
                if probe is not None and reference_file is not None:
                    check.probe(probe, db, reference_file)
            finally:
                close_quietly(db)
                remove(db_dir)
            emit({"kind": "progress", "id": load_id})
            if time.monotonic() >= deadline:
                emit({"kind": "entry", "id": load_id, "status": "timeout", "error": load_over(ceiling_s)})
                return
            if _boxed(times):
                break
    except BaseException as exc:
        reraise_control(exc)
        if time.monotonic() >= deadline:  # the adapter's own timeout stopped a load past the ceiling
            emit({"kind": "entry", "id": load_id, "status": "timeout", "error": load_over(ceiling_s)})
            return
        status = "not_supported" if isinstance(exc, NotSupported) else "failed"
        emit({"kind": "entry", "id": load_id, "status": status, "error": message(exc)})
        return
    emit(load_entry(load_id, int(statistics.median(times)), check, len(times)))


def startup_timeout(case_timeout_s: float) -> float:
    """How long a startup child may stay silent: `STARTUP_TIMEOUT_S`, and at most half the case timeout, so the
    worker kills its own child before a supervisor's silence limit kills the worker (and orphans the child)."""
    return min(STARTUP_TIMEOUT_S, case_timeout_s / 2)


def run_startup(
    db_cls: type[Database],
    db_name: str,
    threads: int | None,
    iterations: int,
    workdir: Path,
    bench_id: str,
    emit: Emit,
    case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
) -> None:
    """The Startup entry: `provision()` here, untimed, then `iterations` startup children (`run_startups`).

    A provisioning error fails the entry ("provisioning: ...") and no child runs.
    """
    emit({"kind": "start", "id": bench_id})
    try:
        db_cls.provision()
    except BaseException as exc:
        reraise_control(exc)
        error = f"provisioning: {message(exc)}"
        emit({"kind": "entry", "id": bench_id, "status": "failed", "error": error, "reference": "python"})
        return
    progress = {"kind": "progress", "id": bench_id}
    timeout_s = startup_timeout(case_timeout_s)
    outcome = run_startups(db_name, threads, iterations, workdir, timeout_s, progress=lambda: emit(progress))
    emit(startup_entry(bench_id, outcome))


def database_line(db_cls: type[Database], workdir: Path, threads: int | None) -> dict[str, Any]:
    """The `database` line: the info of a probe instance created (and removed) in `workdir/info`."""
    probe_dir = workdir / "info"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = db_cls(probe_dir, threads)
    try:
        info = probe.info()
    finally:
        close_quietly(probe)
        remove(probe_dir)
    return {"kind": "database", **asdict(info)}


def case_entry(case: Case, **fields: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": "entry",
        "id": case.bench_id,
        "status": "ok",
        "error": None,
        "timing": None,
        "preflight_ns": None,
        "answer": "unchecked",
        "reference": reference_kind(case.item),
        "answer_digest": None,
        "detail": None,
    }
    entry.update(fields)
    return entry


def reads_first(cases: Sequence[Case]) -> list[Case]:
    """`cases` with every write case after every read case, each part in its given order."""
    return sorted(cases, key=lambda case: isinstance(case.item, Write))


@dataclass(frozen=True)
class _Settings:
    selection: Selection
    out: Path
    budgets: dict[str, float]
    case_timeout_s: float
    case_ceiling_s: float = CASE_CEILING_S
    load_ceiling_s: float = LOAD_CEILING_S


def start_case(
    case: Case, db: Database, base: Path, out: Path | None, budget_s: float, emit: Emit
) -> tuple[Preflight, dict[str, Any]]:
    """Emit the case's `start` line, run its preflight and build (not emit) its entry line.

    The preflight emits the `run` and `progress` lines around its database call. The canonical answers go to
    `answers_path(out, ...)` when `out` is given. The entry has no timing yet.
    """
    emit({"kind": "start", "id": case.bench_id})
    reference_file = catalog.reference_path(case.item, case.scale, base=base)
    pf = preflight(case, db, reference_file, budget_s=budget_s, emit=emit)
    entry = case_entry(
        case,
        status=pf.status,
        error=pf.error,
        preflight_ns=pf.preflight_ns,
        answer=pf.answer,
        detail=pf.detail,
    )
    if pf.answers is not None:
        entry["answer_digest"] = digest(pf.answers)
        if out is not None:
            path = answers_path(out, case.bench_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = catalog.reference_document(case.item, pf.bindings, pf.answers)
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8", newline="\n")
    return pf, entry


def _run_case(case: Case, db: Database, base: Path, settings: _Settings, emit: Emit) -> None:
    """Preflight, then time the case within its limits; its ceiling counts from its `start` line."""
    budget = settings.budgets.get(case.bench_id, settings.case_timeout_s)
    deadline = time.monotonic() + settings.case_ceiling_s
    pf, entry = start_case(case, db, base, settings.out, budget, emit)
    if pf.passed:
        sel = settings.selection
        progress = {"kind": "progress", "id": case.bench_id}
        warmup = max(sel.warmup - 1, 0)
        if pf.write is not None:
            entry.update(
                time_write(
                    pf.write,
                    warmup=warmup,
                    iterations=sel.iterations,
                    on_run=lambda: emit(progress),
                    deadline=deadline,
                    ceiling_s=settings.case_ceiling_s,
                    preflight_ns=pf.preflight_ns,
                )
            )
        else:
            try:
                timing, _ = measure(
                    pf.run,
                    warmup=warmup,
                    iterations=sel.iterations,
                    on_run=lambda: emit(progress),
                    **_limits(deadline, pf.preflight_ns),
                )
                entry["timing"] = asdict(timing)
            except OverCeiling:
                entry["status"], entry["error"] = "timeout", over_ceiling(settings.case_ceiling_s)
            except BaseException as exc:
                reraise_control(exc)
                entry["status"], entry["error"] = "failed", message(exc)
    write_guard(entry, pf.write)
    emit(entry)


def run(
    db_name: str,
    selection_name: str,
    threads: int | None,
    *,
    workdir: Path,
    out: Path,
    skip: frozenset[str] = frozenset(),
    budgets: dict[str, float] | None = None,
    case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
    case_ceiling_s: float = CASE_CEILING_S,
    load_ceiling_s: float = LOAD_CEILING_S,
) -> None:
    """Run every case of the selection for one database and thread setting, appending JSON lines to `out`."""
    sel = catalog.selection(selection_name)
    db_cls = get_database(db_name)
    settings = _Settings(sel, out, budgets or {}, case_timeout_s, case_ceiling_s, load_ceiling_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        emit = emitter(fh)
        emit(database_line(db_cls, workdir, threads))

        startup_id = catalog.startup_id(db_name, threads)
        if startup_id in catalog.startup_ids(sel.name, [db_name]) and startup_id not in skip:
            iterations = sel.startup_iterations
            run_startup(db_cls, db_name, threads, iterations, workdir, startup_id, emit, case_timeout_s)

        # Every (dataset, store, scale) with a supported item gets a load; the selection's other loads are ⊘, as
        # in `catalog.load_ids`.
        groups: dict[tuple[str, Store, str], list[Case]] = {}
        if threads in sel.threads:
            for dataset, store in catalog.load_keys(sel.name, db_cls.models):
                for scale in sel.scales:
                    groups[(dataset, store, scale)] = []
        for case in catalog.cases(sel.name, [db_name]):
            if case.threads != threads or case.bench_id in skip:
                continue
            item = case.item
            if item.supported_by(db_cls.models):
                groups[(item.dataset, item.store, case.scale)].append(case)
            else:
                emit(case_entry(case, status="not_supported", error=f"model {item.model} not supported"))

        load_ids: dict[tuple[str, Store, str], str] = {}
        if "load" in sel.categories and threads in sel.threads:
            for (dataset, store), model in catalog.all_load_keys(sel.name).items():
                for scale in sel.scales:
                    key = (dataset, store, scale)
                    load_id = catalog.load_id(dataset, db_name, scale, threads, store)
                    if key in groups:
                        load_ids[key] = load_id
                    elif load_id not in skip:
                        emit(unsupported_load_entry(load_id, model))
        load_ql_ids: dict[tuple[str, Store, str], str] = {}
        if "load_ql" in sel.categories and threads in sel.threads:
            for dataset, model in catalog.load_ql_keys(sel.name).items():
                for scale in sel.scales:
                    key = (dataset, "graph", scale)
                    load_ql_id = catalog.load_ql_id(dataset, db_name, scale, threads)
                    if load_ql_id in skip:
                        continue
                    if key in groups:
                        load_ql_ids[key] = load_ql_id
                    else:
                        emit(unsupported_load_entry(load_ql_id, model))
        reported = {key: load_id for key, load_id in load_ids.items() if load_id not in skip}
        # An answer-only fixture last: its load is unreported (no `start` line), so a crash while loading it leaves
        # no open id, and the resume rule then fails only what is still unrecorded, its own cases.
        order = sorted(groups, key=lambda key: (key[0] in catalog.ANSWER_ONLY_DATASETS, key))
        needed = [key for key in order if groups[key] or key in reported or key in load_ql_ids]
        if not needed:
            return
        datasets = cache.prepare(sel.name)
        base = cache.data_dir() / cache.dataset_key(sel.name)
        setup_error = prepare_environment(db_cls)
        for key in needed:
            dataset, store, scale = key
            data = datasets[(dataset, scale)]
            _run_dataset(
                db_cls,
                data,
                store,
                reads_first(groups[key]),
                reported.get(key),
                threads,
                workdir,
                base,
                settings,
                emit,
                setup_error,
                load_ql_ids.get(key),
            )


def _run_dataset(
    db_cls: type[Database],
    dataset: Dataset,
    store: Store,
    cases: list[Case],
    load_id: str | None,
    threads: int | None,
    workdir: Path,
    base: Path,
    settings: _Settings,
    emit: Emit,
    setup_error: str | None = None,
    load_ql_id: str | None = None,
) -> None:
    """Load the dataset into `store` and run the cases that read it; then its Load (query language) entry.

    With `setup_error` (from `prepare_environment`) the loads fail with it. The query-language load runs after
    the loaded instance is released, whatever happened to it, and never cascades onto the cases; only a load over
    the load ceiling does: the query-language load of the same data is then ⏱ "load over ..." without running.
    """
    over = False
    if cases or load_id is not None:
        over = _load_and_run(
            db_cls, dataset, store, cases, load_id, threads, workdir, base, settings, emit, setup_error
        )
    if load_ql_id is not None:
        if over:
            emit({"kind": "start", "id": load_ql_id})
            emit({"kind": "entry", "id": load_ql_id, "status": "timeout", "error": load_over(settings.load_ceiling_s)})
            return
        iterations = settings.selection.load_iterations
        run_load_ql(
            db_cls,
            dataset,
            workdir,
            threads,
            iterations,
            base,
            load_ql_id,
            emit,
            setup_error,
            ceiling_s=settings.load_ceiling_s,
        )


def _load_and_run(
    db_cls: type[Database],
    dataset: Dataset,
    store: Store,
    cases: list[Case],
    load_id: str | None,
    threads: int | None,
    workdir: Path,
    base: Path,
    settings: _Settings,
    emit: Emit,
    setup_error: str | None,
) -> bool:
    """Load and run the cases; whether the load ended past the load ceiling (then no case ran: each is ⏱)."""
    if load_id is not None:
        emit({"kind": "start", "id": load_id})
    deadline = time.monotonic() + settings.load_ceiling_s
    loaded: Loaded | None = None
    text = setup_error
    if text is None:
        emit({"kind": "load", "dataset": dataset.name, "store": store, "scale": dataset.scale})
        try:
            progress = {"kind": "progress", "id": load_id}
            # An unreported load (no Load entry: an answer-only fixture, or a selection without the Load row) records
            # no time, so one load is enough.
            iterations = settings.selection.load_iterations if load_id is not None else 1
            check = load_id is not None

            def on_load() -> None:
                emit(progress)

            loaded = load(
                db_cls, dataset, workdir, threads, iterations, on_load, check=check, store=store, deadline=deadline
            )
        except LoadOverCeiling:
            error = load_over(settings.load_ceiling_s)
            if load_id is not None:
                emit({"kind": "entry", "id": load_id, "status": "timeout", "error": error})
            for case in cases:
                emit({"kind": "start", "id": case.bench_id})
                emit(case_entry(case, status="timeout", error=error))
            return True
        except BaseException as exc:
            reraise_control(exc)
            text = message(exc)
    if loaded is None:  # `text` holds the setup or load error
        if load_id is not None:
            emit({"kind": "entry", "id": load_id, "status": "failed", "error": text})
        for case in cases:
            emit({"kind": "start", "id": case.bench_id})
            emit(case_entry(case, status="failed", error=f"load failed: {text}"))
        return False
    if load_id is not None:
        emit(load_entry(load_id, loaded.load_ns, loaded.check, loaded.iterations))
    try:
        for case in cases:
            _run_case(case, loaded.db, base, settings, emit)
    finally:
        close_quietly(loaded.db)
        remove(loaded.workdir)
    return False


def parse_budget(value: str) -> tuple[str, float]:
    bench_id, sep, seconds = value.rpartition("=")
    if not sep or not bench_id:
        raise argparse.ArgumentTypeError(f"expected ID=SECONDS, got {value!r}")
    try:
        return bench_id, float(seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected ID=SECONDS, got {value!r}") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graph_bench.runner.worker",
        description="Run one database over a selection in this process and stream JSON lines.",
    )
    parser.add_argument("--db", required=True, help="Database name.")
    parser.add_argument("--selection", required=True, help="Selection name.")
    parser.add_argument("--threads", required=True, choices=("1", "all"), help="1, or all cores.")
    parser.add_argument("--workdir", help="Directory for database files; default: a fresh temporary directory.")
    parser.add_argument("--out", required=True, help="JSON lines file to append to.")
    parser.add_argument("--skip", nargs="+", action="extend", default=[], metavar="ID", help="Ids not to run.")
    parser.add_argument(
        "--budget",
        nargs="+",
        action="extend",
        default=[],
        type=parse_budget,
        metavar="ID=SECONDS",
        help="Case budgets.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_S,
        metavar="SECONDS",
        help="Budget of a case without --budget.",
    )
    parser.add_argument(
        "--case-ceiling",
        type=float,
        default=CASE_CEILING_S,
        metavar="SECONDS",
        help="Wall time of a case from its start line; no run starts past it.",
    )
    parser.add_argument(
        "--load-ceiling",
        type=float,
        default=LOAD_CEILING_S,
        metavar="SECONDS",
        help="Wall time of a load from its start; a load past it takes its dataset's cases with it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    own_workdir = args.workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="graph-bench-")) if own_workdir else Path(args.workdir)
    try:
        run(
            args.db,
            args.selection,
            1 if args.threads == "1" else None,
            workdir=workdir,
            out=Path(args.out),
            skip=frozenset(args.skip),
            budgets=dict(args.budget),
            case_timeout_s=args.case_timeout,
            case_ceiling_s=args.case_ceiling,
            load_ceiling_s=args.load_ceiling,
        )
    finally:
        if own_workdir:
            remove(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
