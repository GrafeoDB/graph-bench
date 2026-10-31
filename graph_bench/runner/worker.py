"""The worker: runs one database over a selection in its own process and streams JSON lines.

`python -m graph_bench.runner.worker --db NAME --selection S --threads (1|all) --workdir DIR --out FILE.jsonl
[--skip ID ...] [--budget ID=SECONDS ...] [--case-timeout SECONDS]`

Lines, appended to `--out` and flushed one by one:

- `{"kind": "database", <DatabaseInfo fields>}`, always first;
- `{"kind": "start", "id": ...}` before each case and each reported load, so a supervisor knows what hung or crashed;
- `{"kind": "run", "id": ...}` immediately before a case's preflight database call, and a `progress` line
  immediately after it returns (or raises), before any harness work: the window between the two is the database
  call alone, which is what a supervisor's in-run cutoff times;
- `{"kind": "progress", "id": ...}` after a case's preflight call, after each of its warm-up and timed runs, and
  after each load iteration (`id` is null for an unreported load), so a supervisor can tell a long case from a
  hang;
- `{"kind": "entry", "id": ..., "status": ..., ...}` with a case's result, or a load's `load_ns` and its count
  check (`answer` and `detail`, see `load_entry`).

The canonical answers of each case go to `<out dir>/<out stem>/answers/<sanitized id>.json`.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from graph_bench.answers import canonical, canonical_map, check_case, digest
from graph_bench.databases import get_database
from graph_bench.databases.base import Database, NotSupported
from graph_bench.datasets import cache
from graph_bench.datasets.schema import Dataset
from graph_bench.results import Reference
from graph_bench.runner.timing import measure
from graph_bench.suites import Algorithm, Query, catalog
from graph_bench.suites.catalog import Case, Selection
from graph_bench.suites.graphalytics import min_member
from graph_bench.types import MODEL_STORE, AnswerVerdict, Status, Store

DEFAULT_CASE_TIMEOUT_S = 600.0

Emit = Callable[[dict[str, Any]], None]


def reference_kind(item: Query | Algorithm) -> Reference:
    """Where an item's reference answers come from: NetworkX for algorithms, plain Python for queries."""
    return "networkx" if isinstance(item, Algorithm) else "python"


def make_run(db: Database, item: Query | Algorithm, bindings: list[dict[str, Any]]) -> Callable[[], list[Any]]:
    """The timed function: run the item once per binding and return the raw per-binding results.

    A query runs in its model's dialect (`Database.dialect_for`) through `Database.query_model`.
    """
    if isinstance(item, Algorithm):
        name = item.name
        return lambda: [db.algorithm(name, binding) for binding in bindings]
    model = item.model
    text = item.text_for(db.dialect_for(model))
    return lambda: [db.query_model(model, text, binding) for binding in bindings]


def canonical_answers(item: Query | Algorithm, raw: list[Any]) -> list[Any]:
    """Canonicalize raw per-binding results so they compare and hash like the reference answers."""
    if isinstance(item, Query):
        return [canonical(rows, ordered=item.ordered) for rows in raw]
    answers = [canonical_map(result, drop_unreachable=item.name == "bfs") for result in raw]
    if item.name == "wcc":
        answers = [min_member(answer) for answer in answers]
    return answers


def message(exc: BaseException) -> str:
    text = str(exc)
    if isinstance(exc, NotSupported):
        return text or "not supported"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass
class Preflight:
    """The untimed first run of a case: its answers, verdict and whether it may be timed."""

    run: Callable[[], list[Any]]
    bindings: list[dict[str, Any]]
    status: Status = "ok"
    error: str | None = None
    preflight_ns: int | None = None
    answer: AnswerVerdict = "unchecked"
    answers: list[Any] | None = None
    detail: str | None = None

    @property
    def passed(self) -> bool:
        """Completed within budget with a correct answer: only then is the case timed."""
        return self.status == "ok" and self.answer == "agree"


def _no_emit(line: dict[str, Any]) -> None:
    return None


def preflight(case: Case, db: Database, reference_file: Path, *, budget_s: float, emit: Emit = _no_emit) -> Preflight:
    """Run a case once, wall-clocked, check its answers and apply the time budget.

    `NotSupported` gives `not_supported`, any other exception `failed`; a run slower than `budget_s` gives
    `timeout` (its answers are still checked); a wrong answer keeps status `ok` with answer `disagree`.
    The database call alone is bracketed by a `run` line before it and a `progress` line after it (both outside
    the clock), so the reference loading and the answer check never count towards a supervisor's cutoff.
    """
    bindings, _ = catalog.load_reference(reference_file)
    result = Preflight(make_run(db, case.item, bindings), bindings)
    emit({"kind": "run", "id": case.bench_id})
    try:
        start = time.perf_counter_ns()
        raw = result.run()
        result.preflight_ns = time.perf_counter_ns() - start
    except NotSupported as exc:
        result.status, result.error = "not_supported", message(exc)
        return result
    except Exception as exc:
        result.status, result.error = "failed", message(exc)
        return result
    finally:
        emit({"kind": "progress", "id": case.bench_id})
    try:
        result.answers = canonical_answers(case.item, raw)
        result.answer, result.detail = check_case(case, result.answers, reference_file)
    except Exception as exc:
        result.status, result.error = "failed", f"checking the answer: {message(exc)}"
        return result
    seconds = result.preflight_ns / 1e9
    if seconds > budget_s:
        result.status, result.error = "timeout", f"too slow: {seconds:.3g} s > budget {budget_s:.3g} s"
    return result


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
    db_cls: type[Database], dataset: Dataset, workdir: Path, threads: int | None, i: int, store: Store = "graph"
) -> tuple[Database, Path]:
    """A new instance in an empty directory, for the i-th load of a dataset into `store`.

    The directory is `workdir/<dataset>-<scale>-<i>` for the graph store, `workdir/<dataset>-rdf-<scale>-<i>` for
    the RDF store.
    """
    name = dataset.name if store == "graph" else f"{dataset.name}-{store}"
    db_dir = workdir / f"{name}-{dataset.scale}-{i}"
    remove(db_dir)
    db_dir.mkdir(parents=True)
    return db_cls(db_dir, threads), db_dir


def prepare_environment(db_cls: type[Database]) -> str | None:
    """Run the database's one-time environment setup (outside any clock); its error message, or None."""
    try:
        db_cls.prepare_environment()
    except Exception as exc:
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
    """The count check over a load's iterations: the first disagreement or counting error is kept."""

    answer: AnswerVerdict = "unchecked"
    detail: str | None = None
    error: str | None = None

    def add(self, db: Database, dataset: Dataset) -> None:
        """Check one loaded instance (untimed), unless an earlier one already disagreed or failed to count."""
        if self.error is not None or self.answer == "disagree":
            return
        try:
            self.answer, self.detail = count_check(db, dataset)
        except Exception as exc:
            self.error = f"counting the loaded data: {message(exc)}"


def load_entry(load_id: str, load_ns: int, check: LoadCheck) -> dict[str, Any]:
    """A load's entry line: `ok` with the count verdict, or `failed` when counting raised.

    A counting error fails only the load entry: the load itself worked, so the dataset's cases still run on it.
    """
    entry: dict[str, Any] = {
        "kind": "entry",
        "id": load_id,
        "status": "ok",
        "load_ns": load_ns,
        "answer": check.answer,
        "detail": check.detail,
    }
    if check.error is not None:
        entry["status"], entry["error"] = "failed", check.error
    return entry


@dataclass
class Loaded:
    """A loaded instance, its directory, the median load time and the count check of the loads (if run)."""

    db: Database
    workdir: Path
    load_ns: int
    check: LoadCheck


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
) -> Loaded:
    """Load `dataset` into `store` `iterations` times, each into a fresh instance in a fresh directory; keep the last.

    The clock covers `Database.load_store` alone. `progress` is called after each load, outside its clock. With
    `check`, each loaded instance's node and edge counts are checked right after its load, outside the clock too
    (`LoadCheck`).
    """
    times: list[int] = []
    counted = LoadCheck()
    previous: tuple[Database, Path] | None = None
    for i in range(iterations):
        if previous is not None:
            previous[0].close()
            remove(previous[1])
        db, db_dir = fresh_instance(db_cls, dataset, workdir, threads, i, store)
        previous = (db, db_dir)
        start = time.perf_counter_ns()
        try:
            db.load_store(store, dataset)
        except BaseException:
            db.close()
            remove(db_dir)
            raise
        times.append(time.perf_counter_ns() - start)
        if check:
            counted.add(db, dataset)
        progress()
    assert previous is not None
    return Loaded(previous[0], previous[1], int(statistics.median(times)), counted)


def database_line(db_cls: type[Database], workdir: Path, threads: int | None) -> dict[str, Any]:
    """The `database` line: the info of a probe instance created (and removed) in `workdir/info`."""
    probe_dir = workdir / "info"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = db_cls(probe_dir, threads)
    try:
        info = probe.info()
    finally:
        probe.close()
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


@dataclass(frozen=True)
class _Settings:
    selection: Selection
    out: Path
    budgets: dict[str, float]
    case_timeout_s: float


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
    budget = settings.budgets.get(case.bench_id, settings.case_timeout_s)
    pf, entry = start_case(case, db, base, settings.out, budget, emit)
    if pf.passed:
        sel = settings.selection
        progress = {"kind": "progress", "id": case.bench_id}
        try:
            timing, _ = measure(
                pf.run, warmup=max(sel.warmup - 1, 0), iterations=sel.iterations, on_run=lambda: emit(progress)
            )
            entry["timing"] = asdict(timing)
        except Exception as exc:
            entry["status"], entry["error"] = "failed", message(exc)
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
) -> None:
    """Run every case of the selection for one database and thread setting, appending JSON lines to `out`."""
    sel = catalog.selection(selection_name)
    db_cls = get_database(db_name)
    settings = _Settings(sel, out, budgets or {}, case_timeout_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        emit = emitter(fh)
        emit(database_line(db_cls, workdir, threads))

        # Every (dataset, store, scale) with a supported item gets a load, as in `catalog.load_ids`.
        groups: dict[tuple[str, Store, str], list[Case]] = {}
        if threads in sel.threads:
            for dataset, store in catalog.load_keys(sel.name, db_cls.models):
                for scale in sel.scales:
                    groups[(dataset, store, scale)] = []
        for case in catalog.cases(sel.name, [db_name]):
            if case.threads != threads or case.bench_id in skip:
                continue
            model = case.item.model
            if model in db_cls.models:
                groups[(case.item.dataset, MODEL_STORE[model], case.scale)].append(case)
            else:
                emit(case_entry(case, status="not_supported", error=f"model {model} not supported"))

        load_ids: dict[tuple[str, Store, str], str] = {}
        if "load" in sel.categories:
            load_ids = {key: catalog.load_id(key[0], db_name, key[2], threads, key[1]) for key in groups}
        reported = {key: load_id for key, load_id in load_ids.items() if load_id not in skip}
        needed = [key for key in sorted(groups) if groups[key] or key in reported]
        if not needed:
            return
        datasets = cache.prepare(sel.name)
        base = cache.data_dir() / cache.dataset_key(sel.name)
        setup_error = prepare_environment(db_cls)
        for key in needed:
            dataset, store, scale = key
            data = datasets[(dataset, scale)]
            _run_dataset(
                db_cls, data, store, groups[key], reported.get(key), threads, workdir, base, settings, emit, setup_error
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
) -> None:
    """Load the dataset into `store` and run the cases that read it.

    With `setup_error` (from `prepare_environment`) the load fails with it.
    """
    if load_id is not None:
        emit({"kind": "start", "id": load_id})
    loaded: Loaded | None = None
    text = setup_error
    if text is None:
        try:
            progress = {"kind": "progress", "id": load_id}
            iterations = settings.selection.load_iterations
            check = load_id is not None
            loaded = load(
                db_cls, dataset, workdir, threads, iterations, lambda: emit(progress), check=check, store=store
            )
        except Exception as exc:
            text = message(exc)
    if loaded is None:  # `text` holds the setup or load error
        if load_id is not None:
            emit({"kind": "entry", "id": load_id, "status": "failed", "error": text})
        for case in cases:
            emit({"kind": "start", "id": case.bench_id})
            emit(case_entry(case, status="failed", error=f"load failed: {text}"))
        return
    if load_id is not None:
        emit(load_entry(load_id, loaded.load_ns, loaded.check))
    try:
        for case in cases:
            _run_case(case, loaded.db, base, settings, emit)
    finally:
        loaded.db.close()
        remove(loaded.workdir)


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
        )
    finally:
        if own_workdir:
            remove(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
