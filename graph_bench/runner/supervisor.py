"""The supervisor: runs one process per database and thread setting, in sequence, and writes results.json.

The process is a `Runner`: the worker (`run_bench`, mode "bench") or `pytest benches --codspeed` (`run_codspeed`,
mode "codspeed"). Both write the same JSON lines (see `graph_bench.runner.worker`) to their `--out` / `--gb-out`
file, so the supervision below is the same for both; "worker" below means either.

For each database it first probes, in a short subprocess, whether its driver imports (so this process never
imports a driver) and which environment its worker needs. It then starts the worker, reads the worker's JSON
lines from its `--out` file as they arrive and records each entry.

`case_timeout_s` bounds silence; the worker reports progress after every run. The worker writes a `progress`
line after a case's preflight call, after each warm-up and timed run, and after each load iteration, so the
worker is killed only when one single run (or startup step, or harness step) stays silent for `case_timeout_s`;
a whole case may take longer. The same value goes to the worker as `--case-timeout`, the preflight budget of a
case without an in-run budget. Before the worker's first `start` line (startup, driver import, dataset checks)
the limit is at least `STARTUP_TIMEOUT_S`.

Time budgets (spec rule 8): a case gets an in-run `--budget` only when at least 3 earlier databases have an
ok/agree preflight for it: 30x the mean of their 3 fastest, a runaway guard on first (cold) runs. This cutoff
depends on which databases ran earlier. The supervisor enforces it too: it arms the cutoff on the case's `run`
line, which the worker writes immediately before the preflight's database call, and disarms it on the next line
(the `progress` line right after the call). A worker silent for longer than the budget in that window is killed
and the case becomes `timeout` ("too slow: cut off at ..."); harness work (reference loading, the answer check)
is outside the window. At the end `apply_budget_rule` compares every entry with the other databases only, at
10x on the final timings; that decides the marker, and that final rule does not depend on the run order. Startup and
both load rows (`types.BUDGET_EXEMPT`) and the answer-only items (`catalog.answer_only`) have no budget, in the run or
at the end, and are no yardstick.

Resources (the Memory and Disk records, `Selection.resources`): after a database's worker (or pytest) process for
one thread setting has finished, in both modes, the supervisor measures each load the database's models reach, one
fresh child per (dataset, store, scale), sampling the child's process tree from here
(`graph_bench.runner.resources.run_resource`). The child inherits the worker's environment and may stay silent for
`case_timeout_s` (at least `STARTUP_TIMEOUT_S` before its first line), each load in its own directory under
`<workdir>/<db>-<threads>-resources/`. The selection's other loads get ⊘ records, a database that is not
installed ✗ records, and a harness error in this process a `failed` record ("resources harness: ...").
These go to results.json `resources`, never to CodSpeed.

Resume rule: when a worker exits non-zero or is killed for silence, the last `start` id without an entry
becomes `failed` ("worker exited with code <n>") or `timeout`. When that id is a load (the Load row, not Load
(query language), whose instances no case uses), every not yet recorded case of that dataset becomes `failed`
("load failed: <load id>") too, so a broken load is not retried. A new
worker then starts with `--skip` for every recorded id. Without an open `start` id, every unrecorded id gets
the failure and the database is done; so every restart records at least one id and the loop ends. With
`Runner.rerun_passed` (CodSpeed), a killed or crashed process is relaunched even when every id is recorded, as
long as a passed id is not retired, so CodSpeed measures the passed cases again (see `_Run._again`); pytest's
normal exit code 1 with an open id (a harness bug) is not such a crash.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

import psutil

from graph_bench.databases import model_support
from graph_bench.databases.base import DatabaseInfo
from graph_bench.datasets import cache
from graph_bench.results import Entry, GrafeoSource, Machine, Reference, Resource, Results
from graph_bench.runner import message
from graph_bench.runner.resources import run_resource, unmeasured
from graph_bench.runner.startup import STARTUP_TIMEOUT_S
from graph_bench.runner.worker import DEFAULT_CASE_TIMEOUT_S, reference_kind
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Selection
from graph_bench.suites.startup import primary_model
from graph_bench.types import BUDGET_EXEMPT, LOAD_CATEGORIES, Category, Model, Store, Timing

RESULTS_FILE = "results.json"
GRAFEO_FILE = Path(".graph-bench") / "grafeo.json"
# pytest's exit code when its session ended normally with failed tests; pytest-codspeed then kept its measurements.
PYTEST_TESTS_FAILED = 1
BUDGET_FACTOR = 10
IN_RUN_FACTOR = 30
# The in-run guard never cuts a case off below this: it exists to stop runaways, and a cold first run (a JVM
# compiling a new query, a planner's first plan) can take tens of ms where the timed median is within budget. The
# final 10x rule on medians decides the marker; the floor only bounds what a runaway costs.
IN_RUN_FLOOR_S = 10.0
_POLL_S = 0.05
_ROOT = Path(__file__).resolve().parents[2]

# Run in a subprocess: import the database class and report its worker environment, or the import error.
_PROBE = """\
import json, sys
from graph_bench.databases import get_database
try:
    cls = get_database(sys.argv[1])
    doc = {"env": {"1": cls.process_env(1), "all": cls.process_env(None)}}
except Exception as exc:
    doc = {"error": str(exc) or type(exc).__name__}
print(json.dumps(doc))
"""


def _platform_label() -> str:
    """The platform a missing package is reported against, e.g. "linux/aarch64"."""
    return f"{sys.platform}/{platform.machine().lower()}"


def _tag(threads: int | None) -> str:
    return "1" if threads == 1 else "all"


def _not_installed_reason(error: str) -> str:
    """The error of every entry and resource record of a database whose driver does not import."""
    return f"not installed: {error} (the package is missing or has no build for {_platform_label()})"


Kind = Literal["case", "load", "load_ql", "startup"]


@dataclass(frozen=True)
class _Meta:
    """What an entry carries besides its result, known before the worker runs.

    `kind` says what the id is: a catalog item's case, a Load entry, a Load (query language) entry or the Startup
    entry (whose `dataset` is empty).
    """

    category: Category
    model: Model
    benchmark: str
    database: str
    scale: str
    threads: int | None
    dataset: str
    supported: bool
    reference: Reference | None
    kind: Kind = "case"
    store: Store = "graph"

    def entry(self, bench_id: str, **fields: Any) -> Entry:
        reference = fields.pop("reference", None) or self.reference
        return Entry(
            id=bench_id,
            category=self.category,
            model=self.model,
            benchmark=self.benchmark,
            database=self.database,
            scale=self.scale,
            threads=self.threads,
            reference=reference,
            **fields,
        )


def _expected(sel: Selection, db: str, threads: int | None) -> dict[str, _Meta]:
    """Every id one worker reports: its cases (as in `catalog.cases`), its loads (`catalog.load_ids`), its
    query-language loads (`catalog.load_ql_ids`) and its startup (`catalog.startup_ids`)."""
    models = model_support(db)
    meta: dict[str, _Meta] = {}
    startup_id = catalog.startup_id(db, threads)
    if startup_id in catalog.startup_ids(sel.name, [db]):
        meta[startup_id] = _Meta(
            "startup",
            primary_model(models),
            "startup",
            db,
            catalog.STARTUP_SCALE,
            threads,
            "",
            supported=True,
            reference="python",
            kind="startup",
        )
    for case in catalog.cases(sel.name, [db]):
        if case.threads == threads:
            item = case.item
            meta[case.bench_id] = _Meta(
                item.category,
                item.model,
                item.id,
                db,
                case.scale,
                threads,
                item.dataset,
                supported=item.supported_by(models),
                reference=reference_kind(item),
                store=item.store,
            )
    if "load" in sel.categories:
        reached = set(catalog.load_keys(sel.name, models))
        for (dataset, store), model in catalog.all_load_keys(sel.name).items():
            for scale in sel.scales:
                load_id = catalog.load_id(dataset, db, scale, threads, store)
                meta[load_id] = _Meta(
                    "load",
                    model,
                    catalog.load_benchmark(dataset, store),
                    db,
                    scale,
                    threads,
                    dataset,
                    supported=(dataset, store) in reached,
                    reference=None,
                    kind="load",
                    store=store,
                )
    if "load_ql" in sel.categories:
        reached = set(catalog.load_keys(sel.name, models))
        for dataset, model in catalog.load_ql_keys(sel.name).items():
            for scale in sel.scales:
                load_ql_id = catalog.load_ql_id(dataset, db, scale, threads)
                meta[load_ql_id] = _Meta(
                    "load_ql",
                    model,
                    catalog.load_ql_benchmark(dataset),
                    db,
                    scale,
                    threads,
                    dataset,
                    supported=(dataset, "graph") in reached,
                    reference=None,
                    kind="load_ql",
                )
    return meta


def _passed(entry: Entry) -> bool:
    """A result CodSpeed measures: ok/agree, or for either load row ok and not ≠ (a load without `counts` is
    unchecked)."""
    if entry.status != "ok":
        return False
    return entry.answer == "agree" or (entry.category in LOAD_CATEGORIES and entry.answer == "unchecked")


def _top3_mean(values: Sequence[float]) -> float:
    """The mean of the 3 smallest values, or of all of them if there are fewer."""
    return statistics.fmean(sorted(values)[:3])


def _budgets(entries: Sequence[Entry], db: str, meta: dict[str, _Meta]) -> dict[str, float]:
    """Seconds per case: 30x the mean of the 3 smallest `preflight_ns` among earlier databases' ok/agree entries,
    and at least `IN_RUN_FLOOR_S`.

    This in-run cutoff is a runaway guard on first (cold) runs; `apply_budget_rule` at 10x on the final timings
    decides the marker. Only cases with at least 3 such entries get a budget, so the in-run budget never depends
    on which one or two databases happened to run first. `BUDGET_EXEMPT` categories and answer-only items
    (`catalog.answer_only`) get none.
    """
    found: dict[tuple[str, str, int | None], list[int]] = {}
    for e in entries:
        if e.database != db and e.status == "ok" and e.answer == "agree" and e.preflight_ns is not None:
            found.setdefault((e.benchmark, e.scale, e.threads), []).append(e.preflight_ns)
    budgets = {}
    for bench_id, m in meta.items():
        values = found.get((m.benchmark, m.scale, m.threads))
        if values and len(values) >= 3 and m.category not in BUDGET_EXEMPT and not catalog.answer_only(m.benchmark):
            budgets[bench_id] = max(IN_RUN_FACTOR * _top3_mean(values) / 1e9, IN_RUN_FLOOR_S)
    return budgets


def apply_budget_rule(results: Results) -> None:
    """Turn every ok/agree entry slower than 10x the mean of the 3 fastest other databases into `timeout`.

    A benchmark is an item at one scale and thread setting, across databases. Each entry's budget is 10x the
    mean of the 3 fastest other databases' ok/agree values for it (of all others if fewer than 3); with no
    other such database no budget applies. Startup and both load rows (`BUDGET_EXEMPT`) and answer-only items
    (`catalog.answer_only`) are exempt, as entries and as yardsticks. The entry itself never counts, so the outcome
    does not depend on order. Speed is `timing.median_ns` in bench mode and `preflight_ns` in codspeed mode. The
    timing of a marked entry is kept.
    """
    groups: dict[tuple[str, str, int | None], list[tuple[Entry, float]]] = {}
    for e in results.entries:
        if e.status != "ok" or e.answer != "agree" or e.category in BUDGET_EXEMPT or catalog.answer_only(e.benchmark):
            continue  # exempt: no budget and no yardstick (as in the in-run `_budgets`)
        speed = (e.timing.median_ns if e.timing is not None else None) if results.mode == "bench" else e.preflight_ns
        if speed is not None:
            groups.setdefault((e.benchmark, e.scale, e.threads), []).append((e, float(speed)))
    marked: list[tuple[Entry, float]] = []
    for members in groups.values():
        for i, (e, speed) in enumerate(members):
            others = [other for j, (_, other) in enumerate(members) if j != i]
            if not others:
                continue
            average = _top3_mean(others)
            if average > 0 and speed > BUDGET_FACTOR * average:
                marked.append((e, speed / average))
    for e, ratio in marked:
        e.status = "timeout"
        e.error = f"too slow: {ratio:.1f}x the top-3 average"


def graph_bench_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def grafeo_source() -> GrafeoSource:
    """The Grafeo under test: `.graph-bench/grafeo.json` (cwd-relative), else the installed wheel.

    The file counts only while its version is the installed one: after a later `uv sync` replaced the recorded
    build it is stale, and the installed version is reported with source "pypi".
    """
    try:
        version: str | None = importlib.metadata.version("grafeo")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if GRAFEO_FILE.is_file():
        doc = json.loads(GRAFEO_FILE.read_text(encoding="utf-8"))
        if doc.get("version") == version:
            return {
                "version": version,
                "source": doc.get("source", "unknown"),
                "sha": doc.get("sha"),
                "git_sha": doc.get("git_sha"),
            }
    return {"version": version, "source": "pypi", "sha": None, "git_sha": None}


def _native_cpu_name() -> str | None:
    """The marketing CPU name from the OS (Windows registry, /proc/cpuinfo, sysctl), or None."""
    try:
        if sys.platform == "win32":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0])
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                name, sep, value = line.partition(":")
                if sep and name.strip() == "model name":
                    return value
            return None
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10, check=True
            )
            return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def cpu_name() -> str:
    """The CPU name shown in reports, e.g. "AMD Ryzen 9 7940HS"; falls back to `platform.processor()`."""
    name = (_native_cpu_name() or "").strip()
    return name or (platform.processor() or platform.machine()).strip()


def machine() -> Machine:
    return {
        "cpu": cpu_name(),
        "cores": psutil.cpu_count(logical=True) or 0,
        "os": platform.platform(),
        "memory_bytes": psutil.virtual_memory().total,
    }


def _probe(db: str, timeout_s: float) -> dict[str, Any]:
    """Import the database class in a subprocess: `{"env": {"1": ..., "all": ...}}` or `{"error": message}`."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE, db], capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return {"error": f"importing the driver took over {timeout_s:g} s"}
    lines = proc.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        tail = proc.stderr.strip().splitlines()
        return {"error": f"probe exited with code {proc.returncode}" + (f": {tail[-1]}" if tail else "")}


def _launch(argv: list[str], env: dict[str, str], log: IO[bytes], cwd: Path | None = None) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=log)


@dataclass(frozen=True)
class LaunchRequest:
    """What one worker process is asked to do: the ids to skip and the in-run budgets of the rest."""

    db: str
    selection: str
    threads: int | None
    workdir: Path
    out: Path
    case_timeout_s: float
    skip: list[str]
    budgets: dict[str, float]


def worker_argv(r: LaunchRequest) -> list[str]:
    """`python -m graph_bench.runner.worker ...`; `--skip` and `--budget` take several values each."""
    argv = [sys.executable, "-m", "graph_bench.runner.worker", "--db", r.db, "--selection", r.selection]
    argv += ["--threads", _tag(r.threads), "--workdir", str(r.workdir), "--out", str(r.out)]
    argv += ["--case-timeout", repr(r.case_timeout_s)]
    if r.skip:
        argv += ["--skip", *r.skip]
    if r.budgets:
        argv += ["--budget", *(f"{i}={s!r}" for i, s in r.budgets.items())]
    return argv


def pytest_argv(r: LaunchRequest) -> list[str]:
    """`python -m pytest benches --codspeed ...`, run from the repository root; done ids are deselected.

    `-o addopts=` drops the test suite's coverage options from pyproject.toml.
    """
    argv = [sys.executable, "-m", "pytest", "benches", "--codspeed", "-o", "addopts=", "-p", "no:cacheprovider"]
    argv += ["-v", "-rfEs", "--db", r.db, "--selection", r.selection, "--threads", _tag(r.threads)]
    argv += ["--workdir", str(r.workdir), "--gb-out", str(r.out), "--case-timeout", repr(r.case_timeout_s)]
    for bench_id in r.skip:
        argv += ["--deselect", bench_id]
    for bench_id, seconds in r.budgets.items():
        argv += ["--gb-budget", f"{bench_id}={seconds!r}"]
    return argv


@dataclass(frozen=True)
class Runner:
    """How to start the process for one database: its argv, working directory and extra environment.

    `name` names its files under `raw/<db>-<threads>/` and its exit messages; with `echo` its output is copied
    to this process's stdout as it arrives (for the CI log). With `rerun_passed`, a relaunch after a kill or crash
    runs the passed ids again instead of skipping them (see `_Run._skip`).
    """

    name: str
    argv: Callable[[LaunchRequest], list[str]]
    mode: Literal["bench", "codspeed"]
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] = ()
    echo: bool = False
    rerun_passed: bool = False


WORKER = Runner("worker", worker_argv, "bench")
PYTEST = Runner(
    "pytest", pytest_argv, "codspeed", cwd=_ROOT, env=(("PYTHONIOENCODING", "utf-8"),), echo=True, rerun_passed=True
)


class _Echo:
    """Copies what a process appends to its log file to stdout."""

    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.offset = path.stat().st_size if path.exists() else 0

    def pump(self) -> None:
        if not self.enabled or not self.path.exists():
            return
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        if not data:
            return
        self.offset += len(data)
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            sys.stdout.write(data.decode("utf-8", errors="replace"))
            return
        sys.stdout.flush()
        buffer.write(data)
        buffer.flush()


@dataclass
class _Attempt:
    """How one worker process ended."""

    code: int | None = None
    hung: bool = False
    silence_s: float = 0.0
    pending: str | None = None
    cutoff_s: float | None = None  # set when killed at the pending case's in-run budget


@dataclass
class _Run:
    sel: Selection
    out_dir: Path
    workdir: Path
    case_timeout_s: float
    results: Results
    runner: Runner = WORKER
    recorded: dict[str, Entry] = field(default_factory=dict)
    retired: set[str] = field(default_factory=set)

    def record(self, entry: Entry) -> None:
        if entry.id not in self.recorded:
            self.recorded[entry.id] = entry
            self.results.entries.append(entry)

    def database(self, db: str) -> None:
        probe = _probe(db, max(self.case_timeout_s, STARTUP_TIMEOUT_S))
        for threads in self.sel.threads:
            meta = _expected(self.sel, db, threads)
            if "error" in probe:
                self._not_installed(meta, probe["error"])
                self._resources(db, threads, None, missing=_not_installed_reason(probe["error"]))
            else:
                extra_env = probe["env"][_tag(threads)]
                self._worker(db, threads, meta, extra_env)
                self._resources(db, threads, extra_env)

    def _not_installed(self, meta: dict[str, _Meta], error: str) -> None:
        for bench_id, m in meta.items():
            if m.supported:
                self.record(m.entry(bench_id, status="failed", error=_not_installed_reason(error)))
            else:
                self.record(m.entry(bench_id, status="not_supported", error=f"model {m.model} not supported"))

    def _resources(
        self, db: str, threads: int | None, extra_env: dict[str, str] | None, *, missing: str | None = None
    ) -> None:
        """The Memory and Disk records of one database and thread setting, after its worker finished.

        Every (dataset, store) of the selection at every scale gets one: measured in a fresh child where the
        database's models reach the load, ⊘ where they do not, ✗ with `missing` when the database is not installed.
        """
        if not self.sel.resources:
            return
        reached = set(catalog.load_keys(self.sel.name, model_support(db)))
        raw = self.out_dir / "raw" / f"{db}-{_tag(threads)}" / "resources"
        env = {**os.environ, **dict(self.runner.env), **(extra_env or {})}
        for (dataset, store), model in catalog.all_load_keys(self.sel.name).items():
            for scale in self.sel.scales:
                record: Resource
                if (dataset, store) not in reached:
                    error = f"model {model} not supported"
                    record = unmeasured(db, dataset, store, scale, threads, "not_supported", error)
                elif missing is not None:
                    record = unmeasured(db, dataset, store, scale, threads, "failed", missing)
                else:
                    name = f"{catalog.load_benchmark(dataset, store)}-{scale}"
                    try:
                        record = run_resource(
                            db,
                            self.sel.name,
                            dataset,
                            store,
                            scale,
                            threads,
                            workdir=self.workdir / f"{db}-{_tag(threads)}-resources" / name,
                            raw=raw,
                            env=env,
                            timeout_s=self.case_timeout_s,
                        )
                    except Exception as exc:  # a harness error fails this record only; the run goes on
                        error = f"resources harness: {message(exc)}"
                        record = unmeasured(db, dataset, store, scale, threads, "failed", error)
                self.results.resources.append(record)

    def _worker(self, db: str, threads: int | None, meta: dict[str, _Meta], extra_env: dict[str, str]) -> None:
        name = self.runner.name
        raw = self.out_dir / "raw" / f"{db}-{_tag(threads)}"
        raw.mkdir(parents=True, exist_ok=True)
        out = raw / f"{name}.jsonl"
        workdir = self.workdir / f"{db}-{_tag(threads)}"
        budgets = _budgets(self.results.entries, db, meta)
        env = {**os.environ, **dict(self.runner.env), **extra_env}
        try:
            launch = any(i not in self.recorded for i in meta)
            while launch:
                skip = self._skip(meta)
                request = LaunchRequest(
                    db=db,
                    selection=self.sel.name,
                    threads=threads,
                    workdir=workdir,
                    out=out,
                    case_timeout_s=self.case_timeout_s,
                    skip=skip,
                    budgets={i: budgets[i] for i in meta if i in budgets and i not in skip},
                )
                argv = self.runner.argv(request)
                attempt = self._attempt(argv, env, out, raw / f"{name}.log", db, meta, request.budgets)
                launch = self._settle(attempt, meta) and self._again(attempt, meta)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _again(self, attempt: _Attempt, meta: dict[str, _Meta]) -> bool:
        """Whether to relaunch after a hung or crashed attempt that `_settle` let continue.

        While ids are unrecorded, always. With `rerun_passed`, also when every id is recorded but a passed one
        is not retired, and only when the process was killed or crashed: it took its CodSpeed measurements with
        it (e.g. its last case hung), so the relaunch measures them again. Exit code 1 (`PYTEST_TESTS_FAILED`)
        with an open id is a harness bug in a session that ended normally, so the measurements were kept and
        nothing is relaunched; the open id is already `failed`. Each relaunching attempt retired its killed id,
        and a clean exit never gets here (`_settle` returns False), so the loop ends.
        """
        if any(i not in self.recorded for i in meta):
            return True
        lost = attempt.hung or attempt.code not in (0, PYTEST_TESTS_FAILED)
        if not self.runner.rerun_passed or not lost:
            return False
        return any(i not in self.retired and _passed(e) for i, e in self.recorded.items() if i in meta)

    def _skip(self, meta: dict[str, _Meta]) -> list[str]:
        """The ids a (re)launch must not run: every recorded id, or with `rerun_passed` only the marked ones.

        pytest-codspeed writes its measurements when the session ends, so a killed pytest loses all of them;
        the relaunch runs the passed cases (`_passed`) again so CodSpeed measures them. Their
        first entry stays (`record` keeps one per id). An id that was running when a process was killed or
        crashed is retired: never run again, so every relaunch still makes progress.
        """
        if not self.runner.rerun_passed:
            return [i for i in meta if i in self.recorded]
        return [i for i in meta if i in self.retired or (i in self.recorded and not _passed(self.recorded[i]))]

    def _settle(self, attempt: _Attempt, meta: dict[str, _Meta]) -> bool:
        """Record what a finished attempt left open; whether to start another worker."""
        if attempt.cutoff_s is not None:
            budget = f"{attempt.cutoff_s:.3g}"
            status, error = (
                "timeout",
                f"too slow: cut off at {budget} s ({IN_RUN_FACTOR}x the top-3 average of earlier databases, "
                f"at least {IN_RUN_FLOOR_S:g} s)",
            )
        elif attempt.hung:
            status, error = "timeout", f"no output for {attempt.silence_s:g} s"
        elif attempt.code:
            status, error = "failed", f"{self.runner.name} exited with code {attempt.code}"
        else:
            status, error = "failed", f"{self.runner.name} exited without a result"
        pending = attempt.pending
        if pending is None or (not attempt.hung and not attempt.code):
            for bench_id, m in meta.items():
                if bench_id not in self.recorded:
                    self.record(m.entry(bench_id, status=status, error=error))
            return False
        self.retired.add(pending)
        if pending in self.recorded:  # a passed id run again: keep its first verdict
            return True
        failed = meta[pending]
        self.record(failed.entry(pending, status=status, error=error))
        if failed.kind == "load":
            for bench_id, m in meta.items():
                same_data = (m.dataset, m.store, m.scale) == (failed.dataset, failed.store, failed.scale)
                if m.kind == "case" and same_data and bench_id not in self.recorded:
                    self.record(m.entry(bench_id, status="failed", error=f"load failed: {pending}"))
        return True

    def _attempt(
        self,
        argv: list[str],
        env: dict[str, str],
        out: Path,
        log_path: Path,
        db: str,
        meta: dict[str, _Meta],
        budgets: dict[str, float],
    ) -> _Attempt:
        """Run one worker to its end, recording its lines as they arrive; kill it when silent too long (`_follow`)."""
        if out.exists() and out.stat().st_size:
            with out.open("rb+") as fh:  # a killed worker may leave half a line; end it
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
        offset = out.stat().st_size if out.exists() else 0
        attempt = _Attempt()
        echo = _Echo(log_path, self.runner.echo)
        with log_path.open("ab") as log:
            proc = _launch(argv, env, log, self.runner.cwd)
            try:
                return self._follow(proc, out, offset, attempt, db, meta, echo, budgets)
            finally:
                if proc.poll() is None:  # a harness error or Ctrl+C: leave no worker appending to `out`
                    proc.kill()
                    proc.wait()
                echo.pump()
                if self.runner.mode == "codspeed":
                    drop_empty_codspeed_result(env, proc.pid)

    def _follow(
        self,
        proc: "subprocess.Popen[bytes]",
        out: Path,
        offset: int,
        attempt: _Attempt,
        db: str,
        meta: dict[str, _Meta],
        echo: _Echo,
        budgets: dict[str, float],
    ) -> _Attempt:
        """Read the worker's new lines until it exits or stays silent too long.

        Silence is limited to `case_timeout_s`, and right after the `run` line of a case with an in-run budget
        (its preflight's database call) to that budget, until the next line: a preflight call beyond it is cut
        off there instead of running to its end.
        """
        started = False
        cutoff: float | None = None  # the in-run budget of the case whose `run` line is the last line
        buffer = b""
        last = time.monotonic()
        while True:
            code = proc.poll()
            echo.pump()
            chunk = b""
            if out.exists():
                with out.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
            offset += len(chunk)
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                kind = self._line(line, attempt, db, meta)
                started |= kind == "start"
                cutoff = budgets.get(attempt.pending or "") if kind == "run" else None
            if lines:
                last = time.monotonic()
            if code is not None:
                attempt.code = code
                return attempt
            limit = self.case_timeout_s if started else max(self.case_timeout_s, STARTUP_TIMEOUT_S)
            silent = time.monotonic() - last
            if cutoff is not None and cutoff < limit and silent > cutoff:  # `_attempt` kills it
                attempt.hung, attempt.silence_s, attempt.cutoff_s = True, cutoff, cutoff
                return attempt
            if silent > limit:
                attempt.hung, attempt.silence_s = True, limit
                return attempt
            time.sleep(_POLL_S)

    def _line(self, raw: bytes, attempt: _Attempt, db: str, meta: dict[str, _Meta]) -> str | None:
        """Handle one worker line; its kind, or None for an unreadable line.

        Every line resets the silence timer in `_follow`; that is all a `progress` line does, and a `run` line
        also arms the in-run cutoff there.
        """
        try:
            line = json.loads(raw)
        except ValueError:
            return None  # the end of a line a killed worker left unfinished
        kind = line.pop("kind", None)
        if kind == "database":
            self.results.databases[db] = DatabaseInfo(**line)
        elif kind == "start":
            attempt.pending = line["id"]
        elif kind == "entry":
            bench_id = line.pop("id")
            if attempt.pending == bench_id:
                attempt.pending = None
            timing = line.pop("timing", None)
            entry = meta[bench_id].entry(bench_id, timing=Timing(**timing) if timing else None, **line)
            self.record(entry)
        return kind


def run_bench(
    selection: str,
    dbs: Sequence[str],
    out_dir: Path,
    *,
    workdir: Path,
    case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
) -> Results:
    """Run every database of `dbs` in order, one worker per thread setting, and write `out_dir/results.json`.

    Later databases get a preflight budget per case from the earlier ones; at the end `apply_budget_rule`
    marks entries far slower than the top 3. Worker lines, logs and answers go to `out_dir/raw/<db>-<threads>/`.
    """
    return _run_all(selection, dbs, out_dir, workdir, case_timeout_s, WORKER)


def drop_empty_codspeed_result(env: Mapping[str, str], pid: int) -> bool:
    """Remove `<CODSPEED_PROFILE_FOLDER>/results/<pid>.json` when it lists no benchmarks; whether it did.

    pytest-codspeed names the file after the pytest process (plugin.py, `pytest_sessionfinish`), and the CodSpeed
    runner fails the job on a results file with no benchmarks ("No benchmarks found in walltime results file").
    A file with benchmarks, an unreadable file and every other file in the folder are left alone.
    """
    folder = env.get("CODSPEED_PROFILE_FOLDER")
    if not folder:
        return False
    path = Path(folder) / "results" / f"{pid}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("benchmarks"):
        return False
    path.unlink()
    return True


def run_codspeed(
    selection: str,
    dbs: Sequence[str],
    out_dir: Path,
    *,
    workdir: Path,
    case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
) -> Results:
    """As `run_bench`, with one `pytest benches --codspeed` process per database and thread setting.

    CodSpeed measures the benchmarks; results.json gets mode "codspeed": answers and `preflight_ns`, no
    timings. The environment, CODSPEED_PROFILE_FOLDER included, passes to pytest unchanged. Pytest's exit code
    does not matter: a failed or skipped benchmark is already an entry. A pytest process that measured nothing
    (a relaunch left with only skipped cases) leaves a results file without benchmarks, which the CodSpeed
    runner rejects for the whole job; `drop_empty_codspeed_result` removes that one file, and only it.
    """
    return _run_all(selection, dbs, out_dir, workdir, case_timeout_s, PYTEST)


def _run_all(
    selection: str, dbs: Sequence[str], out_dir: Path, workdir: Path, case_timeout_s: float, runner: Runner
) -> Results:
    sel = catalog.selection(selection)
    catalog.cases(selection, dbs)  # unknown names fail here, before anything runs
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache.prepare(selection)  # generate data now, so no worker spends its silence budget on it
    results = Results(
        graph_bench_commit=graph_bench_commit(),
        grafeo=grafeo_source(),
        machine=machine(),
        selection=selection,
        mode=runner.mode,
        started_at=started_at,
        databases={},
    )
    run = _Run(sel, out_dir, workdir, case_timeout_s, results, runner)
    for db in dbs:
        run.database(db)
    results.entries.sort(key=lambda e: e.id)
    results.resources.sort(key=lambda r: r.id)
    apply_budget_rule(results)
    results.write(out_dir / RESULTS_FILE)
    return results


__all__ = [
    "PYTEST",
    "RESULTS_FILE",
    "WORKER",
    "LaunchRequest",
    "Runner",
    "apply_budget_rule",
    "grafeo_source",
    "graph_bench_commit",
    "machine",
    "pytest_argv",
    "run_bench",
    "run_codspeed",
    "worker_argv",
]
