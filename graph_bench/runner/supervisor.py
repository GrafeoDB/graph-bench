"""The supervisor: runs one worker process per database and thread setting, in sequence, and writes results.json.

For each database it first probes, in a short subprocess, whether its driver imports (so this process never
imports a driver) and which environment its worker needs. It then starts the worker, reads the worker's JSON
lines from its `--out` file as they arrive and records each entry.

`case_timeout_s` bounds silence; the worker reports progress after every run. The worker writes a `progress`
line after a case's preflight, after each warm-up and timed run, and after each load iteration, so the worker is
killed only when one single run (or startup step) stays silent for `case_timeout_s`; a whole case may take
longer. The same value goes to the worker as `--case-timeout`, the preflight budget of a case without an in-run
budget. Before the worker's first `start` line (startup, driver import, dataset checks) the limit is at least
`STARTUP_TIMEOUT_S`.

Time budgets (spec rule 8, order independent): a case gets an in-run `--budget` only when at least 3 earlier
databases have an ok/agree preflight for it; at the end `apply_budget_rule` compares every entry with the
other databases only.

Resume rule: when a worker exits non-zero or is killed for silence, the last `start` id without an entry
becomes `failed` ("worker exited with code <n>") or `timeout`. When that id is a load, every not yet recorded
case of that dataset becomes `failed` ("load failed: <load id>") too, so a broken load is not retried. A new
worker then starts with `--skip` for every recorded id. Without an open `start` id, every unrecorded id gets
the failure and the database is done; so every restart records at least one id and the loop ends.
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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import psutil

from graph_bench.databases import model_support
from graph_bench.databases.base import DatabaseInfo
from graph_bench.datasets import cache
from graph_bench.results import Entry, GrafeoSource, Machine, Reference, Results
from graph_bench.runner.worker import DEFAULT_CASE_TIMEOUT_S, reference_kind
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Selection
from graph_bench.types import Category, Model, Timing

RESULTS_FILE = "results.json"
GRAFEO_FILE = Path(".graph-bench") / "grafeo.json"
STARTUP_TIMEOUT_S = 120.0
BUDGET_FACTOR = 10
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


def _tag(threads: int | None) -> str:
    return "1" if threads == 1 else "all"


@dataclass(frozen=True)
class _Meta:
    """What an entry carries besides its result, known before the worker runs."""

    category: Category
    model: Model
    benchmark: str
    database: str
    scale: str
    threads: int | None
    dataset: str
    supported: bool
    reference: Reference | None
    load: bool = False

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
    """Every id one worker reports: its cases (as in `catalog.cases`) and its loads (as in `catalog.load_ids`)."""
    models = model_support(db)
    meta: dict[str, _Meta] = {}
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
                supported=item.model in models,
                reference=reference_kind(item),
            )
    if "load" in sel.categories:
        for item in catalog.items(sel.name):
            if item.model in models:
                for scale in sel.scales:
                    load_id = catalog.load_id(item.dataset, db, scale, threads)
                    load = _Meta(
                        "load",
                        item.model,
                        f"load_{item.dataset}",
                        db,
                        scale,
                        threads,
                        item.dataset,
                        supported=True,
                        reference=None,
                        load=True,
                    )
                    meta.setdefault(load_id, load)
    return meta


def _top3_mean(values: Sequence[float]) -> float:
    """The mean of the 3 smallest values, or of all of them if there are fewer."""
    return statistics.fmean(sorted(values)[:3])


def _budgets(entries: Sequence[Entry], db: str, meta: dict[str, _Meta]) -> dict[str, float]:
    """Seconds per case: 10x the mean of the 3 smallest `preflight_ns` among earlier databases' ok/agree entries.

    Only cases with at least 3 such entries get a budget, so the in-run budget never depends on which one or
    two databases happened to run first.
    """
    found: dict[tuple[str, str, int | None], list[int]] = {}
    for e in entries:
        if e.database != db and e.status == "ok" and e.answer == "agree" and e.preflight_ns is not None:
            found.setdefault((e.benchmark, e.scale, e.threads), []).append(e.preflight_ns)
    budgets = {}
    for bench_id, m in meta.items():
        values = found.get((m.benchmark, m.scale, m.threads))
        if values and len(values) >= 3 and not m.load:
            budgets[bench_id] = BUDGET_FACTOR * _top3_mean(values) / 1e9
    return budgets


def apply_budget_rule(results: Results) -> None:
    """Turn every ok/agree entry slower than 10x the mean of the 3 fastest other databases into `timeout`.

    A benchmark is an item at one scale and thread setting, across databases. Each entry's budget is 10x the
    mean of the 3 fastest other databases' ok/agree values for it (of all others if fewer than 3); with no
    other such database no budget applies. The entry itself never counts, so the outcome does not depend on
    order. Speed is `timing.median_ns` in bench mode and `preflight_ns` in codspeed mode. The timing of a
    marked entry is kept.
    """
    groups: dict[tuple[str, str, int | None], list[tuple[Entry, float]]] = {}
    for e in results.entries:
        if e.status != "ok" or e.answer != "agree":
            continue
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
    """The Grafeo under test: `.graph-bench/grafeo.json` (cwd-relative) when present, else the installed wheel."""
    if GRAFEO_FILE.is_file():
        doc = json.loads(GRAFEO_FILE.read_text(encoding="utf-8"))
        return {"version": doc.get("version"), "source": doc.get("source", "unknown"), "sha": doc.get("sha")}
    try:
        version: str | None = importlib.metadata.version("grafeo")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"version": version, "source": "pypi", "sha": None}


def machine() -> Machine:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cores": psutil.cpu_count(logical=True) or 0,
        "os": platform.platform(),
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


def _launch(argv: list[str], env: dict[str, str], log: IO[bytes]) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)


@dataclass
class _Attempt:
    """How one worker process ended."""

    code: int | None = None
    hung: bool = False
    silence_s: float = 0.0
    pending: str | None = None


@dataclass
class _Run:
    sel: Selection
    out_dir: Path
    workdir: Path
    case_timeout_s: float
    results: Results
    recorded: dict[str, Entry] = field(default_factory=dict)

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
            else:
                self._worker(db, threads, meta, probe["env"][_tag(threads)])

    def _not_installed(self, meta: dict[str, _Meta], error: str) -> None:
        for bench_id, m in meta.items():
            if m.supported:
                self.record(m.entry(bench_id, status="failed", error=f"not installed: {error}"))
            else:
                self.record(m.entry(bench_id, status="not_supported", error=f"model {m.model} not supported"))

    def _worker(self, db: str, threads: int | None, meta: dict[str, _Meta], extra_env: dict[str, str]) -> None:
        raw = self.out_dir / "raw" / f"{db}-{_tag(threads)}"
        raw.mkdir(parents=True, exist_ok=True)
        out = raw / "worker.jsonl"
        workdir = self.workdir / f"{db}-{_tag(threads)}"
        budgets = _budgets(self.results.entries, db, meta)
        env = {**os.environ, **extra_env}
        try:
            while remaining := [i for i in meta if i not in self.recorded]:
                argv = [sys.executable, "-m", "graph_bench.runner.worker", "--db", db, "--selection", self.sel.name]
                argv += ["--threads", _tag(threads), "--workdir", str(workdir), "--out", str(out)]
                argv += ["--case-timeout", repr(self.case_timeout_s)]
                skip = [i for i in meta if i in self.recorded]
                if skip:
                    argv += ["--skip", *skip]
                todo = [f"{i}={budgets[i]!r}" for i in remaining if i in budgets]
                if todo:
                    argv += ["--budget", *todo]
                attempt = self._attempt(argv, env, out, raw / "worker.log", db, meta)
                if not self._settle(attempt, meta):
                    break
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _settle(self, attempt: _Attempt, meta: dict[str, _Meta]) -> bool:
        """Record what a finished attempt left open; whether to start another worker."""
        if attempt.hung:
            status, error = "timeout", f"no output for {attempt.silence_s:g} s"
        elif attempt.code:
            status, error = "failed", f"worker exited with code {attempt.code}"
        else:
            status, error = "failed", "worker exited without a result"
        pending = attempt.pending
        if pending is None or (not attempt.hung and not attempt.code):
            for bench_id, m in meta.items():
                if bench_id not in self.recorded:
                    self.record(m.entry(bench_id, status=status, error=error))
            return False
        failed = meta[pending]
        self.record(failed.entry(pending, status=status, error=error))
        if failed.load:
            for bench_id, m in meta.items():
                same_data = (m.dataset, m.scale) == (failed.dataset, failed.scale)
                if not m.load and same_data and bench_id not in self.recorded:
                    self.record(m.entry(bench_id, status="failed", error=f"load failed: {pending}"))
        return True

    def _attempt(
        self, argv: list[str], env: dict[str, str], out: Path, log_path: Path, db: str, meta: dict[str, _Meta]
    ) -> _Attempt:
        """Run one worker to its end, recording its lines as they arrive; kill it after `case_timeout_s` of silence."""
        if out.exists() and out.stat().st_size:
            with out.open("rb+") as fh:  # a killed worker may leave half a line; end it
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
        offset = out.stat().st_size if out.exists() else 0
        attempt = _Attempt()
        with log_path.open("ab") as log:
            proc = _launch(argv, env, log)
            try:
                return self._follow(proc, out, offset, attempt, db, meta)
            finally:
                if proc.poll() is None:  # a harness error or Ctrl+C: leave no worker appending to `out`
                    proc.kill()
                    proc.wait()

    def _follow(
        self,
        proc: "subprocess.Popen[bytes]",
        out: Path,
        offset: int,
        attempt: _Attempt,
        db: str,
        meta: dict[str, _Meta],
    ) -> _Attempt:
        """Read the worker's new lines until it exits or stays silent too long."""
        started = False
        buffer = b""
        last = time.monotonic()
        while True:
            code = proc.poll()
            chunk = b""
            if out.exists():
                with out.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
            offset += len(chunk)
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                started |= self._line(line, attempt, db, meta)
            if lines:
                last = time.monotonic()
            if code is not None:
                attempt.code = code
                return attempt
            limit = self.case_timeout_s if started else max(self.case_timeout_s, STARTUP_TIMEOUT_S)
            if time.monotonic() - last > limit:  # `_attempt` kills it
                attempt.hung, attempt.silence_s = True, limit
                return attempt
            time.sleep(_POLL_S)

    def _line(self, raw: bytes, attempt: _Attempt, db: str, meta: dict[str, _Meta]) -> bool:
        """Handle one worker line; whether it was a `start` line.

        Every line resets the silence timer in `_follow`; that is all a `progress` line does.
        """
        try:
            line = json.loads(raw)
        except ValueError:
            return False  # the end of a line a killed worker left unfinished
        kind = line.pop("kind", None)
        if kind == "database":
            self.results.databases[db] = DatabaseInfo(**line)
        elif kind == "start":
            attempt.pending = line["id"]
            return True
        elif kind == "entry":
            bench_id = line.pop("id")
            if attempt.pending == bench_id:
                attempt.pending = None
            timing = line.pop("timing", None)
            entry = meta[bench_id].entry(bench_id, timing=Timing(**timing) if timing else None, **line)
            self.record(entry)
        return False


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
        mode="bench",
        started_at=started_at,
        databases={},
    )
    run = _Run(sel, out_dir, workdir, case_timeout_s, results)
    for db in dbs:
        run.database(db)
    results.entries.sort(key=lambda e: e.id)
    apply_budget_rule(results)
    results.write(out_dir / RESULTS_FILE)
    return results


__all__ = ["RESULTS_FILE", "apply_budget_rule", "grafeo_source", "graph_bench_commit", "machine", "run_bench"]
