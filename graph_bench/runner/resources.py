"""The Memory and Disk records: one load in a fresh child process, sampled from outside.

Child: `python -m graph_bench.runner.resources --db X --selection S --dataset D --store (graph|rdf) --scale SC
--threads (1|all) --workdir W --out F` appends JSON lines to F:

1. it makes itself a child subreaper (`sampler.set_child_subreaper`, Linux), so a daemonized descendant (redislite's
   redis-server) stays in its tree;
2. `Database.prepare_environment()`, untimed, as the worker does: the engine is started and warmed, so the
   baseline holds it;
3. a fresh instance in W (`cls(W, threads)`);
4. `{"kind": "memory", "phase": "before", "t_ns", "rss_bytes", "processes"}`: its own tree's RSS (`tree_rss` of its
   pid), then `time.monotonic_ns()`;
5. `load_store(store, dataset)`, the load the Load row times;
6. the `"after"` line: `time.monotonic_ns()`, then the tree's RSS, plus `names`, the tree's process names;
7. the node and edge count check (`worker.count_check`);
8. `paths = store_paths()`, `close()`, then `dir_size(paths)` (None when `paths == []`): the database's own
   shutdown has flushed and checkpointed before the store is measured;
9. `{"kind": "resource", "status", "error", "answer", "detail", "storage", "disk_bytes"}`.

No step forces a garbage collection, in any runtime: memory as a user sees it. The child inherits the worker's
environment (`Database.process_env(threads)`).

Parent (`run_resource`, called by the supervisor): starts the child in its own process group, samples the child's
tree every `SAMPLE_INTERVAL_S` in a thread (`TreeSampler`) and kills the tree when the child ends or stays silent
past `timeout_s`. The record: `baseline_bytes` = before, `after_bytes` = after, `footprint_bytes` = after minus
before (may be negative), `peak_bytes` = the highest of before, after and every sample from before's to after's
time (`window`; one clock across processes), `samples` the samples in that window (0 for a load faster than one
interval), `processes` the most processes the tree held in it.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import psutil

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.datasets import cache
from graph_bench.results import Resource
from graph_bench.runner import message
from graph_bench.runner.sampler import (
    SAMPLE_INTERVAL_S,
    Sample,
    TreeSampler,
    dir_size,
    interpreter_pid,
    set_child_subreaper,
    tree_names,
    tree_rss,
)
from graph_bench.runner.startup import STARTUP_TIMEOUT_S, fresh_dir, kill_tree, new_process_group
from graph_bench.runner.worker import count_check, prepare_environment
from graph_bench.suites import catalog
from graph_bench.types import Status, Store

__all__ = ["EXIT_GRACE_S", "Window", "main", "measure", "run_resource", "unmeasured", "window"]

_POLL_S = 0.05
# How long a child may take to exit once its resource line is in (a runtime's shutdown); then its tree is killed.
EXIT_GRACE_S = 30.0


# The child -----------------------------------------------------------------------------------------------------


def _reading(phase: str) -> dict[str, Any]:
    """This process tree's memory line. `before` reads, then stamps; `after` stamps, then reads: the window between
    the two stamps holds the whole load."""
    pid = os.getpid()
    if phase == "before":
        rss, processes = tree_rss(pid)
        t_ns = time.monotonic_ns()
        return {"kind": "memory", "phase": phase, "t_ns": t_ns, "rss_bytes": rss, "processes": processes}
    t_ns = time.monotonic_ns()
    rss, processes = tree_rss(pid)
    line = {"kind": "memory", "phase": phase, "t_ns": t_ns, "rss_bytes": rss, "processes": processes}
    return {**line, "names": tree_names(pid)}


def _result(status: Status, error: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"kind": "resource", "status": status, "error": error, **fields}


def measure(
    db: str,
    selection: str,
    dataset: str,
    store: Store,
    scale: str,
    threads: int | None,
    workdir: Path,
    emit: Any,
) -> dict[str, Any]:
    """In the child: steps 1 to 8 of the module docstring; `emit` writes the memory lines. The resource line."""
    set_child_subreaper()
    try:
        cls = get_database(db)
        data = cache.prepare(selection)[(dataset, scale)]
    except Exception as exc:
        return _result("failed", message(exc))
    setup_error = prepare_environment(cls)
    if setup_error is not None:
        return _result("failed", setup_error)
    try:
        instance = cls(workdir, threads)
    except Exception as exc:
        return _result("failed", f"opening the instance: {message(exc)}")
    closed = False
    try:
        emit(_reading("before"))
        try:
            instance.load_store(store, data)
        except NotSupported as exc:
            return _result("not_supported", message(exc))
        except Exception as exc:
            return _result("failed", message(exc))
        emit(_reading("after"))
        try:
            answer, detail = count_check(instance, data)
        except Exception as exc:
            return _result("failed", f"counting the loaded data: {message(exc)}")
        try:
            storage = instance.info().storage
        except Exception as exc:
            return _result("failed", f"reading the database info: {message(exc)}")
        try:
            paths = instance.store_paths()
        except Exception as exc:
            return _result("failed", f"reading the store paths: {message(exc)}")
        closed = True
        try:
            instance.close()
        except Exception as exc:
            return _result("failed", f"closing: {message(exc)}")
    finally:
        if not closed:
            try:
                instance.close()
            except Exception as exc:  # the record keeps the first error
                sys.stderr.write(f"closing after the failure also failed: {message(exc)}\n")
    try:
        disk = dir_size(paths) if paths else None
    except OSError as exc:
        return _result("failed", f"measuring the store: {message(exc)}")
    return _result("ok", answer=answer, detail=detail, storage=storage, disk_bytes=disk)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graph_bench.runner.resources",
        description="Load one dataset into a fresh instance in this process; write its memory and disk lines.",
    )
    parser.add_argument("--db", required=True, help="Database name.")
    parser.add_argument("--selection", required=True, help="Selection name (for the dataset cache).")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--store", required=True, choices=("graph", "rdf"), help="The store to load.")
    parser.add_argument("--scale", required=True, help="Dataset scale.")
    parser.add_argument("--threads", required=True, choices=("1", "all"), help="1, or all cores.")
    parser.add_argument("--workdir", required=True, help="A fresh directory for the instance's files.")
    parser.add_argument("--out", required=True, help="JSON lines file to append to.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with Path(args.out).open("a", encoding="utf-8", newline="\n") as fh:

        def emit(line: dict[str, Any]) -> None:
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            fh.flush()

        threads = 1 if args.threads == "1" else None
        emit(measure(args.db, args.selection, args.dataset, args.store, args.scale, threads, Path(args.workdir), emit))
    return 0


# The parent ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The load window's readings: the peak RSS, the supervisor's samples in it and the most processes seen."""

    peak_bytes: int
    samples: int
    processes: int


def window(before: Sample, after: Sample, samples: Sequence[Sample]) -> Window:
    """Peak over `before`, `after` and every sample with `before.t_ns <= t_ns <= after.t_ns`."""
    inside = [s for s in samples if before.t_ns <= s.t_ns <= after.t_ns]
    readings = [before, after, *inside]
    return Window(
        max(r.rss_bytes for r in readings),
        len(inside),
        max(r.processes for r in readings),
    )


def unmeasured(
    db: str, dataset: str, store: Store, scale: str, threads: int | None, status: Status, error: str
) -> Resource:
    """A record without measurements: ⊘ for a load the database's models do not reach, ✗ when not installed."""
    rid = catalog.resource_id(dataset, db, scale, threads, store)
    return Resource(rid, db, dataset, store, scale, threads, status, error)


def _follow(proc: "subprocess.Popen[bytes]", out: Path, timeout_s: float) -> tuple[list[dict[str, Any]], float | None]:
    """The child's lines until it exits, and the silence limit it overran (None when it exited; else it is still
    running).

    The limit is `timeout_s`; before the first line (imports, `prepare_environment`, the instance) at least
    `STARTUP_TIMEOUT_S`, as for the worker's start; once the resource line is in, at most `EXIT_GRACE_S`.
    """
    lines: list[dict[str, Any]] = []
    offset = 0
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
        *complete, buffer = buffer.split(b"\n")
        for raw in complete:
            with contextlib.suppress(ValueError):
                lines.append(json.loads(raw))
        if complete:
            last = time.monotonic()
        if code is not None:
            return lines, None
        limit = timeout_s if lines else max(timeout_s, STARTUP_TIMEOUT_S)
        if any(line.get("kind") == "resource" for line in lines):
            limit = min(timeout_s, EXIT_GRACE_S)
        if time.monotonic() - last > limit:
            return lines, limit
        time.sleep(_POLL_S)


def _child(
    argv: list[str], env: Mapping[str, str] | None, out: Path, log: Path, timeout_s: float
) -> tuple[list[dict[str, Any]], float | None, int | None, list[Sample]]:
    """Run the child with the sampler on its tree; its lines, the silence limit it overran (None when it exited),
    its exit code and the samples.

    Afterwards its whole process tree is killed (`startup.kill_tree`), whether it answered or not.
    """
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    with log.open("wb") as log_fh:
        proc = subprocess.Popen(
            argv,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            **new_process_group(),
        )
        sampler = TreeSampler(interpreter_pid(proc)).start()
        try:
            lines, overran = _follow(proc, out, timeout_s)
        finally:
            samples = sampler.stop()
            kill_tree(proc, sampler.seen, before)
    return lines, overran, (None if overran is not None else proc.returncode), samples


def _line(lines: list[dict[str, Any]], kind: str, phase: str | None = None) -> dict[str, Any] | None:
    return next((x for x in lines if x.get("kind") == kind and x.get("phase") == phase), None)


def _sample(line: dict[str, Any]) -> Sample:
    return Sample(int(line["t_ns"]), int(line["rss_bytes"]), int(line["processes"]))


def run_resource(
    db: str,
    selection: str,
    dataset: str,
    store: Store,
    scale: str,
    threads: int | None,
    *,
    workdir: Path,
    raw: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float,
) -> Resource:
    """Measure one load in a child: its Memory and Disk record.

    The child runs in the fresh directory `workdir` (removed afterwards) with `env` (default: this process's);
    its lines go to `raw/<load benchmark>-<scale>.jsonl`, its output to the `.log` next to it. A child silent for
    longer than `timeout_s` is killed (`timeout`, "no output for <t> s"; before its first line the limit is at
    least `STARTUP_TIMEOUT_S`, and a child that does not exit once its resource line is in is killed after
    `EXIT_GRACE_S`, its record kept); one that exits without its resource line
    is `failed` ("resources child exited with code <n>: <last log line>"). A load error is `failed`, a count
    mismatch `ok` with answer `disagree`. Only an `ok` record carries bytes.
    """
    rid = catalog.resource_id(dataset, db, scale, threads, store)
    name = f"{catalog.load_benchmark(dataset, store)}-{scale}"
    raw.mkdir(parents=True, exist_ok=True)
    out, log = raw / f"{name}.jsonl", raw / f"{name}.log"
    out.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "graph_bench.runner.resources", "--db", db, "--selection", selection]
    argv += ["--dataset", dataset, "--store", store, "--scale", scale, "--threads", "1" if threads == 1 else "all"]
    argv += ["--workdir", str(workdir), "--out", str(out)]
    fresh_dir(workdir)
    try:
        lines, overran, code, samples = _child(argv, env, out, log, timeout_s)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    record = Resource(rid, db, dataset, store, scale, threads, "failed")
    result = _line(lines, "resource")
    if result is None:
        if overran is not None:
            record.status, record.error = "timeout", f"no output for {overran:g} s"
        else:
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            record.error = f"resources child exited with code {code}" + (f": {tail[-1]}" if tail else "")
        return record
    record.status = cast(Status, result["status"])
    record.error = result.get("error")
    record.answer = result.get("answer", "unchecked")
    record.detail = result.get("detail")
    record.storage = result.get("storage")
    record.disk_bytes = result.get("disk_bytes")
    before, after = _line(lines, "memory", "before"), _line(lines, "memory", "after")
    if record.status == "ok" and before is not None and after is not None:
        first, last = _sample(before), _sample(after)
        w = window(first, last, samples)
        record.baseline_bytes, record.after_bytes = first.rss_bytes, last.rss_bytes
        record.footprint_bytes = last.rss_bytes - first.rss_bytes
        record.peak_bytes, record.samples, record.processes = w.peak_bytes, w.samples, w.processes
        record.sample_interval_ms = SAMPLE_INTERVAL_S * 1000
    return record


if __name__ == "__main__":
    sys.exit(main())
