"""Startup children that hang after the engine is up, for the kill-tree tests of FalkorDB and TuringDB (Linux).

Registered through `GRAPH_BENCH_EXTRA_DATABASES` as `<db>_hang=tests.startup_hangs:<Class>`. A hanging class runs
the real adapter's `open_empty` (FalkorDB: the redis-server; TuringDB: the in-process engine and a graph), writes
its pid to the file `MARKER_ENV` names, then sleeps until the parent's timeout kills it. The classes are built on
first access (`__getattr__`), so this module imports without either driver installed.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from graph_bench.runner.startup import Child, run_child

MARKER_ENV = "GRAPH_BENCH_TEST_HANG_MARKER"
# Linux kernel threads are children of kthreadd (pid 2), or pid 0 for kthreadd itself.
KERNEL_THREAD_PARENTS = frozenset({0, 2})
_ADAPTERS = {
    "HangingFalkor": ("graph_bench.databases.falkordb", "FalkorDatabase", "falkordb_hang"),
    "HangingTuring": ("graph_bench.databases.turingdb", "TuringDatabase", "turingdb_hang"),
}


def __getattr__(name: str) -> Any:
    if name not in _ADAPTERS:
        raise AttributeError(name)
    import importlib

    module, base_name, db_name = _ADAPTERS[name]
    base = getattr(importlib.import_module(module), base_name)

    def open_empty(self: Any) -> None:
        base.open_empty(self)
        marker = Path(os.environ[MARKER_ENV])
        # Atomic: the reader polls for the file and must never see it empty or half written.
        partial = marker.with_name(marker.name + ".tmp")
        partial.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(partial, marker)
        time.sleep(600)

    return type(name, (base,), {"name": db_name, "open_empty": open_empty})


def processes_named(name: str) -> set[int]:
    """The pids of the live (not zombie) processes called `name`."""
    pids = set()
    for proc in psutil.process_iter(["name", "status"]):
        if proc.info["name"] == name and proc.info["status"] != psutil.STATUS_ZOMBIE:
            pids.add(proc.pid)
    return pids


def alive(pids: set[int]) -> set[int]:
    """The pids of `pids` still running (a zombie counts as gone)."""
    return {pid for pid in pids if pid in processes_running()}


def processes_running() -> set[int]:
    return {p.pid for p in psutil.process_iter(["status"]) if p.info["status"] != psutil.STATUS_ZOMBIE}


def hang_until_killed(db: str, workdir: Path, marker: Path, timeout_s: float) -> tuple[Child, set[int], list[str]]:
    """Run the hanging startup child `db` (its `MARKER_ENV` set to `marker` by the caller) with `run_child`.

    Once the marker shows the engine is up, the child's process tree is listed, plus every process started since
    (a daemonized redis-server leaves the tree). Returns the child's result, those pids and their names.
    """
    before = processes_running()
    result: list[Child] = []
    thread = threading.Thread(target=lambda: result.append(run_child(db, 1, workdir, timeout_s)))
    thread.start()
    deadline = time.monotonic() + timeout_s
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(marker.read_text(encoding="utf-8")) if marker.exists() else None
    tree: dict[int, str] = {}
    if pid is not None:
        root = psutil.Process(pid)
        for proc in [root, *root.children(recursive=True)]:
            tree[proc.pid] = proc.name()
        for new in processes_running() - before:
            try:
                proc = psutil.Process(new)
                if proc.ppid() in KERNEL_THREAD_PARENTS:  # a kworker the kernel started meanwhile, not ours
                    continue
                tree[new] = proc.name()
            except psutil.Error:
                continue
    thread.join()
    return result[0], set(tree), sorted(tree.values())
