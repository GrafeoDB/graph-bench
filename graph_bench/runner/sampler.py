"""Memory and disk readings for the Memory and Disk records (`graph_bench.runner.resources`).

Memory is the resident set size (RSS, `psutil.Process.memory_info().rss`: Linux /proc, Windows
GetProcessMemoryInfo's working set, macOS task_info) summed over a process and all its descendants. psutil is
cross-platform, walks the process tree and reads RSS cheaply enough to sample every `SAMPLE_INTERVAL_S`. RSS counts
shared pages once per process, does not count reserved but untouched memory (a JVM's unused heap), and keeps
counting memory a garbage collector touched until it is returned to the OS. USS would be more exact but reads
smaps, too slow to sample.

The sampler runs in the supervisor, outside the measured process, as a thread (`TreeSampler`). Its sample times are
`time.monotonic_ns()`, the same clock the measured process stamps its `before` and `after` readings with: the
system-wide monotonic clock, so the two compare. It is CLOCK_MONOTONIC on Linux and, on Windows,
QueryPerformanceCounter from Python 3.13; Python 3.12 on Windows uses GetTickCount64, still one system-wide clock
but with about 15.6 ms resolution, so the window's edges blur by more than one sampling interval there.

A daemonized descendant (a server that double-forks, like redislite's redis-server) is re-parented to init and
leaves the tree, unless the measured process is a child subreaper (`set_child_subreaper`, Linux).

Disk is the apparent size (`st_size`) of the regular files under a database's store paths (`dir_size`).
"""

import atexit
import contextlib
import ctypes
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil

__all__ = [
    "SAMPLE_INTERVAL_S",
    "Sample",
    "TreeSampler",
    "dir_size",
    "interpreter_pid",
    "set_child_subreaper",
    "tree_names",
    "tree_rss",
]

# How often the supervisor reads the measured process tree's RSS.
SAMPLE_INTERVAL_S = 0.01

# PR_SET_CHILD_SUBREAPER (man 2 prctl, https://man7.org/linux/man-pages/man2/prctl.2.html): orphaned descendants
# are re-parented to this process instead of init.
_PR_SET_CHILD_SUBREAPER = 36
# How long, at exit, adopted descendants get to end after SIGTERM before they are killed.
_REAP_GRACE_S = 2.0
_SUBREAPER: dict[int, bool] = {}  # pid -> whether it is a subreaper (a forked child is not: the flag is not inherited)


def _tree(pid: int) -> tuple[int, list[psutil.Process]]:
    """The RSS of the tree under `pid` and the processes counted (the root first); `psutil.Error` when the root is
    gone. A descendant that ended or cannot be read meanwhile is skipped."""
    root = psutil.Process(pid)
    total = root.memory_info().rss
    counted = [root]
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # a zombie raises NoSuchProcess too
            continue
        counted.append(child)
    return total, counted


def tree_rss(pid: int) -> tuple[int, int]:
    """The summed RSS, in bytes, of `pid` and every descendant (`children(recursive=True)`), and how many processes
    were counted. A descendant that ended or denies access is skipped; a gone root raises `psutil.NoSuchProcess`."""
    total, counted = _tree(pid)
    return total, len(counted)


def tree_names(pid: int) -> list[str]:
    """The names of `pid` and its live descendants, sorted (e.g. `["python", "redis-server"]`)."""
    names = []
    for process in _tree(pid)[1]:
        with contextlib.suppress(psutil.Error):
            names.append(process.name())
    return sorted(names)


@dataclass(frozen=True)
class Sample:
    """One reading of a process tree: when (`time.monotonic_ns()`, taken right after the reading), its RSS and how
    many processes it counted."""

    t_ns: int
    rss_bytes: int
    processes: int


class TreeSampler:
    """Reads the RSS of the tree under `pid` every `interval_s` in a thread, from `start()` to `stop()`.

    A reading taken while the root is gone is dropped. `seen` holds every process a reading counted, so the caller
    can kill descendants that outlived their parent.
    """

    def __init__(self, pid: int, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.seen: dict[int, psutil.Process] = {}
        self._samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"tree-sampler-{pid}", daemon=True)

    def start(self) -> "TreeSampler":
        self._thread.start()
        return self

    def stop(self) -> list[Sample]:
        """End the sampling; the samples, oldest first."""
        self._stop.set()
        self._thread.join()
        return list(self._samples)

    def _run(self) -> None:
        start = time.monotonic()
        tick = 0
        while not self._stop.is_set():
            self._read()
            tick += 1
            # A steady cadence: the next reading is due `tick` intervals after the first, however long reading took.
            delay = start + tick * self.interval_s - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                return

    def _read(self) -> None:
        try:
            rss, counted = _tree(self.pid)
        except psutil.Error:
            return
        self._samples.append(Sample(time.monotonic_ns(), rss, len(counted)))
        self.seen.update((p.pid, p) for p in counted[1:])


def interpreter_pid(proc: "subprocess.Popen[bytes] | subprocess.Popen[str]", timeout_s: float = 10.0) -> int:
    """The pid of the interpreter running a Python child started with `sys.executable`.

    That is `proc.pid`, except on Windows in a virtual environment: there `sys.executable` is the venv's launcher,
    which starts the base interpreter (`sys._base_executable`) as its only child. The launcher is not the measured
    program, so its child is returned once it appears.
    """
    base = getattr(sys, "_base_executable", sys.executable)
    if sys.platform != "win32" or os.path.normcase(base) == os.path.normcase(sys.executable):
        return proc.pid
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            children = psutil.Process(proc.pid).children()
        except psutil.Error:
            return proc.pid
        if children:
            return children[0].pid
        if proc.poll() is not None:
            return proc.pid
        time.sleep(0.001)
    return proc.pid


def _reap_adopted() -> None:
    """At exit: end the descendants this subreaper still has (adopted orphans among them), and reap them.

    SIGTERM first, so a server can shut down; SIGKILL after `_REAP_GRACE_S`.
    """
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        with contextlib.suppress(psutil.Error):
            child.terminate()
    _, alive = psutil.wait_procs(children, timeout=_REAP_GRACE_S)
    for child in alive:
        with contextlib.suppress(psutil.Error):
            child.kill()
    psutil.wait_procs(alive, timeout=_REAP_GRACE_S)


def set_child_subreaper() -> bool:
    """Make this process a child subreaper (Linux: `prctl(PR_SET_CHILD_SUBREAPER, 1)` through libc, man 2 prctl), once
    per process; whether it is one. False on other platforms.

    A daemonized descendant is then re-parented to this process, so `children(recursive=True)` still finds it. A
    process that set it also ends and reaps the descendants it still has at exit (`atexit`), adopted orphans
    among them, so none outlives it and none stays a zombie.
    """
    pid = os.getpid()
    if pid not in _SUBREAPER:
        ok = False
        if sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                ok = libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
            except (OSError, AttributeError):
                ok = False
        if ok:
            atexit.register(_reap_adopted)
        _SUBREAPER[pid] = ok
    return _SUBREAPER[pid]


def dir_size(paths: Sequence[Path]) -> int:
    """The apparent size (`st_size`), in bytes, of the regular files at or under `paths`; symlinks are neither
    followed nor counted. A missing path raises `FileNotFoundError`."""
    total = 0
    for path in paths:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            for root, _, files in os.walk(path, followlinks=False):
                for name in files:
                    entry = os.lstat(os.path.join(root, name))
                    if stat.S_ISREG(entry.st_mode):
                        total += entry.st_size
    return total
