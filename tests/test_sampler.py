"""The resource sampler: process-tree RSS, the sampling thread, the shared clock, the subreaper and store sizes."""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

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

MIB = 1 << 20

# Holds `MIB_HELD` MiB, touched so it is resident, then says "ready" and sleeps until killed.
_HOLD = """\
import sys, time
ballast = b"\\x01" * ({mib} << 20)
print("ready", flush=True)
time.sleep(60)
"""

# Holds 64 MiB and starts a grandchild that holds 64 MiB; says "ready" once both hold their memory.
_PARENT = """\
import subprocess, sys, time
ballast = b"\\x01" * (64 << 20)
grandchild = subprocess.Popen([sys.executable, "-c", {hold!r}], stdout=subprocess.PIPE, text=True)
assert grandchild.stdout.readline().strip() == "ready"
print("ready", flush=True)
time.sleep(60)
"""


def _start(code: str) -> "subprocess.Popen[str]":
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def _stop(proc: "subprocess.Popen[str]") -> None:
    for child in psutil.Process(proc.pid).children(recursive=True):
        child.kill()
    proc.kill()
    proc.wait()


def test_tree_rss_counts_children() -> None:
    """A child holding 64 MiB with a grandchild holding 64 MiB: the tree is at least 128 MiB above a bare
    interpreter, and it counts the grandchild's processes."""
    bare = _start(_HOLD.format(mib=0))
    tree = _start(_PARENT.format(hold=_HOLD.format(mib=64)))
    try:
        bare_rss, bare_n = tree_rss(interpreter_pid(bare))
        tree_rss_, tree_n = tree_rss(interpreter_pid(tree))
    finally:
        _stop(bare)
        _stop(tree)
    assert tree_rss_ - bare_rss >= 128 * MIB, (tree_rss_, bare_rss)
    assert tree_n >= bare_n + 1


def test_tree_rss_of_a_gone_process_raises() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    with pytest.raises(psutil.NoSuchProcess):
        tree_rss(proc.pid)


def test_tree_names_lists_the_tree() -> None:
    proc = _start(_PARENT.format(hold=_HOLD.format(mib=0)))
    try:
        names = tree_names(interpreter_pid(proc))
    finally:
        _stop(proc)
    assert len(names) >= 2
    assert all(name.lower().startswith("python") for name in names), names


_TRANSIENT = """\
import time
print("ready", flush=True)
time.sleep(0.3)
ballast = b"\\x01" * (200 << 20)
time.sleep(0.3)
del ballast
time.sleep(0.3)
"""


def test_sampler_sees_a_transient_peak() -> None:
    """A child allocates 200 MiB for 300 ms and frees it: the peak is at least 150 MiB above the baseline."""
    proc = _start(_TRANSIENT)
    sampler = TreeSampler(interpreter_pid(proc)).start()
    proc.wait()
    samples = sampler.stop()
    assert all(isinstance(s, Sample) for s in samples)
    assert len(samples) >= 10
    baseline = samples[0].rss_bytes
    peak = max(s.rss_bytes for s in samples)
    assert peak - baseline >= 150 * MIB, (peak, baseline)
    times = [s.t_ns for s in samples]
    assert times == sorted(times)
    assert SAMPLE_INTERVAL_S == 0.01


def test_sampler_of_a_gone_process_has_no_samples() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    sampler = TreeSampler(proc.pid, interval_s=0.001).start()
    time.sleep(0.05)
    assert sampler.stop() == []


def test_monotonic_clock_is_shared_across_processes() -> None:
    """The child's `time.monotonic_ns()` lies between two parent readings around it: the sampler (parent) and the
    resource child's `before`/`after` lines share one clock."""
    for _ in range(3):
        start = time.monotonic_ns()
        proc = subprocess.run(
            [sys.executable, "-c", "import time; print(time.monotonic_ns())"],
            capture_output=True,
            text=True,
            check=True,
        )
        end = time.monotonic_ns()
        assert start <= int(proc.stdout) <= end


_DAEMON = """\
import os, sys, time
import psutil
from graph_bench.runner.sampler import set_child_subreaper, tree_rss

assert set_child_subreaper()
read, write = os.pipe()
middle = os.fork()
if middle == 0:  # the middle process starts the daemon and exits at once, orphaning it
    if os.fork() == 0:
        os.write(write, str(os.getpid()).encode() + b"\\n")
        time.sleep(60)
    os._exit(0)
os.waitpid(middle, 0)
daemon = int(os.fdopen(read).readline())
time.sleep(0.2)
children = [p.pid for p in psutil.Process().children(recursive=True)]
print(daemon, int(daemon in children), tree_rss(os.getpid())[1], flush=True)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_CHILD_SUBREAPER is Linux only")
def test_subreaper_adopts_a_daemonized_grandchild() -> None:
    """A double-forked process is re-parented to the subreaper, so its tree holds it; at exit the subreaper kills
    and reaps it."""
    proc = subprocess.run([sys.executable, "-c", _DAEMON], capture_output=True, text=True, check=True, timeout=60)
    daemon, adopted, processes = (int(x) for x in proc.stdout.split())
    assert adopted == 1, proc.stdout
    assert processes == 2
    assert not psutil.pid_exists(daemon) or psutil.Process(daemon).status() == psutil.STATUS_ZOMBIE


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="the subreaper exists on Linux")
def test_subreaper_is_linux_only() -> None:
    assert set_child_subreaper() is False


def test_dir_size_is_the_apparent_size_of_regular_files(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "sub").mkdir(parents=True)
    (store / "a.bin").write_bytes(b"x" * 1000)
    (store / "sub" / "b.bin").write_bytes(b"y" * 234)
    (store / "empty").mkdir()
    single = tmp_path / "single.wal"
    single.write_bytes(b"z" * 7)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"o" * 5000)
    try:
        os.symlink(outside, store / "link-dir", target_is_directory=True)
        os.symlink(outside / "big.bin", store / "link-file")
    except OSError:
        pass  # no symlink privilege (Windows): the regular files are still counted
    assert dir_size([store]) == 1234
    assert dir_size([store, single]) == 1241
    assert dir_size([]) == 0


def test_dir_size_of_a_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dir_size([tmp_path / "missing"])


def test_interpreter_pid_is_the_process_running_the_code() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import os; print(os.getpid(), flush=True); import time; time.sleep(5)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        own = int(proc.stdout.readline())
        assert interpreter_pid(proc) == own
    finally:
        _stop(proc)
