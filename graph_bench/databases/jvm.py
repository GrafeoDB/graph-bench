"""What the JVM databases (ArcadeDB, Neo4j Community) share: the heap cap, the JVM start and verified downloads.

A Python process hosts at most one JVM, and the harness runs each database in its own worker process (fairness
rule 1), so each JVM database starts the process's JVM once, from its `prepare_environment`, which the worker and
the CodSpeed session call before any timed load: the JVM start is in no clock (rule 2). `start_once` enforces both.

jpype is imported only inside the functions, so this module imports without any JVM package installed.
"""

import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from graph_bench.datasets import cache

# The maximum heap (`-Xmx`) of every JVM database. Without an explicit cap the JVM takes a quarter of the machine's
# RAM, so the cap would differ per machine and per database; with one constant all JVM databases get the same.
# 4 GiB is ArcadeDB's own default (`start_jvm(heap_size="4g")`), so ArcadeDB runs with the heap it ships with, and
# it leaves room on a GitHub-hosted ubuntu-latest runner (16 GB RAM for public repositories) for off-heap memory
# (Neo4j's page cache) and for the harness process.
MAX_HEAP = "4g"
# The heap flag, for a database that passes JVM options itself (Neo4j: `jpype.startJVM(path, *heap_args(), ...)`).
HEAP_FLAG = f"-Xmx{MAX_HEAP}"
# Downloaded files (JREs, jar archives) go into this directory of the data cache, under their own file name.
DOWNLOADS = "downloads"
_CHUNK = 1 << 20
_OWNER: dict[str, str] = {}


def heap_args() -> list[str]:
    """The JVM options of the shared memory cap."""
    return [HEAP_FLAG]


def _jvm_started() -> bool:
    import jpype

    return bool(jpype.isJVMStarted())


def _input_arguments() -> list[str]:
    """The running JVM's command-line options (`RuntimeMXBean.getInputArguments`)."""
    import jpype

    factory = jpype.JClass("java.lang.management.ManagementFactory")
    return [str(arg) for arg in factory.getRuntimeMXBean().getInputArguments()]


def effective_heap_flag() -> str | None:
    """The `-Xmx` option the running JVM uses (the last one given wins), or None when there is none."""
    flags = [arg for arg in _input_arguments() if arg.startswith("-Xmx")]
    return flags[-1] if flags else None


def system_property(name: str) -> str | None:
    """A system property of the running JVM (`System.getProperty`), e.g. a `-D` setting; None when unset."""
    import jpype

    value = jpype.JClass("java.lang.System").getProperty(name)
    return None if value is None else str(value)


def start_once(owner: str, start: Callable[[], None]) -> None:
    """Start the process's JVM by calling `start`, unless `owner` started it already; then check the heap cap.

    Call it from `prepare_environment` (and, for direct use such as tests, lazily from `load`): the JVM start is
    then outside every clock. `start` must pass the cap, `heap_args()` or `MAX_HEAP`. A JVM that another owner, or
    something outside this module, started raises RuntimeError: one JVM database per process (fairness rule 1).
    A JVM whose effective `-Xmx` is not `HEAP_FLAG` raises RuntimeError too, so a JVM_ARGS-style environment
    variable cannot change the cap unnoticed.
    """
    started_by = _OWNER.get("owner")
    if started_by is not None:
        if started_by != owner:
            raise RuntimeError(f"this process's JVM belongs to {started_by}; {owner} needs a process of its own")
        return
    if _jvm_started():
        raise RuntimeError(f"a JVM is already running in this process; {owner} needs a process of its own")
    start()
    if not _jvm_started():
        raise RuntimeError(f"{owner}: the JVM did not start")
    _OWNER["owner"] = owner
    heap = effective_heap_flag()
    if heap != HEAP_FLAG:
        raise RuntimeError(f"{owner}: the JVM runs with {heap or 'no -Xmx'}, not the shared cap {HEAP_FLAG}")


def owner() -> str | None:
    """The database that started this process's JVM, or None."""
    return _OWNER.get("owner")


_JAVA_VERSION = re.compile(r'^JAVA_VERSION="([^"]+)"\s*$', re.MULTILINE)


def java_version(java_home: Path) -> str:
    """The Java version of a JRE or JDK, from the `JAVA_VERSION` line of its `release` file, without starting it.

    Every OpenJDK build (Temurin, Corretto, ...) ships `<java_home>/release` with that line. `info()` uses it: the
    worker reports the database before `prepare_environment` starts the JVM.
    """
    text = (java_home / "release").read_text(encoding="utf-8")
    match = _JAVA_VERSION.search(text)
    if match is None:
        raise ValueError(f"no JAVA_VERSION line in {java_home / 'release'}")
    return match.group(1)


def downloads_dir() -> Path:
    """Where `fetch` keeps its files: `<data cache>/downloads`."""
    return cache.data_dir() / DOWNLOADS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, sha256: str, name: str | None = None, *, timeout_s: float = 60.0) -> Path:
    """The file at `url` in `downloads_dir()/<name>`, its SHA-256 checked; a file already there with it is reused.

    `name` defaults to the last segment of the URL path; it must be a plain file name. A file there with another
    hash is downloaded again. The download goes to a temporary file in the same directory and is renamed only
    after its hash matched, so an interrupted or wrong download never stays under `name`; a mismatch raises
    ValueError. Redirects are followed (Adoptium's API answers with one). Call it from `prepare_environment`,
    outside every clock.
    """
    expected = sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"not a SHA-256 hex digest: {sha256!r}")
    file_name = name or url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    if not file_name or file_name in (".", "..") or "/" in file_name or "\\" in file_name:
        raise ValueError(f"not a plain file name: {file_name!r}")
    target = downloads_dir() / file_name
    if target.is_file():
        if sha256_of(target) == expected:
            return target
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, part_name = tempfile.mkstemp(prefix=f".{file_name}.", suffix=".part", dir=target.parent)
    part = Path(part_name)
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url, timeout=timeout_s) as response:
            shutil.copyfileobj(response, fh, _CHUNK)
        actual = sha256_of(part)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {url}: got {actual}, expected {expected}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target
