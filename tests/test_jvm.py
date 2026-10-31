"""The shared JVM helper: the heap cap, one JVM owner per process, Java versions and verified downloads.

No JVM is started here (jpype's checks are replaced), so the pytest process stays free for any JVM database.
"""

import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from graph_bench.databases import jvm


@pytest.fixture
def fake_jvm(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, object]]:
    """A pretend JVM: `start_once` sees `state["started"]` and `state["heap"]` instead of jpype's."""
    state: dict[str, object] = {"started": False, "heap": jvm.HEAP_FLAG}
    monkeypatch.setattr(jvm, "_OWNER", {})
    monkeypatch.setattr(jvm, "_jvm_started", lambda: bool(state["started"]))
    monkeypatch.setattr(jvm, "effective_heap_flag", lambda: state["heap"])
    yield state


def test_heap_cap_is_one_constant() -> None:
    assert jvm.MAX_HEAP == "4g"
    assert jvm.heap_args() == ["-Xmx4g"] == [jvm.HEAP_FLAG]


def test_jvm_args_cap_threads_only_when_single_threaded(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = ["-XX:ParallelGCThreads=1", "-XX:ConcGCThreads=1", "-XX:CICompilerCount=2"]
    assert jvm.process_env(1) == {"GRAPH_BENCH_JVM_THREADS": "1"}
    assert jvm.process_env(None) == {jvm.THREADS_ENV: ""}
    # An all-cores run overrides an inherited THREADS_ENV=1 (the worker applies process_env over os.environ).
    monkeypatch.setenv(jvm.THREADS_ENV, "1")
    monkeypatch.setenv(jvm.THREADS_ENV, jvm.process_env(None)[jvm.THREADS_ENV])
    assert not jvm.single_threaded()
    monkeypatch.delenv(jvm.THREADS_ENV, raising=False)
    assert not jvm.single_threaded()
    assert jvm.jvm_args() == ["-Xmx4g"]
    monkeypatch.setenv(jvm.THREADS_ENV, "1")
    assert jvm.single_threaded()
    assert jvm.jvm_args() == ["-Xmx4g", *caps]


def test_check_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    good = {"ParallelGCThreads": "1", "ConcGCThreads": "1", "CICompilerCount": "2"}
    monkeypatch.setattr(jvm, "thread_caps", lambda: dict(good))
    jvm.check_threads(1)
    monkeypatch.setattr(jvm, "thread_caps", lambda: {**good, "ParallelGCThreads": "13"})
    jvm.check_threads(None)  # all cores: nothing to check
    with pytest.raises(RuntimeError, match="thread options are"):
        jvm.check_threads(1)


def test_start_once_starts_once_per_owner(fake_jvm: dict[str, object]) -> None:
    calls: list[str] = []

    def start() -> None:
        calls.append("start")
        fake_jvm["started"] = True

    jvm.start_once("arcadedb", start)
    jvm.start_once("arcadedb", start)
    assert calls == ["start"]
    assert jvm.owner() == "arcadedb"
    with pytest.raises(RuntimeError, match="belongs to arcadedb; neo4j needs a process of its own"):
        jvm.start_once("neo4j", start)


def test_start_once_refuses_a_foreign_jvm(fake_jvm: dict[str, object]) -> None:
    fake_jvm["started"] = True
    with pytest.raises(RuntimeError, match="already running"):
        jvm.start_once("arcadedb", lambda: None)
    assert jvm.owner() is None


def test_start_once_checks_thread_caps_when_single_threaded(
    fake_jvm: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[int | None] = []
    monkeypatch.setattr(jvm, "check_threads", checked.append)
    monkeypatch.setenv(jvm.THREADS_ENV, "1")
    jvm.start_once("arcadedb", lambda: fake_jvm.update(started=True))
    assert checked == [1]


def test_start_once_checks_the_heap_cap(fake_jvm: dict[str, object]) -> None:
    def start() -> None:
        fake_jvm["started"] = True
        fake_jvm["heap"] = "-Xmx8g"

    with pytest.raises(RuntimeError, match="runs with -Xmx8g, not the shared cap -Xmx4g"):
        jvm.start_once("arcadedb", start)


def test_start_once_fails_when_the_jvm_did_not_start(fake_jvm: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="did not start"):
        jvm.start_once("arcadedb", lambda: None)
    assert jvm.owner() is None


def test_java_version_reads_the_release_file(tmp_path: Path) -> None:
    (tmp_path / "release").write_text('IMPLEMENTOR="Eclipse Adoptium"\nJAVA_VERSION="21.0.13"\nOTHER="x"\n')
    assert jvm.java_version(tmp_path) == "21.0.13"
    (tmp_path / "release").write_text('IMPLEMENTOR="x"\n')
    with pytest.raises(ValueError, match="no JAVA_VERSION"):
        jvm.java_version(tmp_path)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, Path]:
    """A file to download (a file:// URL), its SHA-256, and the data cache the downloads go to."""
    payload = tmp_path / "src" / "jre-21.tar.gz"
    payload.parent.mkdir()
    payload.write_bytes(b"not really a JRE" * 1000)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(cache_dir))
    monkeypatch.delenv(jvm.DOWNLOADS_ENV, raising=False)
    return payload.as_uri(), hashlib.sha256(payload.read_bytes()).hexdigest(), cache_dir


def test_fetch_downloads_and_verifies(source: tuple[str, str, Path]) -> None:
    url, digest, cache_dir = source
    path = jvm.fetch(url, digest.upper())
    assert path == cache_dir / "downloads" / "jre-21.tar.gz"
    assert jvm.sha256_of(path) == digest
    assert [p.name for p in path.parent.iterdir()] == ["jre-21.tar.gz"]
    assert jvm.fetch(url, digest, "renamed.tgz").name == "renamed.tgz"


def test_fetch_into_the_downloads_override(
    source: tuple[str, str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GRAPH_BENCH_DOWNLOADS_DIR moves the downloads out of the data cache."""
    url, digest, cache_dir = source
    shared = tmp_path / "shared-downloads"
    monkeypatch.setenv(jvm.DOWNLOADS_ENV, str(shared))
    assert jvm.downloads_dir() == shared
    assert jvm.fetch(url, digest) == shared / "jre-21.tar.gz"
    assert not (cache_dir / "downloads").exists()


def test_fetch_reuses_a_verified_file(source: tuple[str, str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    url, digest, _ = source
    first = jvm.fetch(url, digest)

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloaded again")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert jvm.fetch(url, digest) == first


def test_fetch_replaces_a_corrupt_file(source: tuple[str, str, Path]) -> None:
    url, digest, cache_dir = source
    target = cache_dir / "downloads" / "jre-21.tar.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")
    assert jvm.sha256_of(jvm.fetch(url, digest)) == digest


def test_fetch_mismatch_leaves_nothing(source: tuple[str, str, Path]) -> None:
    url, _, cache_dir = source
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        jvm.fetch(url, "0" * 64)
    assert list((cache_dir / "downloads").iterdir()) == []


def test_fetch_rejects_bad_digests_and_names(source: tuple[str, str, Path]) -> None:
    url, digest, _ = source
    with pytest.raises(ValueError, match="not a SHA-256"):
        jvm.fetch(url, "abc")
    for name in ("../x", "a/b", "a\\b", ".."):
        with pytest.raises(ValueError, match="plain file name"):
            jvm.fetch(url, digest, name)
