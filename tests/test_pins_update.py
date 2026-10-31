"""The pins updater, fed recorded vendor responses (no network)."""

import json
from pathlib import Path

import pytest

from graph_bench import pins_update
from graph_bench.databases import pins
from graph_bench.pins_update import JrePin, Neo4jPin

METADATA = """<metadata><groupId>org.neo4j</groupId><artifactId>neo4j</artifactId><versioning>
<latest>2026.11.0-alpha01</latest><release>2026.10.1</release>
<versions><version>2026.9.0</version><version>2026.10.1</version><version>2026.11.0-alpha01</version></versions>
</versioning></metadata>"""
SIDECAR = ("b" * 64) + "  neo4j-community-2026.10.1-unix.tar.gz\n"


def _asset(name: str, sha: str, release: str = "jdk-21.0.14+7", security: int = 14) -> str:
    asset = {
        "release_name": release,
        "version": {"major": 21, "minor": 0, "security": security, "build": 7},
        "binary": {"package": {"name": name, "checksum": sha, "link": "https://example.invalid/" + name}},
    }
    return json.dumps([asset])


def _responses() -> dict[tuple[str, str], str]:
    return {
        ("linux", "x86_64"): _asset("OpenJDK21U-jre_x64_linux_hotspot_21.0.14_7.tar.gz", "c" * 64),
        ("linux", "aarch64"): _asset("OpenJDK21U-jre_aarch64_linux_hotspot_21.0.14_7.tar.gz", "e" * 64),
        ("win32", "amd64"): _asset("OpenJDK21U-jre_x64_windows_hotspot_21.0.14_7.zip", "d" * 64),
    }


def test_neo4j_release_reads_the_stable_release() -> None:
    assert pins_update.neo4j_release(METADATA) == "2026.10.1"


def test_neo4j_release_rejects_pre_releases_and_missing_tags() -> None:
    with pytest.raises(ValueError, match="not a stable"):
        pins_update.neo4j_release("<release>2026.11.0-alpha01</release>")
    with pytest.raises(ValueError, match="no <release>"):
        pins_update.neo4j_release("<metadata/>")


def test_newest_stable_neo4j_skips_pre_releases() -> None:
    assert pins_update.newest_stable_neo4j(METADATA) == "2026.10.1"
    with pytest.raises(ValueError, match="no stable"):
        pins_update.newest_stable_neo4j("<version>1.0-rc1</version>")


def test_checksum_from_sidecar() -> None:
    assert pins_update.checksum_from_sidecar(SIDECAR) == "b" * 64
    assert pins_update.checksum_from_sidecar(("A" * 64) + "\n") == "a" * 64
    with pytest.raises(ValueError, match="not a SHA-256"):
        pins_update.checksum_from_sidecar("<html>not found</html>")
    with pytest.raises(ValueError, match="not a SHA-256"):
        pins_update.checksum_from_sidecar("")


def test_parse_adoptium() -> None:
    jre = pins_update.parse_adoptium(_responses())
    assert jre.release == "jdk-21.0.14+7"
    assert jre.version == "21.0.14"
    assert jre.files[("win32", "amd64")] == ("OpenJDK21U-jre_x64_windows_hotspot_21.0.14_7.zip", "d" * 64)


def test_parse_adoptium_rejects_disagreeing_platforms_and_empty_answers() -> None:
    mixed = _responses()
    mixed[("win32", "amd64")] = _asset("x.zip", "d" * 64, release="jdk-21.0.13+9", security=13)
    with pytest.raises(ValueError, match="different Temurin releases"):
        pins_update.parse_adoptium(mixed)
    with pytest.raises(ValueError, match="lists no JRE"):
        pins_update.parse_adoptium({("linux", "x86_64"): "[]"})


def _current_pins() -> tuple[Neo4jPin, JrePin]:
    files = {platform: (url.rsplit("/", 1)[1], sha) for platform, (url, sha) in pins.JRES.items()}
    return Neo4jPin(pins.NEO4J_VERSION, pins.NEO4J_SHA256), JrePin(pins.JRE_RELEASE, pins.JRE_VERSION, files)


def test_rewrite_with_the_current_pins_changes_nothing() -> None:
    text = pins_update.PINS_FILE.read_text(encoding="utf-8")
    neo4j, jre = _current_pins()
    assert pins_update.rewrite_pins(text, neo4j, jre) == text


def test_rewrite_to_newer_pins_produces_a_valid_module() -> None:
    text = pins_update.PINS_FILE.read_text(encoding="utf-8")
    jre = pins_update.parse_adoptium(_responses())
    new = pins_update.rewrite_pins(text, Neo4jPin("2026.11.0", "b" * 64), jre)
    namespace: dict[str, object] = {}
    exec(compile(new, "pins.py", "exec"), namespace)  # the module text under test
    assert namespace["NEO4J_VERSION"] == "2026.11.0"
    assert namespace["NEO4J_URL"] == "https://dist.neo4j.org/neo4j-community-2026.11.0-unix.tar.gz"
    assert namespace["NEO4J_SHA256"] == "b" * 64
    assert namespace["JRE_RELEASE"] == "jdk-21.0.14+7"
    assert namespace["JRE_MAJOR"] == 21
    base = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.14%2B7/"
    assert namespace["JRES"] == {
        ("linux", "x86_64"): (base + "OpenJDK21U-jre_x64_linux_hotspot_21.0.14_7.tar.gz", "c" * 64),
        ("linux", "aarch64"): (base + "OpenJDK21U-jre_aarch64_linux_hotspot_21.0.14_7.tar.gz", "e" * 64),
        ("win32", "amd64"): (base + "OpenJDK21U-jre_x64_windows_hotspot_21.0.14_7.zip", "d" * 64),
    }


def test_rewrite_fails_loudly_when_pins_py_changes_shape() -> None:
    neo4j, jre = _current_pins()
    with pytest.raises(ValueError, match="no line matching"):
        pins_update.rewrite_pins("NEO4J_VERSION = 1\n", neo4j, jre)


def test_adoptium_url_asks_for_the_jre_of_the_major() -> None:
    url = pins_update.adoptium_url(21, "linux")
    assert "/latest/21/hotspot" in url
    assert "image_type=jre" in url
    assert "os=linux" in url
    assert "architecture=x64" in url
    assert "architecture=aarch64" in pins_update.adoptium_url(21, "linux", "aarch64")


def test_every_pinned_platform_has_an_adoptium_target() -> None:
    assert set(pins.JRES) == set(pins_update.ADOPTIUM_TARGETS)
    assert pins_update.ADOPTIUM_TARGETS[("linux", "aarch64")] == ("linux", "aarch64")
    assert pins.JRES[("linux", "aarch64")][0].endswith("jre_aarch64_linux_hotspot_21.0.13_9.tar.gz")


def test_main_reports_then_writes_then_finds_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "pins.py"
    target.write_text(pins_update.PINS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(pins_update, "PINS_FILE", target)
    newer = (Neo4jPin("2026.11.0", "b" * 64), pins_update.parse_adoptium(_responses()))
    monkeypatch.setattr(pins_update, "latest_pins", lambda: newer)

    assert pins_update.main([]) == 0
    assert "newer pins" in capsys.readouterr().out
    assert pins.NEO4J_VERSION in target.read_text(encoding="utf-8")  # report only

    assert pins_update.main(["--write"]) == 0
    written = target.read_text(encoding="utf-8")
    assert 'NEO4J_VERSION = "2026.11.0"' in written

    # Fed the pins it just wrote, the updater finds nothing to change.
    current = pins_update.rewrite_pins(written, *newer)
    assert current == written
