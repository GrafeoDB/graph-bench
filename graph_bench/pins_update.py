"""Check for newer pinned downloads and rewrite `graph_bench/databases/pins.py` when there are.

Run by the scheduled `pins` workflow: `python -m graph_bench.pins_update --write`. The parsers take the vendors'
responses as text, so the tests feed them recorded responses and never touch the network.

- Neo4j Community: the newest stable release in Maven Central's `maven-metadata.xml` for `org.neo4j:neo4j`
  (`<release>`), the tarball on dist.neo4j.org, and the SHA-256 Neo4j publishes at `<url>.sha256`.
- Temurin JRE: the newest release of `JRE_MAJOR` (the Java major Neo4j requires, kept in pins.py) from Adoptium's
  API, with the `binary.package.checksum` it publishes for each platform.
"""

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from graph_bench.databases import pins

PINS_FILE = Path(__file__).resolve().parent / "databases" / "pins.py"
MAVEN_METADATA_URL = "https://repo1.maven.org/maven2/org/neo4j/neo4j/maven-metadata.xml"
NEO4J_URL_TEMPLATE = "https://dist.neo4j.org/neo4j-community-{version}-unix.tar.gz"
# Adoptium's (os, architecture) names for the platforms in `pins.JRES`.
ADOPTIUM_TARGETS = {
    ("linux", "x86_64"): ("linux", "x64"),
    ("linux", "aarch64"): ("linux", "aarch64"),
    ("win32", "amd64"): ("windows", "x64"),
}
_STABLE = re.compile(r"\d+(\.\d+)+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Neo4jPin:
    version: str
    sha256: str


@dataclass(frozen=True)
class JrePin:
    release: str  # e.g. jdk-21.0.13+9
    version: str  # e.g. 21.0.13
    files: dict[tuple[str, str], tuple[str, str]]  # platform -> (file name, sha256)


def neo4j_release(metadata_xml: str) -> str:
    """The `<release>` of maven-metadata.xml, which must be a stable version (digits and dots only)."""
    match = re.search(r"<release>([^<]+)</release>", metadata_xml)
    if match is None:
        raise ValueError("no <release> in the Maven metadata")
    version = match.group(1).strip()
    if not _STABLE.fullmatch(version):
        raise ValueError(f"the Maven release {version!r} is not a stable version")
    return version


def newest_stable_neo4j(metadata_xml: str) -> str:
    """The highest stable version listed in the metadata, for when `<release>` is a pre-release."""
    versions = re.findall(r"<version>([^<]+)</version>", metadata_xml)
    stable = [v.strip() for v in versions if _STABLE.fullmatch(v.strip())]
    if not stable:
        raise ValueError("no stable version in the Maven metadata")
    return max(stable, key=lambda v: tuple(int(p) for p in v.split(".")))


def checksum_from_sidecar(text: str) -> str:
    """The SHA-256 in a `<url>.sha256` file (`<hash>  <file name>` or just `<hash>`)."""
    token = text.split()[0].lower() if text.split() else ""
    if not _SHA256.fullmatch(token):
        raise ValueError(f"not a SHA-256 checksum: {token!r}")
    return token


def parse_adoptium(responses: dict[tuple[str, str], str]) -> JrePin:
    """Newest Temurin JRE per platform from Adoptium's `assets/latest` responses; all platforms must agree."""
    releases: set[tuple[str, str]] = set()
    files: dict[tuple[str, str], tuple[str, str]] = {}
    for platform, body in responses.items():
        assets = json.loads(body)
        if not assets:
            raise ValueError(f"Adoptium lists no JRE for {platform}")
        asset = assets[0]
        version = asset["version"]
        releases.add((asset["release_name"], f"{version['major']}.{version['minor']}.{version['security']}"))
        package = asset["binary"]["package"]
        files[platform] = (package["name"], checksum_from_sidecar(package["checksum"]))
    if len(releases) != 1:
        raise ValueError(f"the platforms are on different Temurin releases: {sorted(releases)}")
    release, version = releases.pop()
    return JrePin(release, version, files)


def _sub(text: str, pattern: str, replacement: str) -> str:
    result, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"pins.py has no line matching {pattern!r}")
    return result


def rewrite_pins(text: str, neo4j: Neo4jPin, jre: JrePin) -> str:
    """pins.py with the Neo4j and JRE pins replaced (the rest of the file is untouched)."""
    text = _sub(text, r'^NEO4J_VERSION = ".*"$', f'NEO4J_VERSION = "{neo4j.version}"')
    text = _sub(text, r'^NEO4J_SHA256 = ".*"$', f'NEO4J_SHA256 = "{neo4j.sha256}"')
    text = _sub(text, r'^JRE_RELEASE = ".*"$', f'JRE_RELEASE = "{jre.release}"')
    text = _sub(text, r'^JRE_VERSION = ".*"$', f'JRE_VERSION = "{jre.version}"')
    major = jre.version.split(".")[0]
    base = f"https://github.com/adoptium/temurin{major}-binaries/releases/download/{jre.release.replace('+', '%2B')}"
    text = _sub(text, r'^_JRE_BASE = ".*"$', f'_JRE_BASE = "{base}"')
    for platform, (name, sha256) in jre.files.items():
        extension = "zip" if platform[0] == "win32" else "tar.gz"
        key = rf'\(\s*"{platform[0]}", "{platform[1]}"\): \(\s*'
        pattern = rf'({key}f"\{{_JRE_BASE\}}/)[^"]*(",\s*)"[0-9a-f]{{64}}"'
        text, count = re.subn(
            pattern, lambda m, name=name, sha256=sha256: f'{m.group(1)}{name}{m.group(2)}"{sha256}"', text, count=1
        )
        if count != 1:
            raise ValueError(f"pins.py has no JRE entry for {platform}")
        if not name.endswith(extension):
            raise ValueError(f"unexpected archive type for {platform}: {name}")
    return text


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "graph-bench-pins"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def adoptium_url(major: int, os_name: str, architecture: str = "x64") -> str:
    return (
        f"https://api.adoptium.net/v3/assets/latest/{major}/hotspot"
        f"?architecture={architecture}&image_type=jre&vendor=eclipse&os={os_name}"
    )


def latest_pins() -> tuple[Neo4jPin, JrePin]:
    metadata = _fetch(MAVEN_METADATA_URL)
    version = neo4j_release(metadata)
    sha256 = checksum_from_sidecar(_fetch(NEO4J_URL_TEMPLATE.format(version=version) + ".sha256"))
    responses = {
        platform: _fetch(adoptium_url(pins.JRE_MAJOR, os_name, architecture))
        for platform, (os_name, architecture) in ADOPTIUM_TARGETS.items()
    }
    return Neo4jPin(version, sha256), parse_adoptium(responses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update the Neo4j and Temurin pins to the newest stable releases.")
    parser.add_argument("--write", action="store_true", help="rewrite pins.py (default: only report)")
    args = parser.parse_args(argv)
    neo4j, jre = latest_pins()
    old = PINS_FILE.read_text(encoding="utf-8")
    new = rewrite_pins(old, neo4j, jre)
    if new == old:
        print(f"pins are current: Neo4j {neo4j.version}, Temurin {jre.release}")
        return 0
    print(f"newer pins: Neo4j {pins.NEO4J_VERSION} -> {neo4j.version}, Temurin {pins.JRE_RELEASE} -> {jre.release}")
    if args.write:
        PINS_FILE.write_text(new, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
