"""The pinned downloads of the JVM databases (versions, URLs, SHA-256), in a module with no driver imports.

`graph-bench data key` hashes this file into the CI cache key, so a changed pin refreshes the cache of
`~/.cache/graph-bench`; the key must not need JPype, which is why the pins are not in `neo4j.py`.
"""

# The latest stable Neo4j release (Maven Central `org.neo4j:neo4j`, `<release>` in maven-metadata.xml), its
# Community tarball and the checksum Neo4j publishes at `<url>.sha256`.
NEO4J_VERSION = "2026.10.2"
NEO4J_HOME_NAME = f"neo4j-community-{NEO4J_VERSION}"
NEO4J_URL = f"https://dist.neo4j.org/{NEO4J_HOME_NAME}-unix.tar.gz"
NEO4J_SHA256 = "fcbccee832175fd8b873ca7d8228dbde4cb5cce43f9053833cc7517dc4dd6f31"
GDS_JAR = f"neo4j-graph-data-science-{NEO4J_VERSION}.jar"
# Neo4j 2026.10 runs on Java 21 or 25 ("Java SE 21 and Java SE 25 Platform Specification",
# https://neo4j.com/docs/operations-manual/current/installation/requirements/); its class files are Java 21 (major
# 65), so the JRE is Temurin 21, pinned to one release so the checksums stay valid (Adoptium's API,
# https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture=<x64|aarch64>&image_type=jre&vendor=eclipse&os=<os>,
# `binary.package.checksum`).
# The Java major Neo4j requires; `graph_bench.pins_update` looks for newer Temurin releases of this major only.
JRE_MAJOR = 21
JRE_RELEASE = "jdk-21.0.13.1+2"
JRE_VERSION = "21.0.13"
_JRE_BASE = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13.1%2B2"
JRES: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): (
        f"{_JRE_BASE}/OpenJDK21U-jre_x64_linux_hotspot_21.0.13.1_2.tar.gz",
        "18f44fab9065217f3ba61d0eef8a7b5f3c3713aa7293001b0d4ac10ede3850f0",
    ),
    ("linux", "aarch64"): (
        f"{_JRE_BASE}/OpenJDK21U-jre_aarch64_linux_hotspot_21.0.13.1_2.tar.gz",
        "52bc58323d9266538a6361a3cf8365fc1186dd9510d7f33f623681c4aabb7e72",
    ),
    ("win32", "amd64"): (
        f"{_JRE_BASE}/OpenJDK21U-jre_x64_windows_hotspot_21.0.13.1_2.zip",
        "e15ee1e54a5a3b38027a09cedfa87bc3348d7dd3ce50866dd3d9671d7fffc6c1",
    ),
}
