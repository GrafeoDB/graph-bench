"""The pinned downloads of the JVM databases (versions, URLs, SHA-256), in a module with no driver imports.

`graph-bench data key` hashes this file into the CI cache key, so a changed pin refreshes the cache of
`~/.cache/graph-bench`; the key must not need JPype, which is why the pins are not in `neo4j.py`.
"""

# The latest stable Neo4j release (Maven Central `org.neo4j:neo4j`, `<release>` in maven-metadata.xml), its
# Community tarball and the checksum Neo4j publishes at `<url>.sha256`.
NEO4J_VERSION = "2026.10.1"
NEO4J_HOME_NAME = f"neo4j-community-{NEO4J_VERSION}"
NEO4J_URL = f"https://dist.neo4j.org/{NEO4J_HOME_NAME}-unix.tar.gz"
NEO4J_SHA256 = "afcf4986d47d13dd908427d301d5a9278b2ec7d615a12bb40e22087c7e133e78"
GDS_JAR = f"neo4j-graph-data-science-{NEO4J_VERSION}.jar"
# Neo4j 2026.10 runs on Java 21 or 25 ("Java SE 21 and Java SE 25 Platform Specification",
# https://neo4j.com/docs/operations-manual/current/installation/requirements/); its class files are Java 21 (major
# 65), so the JRE is Temurin 21, pinned to one release so the checksums stay valid (Adoptium's API,
# https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture=x64&image_type=jre&vendor=eclipse&os=<os>,
# `binary.package.checksum`).
# The Java major Neo4j requires; `graph_bench.pins_update` looks for newer Temurin releases of this major only.
JRE_MAJOR = 21
JRE_RELEASE = "jdk-21.0.13+9"
JRE_VERSION = "21.0.13"
_JRE_BASE = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9"
JRES: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): (
        f"{_JRE_BASE}/OpenJDK21U-jre_x64_linux_hotspot_21.0.13_9.tar.gz",
        "a535a1df9c6284d9618872984e026ae0352d1f7e9e0605425c0e6114f47351dd",
    ),
    ("win32", "amd64"): (
        f"{_JRE_BASE}/OpenJDK21U-jre_x64_windows_hotspot_21.0.13_9.zip",
        "2332c0b4cff6bed722da3567c91d5e47568bcba863fa4945d0a8d6fc76158628",
    ),
}
