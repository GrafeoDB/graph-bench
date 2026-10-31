# FalkorDB: issues to file

Upstream: https://github.com/FalkorDB/FalkorDB (engine) and https://github.com/FalkorDB/falkordblite (the embedded Python package).
Latest stable on PyPI, checked 2026-10-31: `falkordblite` 0.11.0 (FalkorDB 4.22.1).

Two drafts.

---

## 1. shortestPath is only allowed in WITH or RETURN, and not for undirected patterns, so OPTIONAL MATCH shortestPath fails

**Status:** draft (feature gap versus openCypher; may be filed as a feature request). **Upstream:** no duplicate found (searched `FalkorDB/FalkorDB`: "shortestPath", "shortestPath OPTIONAL MATCH", "undirected shortestPath"; the hits are unrelated: #3003 is a Rust-port parser hang, #346 and #140 are usage questions).

### Environment

- `falkordblite` 0.11.0 (FalkorDB 4.22.1), PyPI, latest stable.
- Ubuntu 24.04 x86_64 (Docker, glibc 2.39), Python 3.12.

### Minimal reproduction

```python
import importlib.metadata as md
from redislite.falkordb_client import FalkorDB

db = FalkorDB(None, serverconfig={"save": ""})
g = db.select_graph("g")
g.query("CREATE (:P {id:'a'})-[:KNOWS]->(:P {id:'b'})")
print("falkordblite", md.version("falkordblite"))
q_ok = "MATCH (p1:P {id:'a'}), (p2:P {id:'b'}) RETURN length(shortestPath((p1)-[:KNOWS*]-(p2)))"
q = """MATCH (p1:P {id:'a'}), (p2:P {id:'b'})
OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))
RETURN CASE WHEN path IS NULL THEN -1 ELSE length(path) END"""
q_with = """MATCH (p1:P {id:'a'}), (p2:P {id:'b'})
WITH p1, p2, shortestPath((p1)-[:KNOWS*]-(p2)) AS path
RETURN length(path)"""
for name, text in [
    ("directed, in RETURN", q_ok.replace("]-(p2)", "]->(p2)")),
    ("OPTIONAL MATCH", q),
    ("undirected, in WITH", q_with),
]:
    try:
        print(name, "->", g.query(text).result_set)
    except Exception as e:
        print(name, "->", type(e).__name__ + ":", e)
```

### Expected

openCypher (and Neo4j, whose syntax FalkorDB follows) accept `shortestPath((a)-[:R*]-(b))` as a pattern in
`MATCH` and `OPTIONAL MATCH`, and for undirected relationships. LDBC SNB IC13 is written exactly like that
(`OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))`, then `-1` when `path` is NULL). FalkorDB's function
reference (https://docs.falkordb.com/cypher/functions.html, "shortestPath") documents the function but the
restrictions below make it unusable for the unreachable-pair case.

Expected output for the three queries: `[[1]]`, `[[1]]`, `[[1]]`.

### Actual

```text
falkordblite 0.11.0
directed, in RETURN -> [[1]]
OPTIONAL MATCH -> ResponseError: FalkorDB currently only supports shortestPaths in WITH or RETURN clauses
undirected, in WITH -> ResponseError: FalkorDB does not currently support undirected shortestPath traversals
```

So an undirected shortest path (needed when the relationship is stored once per pair) cannot be expressed at all, and
the call cannot be placed in an `OPTIONAL MATCH` to get a NULL path for an unreachable pair.

### Found by

graph-bench (LDBC SNB IC13, case `falkordb/snb_ic13`, fails on every binding), <results page link>.

---

## 2. falkordblite has no wheel for glibc older than 2.38 (Ubuntu 22.04 arm64 and x86_64)

**Status:** draft, packaging request. **Upstream:** no duplicate found (searched `FalkorDB/falkordblite`: "manylinux", "glibc", "arm64"; #25 "The name of the whl files should be fixed" is about naming, not the glibc floor, and is worth linking).

### Environment

- `falkordblite` 0.11.0 (PyPI, latest stable).
- Ubuntu 22.04 aarch64 (Docker `--platform linux/arm64`, glibc 2.35), Python 3.12.16 (uv).

### Minimal reproduction

```bash
docker run --rm --platform linux/arm64 ubuntu:22.04 bash -c '
  ldd --version | head -1
  apt-get update -qq && apt-get install -y -qq curl ca-certificates
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 /v
  uv pip install --python /v/bin/python --only-binary :all: falkordblite'
```

### Expected

A wheel that installs on a mainstream LTS Linux. Ubuntu 22.04 (glibc 2.35) is supported by its vendor until 2027
and is the image of common CI runners (for example CodSpeed's Graviton arm64 runner, which is Ubuntu 22.04). The
PyPI file list for 0.11.0 shows Linux wheels only as `manylinux_2_38_x86_64` and `manylinux_2_38_aarch64`, so
nothing below glibc 2.38 can install them. A `manylinux_2_28` build (or 2_34/2_35) would cover Ubuntu 22.04,
Debian 12 and RHEL 9.

### Actual

```text
aarch64
ldd (Ubuntu GLIBC 2.35-0ubuntu3.15) 2.35
Using Python 3.12.16 environment at: /v
error: No solution found when resolving dependencies
  cause: Because all versions of falkordblite have no usable wheels and you require falkordblite, we can conclude that your requirements are unsatisfiable.
```

Without `--only-binary`, uv falls back to the sdist, which needs a C compiler (the build failed on the clean image
with `error: [Errno 2] No such file or directory: 'gcc'`); that route was not pursued further.

### Found by

graph-bench (the Graviton CI runner cannot install FalkorDB; every FalkorDB case is reported as "not installed"
there), <results page link>.

---

## Not filed (by design or performance only)

- PageRank damping and tolerance are fixed (documented; the answer agrees with the reference).
- THREAD_COUNT and OMP_THREAD_COUNT are module load arguments that falkordblite does not expose: note only.
- Slow point lookups, IC6 and BFS under the 10x budget rule: performance only.
