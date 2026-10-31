# TuringDB: issues to file

Upstream: https://github.com/turing-db/turingdb.
Latest stable on PyPI, checked 2026-10-31: `turingdb` 3.0.2.

One draft.

---

## 1. openCypher `shortestPath((a)-[:R*]-(b))` pattern does not parse

**Status:** draft (feature gap versus openCypher; may be filed as a feature request). **Upstream:** no duplicate found (searched `turing-db/turingdb`: "shortestPath", "OPTIONAL MATCH", "parse error"; the `shortestPath` hits are about TuringDB's own weighted `shortestPath(...)` statement, #617 is a multi-source algorithm request).

### Environment

- `turingdb` 3.0.2 (PyPI, latest stable), embedded client.
- Ubuntu 24.04 x86_64 (Docker), Python 3.12.

### Minimal reproduction

```python
import tempfile
import importlib.metadata as md
from turingdb import TuringDB

db = TuringDB(type="embedded", data_dir=tempfile.mkdtemp())
print("turingdb", md.version("turingdb"))
db.query("CREATE GRAPH g")
db.set_graph("g")
q = """MATCH (p1:Person), (p2:Person)
OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))
RETURN path"""
try:
    print(db.query(q))
except Exception as e:
    print(type(e).__name__ + ":", e)
```

The graph is empty on purpose: the failure is in the parser, before any data is read.

### Expected

openCypher 9 and Neo4j: `shortestPath((a)-[:R*]-(b))` is a pattern function usable in `MATCH`/`OPTIONAL MATCH`, with
undirected relationships and hop counting. LDBC SNB IC13 is written this way. TuringDB's documentation
(https://docs.turingdb.ai/shortest-path) describes a different, weighted `shortestPath(source, target, edgePropName,
dist, path)` statement that follows outgoing edges only, so it cannot answer the hop-count question; the docs point to
variable-length paths, which enumerate every path. The request is a parse of the standard form (unweighted,
undirected), or a clear documented alternative for the unweighted shortest path.

### Actual

```text
turingdb 3.0.2
TuringDBException: PARSE_ERROR: -------* Cypher parser
     1 | MATCH (p1:Person), (p2:Person)
     2 | OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))
       |                       ^^^^^^^^^^^^
-------* syntax error, unexpected SHORTESTPATH, expecting OPAREN
```

### Found by

graph-bench (LDBC SNB IC13, case `turingdb/snb_ic13`, fails on every binding), <results page link>.

---

## Not filed (by design or performance only)

- No native BFS, WCC or PageRank: documented, reported as "not available".
- No value parameters (literals go in the query text): documented design limit, noted as a disadvantage.
- Point lookups, IC2 and IC6 under the 10x budget rule: performance only.
- One engine per process: note only.
