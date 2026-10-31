# LadybugDB: issues to file

Upstream: https://github.com/LadybugDB/ladybug (Python package `ladybug`).
Latest stable on PyPI, checked 2026-10-31: `ladybug` 0.22.1.

Two drafts and one comment on existing issues.

---

## 1. An OPTIONAL MATCH named path with no match is a partial path, not NULL

**Status:** draft. **Upstream:** no duplicate found (searched `LadybugDB/ladybug`: "OPTIONAL MATCH path", "shortest path null"; the only hit, #1457, is a performance issue about IC13).

### Environment

- `ladybug` 0.22.1 (PyPI, latest stable)
- Windows 11 Pro 10.0.26200, Python 3.12.

### Minimal reproduction

```python
import tempfile, os
import ladybug

db = ladybug.Database(os.path.join(tempfile.mkdtemp(), "db"))
conn = ladybug.Connection(db)
conn.execute("CREATE NODE TABLE Person(id STRING, PRIMARY KEY(id))")
conn.execute("CREATE REL TABLE KNOWS(FROM Person TO Person)")
conn.execute("CREATE (:Person {id: 'a'}), (:Person {id: 'b'}), (:Person {id: 'iso'})")
conn.execute("MATCH (x:Person {id:'a'}), (y:Person {id:'b'}) CREATE (x)-[:KNOWS]->(y)")

q = """MATCH (p1:Person {id: 'a'}), (p2:Person {id: 'iso'})
OPTIONAL MATCH path = (p1)-[:KNOWS* SHORTEST]-(p2)
RETURN path IS NULL AS path_is_null, path, length(path) AS len"""
print(ladybug.__version__)
print(conn.execute(q).get_all())
```

### Expected

`iso` has no edges, so the optional pattern has no match and every variable it introduces is NULL, `path` included
(openCypher 9, "OPTIONAL MATCH": where there is no match for the pattern, its variables are bound to null). So
`path_is_null` is `True`, `path` is NULL and `len` is NULL. The common idiom
`CASE WHEN path IS NULL THEN -1 ELSE length(path) END` (the LDBC SNB IC13 "no path" case) then gives -1.

### Actual

```text
0.22.1
[[False, {'_NODES': [{'_ID': {'offset': 0, 'table': 0}, '_LABEL': 'Person', 'id': 'a'}], '_RELS': []}, None]]
```

`path` is a partial path holding only `p1` with no relationships, `path IS NULL` is false, and `length(path)` is NULL,
so the idiom above falls through to `length(path)` and returns NULL instead of -1. A bounded `[:KNOWS*1..3]` pattern
behaves the same; a relationship variable (`OPTIONAL MATCH (p1)-[k:KNOWS]-(p2) RETURN k IS NULL`) is correctly NULL.

### Found by

graph-bench (LDBC SNB IC13 answer checking, case `ladybug/snb_ic13`), <results page link>.

---

## 2. ALGO extension: PageRank does not redistribute the rank of dangling and isolated vertices

**Status:** draft, held until the ALGO extension loads again (see 3). **Upstream:** no duplicate found (searched: "pagerank", "page_rank dangling"; #675 is an unrelated closed crash).

### Environment

- Reproduced on `ladybug` 0.19.1, the newest release where `LOAD EXTENSION ALGO` still works on Linux.
- Cannot be run on the latest stable, 0.22.1, because the extension does not load there (item 3). Retest on 0.22.1 once it loads, and drop this item if it is fixed.
- Ubuntu 24.04 (Docker), Python 3.12.

### Minimal reproduction

```python
import tempfile, os
import ladybug

conn = ladybug.Connection(ladybug.Database(os.path.join(tempfile.mkdtemp(), "db")))
conn.execute("INSTALL ALGO")
conn.execute("LOAD EXTENSION ALGO")
conn.execute("CREATE NODE TABLE V(id INT64, PRIMARY KEY(id))")
conn.execute("CREATE REL TABLE E(FROM V TO V)")
for i in range(4):
    conn.execute(f"CREATE (:V {{id: {i}}})")
for a, b in [(0, 1), (1, 0), (1, 2), (2, 1)]:
    conn.execute(f"MATCH (x:V {{id:{a}}}), (y:V {{id:{b}}}) CREATE (x)-[:E]->(y)")
conn.execute("CALL project_graph('g', ['V'], ['E'])")
rows = conn.execute(
    "CALL page_rank('g', dampingFactor := 0.85, maxIterations := 1000, tolerance := 1e-12) RETURN node.id, rank ORDER BY node.id"
).get_all()
print(ladybug.__version__, rows, "sum =", sum(r[1] for r in rows))
```

Vertex 3 is isolated (no in or out edges).

### Expected

LDBC Graphalytics specification, PageRank: the rank of dangling vertices (no out-edges) is summed and spread evenly
over all vertices each iteration, so the ranks always sum to 1. NetworkX `pagerank` implements the same. For this graph:
`{0: 0.24453, 1: 0.46332, 2: 0.24453, 3: 0.04762}`, sum 1.0.

### Actual

```text
0.19.1 [[0, 0.1925675675676779], [1, 0.3648648648646442], [2, 0.1925675675676779], [3, 0.037500000000000006]] sum = 0.7875
```

Vertex 3 gets exactly (1 - d) / N = 0.0375 and the ranks sum to 0.7875: the dangling mass is dropped instead of
redistributed. A real dataset with isolated vertices therefore gets ranks that are too small and do not match the
Graphalytics reference.

### Found by

graph-bench (LDBC Graphalytics PageRank answer checking), <results page link>.

---

## 3. Comment for existing issues #857 and #1190: ALGO extension fails to load on Linux and Windows (0.20.4 to 0.22.1)

**Status:** comment-on-existing. **Upstream:** https://github.com/LadybugDB/ladybug/issues/857 (algo extension is missing icebug symbols) and https://github.com/LadybugDB/ladybug/issues/1190 (macOS arm64: libnetworkit, libarrow and libomp are not shipped), both open. Not a new issue.

Proposed comment text (post on #1190, mention #857):

> Same root cause on the other platforms, measured with `ladybug` 0.22.1 from PyPI. `INSTALL ALGO` succeeds, then
> `LOAD EXTENSION ALGO` fails.
>
> Linux x86_64 (Ubuntu 24.04, Python 3.12):
>
> ```text
> RuntimeError: IO exception: Failed to load library: /root/.lbdb/extension/0.22.0/linux_amd64/algo/libalgo.lbug_extension which is needed by extension: algo.
> Error: libnetworkit.so: cannot open shared object file: No such file or directory.
> ```
>
> Windows x64 (Windows 11, Python 3.12):
>
> ```text
> RuntimeError: IO exception: Failed to load library: C:\Users\<user>/.lbdb/extension/0.22.0/win_amd64/algo/libalgo.lbug_extension which is needed by extension: algo.
> Error: The specified module could not be found.
> The extension itself was found, so a library it depends on is the one that could not be located. Extensions link against OpenSSL 3 (libssl-3-x64.dll and libcrypto-3-x64.dll), which is not distributed with them.
> ```
>
> Reproduction (3 lines): `conn = ladybug.Connection(ladybug.Database(path)); conn.execute("INSTALL ALGO"); conn.execute("LOAD EXTENSION ALGO")`.
>
> Bisect on Linux: 0.17.1, 0.18.3 and 0.19.1 load the extension; 0.20.4, 0.21.2 and 0.22.1 do not. So the Linux and
> Windows wheels regressed the same way the macOS arm64 one did in 0.20.4. The old `real_ladybug` 0.15.x package can
> no longer fetch the extension either (`extension.ladybugdb.com` serves only v0.17.0 and later, so `INSTALL ALGO`
> returns HTTP 404). Found by graph-bench, <results page link>.

Verification note: both error messages above were reproduced today on 0.22.1, and 0.19.1 was re-confirmed to load on
Linux. The rest of the version bisect (0.17.1, 0.18.3, 0.20.4, 0.21.2) and the `real_ladybug` 404 come from earlier
graph-bench runs (docs/known-disagreements.md) and were not re-run today.

---

## Not filed (by design or performance only)

- Point lookups IS1, IS2, IS5, IS6 slower than the 10x budget rule: agree with the reference, performance only.
- WCC `maxIterations` default 100: documented (the adapter raises the bound to 1000).
