# NeuG

## `MATCH (n) RETURN count(n)` returns no row on a database with no node tables

**Environment:** `neug` 0.3.0 (latest stable on PyPI, 2026-10-31), Windows 11, Python 3.12; embedded,
`neug.Database("")` (in-memory), `max_thread_num=1`.

**Minimal reproduction:**

```python
import neug

db = neug.Database("", max_thread_num=1)
conn = db.connect()
print("no tables:", [tuple(r) for r in conn.execute("MATCH (n) RETURN count(n)")])
list(conn.execute("CREATE NODE TABLE Person(id STRING, PRIMARY KEY(id))"))
print("one empty table:", [tuple(r) for r in conn.execute("MATCH (n) RETURN count(n)")])
conn.close()
db.close()
```

**Expected:** one row `(0,)` in both cases. An aggregation without grouping keys over an empty match returns exactly
one row (openCypher 9 / GQL: `count` over no rows is 0, and an aggregation with no grouping keys always yields one
row), as it already does once any node table exists.

**Actual:**

```text
no tables: []
one empty table: [(0,)]
```

With no node tables the query returns no row at all, so a client that reads `rows[0][0]` fails on a fresh database.

**Found by:** graph-bench's Startup row (an empty database's probe), see `<results page link>`.

**Upstream:** no duplicate found (searched alibaba/neug issues for "aggregation empty", "count no rows").

## Not filed (by design or performance only)

- PageRank is slow under the 10x budget rule. `page_rank` has no tolerance option, so it likely runs a fixed number of
  rounds; the answer agrees. Performance only.
- Load is slow (a checkpoint per COPY statement). Counts and queries agree. Performance only.
