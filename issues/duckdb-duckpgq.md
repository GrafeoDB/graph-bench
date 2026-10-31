# DuckDB + DuckPGQ: issues to file

Upstream: https://github.com/cwida/duckpgq-extension (the extension; DuckDB itself is not at fault).
Latest stable on PyPI, checked 2026-10-31: `duckdb` 2.0.0; DuckPGQ installed with `INSTALL duckpgq FROM community`.

Two drafts and one comment on an existing issue.

---

## 1. SQL/PGQ label disjunction `(m:Post|Comment)` in GRAPH_TABLE is a parse error

**Status:** draft. **Upstream:** no duplicate found (searched `cwida/duckpgq-extension`: "label expression", "disjunction", "label"; open issues #195 and #330 are about anonymous patterns and `COLUMNS (*)`, not label expressions).

### Environment

- `duckdb` 2.0.0 (PyPI, latest stable), `duckpgq` from the community repository (extension build `9c11bc192a`).
- Windows 11 Pro 10.0.26200, Python 3.12.

### Minimal reproduction

```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL duckpgq FROM community")
con.execute("LOAD duckpgq")
con.execute("SET active_grammar_extensions=['duckpgq']")
con.execute("CREATE TABLE Post(id BIGINT)")
con.execute("CREATE TABLE Comment(id BIGINT)")
con.execute("INSERT INTO Post VALUES (1)")
con.execute("INSERT INTO Comment VALUES (2)")
con.execute("CREATE PROPERTY GRAPH g VERTEX TABLES (Post, Comment)")
print(duckdb.__version__)
print(con.execute("FROM GRAPH_TABLE (g MATCH (m:Post) COLUMNS (m.id))").fetchall())  # works
print(con.execute("FROM GRAPH_TABLE (g MATCH (m:Post|Comment) COLUMNS (m.id))").fetchall())
```

### Expected

ISO/IEC 9075-16 (SQL/PGQ) label expressions allow `|` as disjunction in a node pattern, so `(m:Post|Comment)` matches
vertices from either table: `[(1,), (2,)]` (in some order).

### Actual

```text
2.0.0
[(1,)]
_duckdb.ParserException: Parser Error: syntax error at or near "|"

LINE 1: FROM GRAPH_TABLE (g MATCH (m:Post|Comment) COLUMNS (m.id))
                                         ^
```

Related, seen in the same benchmark but not part of this report: the same disjunction inside a path pattern
(`MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)`) fails the same way, and DuckPGQ requires unique labels,
so one edge label cannot span two edge tables.

### Found by

graph-bench (SNB SQL/PGQ cases IS4 and IS6), <results page link>.

---

## 2. SQL/PGQ IC13 (ANY SHORTEST over undirected KNOWS) does not finish on a 10,000 person graph

**Status:** draft. **Upstream:** no exact duplicate found (searched: "ANY SHORTEST", "shortest path performance", "pagerank"). Related but different: #279 "Performance issue with finbench example" (an `ANY SHORTEST` over a longer pattern, 2 minutes). Mention it in the issue.

### Environment

- `duckdb` 2.0.0, DuckPGQ community build `9c11bc192a`. Windows 11 Pro 10.0.26200, Python 3.12, one run per size.
- Data size in the benchmark (LDBC SNB sample `snb-medium`): 10,000 Person vertices and 180,000 KNOWS edges (41,078 vertices and 376,477 edges over all types). The query prints nothing for the 600 s case timeout. On `snb-small` (1,000 persons, 18,000 KNOWS) it takes 3.5 s for 10 bindings.
- The script below uses a random graph with the same degree (18 KNOWS per person) and no other tables, so it needs no benchmark data.

### Minimal reproduction

```python
import random, sys, time, duckdb

n = int(sys.argv[1])  # persons; 18 KNOWS rows per person, as in the LDBC SNB sample
random.seed(1)
con = duckdb.connect()
con.execute("INSTALL duckpgq FROM community")
con.execute("LOAD duckpgq")
con.execute("SET active_grammar_extensions=['duckpgq']")
con.execute("CREATE TABLE Person(id VARCHAR)")
con.execute("CREATE TABLE KNOWS(src VARCHAR, dst VARCHAR)")
con.executemany("INSERT INTO Person VALUES (?)", [(f"person_{i}",) for i in range(n)])
con.executemany(
    "INSERT INTO KNOWS VALUES (?, ?)",
    [(f"person_{random.randrange(n)}", f"person_{random.randrange(n)}") for _ in range(18 * n)],
)
con.execute("""CREATE PROPERTY GRAPH snb VERTEX TABLES (Person)
  EDGE TABLES (KNOWS SOURCE KEY (src) REFERENCES Person (id) DESTINATION KEY (dst) REFERENCES Person (id))""")
t = time.time()
print(
    n,
    con.execute("""SELECT COALESCE(MIN(len), -1) FROM GRAPH_TABLE (snb
  MATCH path = ANY SHORTEST (p1:Person)-[k:KNOWS]-+(p2:Person)
  WHERE p1.id = 'person_0' AND p2.id = 'person_1'
  COLUMNS (PATH_LENGTH(path) AS len))""").fetchall(),
    f"{time.time() - t:.1f} s",
)
```

Run as `python repro.py 1000`, then 2000, 4000, 10000.

### Expected

One source and one target vertex, answered by a shortest path search; a breadth-first search over 180,000 edges
costs milliseconds. Standard SQL/PGQ `ANY SHORTEST` (ISO/IEC 9075-16) returns one shortest path per pair, and LDBC
SNB IC13 asks for exactly that.

### Actual

```text
1000 [(2,)] 0.3 s
2000 [(3,)] 1.7 s
4000 [(2,)] 15.0 s
10000: no output after 400 s, the process was killed by `timeout 400`
```

Time grows about 6x to 9x per doubling of the graph, so the search appears to explore far more than the frontier of a
single-pair breadth-first search.

### Found by

graph-bench (SQL/PGQ IC13 on `snb-medium`, case `duckdb/snb_sql_ic13`, timeout), <results page link>.

---

## 3. Comment for existing issue #187: PageRank tolerance (and iterations) cannot be set

**Status:** comment-on-existing. **Upstream:** https://github.com/cwida/duckpgq-extension/issues/187 ("Add named parameters for PageRank", open). Not a new issue.

Proposed comment text:

> A use case for this, measured with `duckdb` 2.0.0 and the current DuckPGQ community build. `pagerank(g, V, E)`
> stops at the fixed 1e-6 convergence threshold. LDBC Graphalytics validates PageRank against a converged reference
> with an epsilon of 1e-4 relative, and 1e-6 as a stopping threshold is not enough for that on real data.
>
> ```python
> import duckdb, networkx as nx
>
> n = 200
> edges = [(i, (i * 7 + 3) % n) for i in range(n)] + [(i, (i + 1) % n) for i in range(0, n, 3)]
> con = duckdb.connect()
> con.execute("INSTALL duckpgq FROM community")
> con.execute("LOAD duckpgq")
> con.execute("SET active_grammar_extensions=['duckpgq']")
> con.execute("CREATE TABLE V(id BIGINT)")
> con.execute("CREATE TABLE E(src BIGINT, dst BIGINT)")
> con.executemany("INSERT INTO V VALUES (?)", [(i,) for i in range(n)])
> con.executemany("INSERT INTO E VALUES (?, ?)", edges)
> con.execute("CREATE PROPERTY GRAPH g VERTEX TABLES (V) EDGE TABLES (E SOURCE KEY (src) REFERENCES V (id) DESTINATION KEY (dst) REFERENCES V (id))")
> got = dict(con.execute("FROM pagerank(g, V, E)").fetchall())
> ref = nx.pagerank(nx.DiGraph(edges), alpha=0.85, tol=1e-12, max_iter=10000)  # converged
> rel = max(abs(got[v] - ref[v]) / ref[v] for v in ref)
> print(duckdb.__version__, "max relative error vs converged PageRank:", rel)
> ```
>
> Output: `2.0.0 max relative error vs converged PageRank: 0.0015371805284740733` (max absolute error 2.1e-5, ranks
> sum to 0.9985). On the LDBC Graphalytics sample graph (1,001 vertices) the relative error is 1.0e-4 to 3.1e-4 for
> every vertex. Named parameters for the convergence threshold and a maximum iteration count, as proposed in this
> issue, would fix it. Found by graph-bench, <results page link>.

---

## Not filed (by design or performance only)

- BFS, WCC and SQL/PGQ IS1 slower than the 10x budget rule: answers agree, performance only.
- Unique-label requirement (`REPLY_OF__Comment__Comment`): a documented design limit of DuckPGQ property graphs.
