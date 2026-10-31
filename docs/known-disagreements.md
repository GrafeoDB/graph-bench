# Known disagreements

Cases where a database gives a wrong answer, runs far too long, crashes or lacks a feature, and the outcome is
left as it is in the results instead of being worked around in the adapter or the query text. Each section names
the case, the data it was seen on, what the database did, what the reference expects and where the cause lies
(the database engine or our adapter).

The data is what `graph-bench data prepare --selection push` generates (`snb-small` and `graphalytics-small`).
The `- id: <database>/<benchmark>` line of each section is read by the adapter tests as an allowlist: those
entries may fail without failing the test.

## Grafeo: IC13 shortest path does not finish (OPTIONAL MATCH shortestPath)

- id: grafeo/snb_ic13
- Query: `snb_ic13` (IC13 shortest path), the shared openCypher text, no Grafeo override.
- Dataset: `snb-small`.
- Binding: `{"person1Id": "person_0", "person2Id": "person_549"}` (the first of 10; none of them finishes).
- Grafeo 0.6.0 answer: none. The first binding runs for more than 60 s (more than 120 s in the `bench` run)
  and the worker is stopped, so the entry is `timeout` (⏱).
- Reference answer: `[[1]]` (BFS over KNOWS in both directions); the first 8 bindings' answers are 1 or 2,
  binding 8 (`person_0` and the isolated `person_1000`) is -1 and binding 9 (`person_549` with itself) is 0.
- Cause: Grafeo engine (query planner). `explain_cypher` shows the `OPTIONAL MATCH` becoming a `LeftJoin` whose
  right side is `ShortestPath (p1 -> p2)` over `NodeScan (p1:*)` and `NodeScan (p2:*)`, with the `id` filter on
  `p2` applied after it. `p1` and `p2` are not taken from the left side, so it computes shortest paths for all
  node pairs of the graph and filters afterwards. Plan excerpt from a reduced reproduction (parameters x/y,
  simplified RETURN):

  ```text
  Return (length(path) AS l)
    LeftJoin
      Filter (p2.id Eq Parameter("y"))
        NodeScan (p2:Person)
          Filter (p1.id Eq Parameter("x")) [label-first]
            NodeScan (p1:Person)
      Filter (p2.id Eq Parameter("y"))
        ShortestPath (p1 -> p2)
          NodeScan (p2:*)
            NodeScan (p1:*)
  ```

  The same pattern with a plain `MATCH` instead of `OPTIONAL MATCH` plans `ShortestPath` on top of the bound
  `p1` and `p2` and answers all 10 bindings correctly in about 1 ms each, so the slowness is in planning the
  optional form, not in the shortest path itself. A bound `[:KNOWS*..30]` does not help. The query keeps
  `OPTIONAL MATCH` because it must return -1 for unreachable pairs.

## Grafeo: slow but correct (not allowlisted)

These cases agree with the reference and are timed as they are; they are listed for the Grafeo team because
the plans show avoidable work. They carry no `- id:` line, so the adapter test still requires them to agree.

- `snb_is7` on `snb-small` (median 497 ms for 10 bindings, about 50 ms each): the same uncorrelated `LeftJoin` as
  IC13. The `OPTIONAL MATCH (author)-[k:KNOWS]-(p)` side is planned as `Expand (author)--[:KNOWS]--(p)` over
  `NodeScan (author:*)`, so every KNOWS pair of the graph is expanded and then joined, instead of probing the
  bound `author` and `p`. Cause: Grafeo engine (query planner).
- `snb_is4` on `snb-small` (median 45 ms for 10 bindings): `MATCH (m {id: $messageId}) WHERE m:Post OR m:Comment`
  is planned as `NodeScan (m:*)`, a label filter, then the `id` filter, so the property index on `id` is not
  used when the label is a disjunction or absent. Cause: Grafeo engine (query planner).
- `scan_order_limit` and `scan_filter_avg` on `snb-small` (push run medians 3.35 ms and 2.15 ms for 7 and 10
  bindings, against NeuG's 0.38 ms and 0.27 ms; within the 10x rule). `scan_order_limit` plans `Limit (20)` over a
  full `Sort (created DESC, id ASC)` over `Return (m.id, m.content, m.creationDate)`, so every matching comment's
  content is projected and the whole set sorted, where a top-20 sort would keep 20 rows. Cause: Grafeo engine
  (query planner, no top-k for `ORDER BY ... LIMIT`).

## Grafeo: medium-scale RDF and IC13 (not allowlisted)

From the `full` selection on Grafeo release/0.6.1 builds (64a679ed and 9164e307), answers agree, ⏱ by the budget:

- RDF/SPARQL IS1, IS4, IS6 against Oxigraph in the same run (9164e307, medians for 10 bindings): `snb-small`
  101 / 40 / 549 ms against 3.4 / 1.6 / 2.5 ms; `snb-medium` 1,700 / 673 / 8,221 ms against 3.4 / 1.7 / 3.2 ms.
  Per query, IS1 costs about 8.6 ms on small and 74 ms on medium, so it grows with the data: the subject is bound
  only through the literal object of `v:id` from the inline `VALUES`, which Grafeo does not yet turn into a
  per-row lookup. The first query after a load also pays a statistics collection (967 ms on medium for the
  reachability ASK). Reported to grafeo-31 with the exact texts. Cause: Grafeo engine (SPARQL planner).
- Cypher `snb_ic13` on `snb-medium`: 111 ms for 10 bindings on 9164e307 (188 ms on 64a679ed), against 3.3 to 15 ms
  for the three fastest (ArcadeDB, Neo4j, NeuG). Reported; grafeo-28 profiles it on the medium dataset.

## Grafeo: LOAD CSV is 2x to 3x slower than its batch API (not allowlisted)

Load uses the batch API, Grafeo's fastest documented bulk path; LOAD CSV will be tracked as its own benchmark,
Load (query language). The LOAD CSV load is correct (every node and edge count matches the CSVs, and all checked
queries agree afterwards), but slower than the batch API. It is not a disagreement, so there is no `- id:` line.

- Benchmarks: `load_snb`, `load_graphalytics` (the worker's load clock around `GrafeoDatabase.load()`), measured
  with a LOAD CSV version of the adapter (Task 7b, round 0).
- Dataset: `snb-small` (9 node tables, 4,177 nodes; 19 edge tables, 37,860 edges) and `graphalytics-small`
  (1,000 vertices, 36,000 edges).
- Statements (one per table, documented in grafeo `docs/user-guide/cypher/mutations.md`, "LOAD CSV"), e.g. for
  `graphalytics-small`:

  ```cypher
  LOAD CSV WITH HEADERS FROM '<data>/graphalytics-small/V.csv' AS row CREATE (:V {id: toInteger(row.id)})
  // then db.create_property_index("id")
  LOAD CSV WITH HEADERS FROM '<data>/graphalytics-small/E.csv' AS row
  MATCH (a:V {id: toInteger(row.src)}), (b:V {id: toInteger(row.dst)}) CREATE (a)-[:E]->(b)
  ```

- Grafeo 0.6.0 outcome with LOAD CSV: load medians in `bench --selection push --db grafeo,ladybug` of 269 ms
  (`load_snb`) and 202 ms (`load_graphalytics`).
- Expected: at most the batch API path, which parses the CSVs in Python and calls
  `batch_create_nodes_with_props` and `batch_create_edges`: 125 ms and 68 ms, measured with the Task 7/8 adapter
  (the same batch path the adapter uses now). LadybugDB's `COPY FROM` took 177 ms and 55 ms in the LOAD CSV run.
- Where the time goes (one thread, in process): the node statements are fast (`V.csv` 2 ms; all 9 SNB node
  tables 18 ms together). The edge statements take 5 to 7 us per row: `E.csv` about 200 ms for 36,000 rows,
  `KNOWS__Person__Person.csv` 133 ms for 18,000 rows, all 19 SNB edge tables 241 ms together. Reading the rows
  alone (`LOAD CSV ... AS row RETURN count(row)` on `E.csv`) takes 22 ms, and the same `MATCH` plus `CREATE`
  fed from a list parameter (`UNWIND $pairs AS p MATCH (a:V {id: p[0]}), (b:V {id: p[1]}) CREATE (a)-[:E]->(b)`)
  takes about 300 ms, so the CSV reader is not the cost. `EXPLAIN` shows both endpoint lookups use the `id`
  index (`Filter (a.id Eq toInteger(row.src)) [index: id]` over `NodeScan (a:V)`); matching one endpoint costs
  about 85 ms, both 158 ms, and the `CREATE` about 45 ms more. Without the index the edge statement takes 12.9 s.
- Cause: Grafeo engine, per-row cost of an indexed `MATCH` by property plus `CREATE` in a Cypher pipeline
  (about 2 us per index probe), compared with `batch_create_edges` on internal ids. Grafeo has no documented
  bulk path for edges keyed by an external id that avoids the per-row probe.

## Grafeo: SQL/PGQ label disjunction and ANY SHORTEST do not parse yet (IS4, IS6, IC13)

- id: grafeo/snb_sql_is4
- id: grafeo/snb_sql_is6
- id: grafeo/snb_sql_ic13
- Queries: the SQL/PGQ texts in `graph_bench/suites/snb_sql.py`, standard SQL/PGQ (ISO/IEC 9075-16), one text for
  every engine, no Grafeo override (a weaker Grafeo-only text is not allowed). Filters are the graph pattern WHERE
  after the pattern (ruling R6), which Grafeo accepts: `snb_sql_is1` agrees (median 0.226 ms for 10 bindings).
- Dataset: `snb-small`, loaded as the property graph `snb` (Grafeo: a named graph, `create_graph("snb")` and
  `set_graph("snb")`, so `GRAPH_TABLE (snb MATCH ...)` resolves it).
- Grafeo 0.6.0 (local build of release/0.6.1, `D:\gb-grafeo-0.6.1` at 5048055b) outcome: these three fail at parse
  time (✗), on the first binding:
  - `snb_sql_is4`: `GRAFEO-Q001: syntax error: [SQL/PGQ] Expected RParen, found Pipe` at `MATCH (m:Post|Comment)`.
  - `snb_sql_is6`: `Expected LParen, found Identifier` at `MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)`.
  - `snb_sql_ic13`: `Expected LParen, found Identifier` at `MATCH path = ANY SHORTEST (p1:Person)-[k:KNOWS]-+(p2:Person)`.
- Reference answers: the openCypher SNB references (`snb_is4`, `snb_is6`, `snb_ic13`), the same bindings and rows.
- Cause: Grafeo engine (SQL/PGQ parser), confirmed by grafeo-31 on release/0.6.1. IS4: the label disjunction
  `Post|Comment` (grafeo-31: "SQL/PGQ node pattern reads only :A:B"). IS6 and IC13: the `ANY SHORTEST` path
  search prefix (with it, quantified edges inside `GRAPH_TABLE`). For IS6 this is the first failure: the label
  disjunction `Post|Comment` in the same text could fail next. The decision on the target release is the
  user's (decisions.md, 6).

## Grafeo: an SQL/PGQ parameter named `$from` does not parse (Property filter)

- id: grafeo/scan_filter_sql
- Query: `scan_filter_sql` (Property filter (SQL/PGQ)), the shared SQL/PGQ text in `graph_bench/suites/scans.py`,
  no Grafeo override: `MATCH (m:Comment) WHERE m.creationDate >= $from AND m.creationDate < $to`, the same
  parameter names as the openCypher `scan_filter`, whose bindings and references it shares.
- Dataset: `snb-small`, the named graph `snb`.
- Grafeo 0.6.0 outcome: ✗ at parse time on the first binding, `GRAFEO-Q001: syntax error: [SQL/PGQ] Expected
  identifier` pointing at `$from`. The openCypher `scan_filter` with the same `$from` agrees, and DuckDB + DuckPGQ
  runs the same SQL/PGQ text and agrees.
- Reduced: on a one-node graph, `$x`, `$to` and `$min` parse in a graph pattern WHERE; `$from` and `$select` fail
  with the same error, so a parameter whose name is an SQL reserved word is not accepted after `$`. `$name` is the
  binding convention both engines' Python APIs take (see `snb_sql.py`), and a name after `$` is a parameter name,
  not a keyword.
- Reference answer: the openCypher `scan_filter` references (10 windows of about 200 comments each).
- Cause: Grafeo engine (SQL/PGQ parser: the parameter name after `$` is lexed as a keyword). Not renamed in the
  shared text, which would hide it.

## Grafeo: SPARQL IS6 property path is slow (not allowlisted)

- `snb_rdf_is6` on `snb-small` agrees (10/10) with a median of 451 ms for 10 bindings (about 45 ms each), against
  32 to 50 ms for `snb_rdf_is1` and `snb_rdf_is4`. The text starts the group with `VALUES ?messageId { ... }`, then
  `?m v:id ?messageId ; v:REPLY_OF* ?p`. research-b measured the same text at 0.6 ms per binding on Oxigraph and
  found Grafeo's `(v:KNOWS|^v:KNOWS)+` reachability 6.2 s against 12 ms. Cause: Grafeo engine (SPARQL property
  path evaluation). It carries no `- id:` line: the case agrees.
- Load (`load_snb_rdf`): 507 ms for the 62,521 triples of the N-Triples file written at prepare time, sent as
  `INSERT DATA` statements of 10,000 lines: the documented path for bindings without an RDF file loader. This is
  transitional: the adapter calls `db.load_rdf(path, format="ntriples")` (grafeo release/0.6.1, commit 61c4388d)
  as soon as the installed grafeo has it, so the RDF load switches to the file loader with a Grafeo 0.6.1 build,
  and the fallback is then removed.

## LadybugDB: the ALGO extension does not load (PageRank, WCC)

- id: ladybug/graphalytics_pr
- id: ladybug/graphalytics_wcc
- Benchmarks: `graphalytics_pr` (`page_rank`) and `graphalytics_wcc` (`weakly_connected_components`), the two
  that need LadybugDB's ALGO extension. BFS is a Cypher `SHORTEST` pattern and does not need it.
- Dataset: `graphalytics-small`; the dataset does not matter, the extension fails before any data is touched.
- LadybugDB 0.22.1 (`ladybug` on PyPI) answer: none. `INSTALL ALGO` downloads the extension, then
  `LOAD EXTENSION ALGO` fails, so both entries are `failed` (✗) with LadybugDB's error:
  - Linux (`linux_amd64`, the CI runners): `Failed to load library: ~/.lbdb/extension/0.22.0/linux_amd64/algo/
    libalgo.lbug_extension which is needed by extension: algo. Error: libnetworkit.so: cannot open shared object
    file: No such file or directory.`
  - Windows (`win_amd64`): the same `Failed to load library`, with "Extensions link against OpenSSL 3
    (libssl-3-x64.dll and libcrypto-3-x64.dll), which is not distributed with them."
- Reference answer: the NetworkX PageRank and WCC of the graph; nothing is compared.
- Versions: `ladybug` 0.17.1, 0.18.3 and 0.19.1 load the extension on Linux; 0.20.4, 0.21.2 and 0.22.1 do not.
  The older `real_ladybug` package (0.15.x) is no longer usable: its extensions were removed from
  `extension.ladybugdb.com`, which serves only `v0.17.0` and later, so `INSTALL ALGO` returns HTTP 404.
- Cause: LadybugDB packaging. The extension links NetworKit (and on macOS Arrow and OpenMP) without shipping
  them. Upstream: https://github.com/LadybugDB/ladybug/issues/857 and
  https://github.com/LadybugDB/ladybug/issues/1190, both open. The adapter does not install NetworKit itself:
  LadybugDB documents no such step. It loads the extension in the first algorithm call (untimed preflight), so the
  load and BFS are still measured; once the extension loads, these entries are measured and checked as usual.

## LadybugDB: IC13 returns NULL for an unreachable pair (OPTIONAL MATCH path)

- id: ladybug/snb_ic13
- Query: `snb_ic13` (IC13 shortest path), LadybugDB override (`(p1)-[:KNOWS* SHORTEST]-(p2)` in place of
  `shortestPath(...)`, the rest as in the shared text):

  ```cypher
  MATCH (p1:Person {id: $person1Id}), (p2:Person {id: $person2Id})
  OPTIONAL MATCH path = (p1)-[:KNOWS* SHORTEST]-(p2)
  RETURN CASE WHEN p1.id = p2.id THEN 0 WHEN path IS NULL THEN -1 ELSE length(path) END AS shortestPathLength
  ```

- Dataset: `snb-small`, which has one isolated person (`person_1000`, no edges at all) so that IC13 has a pair
  without a path.
- Binding: `{"person1Id": "person_0", "person2Id": "person_1000"}` (binding 8 of 10). The other 9 agree,
  including the pair at distance 0 (`person_549` with itself).
- LadybugDB 0.22.1 answer: `[[null]]`.
- Reference answer: `[[-1]]`, the LDBC IC13 definition for "no path" (and openCypher: a variable of an
  `OPTIONAL MATCH` without a match is null).
- Cause: LadybugDB engine. When the optional pattern has no match, the named path is not null: `RETURN path`
  gives a partial path holding only `p1` (`{'_NODES': [<person_0>], '_RELS': []}`), `path IS NULL` is false and
  `length(path)` is null, so the `CASE` falls through to `length(path)`. The same happens with a bounded
  `[:KNOWS*1..3]` pattern. A relationship variable is null as expected
  (`OPTIONAL MATCH (p1)-[k:KNOWS]-(p2) RETURN k IS NULL` gives true). The query is left as it is.

## LadybugDB: slow point lookups under the 10x budget rule (not allowlisted)

These cases agree with the reference in every run. In a `bench` run next to Grafeo they are marked `timeout`
(⏱) by the budget rule (slower than 10x the mean of the 3 fastest other databases), not by a hang. They carry
no `- id:` line because the adapter test runs LadybugDB alone, where no budget applies and they must agree.

- Queries: `snb_is1`, `snb_is2`, `snb_is5`, `snb_is6` (`snb_is6` with its `ladybug` override).
- Dataset: `snb-small`.
- Binding: all 10 of each.
- LadybugDB 0.15.3 answer: agrees. Medians for 10 bindings, `bench --selection push --db grafeo,ladybug`:
  IS1 9.7 ms, IS2 43.0 ms, IS5 9.2 ms, IS6 39.7 ms (19.8x, 37.8x, 28.0x and 61.1x Grafeo's 0.49, 1.14, 0.33
  and 0.65 ms).
- Reference answer: matched.
- Cause: LadybugDB engine, per-call overhead. `QueryResult.get_compiling_time()` shows every `execute()`
  compiling the query again (IS1 about 0.4 ms compile and 0.2 to 0.5 ms execution per binding, IS2 about 1.3 ms
  compile and 2 ms execution, IS6 about 1.5 ms compile and 0.8 ms execution), so a point lookup costs about
  1 ms where Grafeo answers in about 0.05 ms. A separate `prepare()` plus `execute()` halves IS1 but emits
  "The use of separate prepare + execute of queries is deprecated. Please using a single call to the execute()
  API instead.", so the adapter keeps the documented single `execute()` call.

## LadybugDB: latent (not hit on the push data, not allowlisted)

These agree on the push data and carry no `- id:` line; they are listed because larger or sparser datasets
would likely show them.

- PageRank and dangling or isolated vertices (`graphalytics_pr`). Dataset: a 4-vertex probe (edges 0-1 and 1-2
  in both directions, vertex 3 isolated), loaded and run through the adapter. LadybugDB 0.15.3 gives vertex 3
  (1-d)/N = 0.0375 and the ranks sum to 0.79: the rank of a vertex without out-edges is not redistributed. The
  reference (NetworkX, as LDBC Graphalytics specifies) gives 0.0476 and sums to 1; Grafeo agrees with it.
  `graphalytics-small` now has one isolated vertex (1000, from the isolated SNB person that IC13 needs), so once
  the ALGO extension loads, `graphalytics_pr` is expected to disagree there; it is allowlisted today for the
  extension failure above. Cause: LadybugDB engine (ALGO `page_rank`).
- WCC iteration bound (`graphalytics_wcc`). `weakly_connected_components` stops after `maxIterations`, default
  100 (https://docs.ladybugdb.com/extensions/algo/wcc), so a component whose diameter needs more rounds would be
  split. The adapter passes `maxIterations := 1000`, the bound it also gives PageRank; a component needing more
  than 1000 rounds would still be split. Cause: LadybugDB engine default; the adapter only raises the bound.

## NeuG: PageRank is slow under the 10x budget rule (not allowlisted)

The answer agrees with the reference in every run. In a `bench` run next to Grafeo it is marked `timeout` (⏱) by
the budget rule (slower than 10x the mean of the 3 fastest other databases), not by a hang. It carries no `- id:`
line because the adapter test runs NeuG alone, where no budget applies and it must agree.

- Query: `graphalytics_pr` (`page_rank` of the GDS extension, https://neug.io/docs/extensions/load_gds/).
- Dataset: `graphalytics-small`.
- NeuG 0.3.0 answer: agrees. Median 25 to 59 ms over repeated `bench --selection push --db grafeo,ladybug,neug`
  runs (about 28x to 53x the top-3 average), against Grafeo's 1.1 ms.
- Cause: the adapter passes `max_iterations: 1000`, and `page_rank` has no tolerance option, so (as far as the
  timing shows) it likely runs a fixed number of rounds instead of stopping at convergence. The default is 20
  rounds, which is about 1e-5 off the converged ranks and so is not used: PageRank is compared at convergence.
  The adapter does not lower the bound to win the budget.

## NeuG: load is slow (not allowlisted)

Node and edge counts match and every checked query agrees, but the load clock is long: about 2 to 4 s for
`graphalytics-small` and 4 to 6 s for `snb-small` (Windows, Python 3.12), against Grafeo's 0.07 s and 0.13 s. It is
not a disagreement, so there is no `- id:` line.

- Statements: one explicit-schema `CREATE NODE TABLE` / `CREATE REL TABLE` per table, then `COPY ... (header=true,
  delim=",")` per CSV (https://neug.io/docs/data_io/import_data/). Each COPY commits a private copy-on-write graph
  with a checkpoint (visible in the engine log), and each in-memory database also keeps a temp directory.
  Wrapping the COPY statements in one transaction did not make the load measurably faster.
- Cause: NeuG engine (checkpoint per statement), not measured further.

## NeuG: count over an empty database returns no row (Startup)

- id: neug/startup
- Benchmark: `startup` (the Startup row's probe `MATCH (n) RETURN count(n) AS nodes`, `suites/startup.py`), on
  the empty database `open_empty` opens: `neug.Database("", max_thread_num=...)` and `connect()`, no table yet.
- NeuG 0.3.0 answer: no row (`[]`), at the first child, so the entry is ≠ with no timing (the first child's time
  is in `preflight_ns`). `MATCH (n) RETURN count(*)` gives no row as well, while `RETURN 1` gives `[(1,)]`.
- Reference answer: `[[0]]`. In openCypher an aggregation without grouping keys returns exactly one row, also over
  an empty match; the other seven property-graph databases answer `[[0]]` on their empty databases, LadybugDB (the
  other Kuzu-derived engine) included.
- Once any node table exists (`CREATE NODE TABLE P(id INT64, PRIMARY KEY(id))`, still without data), NeuG answers
  `[(0,)]`, so the loaded cases are not affected. The probe is not changed for NeuG (a case that cannot run the
  shared text keeps its marker).
- Cause: NeuG engine (planner: a pattern over no node table yields no aggregate row). Reproduction:

  ```python
  import neug

  db = neug.Database("", max_thread_num=1)
  conn = db.connect()
  print(list(conn.execute("MATCH (n) RETURN count(n)")))  # [], expected [[0]]
  ```

## FalkorDB: no install on glibc older than 2.38 (the CI runner)

- Package: falkordblite 0.11.0 (FalkorDB 4.22.1). Its PyPI wheels are `manylinux_2_38_x86_64` and
  `manylinux_2_38_aarch64` (plus macOS), so pip and uv refuse them on glibc older than 2.38.
- Where it shows: the CI runner `codspeed-macro-arm64-graviton-ubuntu-22-04` is Ubuntu 22.04 with glibc 2.35, so
  `install-databases.sh` logs the install failure and FalkorDB reports ✗ "not installed" for every case there. It
  runs on newer Linux (the full local run, ci.yml on ubuntu-latest).
- Cause: FalkorDB packaging (wheel platform tag); not worked around in the adapter. A build for glibc 2.35 or
  older would remove the gap.

## FalkorDB: IC13 does not run (shortestPath is limited to WITH or RETURN, and to directed)

- id: falkordb/snb_ic13
- Query: `snb_ic13` (IC13 shortest path), the shared openCypher text, no FalkorDB override.
- Dataset: `snb-small`.
- FalkorDB 4.22.1 (falkordblite 0.11.0) answer: none. Every binding fails with `ResponseError: FalkorDB currently
  only supports shortestPaths in WITH or RETURN clauses`, so the entry is `failed` (✗). The query is
  `OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))`, undirected, as the standard text needs: moving the
  call into `WITH` instead fails with "FalkorDB does not currently support undirected shortestPath traversals".
- Reference answer: `[[1]]` for the first binding; -1 for an unreachable pair and 0 for the same person.
- Cause: FalkorDB engine (openCypher coverage). KNOWS is stored once per pair, so a directed form is not the same
  question. No rewrite is used: a database that cannot run the standard text gets the marker (FAIRNESS.md).

## FalkorDB: PageRank parameters are fixed (answer agrees)

`algo.pageRank(label, relationship)` takes no damping, iteration or tolerance argument and its docs list none
(https://docs.falkordb.com/algorithms/pagerank.html), so the adapter passes only `'V'` and `'E'`. The answer is
within `LDBC_EPS` (1e-4) of the reference on `graphalytics-small`; if a larger graph leaves the epsilon the entry
becomes ≠ and belongs here. No `- id:` line: it agrees today.

## FalkorDB: threads are not configurable (note)

`THREAD_COUNT` and `OMP_THREAD_COUNT` are module load arguments only (https://docs.falkordb.com/configuration.html)
and falkordblite starts the module with none, so the server uses the core count, while every other database runs
single-threaded where it can. The results record `threads_configurable=False`. The server is also a separate
process (`local process`), so its memory is not under the harness's per-process caps.

## FalkorDB: Startup starts a redis-server per child (note)

The Startup time covers falkordblite starting its `redis-server` (with `save ""`, as in a load) and selecting the
graph; the probe on the not yet existing graph answers `[[0]]`. That redis-server daemonizes and leaves the child's
process tree: a child run by hand leaves it running after the child exits (re-parented to init), so the harness
kills it with the child (`kill_tree`, which finds it through the Linux child subreaper), whether the child
answered or timed out. `test_falkordb_startup_leaves_no_redis_server` and
`test_falkordb_startup_timeout_kills_the_redis_server` check both on Linux.

The Memory record needs the same tree: the redis-server is a child of the measuring process only through the Linux
child subreaper, and holds the whole graph, so without it the load would read as a few MB.
`test_falkordb_redis_server_in_tree` checks that the after-load tree holds it. The store is in memory (`save ""`),
so there is no Disk size.

## FalkorDB: slow point lookups and IC6 under the 10x budget rule (not allowlisted)

The answers agree with the references. In a `bench --selection push --db grafeo,falkordb` run on Linux (Docker,
ubuntu:24.04) these are marked `timeout` (⏱) by the budget rule (slower than 10x the mean of the 3 fastest other
databases), not by a hang. They carry no `- id:` line because the adapter test runs FalkorDB alone, where no
budget applies and they must agree.

- Queries and medians (10 bindings): `snb_is1` 4.4 ms (Grafeo 0.20 ms), `snb_is2` 5.7 ms (0.47 ms), `snb_is5`
  7.2 ms (0.16 ms), `snb_is6` 56 ms (0.24 ms), `snb_ic6` 290 ms (25 ms). Every FalkorDB call is a Redis
  round trip over a unix socket with the query parsed and planned by the module, which sets a floor of a few
  milliseconds per call.
- Cause: FalkorDB engine and the out-of-process server (per-call overhead); `snb_is6` and `snb_ic6` also include
  variable-length traversals. The adapter uses the `id` indexes it creates in the load and no query cache.

## FalkorDB: BFS is slow under the 10x budget rule (not allowlisted)

The answer agrees with the reference. BFS levels come from the documented `shortestPath` function, one search per
target vertex (`MATCH (s:V {id: $s}), (b:V) WITH b, shortestPath((s)-[:E*0..]->(b)) AS p ...`,
https://docs.falkordb.com/cypher/functions.html), because `algo.BFS` returns no levels. On `graphalytics-small`
(Linux, Docker) the median is 128.6 ms against Grafeo's 0.42 ms, so it is marked `timeout` (⏱, 302.8x the top-3
average) by the budget rule, not by a hang. No `- id:` line: the adapter test runs FalkorDB alone and it must agree.

## TuringDB: IC13 does not parse (shortestPath is a different, weighted function)

- id: turingdb/snb_ic13
- Query: `snb_ic13` (IC13 shortest path), the shared openCypher text, no TuringDB override.
- Dataset: `snb-small`.
- TuringDB 3.0.2 answer: none. Every binding fails with `PARSE_ERROR ... syntax error, unexpected SHORTESTPATH,
  expecting OPAREN` at `OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))`, so the entry is `failed` (✗).
- Reference answer: `[[1]]` for the first binding; -1 for an unreachable pair and 0 for the same person.
- Cause: TuringDB engine (openCypher coverage). Its own `shortestPath(source, target, edgePropName, dist, path)` is
  a statement after `MATCH`, weighted (Dijkstra on a numeric edge property) and follows outgoing edges only
  (https://docs.turingdb.ai/shortest-path), so it cannot give the undirected hop count. The docs point to
  variable-length paths for hop questions, which "enumerate every path"; no bounded rewrite is used, because a
  database that cannot run the standard text gets the marker (FAIRNESS.md).

## TuringDB: no native graph algorithms (BFS, WCC, PageRank)

- id: turingdb/graphalytics_bfs
- id: turingdb/graphalytics_wcc
- id: turingdb/graphalytics_pr
- All three are `not_supported` (⊘). `CALL db.procedures()` lists only `db.*` and `gnn.*` procedures
  (https://docs.turingdb.ai/query/procedures); `pageRank`, `algo.pageRank` and `gds.pageRank` do not exist.
- BFS levels are not computed with a query: TuringDB documents no shortest-path form giving hop distances from a
  source (`shortestPath` is weighted and directed, see above), and a variable-length `MATCH` enumerates every
  path. LadybugDB and FalkorDB compute BFS through their documented hop-count shortest-path forms; TuringDB has none.

## TuringDB: no value parameters, so each call carries its literals (disadvantage)

TuringDB's parser rejects `$name` in value positions (`Not implemented: Parameters`); parameters work only for
labels and edge types (https://docs.turingdb.ai/query/cypher_subset). Its documented way is literals in the query
text, so the adapter inlines every parameter through one quoting function (`turingdb.literal`: single-quoted
strings with `\` and `'` escaped, ints and floats as numbers, anything else rejected; unit-tested with quotes,
backslashes and unicode). This is not a workaround but the only documented path. The cost is TuringDB's: every
binding is a different text, so the engine cannot reuse a parsed query, where the other databases bind real
parameters. Measured on `snb-small` IS1 (Linux, Docker): 2.5 ms per call with 50 different person ids against
3.1 ms per call with one id repeated, so the lack of plan reuse is not what dominates a call; a call costs a few
milliseconds either way.

## TuringDB: point lookups, IC2 and IC6 under the 10x budget rule (not allowlisted)

The answers agree with the references. In a `bench --selection push --db grafeo,turingdb` run on Linux (Docker,
ubuntu:24.04, PyPI Grafeo 0.6.0) these are marked `timeout` (⏱) by the budget rule (slower than 10x the mean of
the 3 fastest other databases), not by a hang. They carry no `- id:` line because the adapter test runs TuringDB
alone, where no budget applies and they must agree.

- Queries and medians (10 bindings): `snb_is1` 40 ms (Grafeo 0.42 ms), `snb_is2` 99 ms (0.86), `snb_is3` 57 ms
  (2.3), `snb_is5` 33 ms (0.28), `snb_is6` 52 ms (0.39), `snb_ic2` 52 ms (4.1). `snb_is4` 25 ms, `snb_is7`
  88 ms and `snb_ic6` 151 ms stay inside the budget.
- Cause: TuringDB engine, per-call cost of a few milliseconds (the query goes through the pandas-returning
  client); not measured further.
- `snb_is7` agrees on this data: the `IS NOT NULL` over an `OPTIONAL MATCH` variable that failed with a
  `ColumnMask` error on a tiny graph (research) ran correctly on `snb-small`, so the shared text is used as is.

## TuringDB: one engine per process, threads and storage (notes)

- The embedded engine starts a process-wide event handler; a second `TuringDB(type="embedded")` in one process
  fails with `Could not initialize system event handler`, also after the first client is deleted (the client has
  no close method). The worker loads several times per process, so the adapter starts the engine once per process in
  `prepare_environment` (outside every load clock; `data_dir` is a fresh temporary directory removed at exit) and
  gives each load a graph of its own (uuid4 name). TuringDB has no `DROP GRAPH`, so loaded graphs stay in that
  directory until the process exits.
- Startup: a startup child has no engine yet, so its time covers the engine start (in a fresh temporary `data_dir`)
  and a new empty graph made with `CREATE GRAPH` and `set_graph`
  (https://docs.turingdb.ai/graph_dev/create_graph). The engine runs in the child's process ("no daemon"), so it
  ends with the child; a child killed at its timeout leaves its temporary `data_dir` behind (its exit hook does not
  run).
- There is no in-memory mode: the graph is on disk (`storage="on-disk"`), and no thread setting is exposed
  (`threads_configurable=False`).
- Disk: the graph's own directory `<data_dir>/graphs/<graph>` (`store_paths`), measured after `close()`. A listing
  after a load shows its commit log, commits and data parts. The engine's own `graphs/default`, its lock and socket
  files, and the Parquet staging under `data/<graph>` (removed by `close`) are not counted. The files are sized in
  whole MiB blocks, so the apparent size is the engine's allocation, not the bytes of data.
- The load clock covers the CSV to Parquet conversion with pyarrow and
  `LOAD PARQUET` (https://docs.turingdb.ai/import_data/parquet). TuringDB's load needs this conversion because it
  has no CSV bulk path for more than a few thousand edges (https://docs.turingdb.ai/import_data/csv).

## HelixDB: IS3, IS7 and IC6 cannot be expressed in the DSL (not available)

- id: helixdb/snb_is3
- id: helixdb/snb_is7
- id: helixdb/snb_ic6
- HelixDB has no query text in its Python SDK (the openCypher form of the server and cloud API has no method in
  helix-db 0.3.4 and no variable-length paths or `shortestPath`), so each SNB query is a builder written with the
  documented DSL (`graph_bench/databases/helix_queries.py`) that must compute what the shared openCypher text
  computes. A query the DSL cannot express is `not_supported` (⊘) with the reason below and is not run in a weaker
  form; nothing is computed in Python to make up for it.
- `snb_is3`: `ORDER BY friendshipCreationDate DESC, friendId ASC`. `order_by` / `order_by_multiple` sort a stream
  by properties of its own elements. The edges carry `creationDate`, the second key is the `id` of the node at the
  far end of each edge, and `bind` blocks sorting (below), so the two keys cannot be applied together.
- `snb_is7`: `ORDER BY commentCreationDate DESC, replyAuthorId ASC`, the same limit: the comments carry the first
  key and the second is the `id` of each comment's author.
- `snb_ic6`: the ranking `ORDER BY postCount DESC, otherTagName ASC LIMIT 10` over tag groups. `group_count` is
  a terminal step that returns a list of `{name, count}` objects in name order; `order_by` and `fold` reject it
  ("order expected stream input") and `limit` after it keeps the first names, not the largest counts.
- Planner rules that shaped the builders (helix-db-embedded 0.3.3): `order_by`, `repeat` and `value_map` are
  rejected after `bind` ("planner error: order_by is not supported after bind"). IS2 and IC2 therefore sort and limit
  first and bind afterwards; IS2 follows the reply chain with 50 `optional` steps (the data's longest chain is 7)
  instead of `repeat`.
- Cause: HelixDB DSL (feature coverage), not the adapter. The other seven SNB queries agree with the references.

## HelixDB: the Aggregation scan cannot be expressed in the DSL (not available)

- id: helixdb/scan_aggregate
- Query: `scan_aggregate` (Aggregation): Persons grouped by `browserUsed` and `gender`, with `count(*)`,
  `min(birthday)` and `max(creationDate)` per group, ordered by the two keys.
- HelixDB outcome: `not_supported` (⊘), "Aggregation: grouping by two properties (browser and gender) with a count,
  a minimum and a maximum per group cannot be expressed". `group` and `group_count` take one property and are
  terminal steps, and `aggregate_by(fn, property)` computes one function over the whole stream (helix-db 0.3.4,
  https://docs.helix-db.com/database/helix-db/query-guides/projections). Running one entry per known group would need
  the groups in advance, and grouping in Python would compute the answer outside HelixDB; neither is done.
- The other four scans agree. `scan_filter_avg` is two entries of one read batch, `count` and
  `aggregate_by(MEAN, "length")`, each over the same filtered stream, since both steps are terminal: two scans
  where the shared text has one (a disadvantage, not a workaround).
- Cause: HelixDB DSL (feature coverage), not the adapter.

## HelixDB: no BFS, WCC or PageRank (not available)

- id: helixdb/graphalytics_bfs
- id: helixdb/graphalytics_wcc
- id: helixdb/graphalytics_pr
- All three are `not_supported` (⊘). The query language has no WCC or PageRank, and no BFS levels (`repeat` yields
  paths, not distances). The Python SDK also has a `NativeGraph` (`Client.graph(GraphSelection(...))`, then
  `traverse`, `shortest_path`, `louvain_communities`); in a scratch test it gave BFS levels equal to the reference
  (research-b), but the docs mention only `Client.graph(...)` in one sentence
  (https://docs.helix-db.com/database/helix-db/start-here/sdk-setup/python-project-setup) and never describe the
  snapshot or its traversal, so it is not a documented path and the adapter does not use it.
- Cause: HelixDB (no documented algorithm API).

## HelixDB: loading goes through write_batch chunks (disadvantage)

- HelixDB has no bulk loader (no CSV import): nodes and edges are `add_n` / `add_e` entries of `write_batch`
  requests (https://docs.helix-db.com/database/helix-db/query-guides/writing-data), 500 entries per request (the
  docs give no size; a sweep on `snb-small` took 10.2 s at 100, 8.2 s at 500 and 7.8 s at 2000). The internal node
  ids the batches return are mapped to the CSV ids during the load, and a unique equality index on `id` per label is
  created and awaited before the data (https://docs.helix-db.com/database/helix-db/query-guides/secondary-indexes);
  all of it is in the load clock. Cause: HelixDB (no bulk path).
- No thread setting is documented for the embedded runtime (`threads_configurable=False`); storage is
  `InMemory` (`in-memory`).
- IC13 uses `shortest_path(..., max_depth=100, direction=BOTH)`: the shared text is unbounded and the DSL requires
  a bound. The hop count is the number of nodes of the returned path minus one (`length(path)`), -1 for an empty
  path; HelixDB computes the path, the adapter counts it. IS2's reply chain is capped at 50 steps and IS4 to IS6 run
  one source entry per label (Post, Comment) because a source predicate has no label disjunction.

## HelixDB: slow under the 10x budget rule (not allowlisted)

The answers agree with the references. In `bench --selection push --db grafeo,helixdb` (Windows) every query HelixDB runs is
marked `timeout` (⏱) by the budget rule against Grafeo, not by a hang: 10 bindings take 26 to 39 ms for IS1, IS4,
IS5 and IS6 (Grafeo 0.2 to 0.3 ms), 99 ms for IC13, 201 ms for IC2 and 239 ms for IS2 (the 50 `optional` steps).
Load takes 8.3 s (Graphalytics) and 11.6 s (SNB) against 0.09 s and 0.14 s. No `- id:` line: the adapter test runs
HelixDB alone and it must agree.

The Scans it runs agree as well and are ⏱ by the same rule in `bench --selection push` (Windows, eight databases):
`scan_label` 49 ms (32x the top-3 average), `scan_filter` 520 ms for 10 bindings (207x), `scan_order_limit` 543 ms
for 7 bindings (249x) and `scan_filter_avg` 1051 ms for 10 bindings (924x, two filtered scans per binding, see the
Aggregation section). Every filtered scan reads the label's nodes and filters them in the DSL; no range index is
created on `creationDate` or `length`, as no other database gets one.

## HelixDB: intermittent planner error during loads on a busy machine (✗ when it happens, not allowlisted)

- Error: `HelixError: embedded HelixDB error: message='Query error: planner error: unsupported cascades plan:
  selected optimizer result did not contain a best physical alternative'`, raised by a `write_batch` during
  `load()`, so the load and its cases are ✗ with this text.
- Seen in both `full` runs that shared the machine with a second full run: Windows (the `snb-medium` load) and
  Linux in Docker (both Graphalytics loads and the `snb-medium` load), helix-db 0.3.4 and helix-db-embedded 0.3.3.
- Never seen otherwise. HelixDB alone through the worker (Windows and Linux, all four loads). Ten repeated
  Graphalytics loads, under 16 busy CPU loops and with `PYTHONHASHSEED` 0 to 15. A clean sequential `full` run
  with eight databases on Windows: every load ok.
- The batches are the same in every run, so the failure depends on timing or resources, not on the input. Cause:
  HelixDB engine (the cascades optimizer gives up on a write batch under load). Not worked around: a failing load
  stays ✗.

## ArcadeDB: the Graph Importer does not honour CSV quotes (why GraphBatch loads)

- ArcadeDB's Graph Importer (https://docs.arcadedb.com/arcadedb/how-to/migration/graph-importer) is its fastest
  graph loader (research-c: 2.0 s against 3.7 s for GraphBatch on 100k vertices and 500k edges), but its CSV reader
  "splits on the delimiter without honouring quotes" (the docs, under array columns). The SNB text columns
  (`content`, `title`, names) contain commas and quotes, so it would load wrong values. The adapter therefore loads
  with GraphBatch (`create_vertices` and `new_edges` from an id to RID map, the documented Python bulk path), which
  gets its rows from Python's `csv` module. Cause: ArcadeDB (importer); no upstream issue found, the behaviour is
  documented.
- Minimal reproduction (arcadedb-embedded 26.10.2.post1, Windows and Linux):

  ```python
  import json, pathlib, tempfile
  import arcadedb_embedded as adb, jpype

  adb.jvm.start_jvm()
  d = pathlib.Path(tempfile.mkdtemp())
  (d / "V.csv").write_text('id,name\n1,"Smith, John"\n2,"say ""hi"""\n3,plain\n', encoding="utf-8")
  cfg = {"vertices": [{"type": "V", "file": "V.csv", "id": "id", "properties": {"id": "long:id", "name": "name"}}]}
  GI = jpype.JClass("com.arcadedb.integration.importer.graph.GraphImporter")
  J = jpype.JClass("com.arcadedb.serializer.json.JSONObject")(json.dumps(cfg))
  db = adb.create_database(str(d / "db"))
  GI.createSchemaFromConfig(db.get_java_database(), J)
  GI.fromJSON(db.get_java_database(), J, str(d)).run()
  print(db.query("opencypher", "MATCH (n:V) RETURN n.id AS id, n.name AS name ORDER BY id").to_list())
  ```

  Output: `[{'id': 1, 'name': '"Smith'}, {'id': 2, 'name': '"say ""hi"""'}, {'id': 3, 'name': 'plain'}]`; the
  CSV values are `Smith, John` and `say "hi"`.

## ArcadeDB: unlabeled id lookups scan every vertex (IS4, IS5, IS6; not allowlisted)

The answers agree with the references. IS4, IS5 and IS6 start from `(m {id: $messageId})` with no label (the
shared text; a message is a Post or a Comment). ArcadeDB's openCypher planner does not use the unique `id` indexes
of the vertex types for it: `EXPLAIN` shows "Using Traditional Execution (Non-Optimized). Reason: Query pattern not
yet supported by optimizer" and `+ MATCH NODE (m)` (IS4: `[filter: (m:Post OR m:Comment)]`), a scan of all 4178
vertices, while a labeled lookup such as IS3's `(n:Person {id: $personId})` is a `NodeIndexSeek`. On `snb-small`
(Windows, one thread) a single unlabeled lookup takes 2.3 ms against 0.09 ms for `(m:Comment {id: $id})`. In
`bench --selection push --db grafeo,arcadedb` the three are marked `timeout` (⏱) by the budget rule against Grafeo
alone (10 bindings: IS4 11.8 ms, IS5 9.2 ms, IS6 9.8 ms; IS1 0.9 ms). The query text is not rewritten. Cause:
ArcadeDB (query planner). No `- id:` line: the adapter test runs ArcadeDB alone, where no budget applies.

## ArcadeDB: algorithms are slow under the 10x budget rule (not allowlisted)

The answers agree with the references (PageRank within `LDBC_EPS`, run with damping 0.85, `maxIterations` 1000 and
`tolerance` 1e-10 as the other adapters). Against Grafeo alone, BFS (31.7 ms, Grafeo 0.80 ms), WCC (47.8 ms,
Grafeo 1.5 ms) and PageRank (18.7 ms, Grafeo 1.7 ms; 11.3x, close to the limit: 11.7 ms and within it in a run
before the JVM thread caps) are marked `timeout` (⏱) by the budget rule in
`bench --selection push --db grafeo,arcadedb` (Windows). The procedures run without a Graph Analytical View (optional, not
created). `algo.bfs` leaves the start vertex out of its result ("The start node is excluded from the results",
https://docs.arcadedb.com/arcadedb/reference/graph-algorithms/path-finding.html#algo-bfs), so the adapter adds it
at depth 0, the reference's definition. No `- id:` line.

## ArcadeDB: threads, heap, storage and the JVM start (notes)

- Threads: the worker starts the JVM with the documented engine settings `arcadedb.queryParallelismPoolThreads=1`,
  `arcadedb.parallelScanProducerPoolThreads=1`, `arcadedb.queryParallelScan=false` and
  `arcadedb.asyncWorkerThreads=1` (https://docs.arcadedb.com/arcadedb/reference/settings) and
  `java.util.concurrent.ForkJoinPool.common.parallelism=1`, through the package's `ARCADEDB_JVM_ARGS`; GraphBatch
  gets `parallel_flush=False`. The JVM's own threads are capped as for every JVM database (`jvm.py`,
  `SINGLE_THREAD_ARGS`): `-XX:ParallelGCThreads=1` and `-XX:ConcGCThreads=1` (default collector, G1 here) and
  `-XX:CICompilerCount=2`, HotSpot's minimum with tiered compilation, which stays on ("CICompilerCount (1) must be
  at least 2"), so one C1 and one C2 compiler thread
  (https://docs.oracle.com/en/java/javase/25/docs/specs/man/java.html, "Advanced Garbage Collection Options" and
  "Advanced JIT Compiler Options"; the bundled Corretto 25.0.4.1 accepts all three). The defaults on this 16-core
  machine are 13, 3 and 12. A load with threads=1 fails when the JVM lacks any of these settings.
- Not capped: the JVM's other service threads (signal dispatcher, reference handler, finalizer, the JPype
  reference queue) and ArcadeDB's housekeeping threads (coarse clock, cleaners, timer); none of them showed CPU
  use beyond a few percent.
- Measured on `snb-small` and `graphalytics-small` (Windows, 16 cores, shared with other work, `ThreadMXBean` CPU
  time per Java thread and psutil per OS thread): during IC6, IS4, BFS, WCC and PageRank the query work is on the
  calling thread alone (CPU/wall 0.96 to 0.99; no pool thread used CPU). With the caps, IS4 (200 runs after 20
  warm-up runs) peaks at 2.07 cores for the process (2.39 before the caps, 3.75 with all defaults in the same
  probe); the second core is JIT compilation, now on the two compiler threads (the JVM's compilation time over the
  window was 1.26x the wall time, garbage collection 0.05x). After 500 warm-up runs IS4 uses 0.8 to 1.2 cores.
  The results record `threads_configurable=True` with a note listing what is and is not capped.
- Heap: the shared JVM cap `-Xmx4g` (`graph_bench/databases/jvm.py`, `MAX_HEAP`), checked after the start; the page
  cache (`arcadedb.maxPageRAM`) keeps its default, a quarter of the heap.
- Storage: ArcadeDB has no in-memory mode, so each load creates an on-disk database in its fresh workdir
  (`storage="on-disk"`). GraphBatch runs with its default, the write-ahead log off.
- Disk: the database directory `workdir/db` alone (`store_paths`). After a load it holds the bucket files, the index,
  the dictionary and the schema, statistics and configuration files, and nothing else lands in the workdir (the
  engine log goes to a temporary directory).
- The JVM starts once per process in `prepare_environment`, outside every clock, followed by creating and dropping
  one empty database (the engine's one-time start, about 0.25 s). Every load is in the load clock from
  `create_database` on: schema, unique `id` indexes, vertices and edges.
- Startup: each startup child starts its own JVM (no warm start), so the Startup time includes the JVM start and
  the engine's one-time start (its classes and its polyglot engine) before `create_database` on the child's workdir
  and the probe. The JVM thread caps and the single-thread engine settings apply as in a load (the child inherits
  `process_env(threads)`, and `open_empty` checks them).
- Logging: the engine's built-in configuration writes `./log/arcadedb.log` into the working directory; the adapter
  passes `java.util.logging.config.file` with the same configuration and the file in a per-process temporary
  directory, removed at exit.
- Package: arcadedb-embedded 26.10.2.post1 is the latest stable release. ArcadeDB's Python docs announce the JVM
  build's rename to `arcadedb-jvm` from 26.11.1, after which `arcadedb-embedded` becomes a shim that installs the
  native (GraalVM) `arcadedb` package (https://docs.arcadedb.com/arcadedb-python/which-package.html); both exist on
  PyPI only as 26.11.1 pre-releases today.

## Neo4j: IC13 with the same person twice (a documented setting, answer agrees)

- Query: `snb_ic13`, the shared openCypher text, no Neo4j override. One binding pairs a person with itself, where
  the reference is 0 (LDBC IC13).
- With Neo4j's defaults that binding fails: `The shortest path algorithm does not work when the start and end
  nodes are the same.` (research-c, Neo4j 2026.10.1).
- Neo4j documents the setting for it, `dbms.cypher.forbid_shortestpath_common_nodes`
  (https://neo4j.com/docs/operations-manual/current/configuration/configuration-settings/): "The shortest path
  algorithm does not work when the start and end nodes are the same. With this setting set to false no path will
  be returned when that happens. The default value of true will instead throw an exception." The adapter sets it
  to false, so `path` is null for that binding and the shared text's `CASE WHEN p1.id = p2.id THEN 0` returns 0.
  Distinct persons are unaffected. All IC13 bindings agree. No `WHERE p1 <> p2` override is used.

## Neo4j: unlabeled id lookups start from a label scan (IS5, IS6; not allowlisted)

The answers agree with the references. IS5 and IS6 start from `(m {id: $messageId})` with no label (the shared
text). The unique `id` constraints are per label, so the planner has no index for `m`. `PROFILE` on `snb-small`
(comment_5) shows the planner picks another start:
- IS5: `NodeByLabelScan p:Person`, `Expand (p)<-[:HAS_CREATOR]-(m)`, `Filter m.id = $messageId`, 42,259 db hits,
  about 7.4 ms per call warm.
- IS6: `NodeByLabelScan f:Forum`, then the moderator, container and `REPLY_OF*0..` expansions and a filter on
  `m.id`, 14,685 db hits, about 4.0 ms per call.
- IS4 is not affected. Its `WHERE m:Post OR m:Comment` is planned as a `Union` of two `NodeUniqueIndexSeek`s
  (6 db hits, 0.4 ms).
In `bench --selection push --db grafeo,neo4j` (Windows) IS5 and IS6 are marked `timeout` (⏱) by the budget rule
against Grafeo alone (10 bindings: IS5 41.0 ms, IS6 44.4 ms; Grafeo 0.31 and 0.66 ms). The query text is not
rewritten. Cause: Neo4j (the query planner has no index for an unlabeled node). No `- id:` line: the adapter test
runs Neo4j alone, where no budget applies.

## Neo4j: slow cases under the 10x budget rule (not allowlisted)

The answers agree with the references (PageRank within `LDBC_EPS`, see below). Against Grafeo alone, in
`bench --selection push --db grafeo,neo4j` (Windows), these are also ⏱ by the budget rule, not by a hang:
- `snb_is3`: 36.5 ms against 1.9 ms.
- `snb_is4`: 5.6 ms against 0.33 ms.
- `graphalytics_bfs`: 86.9 ms against 0.93 ms.
- `graphalytics_wcc`: 35.7 ms against 1.6 ms.
- `graphalytics_pr`: 155.7 ms against 1.7 ms.
A Cypher call through the embedded API costs about 0.1 ms even for `RETURN 1` (a transaction, the parameter map
and the JPype crossing). The algorithms map each GDS node id back to the app `id` with `gds.util.asNode`, as the
GDS docs do. No `- id:` line.

## Neo4j: GDS algorithms (notes)

- GDS Community ships in the Community tarball (`products/neo4j-graph-data-science-2026.10.1.jar`). The jar goes
  on the JVM classpath and in the plugins directory, with `gds.*` allowed (`dbms.security.procedures.unrestricted`
  and `.allowlist`). Without a licence file GDS runs as its Community Edition: concurrency at most 4,
  https://neo4j.com/docs/graph-data-science/current/introduction/. `gds.license.state()` reports "No valid GDS
  license specified."
- Licence: the jar's `LICENSE.txt` is GPLv3. Its `NOTICE.txt` says the Neo4j-owned software is GPLv3 "However, if
  you have executed ... a commercial license agreement with Neo4j ... the terms of the license in such Commercial
  Agreement will supersede the GPL". Research-c had reported a pre-release agreement text in this jar. The
  2026.10.1 jar's `LICENSE.txt` entries are GPLv3, and no pre-release text was found in its licence files.
  Using GDS is provisional user decision 8.
- Projection: `gds.graph.project('g', 'V', 'E', {readConcurrency: 1})` on the graphalytics dataset, inside the
  load clock, as LadybugDB and NeuG project theirs. It is dropped when the instance closes, because the GDS graph
  catalog outlives a DBMS in the JVM.
- WCC: `gds.wcc.stream` with `concurrency: 1`.
- PageRank: `gds.pageRank.stream` with `dampingFactor: 0.85, maxIterations: 1000, tolerance: 1e-10`, the values
  of the other adapters, and `concurrency: 1`.
  - The defaults are 20 iterations and tolerance 1e-7.
  - Unscaled, GDS scores do not sum to 1, so the documented scaler `L1NORM` ("creates a probability distribution
    where the sum of scores equals 1") is passed, the reference's definition.
  - On `graphalytics-small` (one isolated vertex) the largest relative difference to NetworkX is 5.0e-8, inside
    `LDBC_EPS` (1e-4).
- BFS: GDS has no BFS that yields depths (`gds.bfs.stream` gives the visit order).
  - The hop distances come from `gds.allShortestPaths.dijkstra.stream` without `relationshipWeightProperty`, the
    single-source shortest path procedure. The docs say "If unspecified, the algorithm runs unweighted", so each
    relationship costs 1, the source is included at cost 0, and unreachable vertices have no row.
  - The adapter checks every cost is a whole number and returns it as an int. The answer agrees.
  - This is unweighted Dijkstra, not a BFS implementation. The result definition is the reference's.

## Neo4j: load through neo4j-admin import has a large fixed cost (provisional decision 7; not allowlisted)

The load is `neo4j-admin database import full`, Neo4j's offline bulk importer for an empty database
(https://neo4j.com/docs/operations-manual/current/import/). It runs as a subprocess with the fetched JRE, then
the store is opened embedded. The load clock covers:
1. The header-adapted CSV copies. The importer reads a data source's header from the first line of its first
   file, so each CSV is copied with its header line replaced by a typed one (`id:ID(<label>)`, `name:string`,
   `:START_ID(<label>)`, ...). `--auto-skip-subsequent-headers` cannot use a separate header file here: it skips
   only a line equal to the header, and the CSVs' untyped header lines differ from the typed ones (tested: the
   relationship import then fails on `src`).
2. The importer, two JVM starts included (the `neo4j-admin` launcher and its import process).
3. Opening the store with `DatabaseManagementServiceBuilder`.
4. A unique constraint on `id` per node label, then `db.awaitIndexes`.
5. The GDS projection (graphalytics only).

The importer runs with `--id-type` integer or string (from the schema, one per dataset), `--threads=1` in
single-threaded runs, `--max-off-heap-memory` equal to the embedded page cache (1 GiB) and the JVM heap
`HEAP_SIZE=4g`. The importer's configuration is the tarball's default `conf/neo4j.conf` (for its `--add-opens`
JVM options) plus the instance's data and log directories and, in single-threaded runs, the shared JVM thread caps
as `server.jvm.additional` lines.

Measured on Windows (one thread, push data, three loads per dataset in one process):

| Path | graphalytics-small | snb-small |
| --- | --- | --- |
| neo4j-admin import (adapter): total | 7.4 to 9.1 s | 7.9 to 8.9 s |
| of which the importer subprocess | 4.6 to 4.8 s | 5.0 to 5.4 s |
| of which open, constraints, projection | 2.6 to 4.4 s | 2.9 to 3.4 s |
| `LOAD CSV ... CALL { } IN TRANSACTIONS OF 10000 ROWS` (scratch, not used) | 2.6 to 5.1 s | 3.7 to 4.7 s |

- The importer's own report says `IMPORT DONE in 2s` to `3s` on this data. Most of that is fixed cost, since the
  data is 1,001 nodes and 36,000 relationships. Research-c measured 1.9 s for 100,000 nodes and 500,000
  relationships on Linux, where `LOAD CSV` took 17 s. So the importer is the faster path at scale and the slower one
  on the push data.
- Opening a DBMS costs about 2.1 s even for an empty store.
- In `bench --selection push --db grafeo,neo4j` (Windows, after the fix that keeps the download check out of the
  load) the loads took 9.0 s (Graphalytics) and 7.7 s (SNB), against Grafeo's 0.07 s and 0.12 s. Before the fix,
  each load also re-hashed the cached tarball and JRE (`jvm.fetch`, about 0.2 s). Loads are exempt from the budget
  rule.
- This choice is provisional user decision 7.

## Neo4j: threads, memory, storage and the JVM start (notes)

- Threads (results: `threads_configurable=True` with a note):
  - Community's Cypher runtime runs each query on the calling thread. The parallel runtime and
    `server.cypher.parallel.worker_limit` are Enterprise.
  - GDS gets `concurrency: 1`, and the projection gets `readConcurrency: 1`.
  - The importer gets `--threads=1`. Its help says "(advanced) Max number of worker threads used by the importer";
    its report shows `Max worker threads: 1`.
  - Both JVMs (embedded and importer) get the shared caps `-XX:ParallelGCThreads=1 -XX:ConcGCThreads=1
    -XX:CICompilerCount=2`.
  - Not capped: Neo4j's background threads (page cache flushing, checkpointing, index population) and the JVM's
    other service threads. No documented setting exists for them.
- Memory:
  - The shared heap cap `-Xmx4g`, checked after the start. The importer gets the same through `HEAP_SIZE`.
  - The page cache is set to 1 GiB, ArcadeDB's page cache under the same heap (a quarter of it). Neo4j's own
    default is sized from the machine's RAM.
- Storage: Community has no in-memory mode, so each load writes an on-disk store in its fresh workdir
  (`storage="on-disk"`). The impermanent test builders are internal test utilities that the distribution does not
  ship.
- Disk: `workdir/data` (`store_paths`): `data/databases` and `data/transactions` for the `neo4j` and `system`
  databases, "all data-related content" in Neo4j's words. `conf/` (the importer's configuration), `import/` (the
  CSV copies), `logs/` and `import.report` are the load's scaffolding and are not counted.
- Memory: the importer's JVMs are children of the measuring process, so the peak during the load covers the
  importer, then the embedded DBMS, as the load's real cost; `processes` is at least 2 for it.
- JVM start: once per process in `prepare_environment`, outside every clock. It then opens and shuts down one
  empty DBMS (class loading and GDS registration, about 10 s cold), so the first load does not carry them. It holds
  no data and runs no query. Fetching (SHA-256 checked) and extracting the tarball and the JRE happen in
  `provision`, which `prepare_environment` calls first; it runs once per process (a second call hashes nothing).
- Startup: each startup child starts its own JVM and opens an empty DBMS on its workdir with the same builder
  settings as a load (page cache, GDS plugin directory, procedure allowlist), so the Startup time includes the JVM
  start, class loading and GDS registration: GDS is part of the benchmark's configuration, and every load's DBMS
  registers it too. The shared JVM thread caps apply (the child inherits `process_env(threads)`, and the start is
  checked like a load's). Nothing is fetched or extracted in a child: `provision` ran untimed in the worker before,
  and a missing file fails the entry with "not provisioned: <path>". Measured on Windows: about 8 s per child once
  the files are in the OS cache, 17 s for a cold first child.
- Versions: Neo4j Community 2026.10.1 (the latest stable release, `<release>` in Maven Central's
  `org.neo4j:neo4j` metadata) on Temurin 21.0.13+9. Neo4j 2026.10 supports Java 21 and 25, and its class files are
  Java 21.
- Downloads (for the CI cache), in `<data cache>/downloads`:
  - The tarball is 235 MB and the JRE is 50 MB (Linux) or 47 MB (Windows).
  - Both are extracted into `downloads/unpacked`: 259 MB for Neo4j, 146 MB for the JRE (Windows), plus a 67 MB
    copy of the GDS jar as the plugins directory.
  - About 690 MB in all.

## DuckDB + DuckPGQ: the label disjunction `Post|Comment` does not parse (IS4, IS6; not available as an error)

- id: duckdb/snb_sql_is4
- id: duckdb/snb_sql_is6
- The shared SQL/PGQ texts say "a Post or a Comment" as the standard label expression `m:Post|Comment` (ISO/IEC
  9075-16). DuckPGQ 2.0 (extension build `9c11bc192a`, DuckDB 2.0.0) rejects it while parsing:
  `Parser Error: syntax error at or near "|"`, at `MATCH (m:Post|Comment)` (IS4) and at
  `MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)` (IS6; first failure: the parser stops there). The
  cases show ✗ with that error.
- Not worked around: no union `Message` table and no DuckDB-specific text (a database whose feature does not work has
  to fix it). IS6 would also fail afterwards on the second gap: DuckPGQ needs unique labels, and `REPLY_OF` is two
  edge tables (comment to comment, comment to post) in the dataset, so the property graph labels them
  `REPLY_OF__Comment__Comment` and `REPLY_OF__Comment__Post` and the label `REPLY_OF` does not exist. The same holds
  for `HAS_CREATOR`, `HAS_TAG` and `IS_LOCATED_IN`, which no SQL/PGQ case uses.
- Cause: DuckPGQ (SQL/PGQ label expressions and non-unique edge labels), checked against
  https://duckpgq.org/documentation/property_graph/ and https://duckpgq.org/documentation/sql_pgq/.

## DuckDB + DuckPGQ: PageRank stops at a fixed tolerance (≠)

- id: duckdb/graphalytics_pr
- `pagerank(g, V, E)` (https://duckpgq.org/documentation/graph_functions/) has a fixed damping factor 0.85 and a fixed
  tolerance of 1e-6 that cannot be set. On `graphalytics-small` it stops early: every one of the 1001 ranks is off
  the converged ranks (NetworkX, tolerance 1e-10) by 3e-6 to 4.5e-6 absolute, 1.0e-4 to 3.1e-4 relative, above the
  relative `LDBC_EPS` of 1e-4 (first mismatch: 0.013918058 against 0.013922308). The epsilon is not loosened, so the
  answer is ≠. Cause: DuckPGQ (no tolerance parameter).

## DuckDB + DuckPGQ: slow cases under the 10x budget rule (not allowlisted)

The answers agree with the references. Against Grafeo alone in `bench --selection push --db grafeo,duckdb`
(Windows, one thread): BFS 103 ms (Grafeo 0.5 ms, an `ANY SHORTEST` query that returns the 1000 hop distances) and
WCC 20.7 ms (Grafeo 1.6 ms) are marked ⏱, and SQL/PGQ IS1 15.6 ms for 10 bindings (Grafeo 0.24 ms; the shared text
filters in the graph pattern WHERE, which DuckPGQ evaluates after matching, where the element pattern form
`(n:Person WHERE n.id = $personId)` took 1.2 ms in research). IC13 is 3.5 s for 10 bindings (`ANY SHORTEST` from the
filtered endpoints). No `- id:` line: the adapter test runs DuckDB alone, where no budget applies. The texts are not
rewritten.

## DuckDB + DuckPGQ: SQL/PGQ IC13 does not finish on `snb-medium` (⏱, not allowlisted)

In the `full` selection (`bench --selection full`, Windows and Linux, DuckDB 2.0.0 with duckpgq), `snb_sql_ic13`
on `snb-medium` (about 10x `snb-small`: 41,078 nodes, 376,477 edges) prints no output for the 600 s case timeout,
so the entry is ⏱ with "no output for 600 s". On `snb-small` it agrees (3.5 s for 10 bindings). The text is the
shared standard SQL/PGQ `ANY SHORTEST` over undirected `KNOWS` (`suites/snb_sql.py`), not rewritten. The push
selection has no medium scale, so there is no `- id:` line. Cause: DuckPGQ engine (shortest path search cost grows
much faster than the graph).

## DuckDB + DuckPGQ: threads, storage and the extension (notes)

- Threads: `SET threads=1` (https://duckdb.org/docs/current/configuration/overview) when the run is single-threaded.
- Storage: `duckdb.connect(":memory:")`, in-memory.
- The extension: `INSTALL duckpgq FROM community` runs once per process in `prepare_environment` (it downloads), and
  `LOAD duckpgq` plus `SET active_grammar_extensions=['duckpgq']` run per connection in the load, which DuckDB 2.0
  requires (https://github.com/cwida/duckpgq-extension/blob/main/docs/GRAMMAR_EXTENSION.md). The load is typed
  `CREATE TABLE` plus `COPY` per CSV and one `CREATE PROPERTY GRAPH` over all node and edge tables, all in the load
  clock. A type spread over several node pairs is labelled by its table name (`HAS_TAG__Comment__Tag`), because
  DuckPGQ labels must be unique.
