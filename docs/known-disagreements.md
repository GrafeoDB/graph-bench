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

## Grafeo: LOAD CSV is 2x to 3x slower than its batch API (not allowlisted)

Load uses the batch API, Grafeo's fastest documented bulk path; LOAD CSV will be tracked as its own benchmark,
Load (query language). The LOAD CSV load is correct (every node and edge count matches the CSVs, and all checked
queries agree afterwards), but slower than the batch API. A load entry has no answer, so there is no `- id:` line.

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
  search prefix (with it, quantified edges inside `GRAPH_TABLE`). The decision on the target release is the
  user's (decisions.md, 6).

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
`graphalytics-small` and 4 to 6 s for `snb-small` (Windows, Python 3.12), against Grafeo's 0.07 s and 0.13 s. A
load entry has no answer, so there is no `- id:` line.

- Statements: one explicit-schema `CREATE NODE TABLE` / `CREATE REL TABLE` per table, then `COPY ... (header=true,
  delim=",")` per CSV (https://neug.io/docs/data_io/import_data/). Each COPY commits a private copy-on-write graph
  with a checkpoint (visible in the engine log), and each in-memory database also keeps a temp directory.
  Wrapping the COPY statements in one transaction did not make the load measurably faster.
- Cause: NeuG engine (checkpoint per statement), not measured further.

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
- There is no in-memory mode: the graph is on disk (`storage="on-disk"`), and no thread setting is exposed
  (`threads_configurable=False`).
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
alone (10 bindings: IS4 14.6 ms, IS5 9.3 ms, IS6 9.7 ms; IS1 0.7 ms). The query text is not rewritten. Cause:
ArcadeDB (query planner). No `- id:` line: the adapter test runs ArcadeDB alone, where no budget applies.

## ArcadeDB: algorithms are slow under the 10x budget rule (not allowlisted)

The answers agree with the references (PageRank within `LDBC_EPS`, run with damping 0.85, `maxIterations` 1000 and
`tolerance` 1e-10 as the other adapters). Against Grafeo alone, BFS (21.9 ms, Grafeo 0.95 ms) and WCC (35.2 ms,
Grafeo 3.1 ms) are marked `timeout` (⏱) by the budget rule in `bench --selection push --db grafeo,arcadedb`;
PageRank (11.7 ms, Grafeo 2.4 ms) is within it. The procedures run without a Graph Analytical View (optional, not
created). `algo.bfs` leaves the start vertex out of its result ("The start node is excluded from the results",
https://docs.arcadedb.com/arcadedb/reference/graph-algorithms/path-finding.html#algo-bfs), so the adapter adds it
at depth 0, the reference's definition. No `- id:` line.

## ArcadeDB: threads, heap, storage and the JVM start (notes)

- Threads: the worker starts the JVM with the documented engine settings `arcadedb.queryParallelismPoolThreads=1`,
  `arcadedb.parallelScanProducerPoolThreads=1`, `arcadedb.queryParallelScan=false` and
  `arcadedb.asyncWorkerThreads=1` (https://docs.arcadedb.com/arcadedb/reference/settings) and
  `java.util.concurrent.ForkJoinPool.common.parallelism=1`, through the package's `ARCADEDB_JVM_ARGS`; GraphBatch
  gets `parallel_flush=False`. A load with threads=1 fails when the JVM lacks them. Measured on `snb-small` and
  `graphalytics-small` (Windows, 16 cores, `ThreadMXBean` CPU time per Java thread): during IC6, IS4, BFS, WCC and
  PageRank the query work is on the calling thread alone (CPU/wall 0.86 to 1.0; no pool thread used CPU). The
  process as a whole used 1.04 to 2.4 cores, the rest on the JVM's own threads (garbage collector, JIT compiler),
  which are not Java threads and are not capped: the JVM defaults stay, as they would for any embedding
  application. The results record `threads_configurable=True` with that note.
- Heap: the shared JVM cap `-Xmx4g` (`graph_bench/databases/jvm.py`, `MAX_HEAP`), checked after the start; the page
  cache (`arcadedb.maxPageRAM`) keeps its default, a quarter of the heap.
- Storage: ArcadeDB has no in-memory mode, so each load creates an on-disk database in its fresh workdir
  (`storage="on-disk"`). GraphBatch runs with its default, the write-ahead log off.
- The JVM starts once per process in `prepare_environment`, outside every clock, followed by creating and dropping
  one empty database (the engine's one-time start, about 0.25 s). Every load is in the load clock from
  `create_database` on: schema, unique `id` indexes, vertices and edges.
- Logging: the engine's built-in configuration writes `./log/arcadedb.log` into the working directory; the adapter
  passes `java.util.logging.config.file` with the same configuration and the file in a per-process temporary
  directory, removed at exit.
- Package: arcadedb-embedded 26.10.2.post1 is the latest stable release. ArcadeDB's Python docs announce the JVM
  build's rename to `arcadedb-jvm` from 26.11.1, after which `arcadedb-embedded` becomes a shim that installs the
  native (GraalVM) `arcadedb` package (https://docs.arcadedb.com/arcadedb-python/which-package.html); both exist on
  PyPI only as 26.11.1 pre-releases today.

## DuckDB + DuckPGQ: the label disjunction `Post|Comment` does not parse (IS4, IS6; not available as an error)

- id: duckdb/snb_sql_is4
- id: duckdb/snb_sql_is6
- The shared SQL/PGQ texts say "a Post or a Comment" as the standard label expression `m:Post|Comment` (ISO/IEC
  9075-16). DuckPGQ 2.0 (extension build `9c11bc192a`, DuckDB 2.0.0) rejects it while parsing:
  `Parser Error: syntax error at or near "|"`, at `MATCH (m:Post|Comment)` (IS4) and at
  `MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)` (IS6). The cases show ✗ with that error.
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

## DuckDB + DuckPGQ: threads, storage and the extension (notes)

- Threads: `SET threads=1` (https://duckdb.org/docs/current/configuration/overview) when the run is single-threaded.
- Storage: `duckdb.connect(":memory:")`, in-memory.
- The extension: `INSTALL duckpgq FROM community` runs once per process in `prepare_environment` (it downloads), and
  `LOAD duckpgq` plus `SET active_grammar_extensions=['duckpgq']` run per connection in the load, which DuckDB 2.0
  requires (https://github.com/cwida/duckpgq-extension/blob/main/docs/GRAMMAR_EXTENSION.md). The load is typed
  `CREATE TABLE` plus `COPY` per CSV and one `CREATE PROPERTY GRAPH` over all node and edge tables, all in the load
  clock. A type spread over several node pairs is labelled by its table name (`HAS_TAG__Comment__Tag`), because
  DuckPGQ labels must be unique.
