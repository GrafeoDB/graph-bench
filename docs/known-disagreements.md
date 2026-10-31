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
