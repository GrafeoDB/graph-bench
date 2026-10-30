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
- Reference answer: `[[1]]` (BFS over KNOWS in both directions); every binding's answer is 1 or 2.
- Cause: Grafeo engine (query planner). `explain_cypher` shows the `OPTIONAL MATCH` becoming a `LeftJoin` whose
  right side is `ShortestPath (p1 -> p2)` over `NodeScan (p1:*)` and `NodeScan (p2:*)`, with the `id` filter on
  `p2` applied after it. `p1` and `p2` are not taken from the left side, so it computes shortest paths for all
  node pairs of the graph and filters afterwards:

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

## LadybugDB: slow point lookups under the 10x budget rule (not allowlisted)

These cases agree with the reference in every run. In a `bench` run next to Grafeo they are marked `timeout`
(⏱) by the budget rule (slower than 10x the mean of the 3 fastest other databases), not by a hang. They carry
no `- id:` line because the adapter test runs LadybugDB alone, where no budget applies and they must agree.

- Queries: `snb_is1`, `snb_is2`, `snb_is5`, `snb_is6` (`snb_is6` with its `ladybug` override), on `snb-small`.
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
