# Benchmark Results

**Run:** 2026-02-23 | **Platform:** Windows  
**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)

## Scale Factors (LDBC Standard)

| Scale | Persons | KNOWS Edges | Reference |
|-------|--------:|------------:|-----------|
| SF0.1 (sf01) | 1K | 18K | Quick validation |
| SF1 (sf1) | 10K | 180K | Standard benchmark |
| SF3 (sf3) | 27K | 540K | Medium scale |
| SF10 (sf10) | 73K | 2M | Large scale |
| SF30 (sf30) | 180K | 6.5M | Very large |
| SF100 (sf100) | 280K | 18M | Full scale |

**Current run:** sf01

All times in milliseconds. Best result per benchmark in **bold**. 

---

## Summary

### Overall

| Database | Type | SNB (ms) | SNB Mem | Analytics (ms) | Analytics Mem | ACID (ms) | ACID Mem |
|----------|------| ------:| ------:| ------:| ------:| ------:| ------:|
| **Grafeo Server** | Server | **730.1** |  | **15.2** |  | 198 |  |
| Neo4j | Server | 6,788 | 1458 MB | 253 | 1228 MB | 369 | 1485 MB |
| **Memgraph** | Server | 4,113 | 715 MB | 19 | 636 MB | **107.0** | 613 MB |
| ArangoDB | Server | 40,043 | 371 MB | 22,739 | 283 MB | 2,110 | 382 MB |

### SNB Interactive (total ms)

| Database | Type | Sf01 |
|----------|------|------:|
| **Grafeo Server** | Server | **730.1** |
| Neo4j | Server | 6,788 |
| Memgraph | Server | 4,113 |
| ArangoDB | Server | 40,043 |

### Graph Analytics (native implementations only, total ms)

| Database | Type | Sf01 |
|----------|------|------:|
| **Grafeo Server** | Server | 15.2 |
| **Neo4j** | Server | 252.5 |
| **Memgraph** | Server | 19.0 |

Only databases with native in-database algorithm implementations are included. Databases that fall back to extracting the graph into Python/NetworkX measure extraction overhead, not database performance. That extraction typically adds 100-1,000x overhead.

### Combinatorial Workload (total ms)

| Database | Type | Writes | Reads | Traversals | ACID |
|----------|------|-------:|------:|-----------:|-----:|
| Grafeo Server | Server | | | | |
| Neo4j | Server | | | | |
| Memgraph | Server | | | | |
| ArangoDB | Server | | | | |

### Reading the results

These benchmarks compare databases with fundamentally different architectures. Before drawing conclusions, consider:

- **Embedded vs. server.** Grafeo and LadybugDB run in-process - no network serialization, no protocol overhead. Server databases pay ~0.1-1ms per round-trip.
- **Consistency model.** NebulaGraph uses eventual consistency by default. Its write speeds are not comparable to ACID-compliant databases without qualification.
- **Memory model.** Memgraph is in-memory first with optional WAL persistence. FalkorDB inherits Redis persistence semantics.
- **Scale factor.** Small (10K nodes) fits in L2 cache. Medium and large benchmarks reveal architectural differences.

---

## Per-Database Results

<details>
<summary><h3>Grafeo Server</h3></summary>

| | |
|---|---|
| **Type** | Server (HTTP REST, GQL) |
| **Data model** | LPG + RDF |
| **Query languages** | GQL (ISO) |
| **ACID** | Full (snapshot isolation, WAL) |
| **Consistency** | Strong |
| **License** | Apache 2.0 |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time |
|-----------|-----:|
| snb_is1 | 230.32ms |
| snb_is2 | 23.20ms |
| snb_is3 | 57.75ms |
| snb_is4 | 123.52ms |
| snb_is5 | 52.97ms |
| snb_is6 | 116.69ms |
| snb_is7 | 70.84ms |
| **Total** | **675.28ms** |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time |
|-----------|-----:|
| snb_ic1 | 25.57ms |
| snb_ic2 | 11.74ms |
| snb_ic3 | 5.98ms |
| snb_ic6 | 11.55ms |
| **Total** | **54.84ms** |

#### SNB Interactive - Sf01

| Benchmark | Time |
|-----------|-----:|
| snb_is1 | 230.32ms |
| snb_is2 | 23.20ms |
| snb_is3 | 57.75ms |
| snb_is4 | 123.52ms |
| snb_is5 | 52.97ms |
| snb_is6 | 116.69ms |
| snb_is7 | 70.84ms |
| snb_ic1 | 25.57ms |
| snb_ic2 | 11.74ms |
| snb_ic3 | 5.98ms |
| snb_ic6 | 11.55ms |
| **Total** | **730.12ms** |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time |
|-----------|-----:|
| ldbc_bfs | 2.62ms |
| ldbc_pagerank | 2.10ms |
| ldbc_wcc | 2.36ms |
| ldbc_cdlp | 2.11ms |
| ldbc_lcc | 3.36ms |
| ldbc_sssp | 2.67ms |
| **Total** | **15.23ms** |

#### Graph Analytics - Sf01

| Benchmark | Time |
|-----------|-----:|
| ldbc_bfs | 2.62ms |
| ldbc_pagerank | 2.10ms |
| ldbc_wcc | 2.36ms |
| ldbc_cdlp | 2.11ms |
| ldbc_lcc | 3.36ms |
| ldbc_sssp | 2.67ms |
| **Total** | **15.23ms** |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time |
|-----------|-----:|
| acid_atomicity_c | 5.38ms |
| acid_atomicity_rb | 8.15ms |
| **Total** | **13.53ms** |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time |
|-----------|-----:|
| acid_g0 | 12.62ms |
| acid_g1a | 15.27ms |
| acid_g1b | 7.05ms |
| acid_g1c | 12.27ms |
| acid_imp | 15.97ms |
| acid_pmp | 32.10ms |
| acid_otv | 10.21ms |
| acid_fr | 13.01ms |
| acid_lu | 52.69ms |
| acid_ws | 13.39ms |
| **Total** | **184.58ms** |

</details>

<details>
<summary><h3>Neo4j</h3></summary>

| | |
|---|---|
| **Type** | Server (Bolt RPC) |
| **Data model** | LPG |
| **Query languages** | Cypher |
| **ACID** | Full (read committed isolation) |
| **Consistency** | Strong |
| **License** | GPL / Commercial |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 373.42ms | 1325.1 MB |
| snb_is2 | 267.52ms | 1203.2 MB |
| snb_is3 | 3574.82ms | 1321.0 MB |
| snb_is4 | 97.44ms | 1323.0 MB |
| snb_is5 | 94.41ms | 1324.0 MB |
| snb_is6 | 235.92ms | 1346.6 MB |
| snb_is7 | 164.60ms | 1332.2 MB |
| **Total** | **4808.13ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | FAILED | |
| snb_ic2 | 806.06ms | 1410.0 MB |
| snb_ic3 | 639.89ms | 1458.2 MB |
| snb_ic6 | 533.78ms | 1437.7 MB |
| **Total** | **1979.74ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 373.42ms | 1325.1 MB |
| snb_is2 | 267.52ms | 1203.2 MB |
| snb_is3 | 3574.82ms | 1321.0 MB |
| snb_is4 | 97.44ms | 1323.0 MB |
| snb_is5 | 94.41ms | 1324.0 MB |
| snb_is6 | 235.92ms | 1346.6 MB |
| snb_is7 | 164.60ms | 1332.2 MB |
| snb_ic1 | FAILED | |
| snb_ic2 | 806.06ms | 1410.0 MB |
| snb_ic3 | 639.89ms | 1458.2 MB |
| snb_ic6 | 533.78ms | 1437.7 MB |
| **Total** | **6787.87ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 37.84ms | 917.3 MB |
| ldbc_pagerank | 76.37ms | 932.7 MB |
| ldbc_wcc | 34.45ms | 937.5 MB |
| ldbc_cdlp | 48.98ms | 1224.7 MB |
| ldbc_lcc | 32.71ms | 1225.7 MB |
| ldbc_sssp | 22.16ms | 1227.8 MB |
| **Total** | **252.52ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 37.84ms | 917.3 MB |
| ldbc_pagerank | 76.37ms | 932.7 MB |
| ldbc_wcc | 34.45ms | 937.5 MB |
| ldbc_cdlp | 48.98ms | 1224.7 MB |
| ldbc_lcc | 32.71ms | 1225.7 MB |
| ldbc_sssp | 22.16ms | 1227.8 MB |
| **Total** | **252.52ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 11.81ms | 1479.7 MB |
| acid_atomicity_rb | 15.35ms | 1479.7 MB |
| **Total** | **27.16ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | 24.58ms | 1484.8 MB |
| acid_g1a | 22.94ms | 1483.8 MB |
| acid_g1b | 13.77ms | 1483.8 MB |
| acid_g1c | 20.42ms | 1483.8 MB |
| acid_imp | 16.85ms | 1483.8 MB |
| acid_pmp | 42.32ms | 1482.8 MB |
| acid_otv | 18.19ms | 1483.8 MB |
| acid_fr | 17.50ms | 1482.8 MB |
| acid_lu | 144.96ms | 1483.8 MB |
| acid_ws | 20.77ms | 1484.8 MB |
| **Total** | **342.31ms** | |

</details>

<details>
<summary><h3>Memgraph</h3></summary>

| | |
|---|---|
| **Type** | Server (Bolt RPC) |
| **Data model** | LPG |
| **Query languages** | Cypher |
| **ACID** | Yes (single-node, snapshot isolation) |
| **Consistency** | Strong |
| **License** | BSL / Enterprise |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 181.55ms | 677.3 MB |
| snb_is2 | 137.37ms | 690.9 MB |
| snb_is3 | 2170.60ms | 687.8 MB |
| snb_is4 | 60.24ms | 714.8 MB |
| snb_is5 | 59.34ms | 615.3 MB |
| snb_is6 | 144.97ms | 604.5 MB |
| snb_is7 | 101.83ms | 620.9 MB |
| **Total** | **2855.91ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | FAILED | |
| snb_ic2 | 510.29ms | 624.6 MB |
| snb_ic3 | 385.12ms | 606.7 MB |
| snb_ic6 | 362.12ms | 618.5 MB |
| **Total** | **1257.53ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 181.55ms | 677.3 MB |
| snb_is2 | 137.37ms | 690.9 MB |
| snb_is3 | 2170.60ms | 687.8 MB |
| snb_is4 | 60.24ms | 714.8 MB |
| snb_is5 | 59.34ms | 615.3 MB |
| snb_is6 | 144.97ms | 604.5 MB |
| snb_is7 | 101.83ms | 620.9 MB |
| snb_ic1 | FAILED | |
| snb_ic2 | 510.29ms | 624.6 MB |
| snb_ic3 | 385.12ms | 606.7 MB |
| snb_ic6 | 362.12ms | 618.5 MB |
| **Total** | **4113.44ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 1.33ms | 627.7 MB |
| ldbc_pagerank | 3.14ms | 630.1 MB |
| ldbc_wcc | 2.56ms | 631.9 MB |
| ldbc_cdlp | 3.09ms | 633.2 MB |
| ldbc_lcc | 7.05ms | 634.5 MB |
| ldbc_sssp | 1.84ms | 636.5 MB |
| **Total** | **18.99ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 1.33ms | 627.7 MB |
| ldbc_pagerank | 3.14ms | 630.1 MB |
| ldbc_wcc | 2.56ms | 631.9 MB |
| ldbc_cdlp | 3.09ms | 633.2 MB |
| ldbc_lcc | 7.05ms | 634.5 MB |
| ldbc_sssp | 1.84ms | 636.5 MB |
| **Total** | **18.99ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 2.24ms | 613.0 MB |
| acid_atomicity_rb | 3.47ms | 595.3 MB |
| **Total** | **5.71ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | FAILED | |
| acid_g1a | 10.16ms | 597.0 MB |
| acid_g1b | 3.49ms | 598.7 MB |
| acid_g1c | 5.38ms | 600.5 MB |
| acid_imp | 13.27ms | 597.9 MB |
| acid_pmp | 21.20ms | 596.9 MB |
| acid_otv | 4.58ms | 597.4 MB |
| acid_fr | 9.08ms | 598.2 MB |
| acid_lu | 28.32ms | 598.3 MB |
| acid_ws | 5.78ms | 595.3 MB |
| **Total** | **101.25ms** | |

</details>

<details>
<summary><h3>ArangoDB</h3></summary>

| | |
|---|---|
| **Type** | Server (HTTP / TCP) |
| **Data model** | Multi-model (document, key-value, graph) |
| **Query languages** | AQL, Gremlin, GraphQL |
| **ACID** | Full (read committed isolation) |
| **Consistency** | Strong |
| **License** | Apache 2.0 |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 4496.42ms | 306.6 MB |
| snb_is2 | 5705.91ms | 308.2 MB |
| snb_is3 | 5566.42ms | 328.4 MB |
| snb_is4 | 96.33ms | 337.7 MB |
| snb_is5 | 2205.21ms | 350.0 MB |
| snb_is6 | 6610.82ms | 344.9 MB |
| snb_is7 | 5640.57ms | 348.1 MB |
| **Total** | **30321.68ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | FAILED | |
| snb_ic2 | 9721.13ms | 370.6 MB |
| snb_ic3 | FAILED | |
| snb_ic6 | FAILED | |
| **Total** | **9721.13ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 4496.42ms | 306.6 MB |
| snb_is2 | 5705.91ms | 308.2 MB |
| snb_is3 | 5566.42ms | 328.4 MB |
| snb_is4 | 96.33ms | 337.7 MB |
| snb_is5 | 2205.21ms | 350.0 MB |
| snb_is6 | 6610.82ms | 344.9 MB |
| snb_is7 | 5640.57ms | 348.1 MB |
| snb_ic1 | FAILED | |
| snb_ic2 | 9721.13ms | 370.6 MB |
| snb_ic3 | FAILED | |
| snb_ic6 | FAILED | |
| **Total** | **40042.81ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 44.12ms | 231.9 MB |
| ldbc_pagerank | 4542.07ms | 261.8 MB |
| ldbc_wcc | 4540.06ms | 282.6 MB |
| ldbc_cdlp | 4539.13ms | 269.5 MB |
| ldbc_lcc | 4537.38ms | 283.1 MB |
| ldbc_sssp | 4536.03ms | 281.4 MB |
| **Total** | **22738.79ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 44.12ms | 231.9 MB |
| ldbc_pagerank | 4542.07ms | 261.8 MB |
| ldbc_wcc | 4540.06ms | 282.6 MB |
| ldbc_cdlp | 4539.13ms | 269.5 MB |
| ldbc_lcc | 4537.38ms | 283.1 MB |
| ldbc_sssp | 4536.03ms | 281.4 MB |
| **Total** | **22738.79ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 88.57ms | 367.9 MB |
| acid_atomicity_rb | 132.74ms | 382.3 MB |
| **Total** | **221.31ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | 132.89ms | 372.9 MB |
| acid_g1a | 140.90ms | 348.5 MB |
| acid_g1b | 132.81ms | 352.9 MB |
| acid_g1c | 133.04ms | 351.4 MB |
| acid_imp | 88.47ms | 351.0 MB |
| acid_pmp | 266.86ms | 349.2 MB |
| acid_otv | 177.61ms | 350.0 MB |
| acid_fr | 176.23ms | 352.3 MB |
| acid_lu | 502.62ms | 364.4 MB |
| acid_ws | 137.71ms | 369.3 MB |
| **Total** | **1889.14ms** | |

</details>

---

## Query Languages & Data Models

| |Grafeo Server | Neo4j | Memgraph | ArangoDB |
|---|:---:|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ | ✅ |
| **RDF** | | | | |
| **GQL (ISO)** | | | | |
| **Cypher** | | ✅ | ✅ | |
| **Gremlin** | | | | ✅ |
| **GraphQL** | | | | ✅ |
| **SPARQL** | | | | |
| **AQL** | | | | ✅ |
| **nGQL** | | | | |

---

## Native Algorithm Support

| |Grafeo Server | Neo4j | Memgraph |
|---|:---:|:---:|:---:|
| BFS | ✅ | ✅ | ✅ |
| PageRank | ✅ | ✅ | ✅ |
| WCC | ✅ | ✅ | ✅ |
| CDLP | ✅ | ✅ | ✅ |
| LCC | ✅ | | ✅ |
| SSSP | ✅ | ✅ | ✅ |

---

## Methodology

- **Warmup:** 3 runs discarded before measurement
- **Iterations:** 10 measured runs, median reported
- **Isolation:** Each database gets a clean dataset load before benchmarking
- **Timeout:** 600 seconds per benchmark; exceeded shown as `T/O`

---

## Raw Data

- Results: [`results/bench_20260223_161354.json`](results/bench_20260223_161354.json)
