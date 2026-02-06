# Benchmark Results

## Test Matrix

| Scale | Nodes | Edges | Timeout | SNB Interactive | LDBC ACID | Graph Analytics |
|-------|------:|------:|--------:|-----------------|-----------|-----------------|
| sf01 | 1K | 18K | 60s | All databases | All databases | Native only¹ |
| sf1 | 10K | 180K | 120s | Top 5² | Top 5² | Native only¹ |
| sf100 | 280K | 18M | 600s | Top 2³ | Top 2³ | — |

¹ Graph Analytics for databases with native implementations: Grafeo, Memgraph, Neo4j
² Top 5 by sf01 total time: Grafeo, NebulaGraph, LadybugDB, FalkorDB, Memgraph
³ Top 2 by sf1 total time (ACID-compliant only): Grafeo, LadybugDB

All times in milliseconds. Best result per benchmark in **bold**. Memory in MB.

---

## Summary

### SNB Interactive (total ms)

| Database | Type | sf01 | sf1 | sf100 |
|----------|------|-----:|----:|------:|
| **Grafeo** | Embedded | **3.4** | **245** | **149** |
| LadybugDB | Embedded | 1,426 | — | — |
| FalkorDB | Server | 1,795 | — | — |
| NebulaGraph² | Distributed | 167 | 12,876 | — |
| TuGraph⁴ | Server | 3,401⁴ | — | — |
| Memgraph | Server | 2,530 | 3,169³ | — |
| Neo4j | Server | 3,406 | — | — |
| ArangoDB | Server | 63,747 | — | — |

### Graph Analytics (native only, total ms)

| Database | Type | sf01 | sf1 |
|----------|------|-----:|----:|
| **Grafeo** | Embedded | **4.4** | **5.8** |
| Memgraph | Server | 108 | 260 |
| Neo4j | Server | 304 | 260¹ |

¹ Neo4j LCC failed at sf1 (requires UNDIRECTED relationships)

### LDBC ACID (total ms)

| Database | Type | sf01 | sf1 | Notes |
|----------|------|-----:|----:|-------|
| **Grafeo** | Embedded | **39** | **39** | |
| LadybugDB | Embedded | 106 | 112 | |
| FalkorDB | Server | 90 | 91 | |
| NebulaGraph | Distributed | N/A | — | Eventual consistency |
| TuGraph | Server | 120 | — | |
| Memgraph | Server | ❌ | ❌ | G0, LU failures |
| Neo4j | Server | 237 | — | |
| ArangoDB | Server | 2,123 | — | |

² NebulaGraph uses eventual consistency (`replica_factor=1`), not comparable to ACID databases.
³ Memgraph sf1: 3 timeouts (IS6, IC1, IC3). Total excludes timed-out benchmarks.
⁴ TuGraph IC1 timed out. Total excludes timed-out benchmarks.

### Reading the results

- **Embedded vs. server.** Grafeo and LadybugDB run in-process, so no network overhead. Server databases pay ~0.1–1ms per round-trip.
- **Consistency model.** NebulaGraph uses eventual consistency. Its speeds are not comparable to ACID databases.
- **Memory model.** Memgraph is in-memory first. FalkorDB inherits Redis persistence semantics.
- **Scale factor.** sf01 (1K nodes) fits in cache. sf1/sf100 reveal storage engine and query planner differences.

---

## Per-Database Results

<details>
<summary><h3>Grafeo</h3> Embedded (Rust) | LPG + RDF | GQL, Cypher, Gremlin, GraphQL, SPARQL | Full ACID</summary>

**Why fast:** Columnar storage • Vectorized execution • Lock-free reads • Worst-case optimal joins • Query caching • Zone maps

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 1.11ms | 0.93ms | 0.88ms |
| IS2 — recent posts | — | 6.71ms | 5.6ms |
| IS3 — friends | 0.62ms | 14.46ms | 11.9ms |
| IS4 — content | — | 0.69ms | 0.63ms |
| IS5 — creator posts | — | 48.43ms | 29.8ms |
| IS6 — forum | — | 13.03ms | 5.9ms |
| IS7 — replies | — | 41.66ms | 22.7ms |
| IC1 — friends 3-hop | 0.25ms | 14.48ms | 8.6ms |
| IC2 — recent messages | 0.36ms | 39.01ms | 19.6ms |
| IC3 — friends in cities | 0.44ms | 13.91ms | 15.1ms |
| IC6 — tag co-occurrence | 0.62ms | 52.33ms | 30.9ms |
| *Total* | *3.4ms* | *245ms* | *149ms* |
| *Memory* | *337MB* | *1,339MB* | *1,320MB* |
| **Graph Analytics** ||||
| BFS | 0.12ms | 0.03ms | — |
| PageRank | 0.25ms | 0.61ms | — |
| WCC | 0.55ms | 1.06ms | — |
| CDLP | 0.51ms | 1.05ms | — |
| LCC | 0.31ms | 0.85ms | — |
| SSSP | 2.69ms | 2.17ms | — |
| *Total* | *4.4ms* | *5.8ms* | — |
| *Memory* | *79MB* | *93MB* | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 0.06ms | — | — |
| Atomicity-RB | ✅ 0.04ms | — | — |
| G0 (dirty write) | ✅ 1.8ms | — | — |
| G1a-c (read anomalies) | ✅ 7.2ms | — | — |
| LU (lost update) | ✅ 1.6ms | — | — |
| WS (write skew) | ✅ 0.6ms | — | — |
| *Total* | *39ms* | — | — |

</details>

<details>
<summary><h3>NebulaGraph</h3> Distributed (Thrift) | LPG | nGQL | Eventual consistency</summary>

Results use `replica_factor=1`. Writes are async, data may not be immediately durable.

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 69ms | 131ms | — |
| IS2 — recent posts | — | 197ms | — |
| IS3 — friends | 34ms | 191ms | — |
| IS4 — content | — | 132ms | — |
| IS5 — creator posts | — | 516ms | — |
| IS6 — forum | — | 134ms | — |
| IS7 — replies | — | 586ms | — |
| IC1 — friends 3-hop | 7ms | 8,293ms | — |
| IC2 — recent posts | 15ms | 392ms | — |
| IC3 — friends in cities | 14ms | 1,929ms | — |
| IC6 — tag co-occurrence | 28ms | 375ms | — |
| *Total* | *167ms* | *12,876ms* | — |
| *Memory* | *154MB* | *390MB* | — |
| **LDBC ACID** ||||
| *Result* | N/A | — | — |

</details>

<details>
<summary><h3>LadybugDB</h3> Embedded | LPG | Cypher | Full ACID</summary>

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 30ms | — | — |
| IS3 — friends | 51ms | — | — |
| IC1 — friends 3-hop | 1,041ms | — | — |
| IC2 — recent posts | 20ms | — | — |
| IC3 — friends in cities | 184ms | — | — |
| IC6 — tag co-occurrence | 100ms | — | — |
| *Total* | *1,426ms* | — | — |
| *Memory* | — | — | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 1.6ms | — | — |
| Atomicity-RB | ✅ 1.6ms | — | — |
| G0 (dirty write) | ✅ 4.9ms | — | — |
| G1a-c (read anomalies) | ✅ 15.8ms | — | — |
| LU (lost update) | ✅ 36.6ms | — | — |
| WS (write skew) | ✅ 4.7ms | — | — |
| *Total* | *106ms* | — | — |

</details>

<details>
<summary><h3>FalkorDB</h3> Server (Redis) | LPG | Cypher | Partial ACID</summary>

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 116ms | — | — |
| IS3 — friends | 60ms | — | — |
| IC1 — friends 3-hop | 1,203ms | — | — |
| IC2 — recent posts | 24ms | — | — |
| IC3 — friends in cities | 254ms | — | — |
| IC6 — tag co-occurrence | 138ms | — | — |
| *Total* | *1,795ms* | — | — |
| *Memory* | *157MB* | — | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 2.3ms | — | — |
| Atomicity-RB | ✅ 2.7ms | — | — |
| G0 (dirty write) | ✅ 4.5ms | — | — |
| G1a-c (read anomalies) | ✅ 16.2ms | — | — |
| LU (lost update) | ✅ 15.9ms | — | — |
| WS (write skew) | ✅ 5.1ms | — | — |
| *Total* | *90ms* | — | — |

</details>

<details>
<summary><h3>Memgraph</h3> Server (Bolt) | LPG | Cypher | ACID (conflicts not auto-retried)</summary>

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 161ms | 164ms | — |
| IS2 — recent posts | — | 302ms | — |
| IS3 — friends | 84ms | 33ms | — |
| IS4 — content | — | 56ms | — |
| IS5 — creator posts | — | 773ms | — |
| IS6 — forum | — | ⏱️ | — |
| IS7 — replies | — | 795ms | — |
| IC1 — friends 3-hop | 1,704ms | ⏱️ | — |
| IC2 — recent messages | 34ms | 529ms | — |
| IC3 — friends in cities | 360ms | ⏱️ | — |
| IC6 — tag co-occurrence | 187ms | 517ms | — |
| *Total* | *2,530ms* | *3,169ms*¹ | — |
| *Memory* | *627MB* | *682MB* | — |
| **Graph Analytics** ||||
| BFS | 5.9ms | 2.3ms | — |
| PageRank | 13.7ms | 35.7ms | — |
| WCC | 13.1ms | 36.4ms | — |
| CDLP | 15.7ms | 41.1ms | — |
| LCC | 53.1ms | 141ms | — |
| SSSP | 6.4ms | 3.2ms | — |
| *Total* | *108ms* | *260ms* | — |
| *Memory* | *631MB* | *630MB* | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 2.3ms | ✅ 3.3ms | — |
| Atomicity-RB | ✅ 2.7ms | ✅ 4.0ms | — |
| G0 (dirty write) | ❌ | ❌ | — |
| G1a-c (read anomalies) | ✅ 18.6ms | ✅ 24.2ms | — |
| LU (lost update) | ❌ | ❌ | — |
| WS (write skew) | ✅ 7.5ms | ✅ 6.2ms | — |
| *Total* | ❌ | ❌ | — |

¹ 3 timeouts at sf1: IS6, IC1, IC3. Total excludes timed-out benchmarks.

</details>

<details>
<summary><h3>Neo4j</h3> Server (Bolt) | LPG | Cypher | Full ACID</summary>

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 223ms | — | — |
| IS3 — friends | 117ms | — | — |
| IC1 — friends 3-hop | 2,265ms | — | — |
| IC2 — recent posts | 44ms | — | — |
| IC3 — friends in cities | 496ms | — | — |
| IC6 — tag co-occurrence | 261ms | — | — |
| *Total* | *3,406ms* | — | — |
| *Memory* | — | — | — |
| **Graph Analytics** ||||
| BFS | 115ms | 19.6ms | — |
| PageRank | 64ms | 91.9ms | — |
| WCC | 30ms | 56.9ms | — |
| CDLP | 43ms | 76.2ms | — |
| LCC | — | ❌¹ | — |
| SSSP | 52ms | 14.9ms | — |
| *Total* | *304ms* | *260ms*¹ | — |
| *Memory* | *1,960MB* | *2,500MB* | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 8.5ms | — | — |
| Atomicity-RB | ✅ 6.4ms | — | — |
| G0 (dirty write) | ✅ 17.4ms | — | — |
| G1a-c (read anomalies) | ✅ 48.2ms | — | — |
| LU (lost update) | ✅ 55.3ms | — | — |
| WS (write skew) | ✅ 17.5ms | — | — |
| *Total* | *237ms* | — | — |

¹ LCC failed: requires UNDIRECTED relationships

</details>

<details>
<summary><h3>ArangoDB</h3> Server (HTTP) | Multi-model | AQL | Full ACID</summary>

| Benchmark | sf01 | sf1 | sf100 |
|-----------|-----:|----:|------:|
| **SNB Interactive** ||||
| IS1 — profile lookup | 94ms | — | — |
| IS3 — friends | 2,207ms | — | — |
| IC1 — friends 3-hop | 46,449ms | — | — |
| IC2 — recent posts | 879ms | — | — |
| IC3 — friends in cities | 9,046ms | — | — |
| IC6 — tag co-occurrence | 5,072ms | — | — |
| *Total* | *63,747ms* | — | — |
| *Memory* | *433MB* | — | — |
| **LDBC ACID** ||||
| Atomicity-C | ✅ 89ms | — | — |
| Atomicity-RB | ✅ 89ms | — | — |
| G0 (dirty write) | ✅ 170ms | — | — |
| G1a-c (read anomalies) | ✅ 411ms | — | — |
| LU (lost update) | ✅ 518ms | — | — |
| WS (write skew) | ✅ 138ms | — | — |
| *Total* | *2,123ms* | — | — |

</details>

<details>
<summary><h3>TuGraph</h3> Server (Bolt) | LPG | Cypher | Full ACID</summary>

**Note:** TuGraph ships 34+ native graph algorithms via stored procedures (`algo.*`), but the current benchmark adapter falls back to NetworkX for graph analytics (native procedure signatures differ). IC1 timed out. PageRank failed (numpy not installed for NetworkX fallback). 5 concurrent benchmarks failed (schema field mismatch).

| Benchmark | sf01 |
|-----------|-----:|
| **SNB Interactive** ||
| IS1 — profile lookup | 81ms |
| IS2 — recent posts | 85ms |
| IS3 — friends | 85ms |
| IS4 — content | 81ms |
| IS5 — creator posts | 281ms |
| IS6 — forum | 83ms |
| IS7 — replies | 669ms |
| IC1 — friends 3-hop | ⏱️ |
| IC2 — recent messages | 455ms |
| IC3 — friends in cities | 1,337ms |
| IC6 — tag co-occurrence | 244ms |
| *Total* | *3,401ms*¹ |
| **Graph Analytics** (NetworkX fallback) ||
| BFS | 86ms |
| PageRank | ❌² |
| WCC | 88ms |
| CDLP | 90ms |
| LCC | 86ms |
| SSSP | 86ms |
| *Total* | *436ms*³ |
| **LDBC ACID** ||
| Atomicity-C | ✅ 2.7ms |
| Atomicity-RB | ✅ 3.3ms |
| G0 (dirty write) | ✅ 5.9ms |
| G1a-c (read anomalies) | ✅ 19.9ms |
| IMP (item-many-preceders) | ✅ 13.6ms |
| PMP (pred-many-preceders) | ✅ 23.7ms |
| OTV (observed txn vanishes) | ✅ 5.1ms |
| FR (fractured read) | ✅ 9.4ms |
| LU (lost update) | ✅ 28.9ms |
| WS (write skew) | ✅ 7.0ms |
| *Total* | *120ms* |

¹ IC1 timed out, excluded from total
² PageRank: numpy not installed for NetworkX fallback
³ Using NetworkX fallback (native stored procedures not yet wired); excludes PageRank

</details>

---

## Methodology

- **Warmup:** 3 runs discarded before measurement (1 for sf100)
- **Iterations:** 10 measured runs, median reported (1 for sf100)
- **Isolation:** Each database gets a clean dataset load before benchmarking
- **Memory:** Peak RSS measured during benchmark execution

> **Ain't nobody got time for that:** sf100 uses 1 warmup + 1 measurement iteration, which I changed after running Grafeo and getting stuck with FalkorDB. At this scale, queries can take minutes per run, making 10 iterations very time consuming for the other databases.

### Hardware used

**Run:** 2026-02-05 | **Platform:** Windows | **CPU:** AMD Ryzen 7 7800X3D | **RAM:** 64GB
**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)


### Benchmark categories

**SNB Interactive** implements a subset of the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/) Interactive workload: profile lookups, friend traversals, multi-hop path queries, temporal filtering, and tag co-occurrence.

**Graph Analytics** implements core algorithms from [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/): BFS, PageRank, WCC (weakly connected components), CDLP (community detection via label propagation), LCC (local clustering coefficient), and SSSP (single-source shortest path).

**LDBC ACID** implements the [LDBC ACID test suite](https://github.com/ldbc/ldbc_acid) for transactional consistency: atomicity tests (commit visibility, rollback correctness) and isolation anomaly detection (G0 dirty write, G1a-c read anomalies, IMP/PMP multi-preceders, OTV observed transaction vanishes, FR fractured read, LU lost update, WS write skew).

### Datasets

| Scale | Source | Nodes | Edges | Description |
|-------|--------|------:|------:|-------------|
| sf01 | LDBC SNB SF0.1 | 10K | 50K | Cache-friendly, baseline comparison |
| sf1 | LDBC SNB SF1 | 10K | 180K | Standard scale, reveals optimizer differences |
| sf100 | LDBC SNB SF100 | 10M | 50M | Production scale, stress test |

---

## Query Languages & Data Models

| | Grafeo | LadybugDB | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RDF** | ✅ | | | | | | | |
| **GQL (ISO)** | ✅ | | | | | | | |
| **Cypher** | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | |
| **Gremlin** | ✅ | | | | | ✅ | | |
| **GraphQL** | ✅ | | | | | ✅ | | |
| **SPARQL** | ✅ | | | | | | | |
| **AQL** | | | | | | ✅ | | |
| **nGQL** | | | | | | | | ✅ |

---

## Native Algorithm Support

| | Grafeo | Neo4j | Memgraph | TuGraph¹ |
|---|:---:|:---:|:---:|:---:|
| BFS | ✅ | ✅ | ✅ | ✅ |
| PageRank | ✅ | ✅ | ✅ | ✅ |
| WCC | ✅ | ✅ | ✅ | ✅ |
| CDLP | ✅ | ✅ | ✅ | ✅ |
| LCC | ✅ | | ✅ | ✅ |
| SSSP | ✅ | ✅ | ✅ | ✅ |

¹ TuGraph ships 34+ native algorithms, but the current benchmark adapter uses NetworkX fallback (native stored procedure signatures differ).

Other databases (LadybugDB, FalkorDB, ArangoDB, NebulaGraph) do not ship native implementations of LDBC Graph Analytics algorithms.
