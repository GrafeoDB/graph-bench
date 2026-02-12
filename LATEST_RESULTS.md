# Benchmark Results

65 benchmarks across 12 categories, run at scale factor sf01 (~10K nodes, ~50K edges). Grafeo passed all 65 benchmarks and placed first in 10 of 12 categories (TuGraph led Vector and Hybrid).

---

## Test Configuration

| | |
|---|---|
| **Scale** | sf01 (1,031 persons, ~10K nodes total, ~50K edges) |
| **Timeout** | 60s per benchmark |
| **Warmup** | 3 iterations discarded |
| **Measured** | 10 iterations, mean reported |
| **Hardware** | AMD Ryzen 7 7800X3D, 64GB RAM, Windows, Python 3.13 |
| **Date** | 2026-02-11 (Grafeo, Neo4j, TuringDB); 2026-02-06 (others) |

---

## Results at a Glance

All times in milliseconds. **Bold** = best in category. \* = some benchmarks failed (total excludes failures). — = all failed.

| Category | Grafeo | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB | TuringDB |
|----------|-------:|----------:|------:|---------:|---------:|--------:|---------:|---------:|
| SNB Interactive (11) | **1,269** | 15,157 | 6,294\* | 2,608\* | 7,410\* | 5,978\* | 24,700\* | —\* |
| Graph Analytics (6) | **0.5** | 380\* | 132 | 312\* | 30 | 466\* | 18,900\* | 27,233 |
| ACID (12) | **41** | 109 | 201 | 94 | 110\* | 118 | 2,131 | 5,589\* |
| Algorithms (6) | **13** | 436\* | 629 | 134\* | 287\* | 539\* | 18,500\* | 25,659 |
| Query (3) | **5.3** | 51 | 99 | 62 | 75 | 355 | 3,961 | 3,968 |
| Read (2) | **2.5** | 41 | 129 | 77 | 108 | 110 | 151 | 4,465 |
| Write (5) | **6.0** | 149 | 517 | 150 | 209 | 636 | 10,385 | 17,886 |
| Traversal (5) | **6.8** | 152 | 241 | 98\* | 147\* | 225 | 4,670 | 8,959 |
| Graph Stats (4) | **5.1** | 125 | 202 | 91 | 150 | 183 | 2,713 | 6,486 |
| Concurrent (5) | **152** | 1,932 | 2,183 | 635 | 833\* | —\* | 13,400\* | —\* |
| Vector (4) | 2,494 | 6,015 | 12,738 | 7,789\* | 3,128\* | **910** | 16,310 | —\* |
| Hybrid (2) | 334 | 712 | 1,557 | 4,476 | 1,724 | **89** | 10,300 | —\* |
| **Pass Rate** | **65/65** | 63/65 | 64/65 | 58/65 | 58/65 | 57/65 | 57/65 | 40/65 |

### Reading the results

- **Embedded vs. Server.** Grafeo and LadybugDB run in-process (no network overhead). Server databases (Neo4j, FalkorDB, Memgraph, TuGraph, ArangoDB, TuringDB) pay ~0.1–3ms per round-trip. TuringDB uses HTTP REST (higher latency than Bolt/Redis protocols).
- **Native algorithms.** Grafeo has native graph analytics. Neo4j uses GDS (Graph Data Science) library. Memgraph uses MAGE plugins. Others fall back to NetworkX in Python.
- **Vector search.** All databases use the brute-force Python fallback (scan + cosine similarity). No native vector indexes are wired yet. TuringDB crashes on vector workloads (server OOM with 128-dim float arrays).
- **Concurrent.** TuGraph and TuringDB failed all 5 concurrent benchmarks. Memgraph failed 2 (lost_update, concurrent_acid). TuringDB's git-like change model doesn't support concurrent writes.
- **TuringDB.** HTTP REST API with OpenCypher. Passed all core graph benchmarks (31/31) but failed concurrent (server crash), vector/hybrid (server OOM), 3 ACID tests (change model limitations), and all SNB (data setup too slow via HTTP).
- **Pass rate.** Failures include timeouts (IC1 at sf01 = 3-hop BFS over all KNOWS edges), missing dependencies (PageRank without scipy), and server crashes.

---

## LDBC SNB Interactive

Subset of the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/) Interactive workload. Times in ms. TuringDB omitted (all SNB benchmarks failed — data setup too slow via HTTP REST).

| Query | Description | Grafeo | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB |
|-------|-------------|-------:|----------:|------:|---------:|---------:|--------:|---------:|
| IS1 | Profile lookup | **4.2** | 150 | 274 | 240 | 325 | 257 | 4,700 |
| IS2 | Recent messages | **3.6** | 158 | 209 | 200 | 261 | 200 | 1,100 |
| IS3 | Friends of person | **53** | 1,100 | 3,338 | FAIL | 3,900 | 3,000 | 6,500 |
| IS4 | Message content | **1.2** | 31 | 90 | 78 | 107 | 85 | 103 |
| IS5 | Creator of message | **1.3** | 57 | 88 | 82 | 107 | 85 | 2,200 |
| IS6 | Forum of message | **4.7** | 154 | 216 | 199 | 273 | 204 | FAIL |
| IS7 | Replies to message | **3.8** | 135 | 153 | 188 | 178 | 145 | FAIL |
| IC1 | Friends 3-hop by name | **1,112** | 12,000 | FAIL | FAIL | FAIL | FAIL | FAIL |
| IC2 | Friends' recent messages | **28** | 372 | 767 | 607 | 894 | 671 | 10,100 |
| IC3 | Friends in countries | **25** | 549 | 611 | 550 | 721 | 679 | FAIL |
| IC6 | Tag co-occurrence | **32** | 444 | 548 | 463 | 618 | 652 | FAIL |
| *Total* | | ***1,269*** | *15,157* | *6,294* | *2,608* | *7,410* | *5,978* | *24,700* |

## LDBC Graph Analytics

Core algorithms from [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/). Times in ms. Grafeo, Neo4j (GDS), and Memgraph (MAGE) use native implementations; others use NetworkX fallback.

| Algorithm | Grafeo | Memgraph | Neo4j | TuringDB | FalkorDB | LadybugDB | TuGraph | ArangoDB |
|-----------|-------:|---------:|------:|---------:|---------:|----------:|--------:|---------:|
| BFS | **0.03** | 1.3 | 14.4 | 4,539 | 61 | 79 | 94 | 44 |
| PageRank | **0.07** | 8.6 | 43.9 | 4,540 | FAIL | FAIL | FAIL | FAIL |
| WCC | **0.08** | 2.9 | 13.2 | 4,544 | 62 | 76 | 91 | 4,600 |
| CDLP | **0.12** | 6.1 | 28.6 | 4,539 | 63 | 77 | 92 | 5,100 |
| LCC | **0.09** | 7.3 | 20.1 | 4,536 | 63 | 75 | 95 | 4,600 |
| SSSP | **0.13** | 3.2 | 12.2 | 4,536 | 64 | 73 | 94 | 4,600 |
| *Total* | ***0.5*** | *30* | *132* | *27,233* | *312* | *380* | *466* | *18,900* |

## LDBC ACID

[LDBC ACID test suite](https://github.com/ldbc/ldbc_acid) for transactional consistency. Times in ms.

| Test | Grafeo | FalkorDB | LadybugDB | Neo4j | Memgraph | TuGraph | ArangoDB | TuringDB |
|------|-------:|---------:|----------:|------:|---------:|--------:|---------:|---------:|
| Atomicity-C | **0.10** | 1.8 | 1.6 | 6.3 | 2.6 | 2.8 | 89 | 395 |
| Atomicity-RB | **0.09** | 3.1 | 2.2 | 8.3 | 4.0 | 4.4 | 133 | 396 |
| G0 (dirty write) | **2.1** | 4.8 | 5.2 | 14.4 | FAIL | 6.0 | 133 | 573 |
| G1a (aborted read) | **6.5** | 9.9 | 9.8 | 15.7 | 13 | 11 | 137 | FAIL |
| G1b (interm. read) | **0.10** | 2.6 | 2.3 | 10.2 | 3.7 | 3.7 | 132 | 572 |
| G1c (circular info) | **0.74** | 4.5 | 4.3 | 14.7 | 5.3 | 6.3 | 138 | 660 |
| IMP (item-many-prec) | **11** | 13 | 12 | 15.7 | 13 | 14 | 87 | 395 |
| PMP (pred-many-prec) | **11** | 20 | 18 | 28.0 | 22 | 24 | 269 | FAIL |
| OTV (observed txn vanish) | **0.11** | 3.8 | 3.2 | 13.2 | 4.6 | 5.4 | 183 | 792 |
| FR (fractured read) | **5.9** | 8.7 | 7.9 | 13.7 | 9.1 | 9.4 | 178 | 704 |
| LU (lost update) | **2.2** | 17 | 38 | 46.4 | 27 | 25 | 515 | FAIL |
| WS (write skew) | **0.70** | 5.4 | 4.9 | 14.8 | 6.1 | 7.0 | 138 | 527 |
| *Total* | ***41*** | *94* | *109* | *201* | *110* | *118* | *2,131* | *5,589* |
| *Result* | *PASS* | *PASS* | *PASS* | *PASS* | *G0 FAIL* | *PASS* | *PASS* | *3 FAIL* |

---

## Per-Database Details

<details>
<summary><h3>Grafeo</h3> Embedded (Rust) | LPG + RDF | GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ | Full ACID</summary>

**65/65 benchmarks passed.** Native graph analytics engine, columnar storage, vectorized execution, lock-free reads, worst-case optimal joins. Version 0.5.1.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 — profile lookup | 4.2ms | | BFS | 0.03ms |
| IS2 — recent messages | 3.6ms | | PageRank | 0.07ms |
| IS3 — friends | 53ms | | WCC | 0.08ms |
| IS4 — message content | 1.2ms | | CDLP | 0.12ms |
| IS5 — creator | 1.3ms | | LCC | 0.09ms |
| IS6 — forum | 4.7ms | | SSSP | 0.13ms |
| IS7 — replies | 3.8ms | | *GA Total* | *0.5ms* |
| IC1 — friends 3-hop | 1,112ms | | | |
| IC2 — friends' messages | 28ms | | **Algorithms** | |
| IC3 — friends in countries | 25ms | | PageRank | 0.15ms |
| IC6 — tag co-occurrence | 32ms | | Community Detection | 0.13ms |
| *SNB Total* | *1,269ms* | | Betweenness Centrality | 2.0ms |
| | | | Closeness Centrality | 1.3ms |
| **Query** | | | Triangle Count | 7.9ms |
| Aggregation | 2.5ms | | Common Neighbors | 1.7ms |
| Filter (equality) | 2.1ms | | *Algo Total* | *13ms* |
| Filter (range) | 0.77ms | | | |
| *Query Total* | *5.3ms* | | **Vector** | |
| | | | Insert | 60ms |
| **Read** | | | k-NN Search | 173ms |
| Single Read | 0.87ms | | Batch Search (100x) | 2,058ms |
| Batch Read | 1.7ms | | Recall@10 | 203ms |
| *Read Total* | *2.5ms* | | *Vector Total* | *2,494ms* |
| | | | | |
| **Write** | | | **Hybrid** | |
| Node Insertion | 0.51ms | | Graph → Vector | 280ms |
| Edge Insertion | 1.0ms | | Vector → Graph | 54ms |
| Property Update | 0.84ms | | *Hybrid Total* | *334ms* |
| Edge Add (existing nodes) | 0.07ms | | | |
| Mixed Workload | 3.5ms | | **Concurrent** | |
| *Write Total* | *6.0ms* | | Throughput Scaling | 59ms |
| | | | Lost Update | 3.4ms |
| **Traversal** | | | Read-After-Write | 3.7ms |
| 1-hop | 1.4ms | | Mixed | 13ms |
| 2-hop | 1.4ms | | ACID | 72ms |
| BFS | 1.3ms | | *Concurrent Total* | *152ms* |
| DFS | 1.3ms | | | |
| Shortest Path | 1.5ms | | **ACID** | |
| *Traversal Total* | *6.8ms* | | *12/12 PASS* | *41ms* |

</details>

<details>
<summary><h3>TuringDB</h3> Server (HTTP REST) | Column-oriented | OpenCypher | Git-like versioning</summary>

**40/65 benchmarks passed.** C++23 in-memory column-oriented graph database. Passed all core graph benchmarks (31/31). Failed: all SNB (data setup too slow via HTTP), all concurrent (server crash under thread load), all vector/hybrid (server OOM with float arrays), 3 ACID tests (change model limitations).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **Algorithms** | | | **LDBC Graph Analytics** | |
| PageRank | 4,447ms | | BFS | 4,539ms |
| Community Detection | 4,454ms | | PageRank | 4,540ms |
| Betweenness Centrality | 3,071ms | | WCC | 4,544ms |
| Closeness Centrality | 1,586ms | | CDLP | 4,539ms |
| Triangle Count | 9,695ms | | LCC | 4,536ms |
| Common Neighbors | 2,406ms | | SSSP | 4,536ms |
| *Algo Total* | *25,659ms* | | *GA Total* | *27,233ms* |
| | | | | |
| **Query** | | | **Write** | |
| Aggregation | 2,207ms | | Node Insertion | 177ms |
| Filter (equality) | 880ms | | Edge Insertion | 1,629ms |
| Filter (range) | 882ms | | Property Update | 8,810ms |
| *Query Total* | *3,968ms* | | Edge Add (existing) | 222ms |
| | | | Mixed Workload | 7,048ms |
| **Read** | | | *Write Total* | *17,886ms* |
| Single Read | 4,403ms | | | |
| Batch Read | 62ms | | **Traversal** | |
| *Read Total* | *4,465ms* | | 1-hop | 2,202ms |
| | | | 2-hop | 1,760ms |
| **Graph Stats** | | | BFS | 1,307ms |
| Connected Components | 1,453ms | | DFS | 1,679ms |
| Degree Distribution | 1,453ms | | Shortest Path | 2,012ms |
| Graph Density | 86ms | | *Traversal Total* | *8,959ms* |
| Reachability | 3,493ms | | | |
| *Stats Total* | *6,486ms* | | **ACID** | |
| | | | *9/12 (3 FAIL)* | *5,589ms* |
| **SNB Interactive** | FAIL | | **Concurrent** | FAIL |
| **Vector** | FAIL | | **Hybrid** | FAIL |

</details>

<details>
<summary><h3>Neo4j</h3> Server (Bolt) | LPG | Cypher | Native graph analytics (GDS)</summary>

**64/65 benchmarks passed.** Uses GDS (Graph Data Science) library for native graph analytics. Only failure: IC1 (3-hop friend search timeout — same as FalkorDB, Memgraph, TuGraph, ArangoDB). All 12 ACID tests passed. Version 5.23.0.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (native GDS)** | |
| IS1 — profile lookup | 274ms | | BFS | 14.4ms |
| IS2 — recent messages | 209ms | | PageRank | 43.9ms |
| IS3 — friends | 3,338ms | | WCC | 13.2ms |
| IS4 — message content | 90ms | | CDLP | 28.6ms |
| IS5 — creator | 88ms | | LCC | 20.1ms |
| IS6 — forum | 216ms | | SSSP | 12.2ms |
| IS7 — replies | 153ms | | *GA Total* | *132ms* |
| IC1 — friends 3-hop | FAIL | | | |
| IC2 — friends' messages | 767ms | | **Algorithms** | |
| IC3 — friends in countries | 611ms | | PageRank | 44.6ms |
| IC6 — tag co-occurrence | 548ms | | Community Detection | 186ms |
| *SNB Total* | *6,294ms* | | Betweenness Centrality | 20.3ms |
| | | | Closeness Centrality | 43.3ms |
| **Query** | | | Triangle Count | 264ms |
| Aggregation | 42.8ms | | Common Neighbors | 71.7ms |
| Filter (equality) | 19.0ms | | *Algo Total* | *629ms* |
| Filter (range) | 37.0ms | | | |
| *Query Total* | *99ms* | | **Vector** | |
| | | | Insert | 250ms |
| **Read** | | | k-NN Search | 1,007ms |
| Single Read | 89.4ms | | Batch Search (100x) | 10,435ms |
| Batch Read | 39.9ms | | Recall@10 | 1,046ms |
| *Read Total* | *129ms* | | *Vector Total* | *12,738ms* |
| | | | | |
| **Write** | | | **Hybrid** | |
| Node Insertion | 8.0ms | | Graph → Vector | 1,232ms |
| Edge Insertion | 58.3ms | | Vector → Graph | 325ms |
| Property Update | 302ms | | *Hybrid Total* | *1,557ms* |
| Edge Add (existing nodes) | 6.4ms | | | |
| Mixed Workload | 142ms | | **Concurrent** | |
| *Write Total* | *517ms* | | Throughput Scaling | 699ms |
| | | | Lost Update | 114ms |
| **Traversal** | | | Read-After-Write | 131ms |
| 1-hop | 63.5ms | | Mixed | 148ms |
| 2-hop | 50.4ms | | ACID | 1,091ms |
| BFS | 48.5ms | | *Concurrent Total* | *2,183ms* |
| DFS | 60.4ms | | | |
| Shortest Path | 18.6ms | | **ACID** | |
| *Traversal Total* | *241ms* | | *12/12 PASS* | *201ms* |

</details>

<details>
<summary><h3>LadybugDB</h3> Embedded | LPG | Cypher | Full ACID</summary>

**63/65 benchmarks passed.** Failures: PageRank (2x, scipy not installed for NetworkX fallback).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 — profile lookup | 150ms | | BFS | 79ms |
| IS2 — recent messages | 158ms | | PageRank | FAIL |
| IS3 — friends | 1,100ms | | WCC | 76ms |
| IS4 — message content | 31ms | | CDLP | 77ms |
| IS5 — creator | 57ms | | LCC | 75ms |
| IS6 — forum | 154ms | | SSSP | 73ms |
| IS7 — replies | 135ms | | *GA Total* | *380ms* |
| IC1 — friends 3-hop | 12,000ms | | | |
| IC2 — friends' messages | 372ms | | **Algorithms** | |
| IC3 — friends in countries | 549ms | | Community Detection | 95ms |
| IC6 — tag co-occurrence | 444ms | | Betweenness Centrality | 49ms |
| *SNB Total* | *15,157ms* | | Closeness Centrality | 32ms |
| | | | Triangle Count | 203ms |
| **ACID** | | | Common Neighbors | 56ms |
| *12/12 PASS* | *109ms* | | *Algo Total* | *436ms* |

</details>

<details>
<summary><h3>FalkorDB</h3> Server (Redis) | LPG | Cypher | Full ACID</summary>

**58/65 benchmarks passed.** Failures: IS3, IC1 (timeout), PageRank (2x), Shortest Path, Betweenness Centrality, Vector Batch Search.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 — profile lookup | 240ms | | BFS | 61ms |
| IS2 — recent messages | 200ms | | PageRank | FAIL |
| IS3 — friends | FAIL | | WCC | 62ms |
| IS4 — message content | 78ms | | CDLP | 63ms |
| IS5 — creator | 82ms | | LCC | 63ms |
| IS6 — forum | 199ms | | SSSP | 64ms |
| IS7 — replies | 188ms | | *GA Total* | *312ms* |
| IC1 — friends 3-hop | FAIL | | | |
| IC2 — friends' messages | 607ms | | **ACID** | |
| IC3 — friends in countries | 550ms | | *12/12 PASS* | *94ms* |
| IC6 — tag co-occurrence | 463ms | | | |
| *SNB Total* | *2,608ms* | | | |

</details>

<details>
<summary><h3>Memgraph</h3> Server (Bolt) | LPG | Cypher | Native graph analytics (MAGE)</summary>

**58/65 benchmarks passed.** Failures: IC1 (timeout), Betweenness Centrality, Shortest Path, ACID G0 (dirty write), Lost Update, Concurrent ACID, Vector Batch Search.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (native MAGE)** | |
| IS1 — profile lookup | 325ms | | BFS | 1.3ms |
| IS2 — recent messages | 261ms | | PageRank | 8.6ms |
| IS3 — friends | 3,900ms | | WCC | 2.9ms |
| IS4 — message content | 107ms | | CDLP | 6.1ms |
| IS5 — creator | 107ms | | LCC | 7.3ms |
| IS6 — forum | 273ms | | SSSP | 3.2ms |
| IS7 — replies | 178ms | | *GA Total* | *30ms* |
| IC1 — friends 3-hop | FAIL | | | |
| IC2 — friends' messages | 894ms | | **ACID** | |
| IC3 — friends in countries | 721ms | | *11/12 (G0 FAIL)* | *110ms* |
| IC6 — tag co-occurrence | 618ms | | | |
| *SNB Total* | *7,410ms* | | | |

</details>

<details>
<summary><h3>TuGraph</h3> Server (Bolt) | LPG | Cypher | Full ACID</summary>

**57/65 benchmarks passed.** Ships 34+ native algorithms via stored procedures, but the benchmark adapter uses NetworkX fallback (procedure signatures differ). Failures: IC1 (timeout), PageRank (2x, scipy), all 5 concurrent benchmarks (schema field mismatch).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (NetworkX)** | |
| IS1 — profile lookup | 257ms | | BFS | 94ms |
| IS2 — recent messages | 200ms | | PageRank | FAIL |
| IS3 — friends | 3,000ms | | WCC | 91ms |
| IS4 — message content | 85ms | | CDLP | 92ms |
| IS5 — creator | 85ms | | LCC | 95ms |
| IS6 — forum | 204ms | | SSSP | 94ms |
| IS7 — replies | 145ms | | *GA Total* | *466ms* |
| IC1 — friends 3-hop | FAIL | | | |
| IC2 — friends' messages | 671ms | | **ACID** | |
| IC3 — friends in countries | 679ms | | *12/12 PASS* | *118ms* |
| IC6 — tag co-occurrence | 652ms | | | |
| *SNB Total* | *5,978ms* | | | |

</details>

<details>
<summary><h3>ArangoDB</h3> Server (HTTP) | Multi-model | AQL | Full ACID</summary>

**57/65 benchmarks passed.** Failures: IS6, IS7, IC1, IC3, IC6 (timeout), PageRank (2x), Concurrent ACID.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (NetworkX)** | |
| IS1 — profile lookup | 4,700ms | | BFS | 44ms |
| IS2 — recent messages | 1,100ms | | PageRank | FAIL |
| IS3 — friends | 6,500ms | | WCC | 4,600ms |
| IS4 — message content | 103ms | | CDLP | 5,100ms |
| IS5 — creator | 2,200ms | | LCC | 4,600ms |
| IS6 — forum | FAIL | | SSSP | 4,600ms |
| IS7 — replies | FAIL | | *GA Total* | *18,900ms* |
| IC1 — friends 3-hop | FAIL | | | |
| IC2 — friends' messages | 10,100ms | | **ACID** | |
| IC3 — friends in countries | FAIL | | *12/12 PASS* | *2,131ms* |
| IC6 — tag co-occurrence | FAIL | | | |
| *SNB Total* | *24,700ms* | | | |

</details>

---

## Methodology

### Benchmark Categories (65 total)

| Category | Count | Source | Description |
|----------|------:|--------|-------------|
| SNB Interactive | 11 | [LDBC SNB](https://ldbcouncil.org/benchmarks/snb/) | Profile lookups, friend traversals, multi-hop queries, temporal filtering |
| Graph Analytics | 6 | [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/) | BFS, PageRank, WCC, CDLP, LCC, SSSP |
| ACID | 12 | [LDBC ACID](https://github.com/ldbc/ldbc_acid) | Atomicity, isolation anomaly detection (G0–G1c, IMP, PMP, OTV, FR, LU, WS) |
| Algorithms | 6 | Custom | PageRank, community detection, centrality, triangles, common neighbors |
| Query | 3 | Custom | Aggregation, equality filter, range filter |
| Read | 2 | Custom | Single node read, batch read |
| Write | 5 | Custom | Node/edge insertion, property update, mixed workload |
| Traversal | 5 | Custom | 1-hop, 2-hop, BFS, DFS, shortest path |
| Graph Stats | 4 | Custom | Connected components, degree distribution, density, reachability |
| Concurrent | 5 | Custom | Throughput scaling, lost update, read-after-write, mixed, ACID under concurrency |
| Vector | 4 | Custom | Insert, k-NN, batch search, recall@10 (128-dim, cosine, brute-force fallback) |
| Hybrid | 2 | Custom | Graph→vector search, vector→graph expansion |

### Dataset

LDBC SNB-derived social network: Persons, Cities, Tags, Countries, Forums, Posts, Comments, Universities, Companies. Relationships: KNOWS, LIVES_IN, HAS_INTEREST, HAS_CREATOR, REPLY_OF, HAS_TAG, CONTAINER_OF, STUDY_AT, WORK_AT, IS_LOCATED_IN, and reverse helper edges for efficient traversal.

### Fairness

- All adapters create property indexes on `id` during node insertion for fair edge-insertion lookup performance.
- Graph Analytics: databases without native implementations use a NetworkX fallback. The graph extraction + algorithm overhead is included in the measured time.
- Vector: all databases use the same brute-force Python fallback (no native vector indexes wired).
- Embedded databases (Grafeo, LadybugDB) have zero network overhead. Server databases communicate over TCP.

### Query Languages & Data Models

| | Grafeo | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB | TuringDB |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RDF** | ✅ | | | | | | | |
| **GQL (ISO)** | ✅ | | | | | | | |
| **Cypher** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| **Gremlin** | ✅ | | | | | | | |
| **GraphQL** | ✅ | | | | | | | |
| **SPARQL** | ✅ | | | | | | | |
| **SQL/PGQ** | ✅ | | | | | | | |
| **AQL** | | | | | | | ✅ | |

Grafeo supports all listed query languages at the engine level. The benchmark adapter uses Cypher syntax via `execute()`, but performance is identical regardless of query language — they all compile to the same execution plan.

### Native Algorithm Support

| | Grafeo | Neo4j¹ | Memgraph | TuGraph² |
|---|:---:|:---:|:---:|:---:|
| BFS | ✅ | ✅ | ✅ | ✅ |
| PageRank | ✅ | ✅ | ✅ | ✅ |
| WCC | ✅ | ✅ | ✅ | ✅ |
| CDLP | ✅ | ✅ | ✅ | ✅ |
| LCC | ✅ | ✅ | ✅ | ✅ |
| SSSP | ✅ | ✅ | ✅ | ✅ |

¹ Neo4j uses GDS (Graph Data Science) library procedures. The benchmark adapter calls GDS streaming procedures for all 6 LDBC algorithms natively.
² TuGraph ships 34+ native algorithms, but the benchmark adapter uses NetworkX fallback (native stored procedure signatures differ from adapter API).

Other databases (LadybugDB, FalkorDB, ArangoDB, TuringDB) do not ship native implementations of LDBC Graph Analytics algorithms.

**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)
