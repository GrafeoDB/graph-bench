# Benchmark Results

65 benchmarks across 12 categories on 9 databases at scale factor sf01. Grafeo passed 65/65 and placed first in 9 of 12 categories. Grafeo Server led SNB Interactive and Vector. TuGraph led Hybrid.

---

## Test Configuration

| | |
|---|---|
| **Scale** | sf01 (1,031 persons, ~10K nodes total, ~50K edges) |
| **Timeout** | 60s per benchmark |
| **Warmup** | 3 iterations discarded |
| **Measured** | 3 iterations, mean reported |
| **Hardware** | AMD Ryzen 7 7800X3D, 64GB RAM, Windows, Python 3.13 |

---

## Results at a Glance

All times in milliseconds. **Bold** = best in category. \* = some benchmarks failed (total excludes failures). - = not tested.

| Category | Grafeo | Grafeo Server | Memgraph | FalkorDB | Neo4j | TuGraph | LadybugDB | ArangoDB | TuringDB |
|----------|-------:|--------------:|---------:|---------:|------:|--------:|----------:|---------:|---------:|
| SNB Interactive (11) | 876 | **444** | 7,410\* | 2,608\* | 6,294\* | 5,978\* | 15,157 | 24,700\* | -\* |
| Graph Analytics (6) | **1.8** | 16 | 30 | 312\* | 132 | 466\* | 380\* | 18,900\* | 27,233 |
| ACID (12) | **39** | 215 | 110\* | 94 | 201 | 118 | 129 | 2,131 | 5,589\* |
| Algorithms (6) | **13** | 99 | 287\* | 134\* | 629 | 539\* | 436\* | 18,500\* | 25,659 |
| Query (3) | **4.9** | 71 | 75 | 62 | 99 | 355 | 31 | 3,961 | 3,968 |
| Read (2) | **2.4** | 78 | 108 | 77 | 129 | 110 | 49 | 151 | 4,465 |
| Write (5) | **19** | 243 | 209 | 150 | 517 | 636 | 176 | 10,385 | 17,886 |
| Traversal (5) | **6.1** | 87 | 147\* | 98\* | 241 | 225 | 141 | 4,670 | 8,959 |
| Graph Stats (4) | **5.1** | 68 | 150 | 91 | 202 | 183 | 125 | 2,713 | 6,486 |
| Concurrent (5) | **152** | 1,418 | 833\* | 635 | 2,183 | -\* | 1,932 | 13,400\* | -\* |
| Vector (4) | 2,494 | **305** | 3,128\* | 7,789\* | 12,738 | 910 | 6,015 | 16,310 | -\* |
| Hybrid (2) | 334 | 121 | 1,724 | 4,476 | 1,557 | **89** | 712 | 10,300 | -\* |
| **Pass Rate** | **65/65** | **65/65** | 58/65 | 58/65 | 64/65 | 57/65 | 63/65 | 57/65 | 40/65 |

### Notes

- **Embedded vs. Server.** Grafeo and LadybugDB run in-process (no network overhead). Server databases pay ~0.1–3ms per round-trip. Grafeo Server uses GWP (gRPC binary protocol) with ~0.1ms per-query wire overhead.
- **Native algorithms.** Grafeo, Neo4j (GDS), and Memgraph (MAGE) run graph analytics natively. Grafeo Server has the same engine but algorithms are not yet exposed over the wire protocol. All others use a NetworkX fallback in Python - graph extraction overhead is included in measured time.
- **Vector search.** All databases use the same brute-force Python fallback (scan + cosine similarity). No native vector indexes are wired.
- **Edge insertion.** All adapters look up source and target nodes by property via the query engine before creating edges. All adapters create property indexes on `id` for fair lookup performance.
- **Failures.** Common failure causes: IC1 timeout (3-hop BFS over all KNOWS edges), missing scipy (PageRank via NetworkX), server crashes under concurrent load.

---

## LDBC SNB Interactive

Subset of the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/) Interactive workload. Times in ms. TuringDB omitted (all SNB failed).

| Query | Description | Grafeo | Grafeo Server | Memgraph | FalkorDB | Neo4j | TuGraph | LadybugDB | ArangoDB |
|-------|-------------|-------:|--------------:|---------:|---------:|------:|--------:|----------:|---------:|
| IS1 | Profile lookup | **3.5** | 76 | 325 | 240 | 274 | 257 | 150 | 4,700 |
| IS2 | Recent messages | **3.0** | 15 | 261 | 200 | 209 | 200 | 158 | 1,100 |
| IS3 | Friends of person | 48 | **39** | 3,900 | FAIL | 3,338 | 3,000 | 1,100 | 6,500 |
| IS4 | Message content | **1.1** | 76 | 107 | 78 | 90 | 85 | 31 | 103 |
| IS5 | Creator of message | **1.3** | 42 | 107 | 82 | 88 | 85 | 57 | 2,200 |
| IS6 | Forum of message | **3.2** | 76 | 273 | 199 | 216 | 204 | 154 | FAIL |
| IS7 | Replies to message | **3.6** | 45 | 178 | 188 | 153 | 145 | 135 | FAIL |
| IC1 | Friends 3-hop by name | 758 | **16** | FAIL | FAIL | FAIL | FAIL | 12,000 | FAIL |
| IC2 | Friends' recent messages | 18 | **8.5** | 894 | 607 | 767 | 671 | 372 | 10,100 |
| IC3 | Friends in countries | 15 | **3.9** | 721 | 550 | 611 | 679 | 549 | FAIL |
| IC6 | Tag co-occurrence | **21** | 46 | 618 | 463 | 548 | 652 | 444 | FAIL |
| *Total* | | *876* | ***444*** | *7,410* | *2,608* | *6,294* | *5,978* | *15,157* | *24,700* |

## LDBC Graph Analytics

Core algorithms from [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/). Times in ms. Grafeo, Neo4j (GDS), and Memgraph (MAGE) use native implementations; others use NetworkX fallback.

| Algorithm | Grafeo | Grafeo Server | Memgraph | FalkorDB | Neo4j | TuGraph | LadybugDB | ArangoDB | TuringDB |
|-----------|-------:|--------------:|---------:|---------:|------:|--------:|----------:|---------:|---------:|
| BFS | **0.03** | 2.7 | 1.3 | 61 | 14.4 | 94 | 79 | 44 | 4,539 |
| PageRank | **0.04** | 2.7 | 8.6 | FAIL | 43.9 | FAIL | FAIL | FAIL | 4,540 |
| WCC | **0.05** | 2.5 | 2.9 | 62 | 13.2 | 91 | 76 | 4,600 | 4,544 |
| CDLP | **0.08** | 2.6 | 6.1 | 63 | 28.6 | 92 | 77 | 5,100 | 4,539 |
| LCC | **0.06** | 2.6 | 7.3 | 63 | 20.1 | 95 | 75 | 4,600 | 4,536 |
| SSSP | **1.5** | 2.6 | 3.2 | 64 | 12.2 | 94 | 73 | 4,600 | 4,536 |
| *Total* | ***1.8*** | *16* | *30* | *312* | *132* | *466* | *380* | *18,900* | *27,233* |

## LDBC ACID

[LDBC ACID test suite](https://github.com/ldbc/ldbc_acid) for transactional consistency. Times in ms.

| Test | Grafeo | Grafeo Server | Memgraph | FalkorDB | Neo4j | TuGraph | LadybugDB | ArangoDB | TuringDB |
|------|-------:|--------------:|---------:|---------:|------:|--------:|----------:|---------:|---------:|
| Atomicity-C | **0.05** | 5.5 | 2.6 | 1.8 | 6.3 | 2.8 | 2.0 | 89 | 395 |
| Atomicity-RB | **0.07** | 5.4 | 4.0 | 3.1 | 8.3 | 4.4 | 2.7 | 133 | 396 |
| G0 (dirty write) | **2.0** | 9.7 | FAIL | 4.8 | 14.4 | 6.0 | 5.7 | 133 | 573 |
| G1a (aborted read) | **6.4** | 26 | 13 | 9.9 | 15.7 | 11 | 10 | 137 | FAIL |
| G1b (interm. read) | **0.07** | 7.5 | 3.7 | 2.6 | 10.2 | 3.7 | 2.7 | 132 | 572 |
| G1c (circular info) | **0.53** | 9.3 | 5.3 | 4.5 | 14.7 | 6.3 | 5.3 | 138 | 660 |
| IMP (item-many-prec) | **11** | 15 | 13 | 13 | 15.7 | 14 | 13 | 87 | 395 |
| PMP (pred-many-prec) | **11** | 29 | 22 | 20 | 28.0 | 24 | 20 | 269 | FAIL |
| OTV (observed txn vanish) | **0.07** | 9.4 | 4.6 | 3.8 | 13.2 | 5.4 | 3.7 | 183 | 792 |
| FR (fractured read) | **5.6** | 11 | 9.1 | 8.7 | 13.7 | 9.4 | 8.5 | 178 | 704 |
| LU (lost update) | **1.7** | 78 | 27 | 17 | 46.4 | 25 | 50 | 515 | FAIL |
| WS (write skew) | **0.58** | 8.5 | 6.1 | 5.4 | 14.8 | 7.0 | 6.1 | 138 | 527 |
| *Total* | ***39*** | *215* | *110* | *94* | *201* | *118* | *129* | *2,131* | *5,589* |
| *Result* | *PASS* | *PASS* | *G0 FAIL* | *PASS* | *PASS* | *PASS* | *PASS* | *PASS* | *3 FAIL* |

---

## Per-Database Details

<details>
<summary><h3>Grafeo</h3> Embedded (Rust) | LPG + RDF | GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ | Full ACID</summary>

**65/65 passed.** v0.5.1. Native graph analytics (BFS, PageRank, WCC, CDLP, LCC). SSSP uses NetworkX fallback. Edge insertion uses GQL MATCH for node lookup + `create_edge()`.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 - profile lookup | 3.5ms | | BFS | 0.03ms |
| IS2 - recent messages | 3.0ms | | PageRank | 0.04ms |
| IS3 - friends | 48ms | | WCC | 0.05ms |
| IS4 - message content | 1.1ms | | CDLP | 0.08ms |
| IS5 - creator | 1.3ms | | LCC | 0.06ms |
| IS6 - forum | 3.2ms | | SSSP | 1.5ms |
| IS7 - replies | 3.6ms | | *GA Total* | *1.8ms* |
| IC1 - friends 3-hop | 758ms | | | |
| IC2 - friends' messages | 18ms | | **Algorithms** | |
| IC3 - friends in countries | 15ms | | PageRank | 0.15ms |
| IC6 - tag co-occurrence | 21ms | | Community Detection | 0.13ms |
| *SNB Total* | *876ms* | | Betweenness Centrality | 2.0ms |
| | | | Closeness Centrality | 1.3ms |
| **Query** | | | Triangle Count | 7.9ms |
| Aggregation | 2.2ms | | Common Neighbors | 1.7ms |
| Filter (equality) | 2.0ms | | *Algo Total* | *13ms* |
| Filter (range) | 0.74ms | | | |
| *Query Total* | *4.9ms* | | **Vector** | |
| | | | Insert | 60ms |
| **Read** | | | k-NN Search | 173ms |
| Single Read | 0.82ms | | Batch Search (100x) | 2,058ms |
| Batch Read | 1.6ms | | Recall@10 | 203ms |
| *Read Total* | *2.4ms* | | *Vector Total* | *2,494ms* |
| | | | | |
| **Write** | | | **Hybrid** | |
| Node Insertion | 0.55ms | | Graph → Vector | 280ms |
| Edge Insertion | 16ms | | Vector → Graph | 54ms |
| Property Update | 0.43ms | | *Hybrid Total* | *334ms* |
| Edge Add (existing nodes) | 0.54ms | | | |
| Mixed Workload | 1.6ms | | **Concurrent** | |
| *Write Total* | *19ms* | | Throughput Scaling | 59ms |
| | | | Lost Update | 3.4ms |
| **Traversal** | | | Read-After-Write | 3.7ms |
| 1-hop | 1.3ms | | Mixed | 13ms |
| 2-hop | 1.4ms | | ACID | 72ms |
| BFS | 1.1ms | | *Concurrent Total* | *152ms* |
| DFS | 1.3ms | | | |
| Shortest Path | 1.1ms | | **ACID** | |
| *Traversal Total* | *6.1ms* | | *12/12 PASS* | *39ms* |

</details>

<details>
<summary><h3>Grafeo Server</h3> Server (GWP/gRPC) | LPG | GQL | Full ACID</summary>

**65/65 passed.** Same Grafeo engine (v0.5.1) over GWP (GQL Wire Protocol, gRPC) via grafeo-server:lite. Graph analytics use NetworkX fallback (algorithms not yet exposed over the wire protocol).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (NetworkX)** | |
| IS1 - profile lookup | 76ms | | BFS | 2.7ms |
| IS2 - recent messages | 15ms | | PageRank | 2.7ms |
| IS3 - friends | 39ms | | WCC | 2.5ms |
| IS4 - message content | 76ms | | CDLP | 2.6ms |
| IS5 - creator | 42ms | | LCC | 2.6ms |
| IS6 - forum | 76ms | | SSSP | 2.6ms |
| IS7 - replies | 45ms | | *GA Total* | *16ms* |
| IC1 - friends 3-hop | 16ms | | | |
| IC2 - friends' messages | 8.5ms | | **Algorithms (NetworkX)** | |
| IC3 - friends in countries | 3.9ms | | PageRank | 2.6ms |
| IC6 - tag co-occurrence | 46ms | | Community Detection | 2.6ms |
| *SNB Total* | *444ms* | | Betweenness Centrality | 26ms |
| | | | Closeness Centrality | 7.8ms |
| **Query** | | | Triangle Count | 16ms |
| Aggregation | 40ms | | Common Neighbors | 45ms |
| Filter (equality) | 16ms | | *Algo Total* | *99ms* |
| Filter (range) | 15ms | | | |
| *Query Total* | *71ms* | | **Vector** | |
| | | | Insert | 173ms |
| **Read** | | | k-NN Search | 8.3ms |
| Single Read | 77ms | | Batch Search (100x) | 115ms |
| Batch Read | 0.93ms | | Recall@10 | 8.7ms |
| *Read Total* | *78ms* | | *Vector Total* | *305ms* |
| | | | | |
| **Write** | | | **Hybrid** | |
| Node Insertion | 2.7ms | | Graph → Vector | 111ms |
| Edge Insertion | 12ms | | Vector → Graph | 10ms |
| Property Update | 124ms | | *Hybrid Total* | *121ms* |
| Edge Add (existing nodes) | 2.5ms | | | |
| Mixed Workload | 102ms | | **Concurrent** | |
| *Write Total* | *243ms* | | Throughput Scaling | 343ms |
| | | | Lost Update | 76ms |
| **Traversal** | | | Read-After-Write | 109ms |
| 1-hop | 40ms | | Mixed | 261ms |
| 2-hop | 16ms | | ACID | 629ms |
| BFS | 8.2ms | | *Concurrent Total* | *1,418ms* |
| DFS | 8.3ms | | | |
| Shortest Path | 15ms | | **Graph Stats** | |
| *Traversal Total* | *87ms* | | Connected Components | 25ms |
| | | | Degree Distribution | 25ms |
| | | | Graph Density | 1.8ms |
| | | | Reachability | 16ms |
| | | | *Stats Total* | *68ms* |
| | | | | |
| | | | **ACID** | |
| | | | *12/12 PASS* | *215ms* |

</details>

<details>
<summary><h3>Memgraph</h3> Server (Bolt) | LPG | Cypher | Native graph analytics (MAGE)</summary>

**58/65 passed.** Failures: IC1 (timeout), Betweenness Centrality, Shortest Path, ACID G0 (dirty write), Lost Update, Concurrent ACID, Vector Batch Search.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (native MAGE)** | |
| IS1 - profile lookup | 325ms | | BFS | 1.3ms |
| IS2 - recent messages | 261ms | | PageRank | 8.6ms |
| IS3 - friends | 3,900ms | | WCC | 2.9ms |
| IS4 - message content | 107ms | | CDLP | 6.1ms |
| IS5 - creator | 107ms | | LCC | 7.3ms |
| IS6 - forum | 273ms | | SSSP | 3.2ms |
| IS7 - replies | 178ms | | *GA Total* | *30ms* |
| IC1 - friends 3-hop | FAIL | | | |
| IC2 - friends' messages | 894ms | | **ACID** | |
| IC3 - friends in countries | 721ms | | *11/12 (G0 FAIL)* | *110ms* |
| IC6 - tag co-occurrence | 618ms | | | |
| *SNB Total* | *7,410ms* | | | |

</details>

<details>
<summary><h3>FalkorDB</h3> Server (Redis) | LPG | Cypher | Full ACID</summary>

**58/65 passed.** Failures: IS3, IC1 (timeout), PageRank (2x), Shortest Path, Betweenness Centrality, Vector Batch Search.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 - profile lookup | 240ms | | BFS | 61ms |
| IS2 - recent messages | 200ms | | PageRank | FAIL |
| IS3 - friends | FAIL | | WCC | 62ms |
| IS4 - message content | 78ms | | CDLP | 63ms |
| IS5 - creator | 82ms | | LCC | 63ms |
| IS6 - forum | 199ms | | SSSP | 64ms |
| IS7 - replies | 188ms | | *GA Total* | *312ms* |
| IC1 - friends 3-hop | FAIL | | | |
| IC2 - friends' messages | 607ms | | **ACID** | |
| IC3 - friends in countries | 550ms | | *12/12 PASS* | *94ms* |
| IC6 - tag co-occurrence | 463ms | | | |
| *SNB Total* | *2,608ms* | | | |

</details>

<details>
<summary><h3>Neo4j</h3> Server (Bolt) | LPG | Cypher | Native graph analytics (GDS)</summary>

**64/65 passed.** v5.23.0. Uses GDS library for native graph analytics. Only failure: IC1 (timeout).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (native GDS)** | |
| IS1 - profile lookup | 274ms | | BFS | 14.4ms |
| IS2 - recent messages | 209ms | | PageRank | 43.9ms |
| IS3 - friends | 3,338ms | | WCC | 13.2ms |
| IS4 - message content | 90ms | | CDLP | 28.6ms |
| IS5 - creator | 88ms | | LCC | 20.1ms |
| IS6 - forum | 216ms | | SSSP | 12.2ms |
| IS7 - replies | 153ms | | *GA Total* | *132ms* |
| IC1 - friends 3-hop | FAIL | | | |
| IC2 - friends' messages | 767ms | | **Algorithms** | |
| IC3 - friends in countries | 611ms | | PageRank | 44.6ms |
| IC6 - tag co-occurrence | 548ms | | Community Detection | 186ms |
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
<summary><h3>TuGraph</h3> Server (Bolt) | LPG | Cypher | Full ACID</summary>

**57/65 passed.** 34+ native algorithms via stored procedures (not wired to benchmark adapter - uses NetworkX fallback). Failures: IC1 (timeout), PageRank (2x, scipy), all 5 concurrent (schema field mismatch).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (NetworkX)** | |
| IS1 - profile lookup | 257ms | | BFS | 94ms |
| IS2 - recent messages | 200ms | | PageRank | FAIL |
| IS3 - friends | 3,000ms | | WCC | 91ms |
| IS4 - message content | 85ms | | CDLP | 92ms |
| IS5 - creator | 85ms | | LCC | 95ms |
| IS6 - forum | 204ms | | SSSP | 94ms |
| IS7 - replies | 145ms | | *GA Total* | *466ms* |
| IC1 - friends 3-hop | FAIL | | | |
| IC2 - friends' messages | 671ms | | **ACID** | |
| IC3 - friends in countries | 679ms | | *12/12 PASS* | *118ms* |
| IC6 - tag co-occurrence | 652ms | | | |
| *SNB Total* | *5,978ms* | | | |

</details>

<details>
<summary><h3>LadybugDB</h3> Embedded | LPG | Cypher | Full ACID</summary>

**63/65 passed.** Failures: PageRank (2x, scipy not installed). Uses dedicated columns for common properties with JSON overflow for uncommon fields.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 - profile lookup | 150ms | | BFS | 79ms |
| IS2 - recent messages | 158ms | | PageRank | FAIL |
| IS3 - friends | 1,100ms | | WCC | 76ms |
| IS4 - message content | 31ms | | CDLP | 77ms |
| IS5 - creator | 57ms | | LCC | 75ms |
| IS6 - forum | 154ms | | SSSP | 73ms |
| IS7 - replies | 135ms | | *GA Total* | *380ms* |
| IC1 - friends 3-hop | 12,000ms | | | |
| IC2 - friends' messages | 372ms | | **Algorithms** | |
| IC3 - friends in countries | 549ms | | Community Detection | 95ms |
| IC6 - tag co-occurrence | 444ms | | Betweenness Centrality | 49ms |
| *SNB Total* | *15,157ms* | | Closeness Centrality | 32ms |
| | | | Triangle Count | 203ms |
| **ACID** | | | Common Neighbors | 56ms |
| *12/12 PASS* | *129ms* | | *Algo Total* | *436ms* |
| | | | | |
| **Query** | | | **Read** | |
| Aggregation | 27ms | | Single Read | 46ms |
| Filter (equality) | 1.5ms | | Batch Read | 2.4ms |
| Filter (range) | 2.3ms | | *Read Total* | *49ms* |
| *Query Total* | *31ms* | | | |
| | | | **Write** | |
| **Traversal** | | | Node Insertion | 48ms |
| 1-hop | 35ms | | Edge Insertion | 17ms |
| 2-hop | 32ms | | Property Update | 29ms |
| BFS | 22ms | | Edge Add (existing) | 2.0ms |
| DFS | 27ms | | Mixed Workload | 79ms |
| Shortest Path | 25ms | | *Write Total* | *176ms* |
| *Traversal Total* | *141ms* | | | |

</details>

<details>
<summary><h3>ArangoDB</h3> Server (HTTP) | Multi-model | AQL | Full ACID</summary>

**57/65 passed.** Failures: IS6, IS7, IC1, IC3, IC6 (timeout), PageRank (2x), Concurrent ACID.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics (NetworkX)** | |
| IS1 - profile lookup | 4,700ms | | BFS | 44ms |
| IS2 - recent messages | 1,100ms | | PageRank | FAIL |
| IS3 - friends | 6,500ms | | WCC | 4,600ms |
| IS4 - message content | 103ms | | CDLP | 5,100ms |
| IS5 - creator | 2,200ms | | LCC | 4,600ms |
| IS6 - forum | FAIL | | SSSP | 4,600ms |
| IS7 - replies | FAIL | | *GA Total* | *18,900ms* |
| IC1 - friends 3-hop | FAIL | | | |
| IC2 - friends' messages | 10,100ms | | **ACID** | |
| IC3 - friends in countries | FAIL | | *12/12 PASS* | *2,131ms* |
| IC6 - tag co-occurrence | FAIL | | | |
| *SNB Total* | *24,700ms* | | | |

</details>

<details>
<summary><h3>TuringDB</h3> Server (HTTP REST) | Column-oriented | OpenCypher | Git-like versioning</summary>

**40/65 passed.** C++23 in-memory column-oriented. Failed: all SNB (HTTP data setup too slow), all concurrent (server crash), all vector/hybrid (server OOM), 3 ACID (change model limitations).

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

LDBC SNB-derived social network: Persons, Cities, Tags, Countries, Forums, Posts, Comments, Universities, Companies. Relationships: KNOWS, LIVES_IN, HAS_INTEREST, HAS_CREATOR, REPLY_OF, HAS_TAG, CONTAINER_OF, STUDY_AT, WORK_AT, IS_LOCATED_IN.

### Query Languages & Data Models

| | Grafeo | Grafeo Server | Memgraph | FalkorDB | Neo4j | TuGraph | LadybugDB | ArangoDB | TuringDB |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RDF** | ✅ | ✅ | | | | | | | |
| **GQL (ISO)** | ✅ | ✅ | | | | | | | |
| **Cypher** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| **Gremlin** | ✅ | ✅ | | | | | | | |
| **GraphQL** | ✅ | ✅ | | | | | | | |
| **SPARQL** | ✅ | ✅ | | | | | | | |
| **SQL/PGQ** | ✅ | ✅ | | | | | | | |
| **AQL** | | | | | | | | ✅ | |

### Native Algorithm Support

| | Grafeo | Grafeo Server | Memgraph | Neo4j | TuGraph |
|---|:---:|:---:|:---:|:---:|:---:|
| BFS | ✅ | | ✅ | ✅ | ✅ |
| PageRank | ✅ | | ✅ | ✅ | ✅ |
| WCC | ✅ | | ✅ | ✅ | ✅ |
| CDLP | ✅ | | ✅ | ✅ | ✅ |
| LCC | ✅ | | ✅ | ✅ | ✅ |
| SSSP | | | ✅ | ✅ | ✅ |

- **Neo4j** uses GDS (Graph Data Science) library procedures.
- **Grafeo Server** has the same native algorithms as Grafeo embedded but they are not yet exposed over GWP. Uses NetworkX fallback.
- **TuGraph** ships 34+ native algorithms but the benchmark adapter uses NetworkX fallback (procedure signatures differ).
- **LadybugDB, FalkorDB, ArangoDB, TuringDB** do not ship native implementations of LDBC Graph Analytics algorithms.

**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)
