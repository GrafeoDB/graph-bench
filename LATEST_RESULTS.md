# Benchmark Results

65 benchmarks across 12 categories on 9 databases at scale factor sf01. Grafeo passed 65/65 and placed first in 10 of 12 categories. TuGraph led Vector and Hybrid.

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

All times in milliseconds. **Bold** = best in category. \* = some benchmarks failed (total excludes failures). — = not tested. † = 33/65 attempted.

| Category | Grafeo | Grafeo Server | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB | TuringDB |
|----------|-------:|--------------:|----------:|------:|---------:|---------:|--------:|---------:|---------:|
| SNB Interactive (11) | **876** | — | 15,157 | 6,294\* | 2,608\* | 7,410\* | 5,978\* | 24,700\* | —\* |
| Graph Analytics (6) | **1.8** | 785 | 380\* | 132 | 312\* | 30 | 466\* | 18,900\* | 27,233 |
| ACID (12) | **39** | 5,532 | 129 | 201 | 94 | 110\* | 118 | 2,131 | 5,589\* |
| Algorithms (6) | **13** | — | 436\* | 629 | 134\* | 287\* | 539\* | 18,500\* | 25,659 |
| Query (3) | **4.9** | 3,962 | 31 | 99 | 62 | 75 | 355 | 3,961 | 3,968 |
| Read (2) | **2.4** | 4,446 | 49 | 129 | 77 | 108 | 110 | 151 | 4,465 |
| Write (5) | **19** | 10,059 | 176 | 517 | 150 | 209 | 636 | 10,385 | 17,886 |
| Traversal (5) | **6.1** | 4,839 | 141 | 241 | 98\* | 147\* | 225 | 4,670 | 8,959 |
| Graph Stats (4) | **5.1** | — | 125 | 202 | 91 | 150 | 183 | 2,713 | 6,486 |
| Concurrent (5) | **152** | — | 1,932 | 2,183 | 635 | 833\* | —\* | 13,400\* | —\* |
| Vector (4) | 2,494 | — | 6,015 | 12,738 | 7,789\* | 3,128\* | **910** | 16,310 | —\* |
| Hybrid (2) | 334 | — | 712 | 1,557 | 4,476 | 1,724 | **89** | 10,300 | —\* |
| **Pass Rate** | **65/65** | 33/33† | 63/65 | 64/65 | 58/65 | 58/65 | 57/65 | 57/65 | 40/65 |

### Notes

- **Embedded vs. Server.** Grafeo and LadybugDB run in-process (no network overhead). Server databases pay ~0.1–3ms per round-trip. Grafeo Server uses HTTP REST with GQL (higher per-query overhead than Bolt/Redis protocols).
- **Native algorithms.** Grafeo, Neo4j (GDS), and Memgraph (MAGE) run graph analytics natively. Grafeo Server has the same engine but algorithms are not yet exposed over HTTP. All others use a NetworkX fallback in Python — graph extraction overhead is included in measured time.
- **Vector search.** All databases use the same brute-force Python fallback (scan + cosine similarity). No native vector indexes are wired.
- **Edge insertion.** All adapters look up source and target nodes by property via the query engine before creating edges. All adapters create property indexes on `id` for fair lookup performance.
- **Failures.** Common failure causes: IC1 timeout (3-hop BFS over all KNOWS edges), missing scipy (PageRank via NetworkX), server crashes under concurrent load.

---

## LDBC SNB Interactive

Subset of the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/) Interactive workload. Times in ms. TuringDB and Grafeo Server omitted (not tested on SNB).

| Query | Description | Grafeo | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB |
|-------|-------------|-------:|----------:|------:|---------:|---------:|--------:|---------:|
| IS1 | Profile lookup | **3.5** | 150 | 274 | 240 | 325 | 257 | 4,700 |
| IS2 | Recent messages | **3.0** | 158 | 209 | 200 | 261 | 200 | 1,100 |
| IS3 | Friends of person | **48** | 1,100 | 3,338 | FAIL | 3,900 | 3,000 | 6,500 |
| IS4 | Message content | **1.1** | 31 | 90 | 78 | 107 | 85 | 103 |
| IS5 | Creator of message | **1.3** | 57 | 88 | 82 | 107 | 85 | 2,200 |
| IS6 | Forum of message | **3.2** | 154 | 216 | 199 | 273 | 204 | FAIL |
| IS7 | Replies to message | **3.6** | 135 | 153 | 188 | 178 | 145 | FAIL |
| IC1 | Friends 3-hop by name | **758** | 12,000 | FAIL | FAIL | FAIL | FAIL | FAIL |
| IC2 | Friends' recent messages | **18** | 372 | 767 | 607 | 894 | 671 | 10,100 |
| IC3 | Friends in countries | **15** | 549 | 611 | 550 | 721 | 679 | FAIL |
| IC6 | Tag co-occurrence | **21** | 444 | 548 | 463 | 618 | 652 | FAIL |
| *Total* | | ***876*** | *15,157* | *6,294* | *2,608* | *7,410* | *5,978* | *24,700* |

## LDBC Graph Analytics

Core algorithms from [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/). Times in ms. Grafeo, Neo4j (GDS), and Memgraph (MAGE) use native implementations; others use NetworkX fallback.

| Algorithm | Grafeo | Memgraph | Neo4j | Grafeo Server | FalkorDB | LadybugDB | TuGraph | ArangoDB | TuringDB |
|-----------|-------:|---------:|------:|--------------:|---------:|----------:|--------:|---------:|---------:|
| BFS | **0.03** | 1.3 | 14.4 | 130 | 61 | 79 | 94 | 44 | 4,539 |
| PageRank | **0.04** | 8.6 | 43.9 | 132 | FAIL | FAIL | FAIL | FAIL | 4,540 |
| WCC | **0.05** | 2.9 | 13.2 | 133 | 62 | 76 | 91 | 4,600 | 4,544 |
| CDLP | **0.08** | 6.1 | 28.6 | 130 | 63 | 77 | 92 | 5,100 | 4,539 |
| LCC | **0.06** | 7.3 | 20.1 | 131 | 63 | 75 | 95 | 4,600 | 4,536 |
| SSSP | **1.5** | 3.2 | 12.2 | 129 | 64 | 73 | 94 | 4,600 | 4,536 |
| *Total* | ***1.8*** | *30* | *132* | *785* | *312* | *380* | *466* | *18,900* | *27,233* |

## LDBC ACID

[LDBC ACID test suite](https://github.com/ldbc/ldbc_acid) for transactional consistency. Times in ms.

| Test | Grafeo | FalkorDB | Memgraph | TuGraph | LadybugDB | Neo4j | ArangoDB | Grafeo Server | TuringDB |
|------|-------:|---------:|---------:|--------:|----------:|------:|---------:|--------------:|---------:|
| Atomicity-C | **0.05** | 1.8 | 2.6 | 2.8 | 2.0 | 6.3 | 89 | 221 | 395 |
| Atomicity-RB | **0.07** | 3.1 | 4.0 | 4.4 | 2.7 | 8.3 | 133 | 220 | 396 |
| G0 (dirty write) | **2.0** | 4.8 | FAIL | 6.0 | 5.7 | 14.4 | 133 | 312 | 573 |
| G1a (aborted read) | **6.4** | 9.9 | 13 | 11 | 10 | 15.7 | 137 | 352 | FAIL |
| G1b (interm. read) | **0.07** | 2.6 | 3.7 | 3.7 | 2.7 | 10.2 | 132 | 307 | 572 |
| G1c (circular info) | **0.53** | 4.5 | 5.3 | 6.3 | 5.3 | 14.7 | 138 | 396 | 660 |
| IMP (item-many-prec) | **11** | 13 | 13 | 14 | 13 | 15.7 | 87 | 220 | 395 |
| PMP (pred-many-prec) | **11** | 20 | 22 | 24 | 20 | 28.0 | 269 | 890 | FAIL |
| OTV (observed txn vanish) | **0.07** | 3.8 | 4.6 | 5.4 | 3.7 | 13.2 | 183 | 440 | 792 |
| FR (fractured read) | **5.6** | 8.7 | 9.1 | 9.4 | 8.5 | 13.7 | 178 | 352 | 704 |
| LU (lost update) | **1.7** | 17 | 27 | 25 | 50 | 46.4 | 515 | 1,470 | FAIL |
| WS (write skew) | **0.58** | 5.4 | 6.1 | 7.0 | 6.1 | 14.8 | 138 | 352 | 527 |
| *Total* | ***39*** | *94* | *110* | *118* | *129* | *201* | *2,131* | *5,532* | *5,589* |
| *Result* | *PASS* | *PASS* | *G0 FAIL* | *PASS* | *PASS* | *PASS* | *PASS* | *PASS* | *3 FAIL* |

---

## Per-Database Details

<details>
<summary><h3>Grafeo</h3> Embedded (Rust) | LPG + RDF | GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ | Full ACID</summary>

**65/65 passed.** v0.5.1. Native graph analytics (BFS, PageRank, WCC, CDLP, LCC). SSSP uses NetworkX fallback. Edge insertion uses GQL MATCH for node lookup + `create_edge()`.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 — profile lookup | 3.5ms | | BFS | 0.03ms |
| IS2 — recent messages | 3.0ms | | PageRank | 0.04ms |
| IS3 — friends | 48ms | | WCC | 0.05ms |
| IS4 — message content | 1.1ms | | CDLP | 0.08ms |
| IS5 — creator | 1.3ms | | LCC | 0.06ms |
| IS6 — forum | 3.2ms | | SSSP | 1.5ms |
| IS7 — replies | 3.6ms | | *GA Total* | *1.8ms* |
| IC1 — friends 3-hop | 758ms | | | |
| IC2 — friends' messages | 18ms | | **Algorithms** | |
| IC3 — friends in countries | 15ms | | PageRank | 0.15ms |
| IC6 — tag co-occurrence | 21ms | | Community Detection | 0.13ms |
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
<summary><h3>Grafeo Server</h3> Server (HTTP REST) | LPG | GQL | Full ACID</summary>

**33/33 passed (33/65 attempted).** Same Grafeo engine (v0.5.1) over HTTP with GQL queries via grafeo-server:lite. HTTP round-trip overhead (~1–4ms per query) dominates at this scale. Graph analytics use NetworkX fallback (algorithms not yet exposed over HTTP).

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **ACID** | | | **Graph Analytics (NetworkX)** | |
| Atomicity-C | 221ms | | BFS | 130ms |
| Atomicity-RB | 220ms | | PageRank | 132ms |
| G0 (dirty write) | 312ms | | WCC | 133ms |
| G1a (aborted read) | 352ms | | CDLP | 130ms |
| G1b (interm. read) | 307ms | | LCC | 131ms |
| G1c (circular info) | 396ms | | SSSP | 129ms |
| IMP | 220ms | | *GA Total* | *785ms* |
| PMP | 890ms | | | |
| OTV | 440ms | | **Query** | |
| FR | 352ms | | Aggregation | 2,202ms |
| LU | 1,470ms | | Filter (equality) | 880ms |
| WS | 352ms | | Filter (range) | 880ms |
| *ACID Total* | *5,532ms* | | *Query Total* | *3,962ms* |
| *Result* | *12/12 PASS* | | | |
| | | | **Read** | |
| **Write** | | | Single Read | 4,402ms |
| Node Insertion | 89ms | | Batch Read | 44ms |
| Edge Insertion | 265ms | | *Read Total* | *4,446ms* |
| Property Update | 4,402ms | | | |
| Edge Add (existing nodes) | 89ms | | **Traversal** | |
| Mixed Workload | 5,214ms | | 1-hop | 2,200ms |
| *Write Total* | *10,059ms* | | 2-hop | 879ms |
| | | | BFS | 440ms |
| | | | DFS | 440ms |
| | | | Shortest Path | 880ms |
| | | | *Traversal Total* | *4,839ms* |

</details>

<details>
<summary><h3>LadybugDB</h3> Embedded | LPG | Cypher | Full ACID</summary>

**63/65 passed.** Failures: PageRank (2x, scipy not installed). Uses dedicated columns for common properties with JSON overflow for uncommon fields.

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
<summary><h3>Neo4j</h3> Server (Bolt) | LPG | Cypher | Native graph analytics (GDS)</summary>

**64/65 passed.** v5.23.0. Uses GDS library for native graph analytics. Only failure: IC1 (timeout).

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
<summary><h3>FalkorDB</h3> Server (Redis) | LPG | Cypher | Full ACID</summary>

**58/65 passed.** Failures: IS3, IC1 (timeout), PageRank (2x), Shortest Path, Betweenness Centrality, Vector Batch Search.

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

**58/65 passed.** Failures: IC1 (timeout), Betweenness Centrality, Shortest Path, ACID G0 (dirty write), Lost Update, Concurrent ACID, Vector Batch Search.

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

**57/65 passed.** 34+ native algorithms via stored procedures (not wired to benchmark adapter — uses NetworkX fallback). Failures: IC1 (timeout), PageRank (2x, scipy), all 5 concurrent (schema field mismatch).

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

**57/65 passed.** Failures: IS6, IS7, IC1, IC3, IC6 (timeout), PageRank (2x), Concurrent ACID.

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

| | Grafeo | Grafeo Server | LadybugDB | Neo4j | FalkorDB | Memgraph | TuGraph | ArangoDB | TuringDB |
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

| | Grafeo | Neo4j | Memgraph | Grafeo Server | TuGraph |
|---|:---:|:---:|:---:|:---:|:---:|
| BFS | ✅ | ✅ | ✅ | | ✅ |
| PageRank | ✅ | ✅ | ✅ | | ✅ |
| WCC | ✅ | ✅ | ✅ | | ✅ |
| CDLP | ✅ | ✅ | ✅ | | ✅ |
| LCC | ✅ | ✅ | ✅ | | ✅ |
| SSSP | | ✅ | ✅ | | ✅ |

- **Neo4j** uses GDS (Graph Data Science) library procedures.
- **Grafeo Server** has the same native algorithms as Grafeo embedded but they are not yet exposed over the HTTP API. Uses NetworkX fallback.
- **TuGraph** ships 34+ native algorithms but the benchmark adapter uses NetworkX fallback (procedure signatures differ).
- **LadybugDB, FalkorDB, ArangoDB, TuringDB** do not ship native implementations of LDBC Graph Analytics algorithms.

**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)
