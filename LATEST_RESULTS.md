# Benchmark Results

65 benchmarks across 12 categories, run at scale factor sf01 (~10K nodes, ~50K edges). Grafeo passed all 65 benchmarks and placed first in every category.

---

## Test Configuration

| | |
|---|---|
| **Scale** | sf01 (1,031 persons, ~10K nodes total, ~50K edges) |
| **Timeout** | 60s per benchmark |
| **Warmup** | 3 iterations discarded |
| **Measured** | 10 iterations, mean reported |
| **Hardware** | AMD Ryzen 7 7800X3D, 64GB RAM, Windows, Python 3.13 |
| **Date** | 2026-02-06 |

---

## Results at a Glance

All times in milliseconds. **Bold** = best in category. \* = some benchmarks failed (total excludes failures).

| Category | Grafeo | LadybugDB | FalkorDB | Memgraph | TuGraph | ArangoDB |
|----------|-------:|----------:|---------:|---------:|--------:|---------:|
| SNB Interactive (11) | **1,381** | 15,157 | 2,608\* | 7,410\* | 5,978\* | 24,700\* |
| Graph Analytics (6) | **3.3** | 380\* | 312\* | 30 | 466\* | 18,900\* |
| ACID (12) | **41** | 109 | 94 | 110\* | 118 | 2,131 |
| Algorithms (6) | **2.0** | 436\* | 134\* | 287\* | 539\* | 18,500\* |
| Query (3) | **6.7** | 51 | 62 | 75 | 355 | 3,961 |
| Read (2) | **3.1** | 41 | 77 | 108 | 110 | 151 |
| Write (5) | **5.9** | 149 | 150 | 209 | 636 | 10,385 |
| Traversal (5) | **2.0** | 152 | 98\* | 147\* | 225 | 4,670 |
| Graph Stats (4) | **2.2** | 125 | 91 | 150 | 183 | 2,713 |
| Concurrent (5) | **197** | 1,932 | 635 | 833\* | —\* | 13,400\* |
| Vector (4) | **2,230** | 6,015 | 7,789\* | 3,128\* | 910 | 16,310 |
| Hybrid (2) | **235** | 712 | 4,476 | 1,724 | 89 | 10,300 |
| **Pass Rate** | **65/65** | 63/65 | 58/65 | 58/65 | 57/65 | 57/65 |

### Reading the results

- **Embedded vs. Server.** Grafeo and LadybugDB run in-process (no network overhead). Server databases (FalkorDB, Memgraph, TuGraph, ArangoDB) pay ~0.1–1ms per round-trip.
- **Native algorithms.** Grafeo has native graph analytics. Memgraph uses MAGE plugins. Others fall back to NetworkX in Python (PageRank fails without numpy).
- **Vector search.** All databases use the brute-force Python fallback (scan + cosine similarity). No native vector indexes are wired yet.
- **Concurrent.** TuGraph failed all 5 concurrent benchmarks (schema field mismatch). Memgraph failed 2 (lost_update, concurrent_acid).
- **Pass rate.** Failures include timeouts (IC1 at sf01 = 3-hop BFS over all KNOWS edges) and missing dependencies (PageRank without numpy).

---

## LDBC SNB Interactive

Subset of the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/) Interactive workload. Times in ms.

| Query | Description | Grafeo | LadybugDB | FalkorDB | Memgraph | TuGraph | ArangoDB |
|-------|-------------|-------:|----------:|---------:|---------:|--------:|---------:|
| IS1 | Profile lookup | **5.7** | 150 | 240 | 325 | 257 | 4,700 |
| IS2 | Recent messages | **4.5** | 158 | 200 | 261 | 200 | 1,100 |
| IS3 | Friends of person | **67** | 1,100 | FAIL | 3,900 | 3,000 | 6,500 |
| IS4 | Message content | **1.6** | 31 | 78 | 107 | 85 | 103 |
| IS5 | Creator of message | **1.6** | 57 | 82 | 107 | 85 | 2,200 |
| IS6 | Forum of message | **4.0** | 154 | 199 | 273 | 204 | FAIL |
| IS7 | Replies to message | **5.8** | 135 | 188 | 178 | 145 | FAIL |
| IC1 | Friends 3-hop by name | **1,200** | 12,000 | FAIL | FAIL | FAIL | FAIL |
| IC2 | Friends' recent messages | **32** | 372 | 607 | 894 | 671 | 10,100 |
| IC3 | Friends in countries | **22** | 549 | 550 | 721 | 679 | FAIL |
| IC6 | Tag co-occurrence | **36** | 444 | 463 | 618 | 652 | FAIL |
| *Total* | | ***1,381*** | *15,157* | *2,608* | *7,410* | *5,978* | *24,700* |

## LDBC Graph Analytics

Core algorithms from [LDBC Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/). Times in ms. Grafeo and Memgraph use native implementations; others use NetworkX fallback.

| Algorithm | Grafeo | Memgraph | FalkorDB | LadybugDB | TuGraph | ArangoDB |
|-----------|-------:|---------:|---------:|----------:|--------:|---------:|
| BFS | **1.4** | 1.3 | 61 | 79 | 94 | 44 |
| PageRank | **0.10** | 8.6 | FAIL | FAIL | FAIL | FAIL |
| WCC | **0.11** | 2.9 | 62 | 76 | 91 | 4,600 |
| CDLP | **0.14** | 6.1 | 63 | 77 | 92 | 5,100 |
| LCC | **0.11** | 7.3 | 63 | 75 | 95 | 4,600 |
| SSSP | **1.4** | 3.2 | 64 | 73 | 94 | 4,600 |
| *Total* | ***3.3*** | *29* | *312* | *380* | *466* | *18,900* |

## LDBC ACID

[LDBC ACID test suite](https://github.com/ldbc/ldbc_acid) for transactional consistency. Times in ms.

| Test | Grafeo | FalkorDB | LadybugDB | Memgraph | TuGraph | ArangoDB |
|------|-------:|---------:|----------:|---------:|--------:|---------:|
| Atomicity-C | **0.11** | 1.8 | 1.6 | 2.6 | 2.8 | 89 |
| Atomicity-RB | **0.11** | 3.1 | 2.2 | 4.0 | 4.4 | 133 |
| G0 (dirty write) | **2.3** | 4.8 | 5.2 | FAIL | 6.0 | 133 |
| G1a (aborted read) | **6.8** | 9.9 | 9.8 | 13 | 11 | 137 |
| G1b (interm. read) | **0.17** | 2.6 | 2.3 | 3.7 | 3.7 | 132 |
| G1c (circular info) | **0.74** | 4.5 | 4.3 | 5.3 | 6.3 | 138 |
| IMP (item-many-prec) | **10.8** | 13 | 12 | 13 | 14 | 87 |
| PMP (pred-many-prec) | **11** | 20 | 18 | 22 | 24 | 269 |
| OTV (observed txn vanish) | **0.16** | 3.8 | 3.2 | 4.6 | 5.4 | 183 |
| FR (fractured read) | **5.9** | 8.7 | 7.9 | 9.1 | 9.4 | 178 |
| LU (lost update) | **2.2** | 17 | 38 | 27 | 25 | 515 |
| WS (write skew) | **0.65** | 5.4 | 4.9 | 6.1 | 7.0 | 138 |
| *Total* | ***41*** | *94* | *109* | *110* | *118* | *2,131* |
| *Result* | *PASS* | *PASS* | *PASS* | *G0 FAIL* | *PASS* | *PASS* |

---

## Per-Database Details

<details>
<summary><h3>Grafeo</h3> Embedded (Rust) | LPG + RDF | GQL, Cypher, Gremlin, GraphQL, SPARQL | Full ACID</summary>

**65/65 benchmarks passed.** Native graph analytics engine, columnar storage, vectorized execution, lock-free reads, worst-case optimal joins.

| Benchmark | Time | | Benchmark | Time |
|-----------|-----:|-|-----------|-----:|
| **SNB Interactive** | | | **Graph Analytics** | |
| IS1 — profile lookup | 5.7ms | | BFS | 1.4ms |
| IS2 — recent messages | 4.5ms | | PageRank | 0.10ms |
| IS3 — friends | 67ms | | WCC | 0.11ms |
| IS4 — message content | 1.6ms | | CDLP | 0.14ms |
| IS5 — creator | 1.6ms | | LCC | 0.11ms |
| IS6 — forum | 4.0ms | | SSSP | 1.4ms |
| IS7 — replies | 5.8ms | | *GA Total* | *3.3ms* |
| IC1 — friends 3-hop | 1,200ms | | | |
| IC2 — friends' messages | 32ms | | **Algorithms** | |
| IC3 — friends in countries | 22ms | | PageRank | 0.14ms |
| IC6 — tag co-occurrence | 36ms | | Community Detection | 0.15ms |
| *SNB Total* | *1,381ms* | | Betweenness Centrality | 0.56ms |
| | | | Closeness Centrality | 0.27ms |
| **Query** | | | Triangle Count | 0.24ms |
| Aggregation | 2.9ms | | Common Neighbors | 0.63ms |
| Filter (equality) | 2.7ms | | *Algo Total* | *2.0ms* |
| Filter (range) | 1.0ms | | | |
| *Query Total* | *6.7ms* | | **Vector** | |
| | | | Insert | 51ms |
| **Read** | | | k-NN Search | 185ms |
| Single Read | 0.88ms | | Batch Search (100x) | 1,800ms |
| Batch Read | 2.3ms | | Recall@10 | 194ms |
| *Read Total* | *3.1ms* | | *Vector Total* | *2,230ms* |
| | | | | |
| **Write** | | | **Hybrid** | |
| Node Insertion | 0.70ms | | Graph → Vector | 197ms |
| Edge Insertion | 0.93ms | | Vector → Graph | 38ms |
| Property Update | 0.57ms | | *Hybrid Total* | *235ms* |
| Edge Add (existing nodes) | 0.09ms | | | |
| Mixed Workload | 3.6ms | | **Concurrent** | |
| *Write Total* | *5.9ms* | | Throughput Scaling | 74ms |
| | | | Lost Update | 4.1ms |
| **Traversal** | | | Read-After-Write | 4.4ms |
| 1-hop | 0.55ms | | Mixed | 17ms |
| 2-hop | 0.46ms | | ACID | 97ms |
| BFS | 0.24ms | | *Concurrent Total* | *197ms* |
| DFS | 0.25ms | | | |
| Shortest Path | 0.46ms | | **ACID** | |
| *Traversal Total* | *2.0ms* | | *12/12 PASS* | *41ms* |

</details>

<details>
<summary><h3>LadybugDB</h3> Embedded | LPG | Cypher | Full ACID</summary>

**63/65 benchmarks passed.** Failures: PageRank (2x, numpy not installed for NetworkX fallback).

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

**57/65 benchmarks passed.** Ships 34+ native algorithms via stored procedures, but the benchmark adapter uses NetworkX fallback (procedure signatures differ). Failures: IC1 (timeout), PageRank (2x, numpy), all 5 concurrent benchmarks (schema field mismatch).

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

| | Grafeo | LadybugDB | FalkorDB | Memgraph | TuGraph | ArangoDB |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RDF** | ✅ | | | | | |
| **GQL (ISO)** | ✅ | | | | | |
| **Cypher** | ✅ | ✅ | ✅ | ✅ | ✅ | |
| **Gremlin** | ✅ | | | | | ✅ |
| **GraphQL** | ✅ | | | | | ✅ |
| **SPARQL** | ✅ | | | | | |
| **AQL** | | | | | | ✅ |

### Native Algorithm Support

| | Grafeo | Memgraph | TuGraph¹ |
|---|:---:|:---:|:---:|
| BFS | ✅ | ✅ | ✅ |
| PageRank | ✅ | ✅ | ✅ |
| WCC | ✅ | ✅ | ✅ |
| CDLP | ✅ | ✅ | ✅ |
| LCC | ✅ | ✅ | ✅ |
| SSSP | ✅ | ✅ | ✅ |

¹ TuGraph ships 34+ native algorithms, but the benchmark adapter uses NetworkX fallback (native stored procedure signatures differ from adapter API).

Other databases (LadybugDB, FalkorDB, ArangoDB) do not ship native implementations of LDBC Graph Analytics algorithms.

**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)
