# Benchmark Results

**Run:** 2026-02-23 | **Platform:** Windows  
**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)

> **Note:** These are not official LDBC Benchmark results. Workloads are inspired by [LDBC](https://ldbcouncil.org/) specifications but have not been audited by the LDBC Council. See the [LDBC disclaimer](README.md#ldbc-disclaimer) for details.

## Scale Factors (based on LDBC SNB)

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
| **Grafeo** | Embedded | **2,904** | 136 MB | **0.4** | 43 MB | **39.6** | 67 MB |
| LadybugDB | Embedded | 5,333 | 4890 MB | 225 | 250 MB | 128 | 4914 MB |
| FalkorDB Lite | Embedded | 7,454 | 156 MB | 89 | 88 MB | 72 | 144 MB |

### SNB Interactive (total ms)

| Database | Type | Sf01 |
|----------|------|------:|
| **Grafeo** | Embedded | **2,904** |
| LadybugDB | Embedded | 5,333 |
| FalkorDB Lite | Embedded | 7,454 |

### Graph Analytics (native implementations only, total ms)

| Database | Type | Sf01 |
|----------|------|------:|
| **Grafeo** | Embedded | 0.4 |
| **LadybugDB** | Embedded | 224.6 |
| **FalkorDB Lite** | Embedded | 89.2 |

Only databases with native in-database algorithm implementations are included. Databases that fall back to extracting the graph into Python/NetworkX measure extraction overhead, not database performance. That extraction typically adds 100-1,000x overhead.

### Combinatorial Workload (total ms)

| Database | Type | Writes | Reads | Traversals | ACID |
|----------|------|-------:|------:|-----------:|-----:|
| Grafeo | Embedded | | | | |
| LadybugDB | Embedded | | | | |
| FalkorDB Lite | Embedded | | | | |

### Reading the results

These benchmarks compare databases with fundamentally different architectures. Before drawing conclusions, consider:

- **Embedded vs. server.** Grafeo and LadybugDB run in-process - no network serialization, no protocol overhead. Server databases pay ~0.1-1ms per round-trip.
- **Consistency model.** NebulaGraph uses eventual consistency by default. Its write speeds are not comparable to ACID-compliant databases without qualification.
- **Memory model.** Memgraph is in-memory first with optional WAL persistence. FalkorDB inherits Redis persistence semantics.
- **Scale factor.** Small (10K nodes) fits in L2 cache. Medium and large benchmarks reveal architectural differences.

---

## Per-Database Results

<details>
<summary><h3>Grafeo</h3></summary>

| | |
|---|---|
| **Type** | Embedded (in-process, Rust) |
| **Data model** | LPG + RDF |
| **Query languages** | GQL (ISO), Cypher, Gremlin, GraphQL, SPARQL |
| **ACID** | Full (snapshot isolation, WAL) |
| **Consistency** | Strong |
| **License** | Apache 2.0 |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 25.21ms | 130.0 MB |
| snb_is2 | 20.27ms | 132.2 MB |
| snb_is3 | 304.58ms | 133.0 MB |
| snb_is4 | 8.34ms | 131.3 MB |
| snb_is5 | 8.46ms | 132.6 MB |
| snb_is6 | 20.77ms | 134.1 MB |
| snb_is7 | 16.01ms | 134.1 MB |
| **Total** | **403.65ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | 2303.68ms | 133.6 MB |
| snb_ic2 | 76.39ms | 134.8 MB |
| snb_ic3 | 61.80ms | 135.5 MB |
| snb_ic6 | 58.03ms | 136.0 MB |
| **Total** | **2499.90ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 25.21ms | 130.0 MB |
| snb_is2 | 20.27ms | 132.2 MB |
| snb_is3 | 304.58ms | 133.0 MB |
| snb_is4 | 8.34ms | 131.3 MB |
| snb_is5 | 8.46ms | 132.6 MB |
| snb_is6 | 20.77ms | 134.1 MB |
| snb_is7 | 16.01ms | 134.1 MB |
| snb_ic1 | 2303.68ms | 133.6 MB |
| snb_ic2 | 76.39ms | 134.8 MB |
| snb_ic3 | 61.80ms | 135.5 MB |
| snb_ic6 | 58.03ms | 136.0 MB |
| **Total** | **2903.54ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 0.03ms | 42.5 MB |
| ldbc_pagerank | 0.04ms | 42.7 MB |
| ldbc_wcc | 0.06ms | 42.7 MB |
| ldbc_cdlp | 0.08ms | 42.8 MB |
| ldbc_lcc | 0.06ms | 42.8 MB |
| ldbc_sssp | 0.13ms | 42.9 MB |
| **Total** | **0.40ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 0.03ms | 42.5 MB |
| ldbc_pagerank | 0.04ms | 42.7 MB |
| ldbc_wcc | 0.06ms | 42.7 MB |
| ldbc_cdlp | 0.08ms | 42.8 MB |
| ldbc_lcc | 0.06ms | 42.8 MB |
| ldbc_sssp | 0.13ms | 42.9 MB |
| **Total** | **0.40ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 0.06ms | 66.6 MB |
| acid_atomicity_rb | 0.08ms | 65.9 MB |
| **Total** | **0.14ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | 2.05ms | 66.0 MB |
| acid_g1a | 6.71ms | 66.0 MB |
| acid_g1b | 0.07ms | 66.0 MB |
| acid_g1c | 0.68ms | 66.0 MB |
| acid_imp | 10.60ms | 66.0 MB |
| acid_pmp | 10.92ms | 66.0 MB |
| acid_otv | 0.10ms | 66.0 MB |
| acid_fr | 5.66ms | 66.0 MB |
| acid_lu | 1.97ms | 66.0 MB |
| acid_ws | 0.70ms | 66.0 MB |
| **Total** | **39.47ms** | |

</details>

<details>
<summary><h3>LadybugDB</h3></summary>

| | |
|---|---|
| **Type** | Embedded |
| **Data model** | LPG |
| **Query languages** | Cypher |
| **ACID** | Full (snapshot isolation) |
| **Consistency** | Strong |
| **License** | MIT |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 181.16ms | 404.9 MB |
| snb_is2 | 159.71ms | 549.9 MB |
| snb_is3 | 1770.57ms | 930.0 MB |
| snb_is4 | 50.31ms | 1065.3 MB |
| snb_is5 | 66.66ms | 1179.9 MB |
| snb_is6 | 193.14ms | 1342.2 MB |
| snb_is7 | 133.67ms | 1482.8 MB |
| **Total** | **2555.24ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | FAILED | |
| snb_ic2 | 687.86ms | 4454.1 MB |
| snb_ic3 | 1150.17ms | 4683.0 MB |
| snb_ic6 | 939.63ms | 4889.6 MB |
| **Total** | **2777.66ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 181.16ms | 404.9 MB |
| snb_is2 | 159.71ms | 549.9 MB |
| snb_is3 | 1770.57ms | 930.0 MB |
| snb_is4 | 50.31ms | 1065.3 MB |
| snb_is5 | 66.66ms | 1179.9 MB |
| snb_is6 | 193.14ms | 1342.2 MB |
| snb_is7 | 133.67ms | 1482.8 MB |
| snb_ic1 | FAILED | |
| snb_ic2 | 687.86ms | 4454.1 MB |
| snb_ic3 | 1150.17ms | 4683.0 MB |
| snb_ic6 | 939.63ms | 4889.6 MB |
| **Total** | **5332.90ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 71.41ms | 203.8 MB |
| ldbc_pagerank | 3.30ms | 212.5 MB |
| ldbc_wcc | 2.29ms | 215.1 MB |
| ldbc_cdlp | 5.51ms | 215.5 MB |
| ldbc_lcc | 70.25ms | 231.6 MB |
| ldbc_sssp | 71.84ms | 249.9 MB |
| **Total** | **224.60ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 71.41ms | 203.8 MB |
| ldbc_pagerank | 3.30ms | 212.5 MB |
| ldbc_wcc | 2.29ms | 215.1 MB |
| ldbc_cdlp | 5.51ms | 215.5 MB |
| ldbc_lcc | 70.25ms | 231.6 MB |
| ldbc_sssp | 71.84ms | 249.9 MB |
| **Total** | **224.60ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 2.10ms | 4894.2 MB |
| acid_atomicity_rb | 3.04ms | 4894.3 MB |
| **Total** | **5.13ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | 5.81ms | 4894.5 MB |
| acid_g1a | 9.93ms | 4895.7 MB |
| acid_g1b | 2.64ms | 4895.7 MB |
| acid_g1c | 5.47ms | 4896.1 MB |
| acid_imp | 12.85ms | 4899.3 MB |
| acid_pmp | 19.57ms | 4899.3 MB |
| acid_otv | 3.71ms | 4899.4 MB |
| acid_fr | 8.38ms | 4901.1 MB |
| acid_lu | 48.05ms | 4911.7 MB |
| acid_ws | 6.39ms | 4914.0 MB |
| **Total** | **122.79ms** | |

</details>

<details>
<summary><h3>FalkorDB Lite</h3></summary>

| | |
|---|---|
| **Type** | Embedded (redislite + FalkorDB module) |
| **Data model** | LPG |
| **Query languages** | Cypher |
| **ACID** | Partial (Redis-level durability) |
| **Consistency** | Strong |
| **License** | SSPL |

#### SNB Interactive - Short Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 78.92ms | 143.6 MB |
| snb_is2 | 62.00ms | 148.0 MB |
| snb_is3 | 898.14ms | 148.5 MB |
| snb_is4 | 25.37ms | 151.0 MB |
| snb_is5 | 26.51ms | 151.1 MB |
| snb_is6 | 66.81ms | 152.7 MB |
| snb_is7 | 44.66ms | 155.2 MB |
| **Total** | **1202.41ms** | |

#### SNB Interactive - Complex Reads - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_ic1 | 5676.43ms | 154.6 MB |
| snb_ic2 | 199.06ms | 152.0 MB |
| snb_ic3 | 198.70ms | 154.0 MB |
| snb_ic6 | 177.89ms | 156.4 MB |
| **Total** | **6252.08ms** | |

#### SNB Interactive - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| snb_is1 | 78.92ms | 143.6 MB |
| snb_is2 | 62.00ms | 148.0 MB |
| snb_is3 | 898.14ms | 148.5 MB |
| snb_is4 | 25.37ms | 151.0 MB |
| snb_is5 | 26.51ms | 151.1 MB |
| snb_is6 | 66.81ms | 152.7 MB |
| snb_is7 | 44.66ms | 155.2 MB |
| snb_ic1 | 5676.43ms | 154.6 MB |
| snb_ic2 | 199.06ms | 152.0 MB |
| snb_ic3 | 198.70ms | 154.0 MB |
| snb_ic6 | 177.89ms | 156.4 MB |
| **Total** | **7454.49ms** | |

#### LDBC Graphanalytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 29.20ms | 84.8 MB |
| ldbc_pagerank | 0.80ms | 85.5 MB |
| ldbc_wcc | 0.29ms | 85.8 MB |
| ldbc_cdlp | 0.26ms | 85.8 MB |
| ldbc_lcc | 29.12ms | 86.8 MB |
| ldbc_sssp | 29.51ms | 87.7 MB |
| **Total** | **89.18ms** | |

#### Graph Analytics - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| ldbc_bfs | 29.20ms | 84.8 MB |
| ldbc_pagerank | 0.80ms | 85.5 MB |
| ldbc_wcc | 0.29ms | 85.8 MB |
| ldbc_cdlp | 0.26ms | 85.8 MB |
| ldbc_lcc | 29.12ms | 86.8 MB |
| ldbc_sssp | 29.51ms | 87.7 MB |
| **Total** | **89.18ms** | |

#### LDBC ACID - Atomicity - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_atomicity_c | 1.10ms | 144.0 MB |
| acid_atomicity_rb | 1.40ms | 142.7 MB |
| **Total** | **2.50ms** | |

#### LDBC ACID - Isolation - Sf01

| Benchmark | Time | Memory |
|-----------|-----:|-------:|
| acid_g0 | 3.23ms | 142.8 MB |
| acid_g1a | 7.52ms | 142.8 MB |
| acid_g1b | 1.14ms | 142.8 MB |
| acid_g1c | 2.36ms | 142.6 MB |
| acid_imp | 11.63ms | 142.6 MB |
| acid_pmp | 14.75ms | 142.6 MB |
| acid_otv | 1.68ms | 142.6 MB |
| acid_fr | 6.83ms | 142.6 MB |
| acid_lu | 17.73ms | 142.7 MB |
| acid_ws | 2.82ms | 142.7 MB |
| **Total** | **69.68ms** | |

</details>

---

## Query Languages & Data Models

| |Grafeo | LadybugDB | FalkorDB Lite |
|---|:---:|:---:|:---:|
| **LPG** | ✅ | ✅ | ✅ |
| **RDF** | ✅ | | |
| **GQL (ISO)** | ✅ | | |
| **Cypher** | ✅ | ✅ | ✅ |
| **Gremlin** | ✅ | | |
| **GraphQL** | ✅ | | |
| **SPARQL** | ✅ | | |

---

## Native Algorithm Support

| |Grafeo | LadybugDB | FalkorDB Lite |
|---|:---:|:---:|:---:|
| BFS | ✅ | | |
| PageRank | ✅ | ✅ | ✅ |
| WCC | ✅ | ✅ | ✅ |
| CDLP | ✅ | ✅ | ✅ |
| LCC | ✅ | | |
| SSSP | | | |

---

## Methodology

- **Warmup:** 3 runs discarded before measurement
- **Iterations:** 10 measured runs, median reported
- **Isolation:** Each database gets a clean dataset load before benchmarking
- **Timeout:** 600 seconds per benchmark; exceeded shown as `T/O`

---

## Raw Data

- Results: [`results/bench_20260223_142425.json`](results/bench_20260223_142425.json)
