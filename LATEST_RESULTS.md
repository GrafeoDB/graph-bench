# Latest Benchmark Results

Run: 2026-02-04 | Scale: small (10K nodes, 50K edges) | Platform: Windows

## LDBC Graph Analytics (Native Implementations Only)

Only databases with native in-database algorithm implementations are shown. Databases using NetworkX fallback (which extracts the graph to Python memory) are excluded as they measure extraction overhead, not database performance.

| Benchmark | Grafeo* | Neo4j | Memgraph |
|-----------|---------|-------|----------|
| BFS | **0.12ms** | 115ms | 5.9ms |
| PageRank | **0.25ms** | 64ms | 13.7ms |
| WCC | **0.55ms** | 30ms | 13.1ms |
| CDLP | **0.51ms** | 43ms | 15.7ms |
| LCC | **0.31ms** | - | 53.1ms |
| SSSP | **2.69ms** | 52ms | 6.4ms |
| **Total** | **4.43ms** | 304ms+ | 108ms+ |

Grafeo's algorithm performance comes from its vectorized execution engine and native Rust implementations that operate directly on columnar storage, avoiding any data movement or serialization.

## Write Operations

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph⁶ |
|-----------|---------|------------|---------|-------|----------|----------|----------|--------------|
| node_insertion | 3.3ms | 255ms | 6773ms | 60ms | 62ms | 522ms | 53ms | **1.8ms**⁶ |
| edge_insertion | 7.0ms | 270ms | 4269ms | 1865ms | 1822ms | 334ms | **49ms** | **1.3ms**⁶ |
| **Total** | **10.3ms** | 525ms | 11042ms | 1925ms | 1884ms | 856ms | 102ms | 3.1ms⁶ |

Grafeo uses chunked columnar storage with delta buffers for writes, allowing batch inserts without rebuilding indexes. NebulaGraph's speed reflects async writes with eventual consistency, not durable commits. Among ACID-compliant databases, **Grafeo is fastest** for node insertion and **ArangoDB** for edge insertion.

## Read Operations

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|-----------|---------|------------|---------|-------|----------|----------|----------|-------------|
| single_read | **0.6ms** | 30ms | 74ms | 283ms | 269ms | 113ms | 98ms | 70ms |
| batch_read | 5.7ms | 2.9ms | 2.8ms | 24ms | 25ms | 10ms | 45ms | **0.8ms** |
| **Total** | **6.3ms** | 32.9ms | 76.8ms | 307ms | 294ms | 123ms | 143ms | 70.8ms |

Grafeo's single_read uses O(1) hash index lookups with lock-free concurrent access (DashMap). Batch reads benefit from vectorized execution and zone map filtering.

## Traversals

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|-----------|---------|------------|---------|-------|----------|----------|----------|-------------|
| bfs | **0.3ms** | 50ms | 25ms | 36ms | 36ms | 20ms | 439ms | 7.4ms |
| dfs | **0.3ms** | 64ms | 32ms | 49ms | 48ms | 25ms | 439ms | 7.5ms |
| hop_1 | **0.8ms** | 81ms | 40ms | 65ms | 61ms | 31ms | 2211ms | 35ms |
| hop_2 | **0.5ms** | 65ms | 34ms | 49ms | 49ms | 26ms | 879ms | 15ms |
| triangle_count | **0.5ms** | 190ms | 46ms | 356ms | 346ms | 31ms | 13226ms | 35ms |
| common_neighbors | **0.7ms** | 38ms | 48ms | 71ms | 70ms | 37ms | 2648ms | 41ms |
| **Total** | **3.1ms** | 488ms | 225ms | 626ms | 610ms | 170ms | 19842ms | 140.9ms |

Grafeo's traversal performance comes from chunked adjacency lists with cache-friendly memory layout and worst-case optimal join algorithms (Leapfrog TrieJoin) for pattern matching.

## LDBC SNB Interactive

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|-----------|---------|------------|---------|-------|----------|----------|----------|-------------|
| snb_is1 (profile lookup) | **1.11ms** | 30ms | 73ms | 223ms | 161ms | 116ms | 94ms | 69ms |
| snb_is3 (friends) | **0.62ms** | 51ms | 81ms | 117ms | 84ms | 60ms | 2207ms | 34ms |
| snb_ic1 (friends 3-hop) | **0.25ms** | 1041ms | 2045ms | 2265ms | 1704ms | 1203ms | 46449ms | 7ms |
| snb_ic2 (recent posts) | **0.36ms** | 20ms | 39ms | 44ms | 34ms | 24ms | 879ms | 15ms |
| snb_ic3 (friends in cities) | **0.44ms** | 184ms | 378ms | 496ms | 360ms | 254ms | 9046ms | 14ms |
| snb_ic6 (tag co-occurrence) | **0.62ms** | 100ms | 202ms | 261ms | 187ms | 138ms | 5072ms | 28ms |
| **Total** | **3.4ms** | 1426ms | 2818ms | 3406ms | 2530ms | 1795ms | 63747ms | 167ms |

Grafeo's complex query performance benefits from cost-based optimization with cardinality estimation, query plan caching (5-10x speedup for repeated queries), and factorized query processing that avoids Cartesian products in multi-hop traversals.

## Concurrent ACID

Tests parallel throughput and consistency under concurrent workloads using LDBC SNB dataset.

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|-----------|---------|------------|---------|-------|----------|----------|----------|-------------|
| mixed_workload (80/20 r/w) | **6.6ms** | 192ms | 229ms | 161ms | 96ms | 89ms | 2210ms | 118ms |
| throughput_scaling (1-8 workers) | **12.2ms** | 797ms | 980ms | 597ms | 474ms | 270ms | 8859ms | 327ms |
| lost_update (counter increment) | **1.4ms** | 93ms | FAILED⁹ | 171ms | FAILED¹⁰ | 114ms | 1179ms | 140ms |
| read_after_write (visibility) | **2.3ms** | 96ms | 239ms | 190ms | 112ms | 109ms | 1135ms | 132ms |
| concurrent_acid (aggregate) | **28.5ms** | 1210ms | FAILED⁹ | 1103ms | FAILED¹⁰ | 561ms | 13373ms | 697ms |

⁹ DuckDB: Optimistic concurrency, fails on conflicting updates instead of serializing ("Conflict on update!").
¹⁰ Memgraph: Transaction conflicts not auto-retried ("Cannot resolve conflicting transactions").
⁶ NebulaGraph: Results obtained with `replica_factor=1` (single-node); distributed deployments with multiple replicas may exhibit different behavior due to eventual consistency.

**What these benchmarks measure:**

- **mixed_workload**: 80% reads, 20% writes running concurrently across 4 workers
- **throughput_scaling**: Read throughput scaling from 1→2→4→8 parallel workers
- **lost_update**: 4 threads incrementing same counter 25 times each (tests atomicity)
- **read_after_write**: Write then immediate read-back (tests visibility/consistency)
- **concurrent_acid**: Aggregate of all tests

Grafeo's concurrent performance benefits from lock-free reads (DashMap) and optimistic concurrency control that minimizes contention.

---

\* Embedded databases run in-process without network overhead. Server databases include network latency inherent to their architecture, making direct timing comparisons not entirely apples-to-apples.

"-" indicates the algorithm is not supported natively.

---

## Feature Availability

| Feature | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|---------|---------|------------|---------|-------|----------|----------|----------|-------------|
| Native PageRank | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Native WCC | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Native CDLP | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Native LCC | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Native SSSP | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Native BFS | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Deployment | Embedded | Embedded | Embedded | Server | Server | Server | Server | Distributed |

---

## Data Models and Query Languages

| Feature | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|---------|---------|------------|---------|-------|----------|----------|----------|-------------|
| **Data Model** | LPG + RDF | LPG | Relational | LPG | LPG | LPG | Multi-model⁸ | LPG |
| **LPG Support** | ✅ | ✅ | Via SQL/PGQ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RDF Support** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **GQL (ISO)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Cypher** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Gremlin** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **GraphQL** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **SPARQL** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **SQL/PGQ** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **AQL** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **nGQL** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

⁸ ArangoDB is multi-model: document, key-value, and graph in one database.

**Data Model Definitions:**

- **LPG (Labeled Property Graph)**: Nodes have labels and properties, edges have types and properties. Used by most graph databases.
- **RDF (Resource Description Framework)**: Triple-based model (subject-predicate-object). W3C standard for linked data and knowledge graphs.
- **Multi-model**: Combines multiple data models (document, graph, key-value) in one database.

**Grafeo's Query Language Support:**

Grafeo supports 5 query languages through a unified translation pipeline:

1. **GQL** (ISO/IEC 39075) - The new international standard for graph queries
2. **Cypher** (openCypher 9.0) - Neo4j-compatible ASCII-art syntax
3. **Gremlin** (Apache TinkerPop) - Traversal-based DSL
4. **GraphQL** - Schema-driven queries for both LPG and RDF
5. **SPARQL** (W3C 1.1) - Full RDF query support with REGEX, EXISTS, functions

All languages compile to the same logical plan, so performance is consistent regardless of syntax choice.

---

## Database Characteristics

Understanding these differences is critical for fair benchmark interpretation.

| Characteristic | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB | ArangoDB | NebulaGraph |
|----------------|---------|------------|---------|-------|----------|----------|----------|-------------|
| **ACID Transactions** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅⁴ | ⚠️⁵ | ✅ Full | ⚠️⁶ |
| **Consistency Model** | Strong | Strong | Strong | Strong | Strong | Strong | Strong | Eventual |
| **Default Durability** | WAL | Configurable | WAL | Sync | Optional WAL | AOF/RDB | RocksDB | Async |
| **Write Guarantee** | Durable | Durable | Durable | Durable | In-memory⁴ | Redis-level | Durable | Eventual |
| **Isolation Level** | Snapshot | Snapshot | Snapshot | Read Committed | Snapshot | None | Read Committed | None |
| **Network Overhead** | None | None | None | Bolt RPC | Bolt RPC | Redis protocol | HTTP/TCP | Thrift RPC |
| **Clustering** | Single | Single | Single | Causal⁷ | Single⁴ | Redis Cluster | Multi-model | Native sharding |
| **License** | Apache 2.0 | MIT | MIT | GPL/Commercial | BSL/Enterprise | SSPL | Apache 2.0 | Apache 2.0 |

⁴ Memgraph is in-memory first, WAL persistence is optional. ACID applies to single-node only.
⁵ FalkorDB inherits Redis persistence semantics (AOF fsync policy determines durability).
⁶ NebulaGraph uses eventual consistency with tunable replica factor. Benchmark uses `replica_factor=1` (no redundancy).
⁷ Neo4j Enterprise supports causal clustering, Community Edition is single-node.

---

## How Grafeo Compares

### Unique Capabilities

Grafeo is the only database in this benchmark that supports:

- **Dual data models** (LPG and RDF) with optimized storage for each
- **5 query languages** (GQL, Cypher, Gremlin, GraphQL, SPARQL) through a unified execution engine
- **ISO GQL** (ISO/IEC 39075), the new international standard for graph queries

This flexibility means you can use Cypher for property graph queries, SPARQL for RDF/knowledge graph queries, and GQL for standards-compliant code, all on the same database.

### Why Grafeo is Fast

Grafeo combines several modern database techniques:

- **Columnar storage** with type-specific compression (dictionary encoding for strings, delta encoding for integers, bit-packing for booleans)
- **Vectorized push-based execution** inspired by DuckDB, processing data in cache-friendly chunks
- **Lock-free concurrent reads** using DashMap for hash indexes
- **Worst-case optimal joins** (Leapfrog TrieJoin) for pattern matching, O(N^1.5) for triangles vs O(N²) with naive joins
- **Query plan caching** that provides 5-10x speedup for repeated queries
- **Zone maps** for predicate pushdown, skipping irrelevant data chunks
- **Ring Index** for RDF data, using wavelet trees for 3x space reduction

### Embedded vs Server Trade-offs

Grafeo is embedded, meaning it runs in-process with your application. This eliminates network overhead but limits it to single-node deployments. For comparison:

| Scenario | Grafeo (Embedded) | Neo4j (Server) |
|----------|-------------------|----------------|
| Single read | **~0.6ms** (direct memory) | ~283ms (network + disk) |
| Best for | Analytics, local apps, edge | Multi-user, distributed |

---

## Benchmark Caveats

### Write Operations

- **NebulaGraph's fast writes** (1.8ms node, 1.3ms edge) use async replication with eventual consistency, writes may not be immediately durable or visible on all replicas
- **Memgraph** runs in-memory by default, persistence adds latency
- **Embedded databases** avoid serialization and network overhead entirely

### Algorithm Benchmarks

- Only native implementations are shown in the LDBC Graph Analytics table
- Databases without native support would need to extract the graph to Python/NetworkX, which measures extraction overhead rather than database performance
- This extraction can add 100-1000x overhead

### Server vs Embedded

- Server databases include ~0.1-1ms network round-trip per operation
- For single-operation benchmarks (single_read), network latency dominates
- For batch operations, network overhead is amortized

---

Full results:

- Main benchmarks: [results/bench_20260204_054918.json](results/bench_20260204_054918.json)
- Concurrent ACID: [results/bench_20260204_201035.json](results/bench_20260204_201035.json)
- Concurrent (DuckDB fix): [results/bench_20260204_225828.json](results/bench_20260204_225828.json)
- Concurrent (NebulaGraph fix): [results/bench_20260204_222424.json](results/bench_20260204_222424.json)
