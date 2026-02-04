# Latest Benchmark Results

Run: 2026-02-04 | Scale: small (10K nodes, 50K edges) | Platform: Windows

## LDBC Graph Analytics

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB |
|-----------|---------|------------|---------|-------|----------|----------|
| BFS | 0.12ms | 3273ms | 3722ms | 115ms | 5.9ms | 3252ms |
| PageRank | 0.25ms | 661ms | 719ms | 64ms | 13.7ms | 659ms |
| WCC | 0.55ms | 651ms | 718ms | 30ms | 13.1ms | 658ms |
| CDLP | 0.51ms | 681ms | 726ms | 43ms | 15.7ms | 664ms |
| LCC | 0.31ms | 657ms | 709ms | - | 53.1ms | 660ms |
| SSSP | 2.69ms | 1981ms | 2151ms | 52ms | 6.4ms | 1963ms |

## Write Operations

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB |
|-----------|---------|------------|---------|-------|----------|----------|
| node_insertion | 3.3ms | 255ms | 6773ms | 60ms | 62ms | 522ms |
| edge_insertion | 7.0ms | 270ms | 4269ms | 1865ms | 1822ms | 334ms |

## Read Operations

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB |
|-----------|---------|------------|---------|-------|----------|----------|
| single_read | 0.6ms | 30ms | 74ms | 283ms | 269ms | 113ms |
| batch_read | 5.7ms | 2.9ms | 2.8ms | 24ms | 25ms | 10ms |

## Traversals

| Benchmark | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB |
|-----------|---------|------------|---------|-------|----------|----------|
| bfs | 0.3ms | 50ms | 25ms | 36ms | 36ms | 20ms |
| dfs | 0.3ms | 64ms | 32ms | 49ms | 48ms | 25ms |
| hop_1 | 0.8ms | 81ms | 40ms | 65ms | 61ms | 31ms |
| hop_2 | 0.5ms | 65ms | 34ms | 49ms | 49ms | 26ms |
| triangle_count | 0.5ms | 190ms | 46ms | 356ms | 346ms | 31ms |
| common_neighbors | 0.7ms | 38ms | 48ms | 71ms | 70ms | 37ms |

---

\* Embedded databases run in-process without network overhead. Server databases (Neo4j, Memgraph, FalkorDB) include network latency, which is inherent to their architecture. This makes direct timing comparisons between embedded and server databases not entirely apples-to-apples.

"-" indicates algorithm failed on this dataset.

---

## Feature Availability

| Feature | Grafeo* | LadybugDB* | DuckDB* | Neo4j | Memgraph | FalkorDB |
|---------|---------|------------|---------|-------|----------|----------|
| Native PageRank | ✅ | NetworkX | NetworkX | ✅ | ✅ | NetworkX |
| Native WCC | ✅ | NetworkX | NetworkX | ✅ | ✅ | NetworkX |
| Native CDLP | ✅ | NetworkX | NetworkX | ✅ | ✅ | NetworkX |
| Native LCC | ✅ | NetworkX | NetworkX | ❌ | ✅ | NetworkX |
| Native SSSP | ✅ | NetworkX | NetworkX | ✅ | ✅ | NetworkX |
| Native BFS | ✅ | NetworkX | NetworkX | ✅ | ✅ | NetworkX |
| Query Language | GQL (ISO) | Cypher | SQL/PGQ | Cypher | Cypher | Cypher |
| Deployment | Embedded | Embedded | Embedded | Server | Server | Server |

---

Full results: [results/bench_20260204_054918.json](results/bench_20260204_054918.json)
