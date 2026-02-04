# Latest Benchmark Results

Run: 2026-02-04 | Scale: small (10K nodes, 50K edges) | Platform: Windows | **174/186 passed**

## Databases Tested

| Type | Database | Status |
|------|----------|--------|
| Embedded | Grafeo 0.2.6 | Native algorithms |
| Embedded | LadybugDB | NetworkX fallback |
| Embedded | DuckDB | NetworkX fallback |
| Server | Neo4j | Partial support |
| Server | Memgraph | Partial support |
| Server | FalkorDB | Partial support |

## Core Benchmarks (6 databases)

| Benchmark | Grafeo | LadybugDB | DuckDB | Neo4j | Memgraph | FalkorDB |
|-----------|--------|-----------|--------|-------|----------|----------|
| node_insertion | **3.3ms** | 255ms | 6773ms | 60ms | 62ms | 522ms |
| edge_insertion | **7.0ms** | 270ms | 4269ms | 1865ms | 1822ms | 334ms |
| single_read | **0.6ms** | 30ms | 74ms | 283ms | 269ms | 113ms |
| batch_read | 5.7ms | **2.9ms** | 2.8ms | 24ms | 25ms | 10ms |
| bfs | **0.3ms** | 50ms | 25ms | 36ms | 36ms | 20ms |
| dfs | **0.3ms** | 64ms | 32ms | 49ms | 48ms | 25ms |
| hop_1 | **0.8ms** | 81ms | 40ms | 65ms | 61ms | 31ms |
| hop_2 | **0.5ms** | 65ms | 34ms | 49ms | 49ms | 26ms |
| triangle_count | **0.5ms** | 190ms | 46ms | 356ms | 346ms | 31ms |
| common_neighbors | **0.7ms** | 38ms | 48ms | 71ms | 70ms | 37ms |

**Grafeo wins 9/10 core benchmarks** across all 6 databases.

## LDBC Graphanalytics Benchmarks

Standard graph analytics benchmarks from [LDBC Graphanalytics](https://github.com/ldbc/ldbc_graphalytics).

| Benchmark | Grafeo | LadybugDB | DuckDB | Winner |
|-----------|--------|-----------|--------|--------|
| ldbc_bfs | **0.18ms** | 3400ms | 4097ms | Grafeo 19,052x |
| ldbc_pagerank | **0.41ms** | 698ms | 814ms | Grafeo 1,687x |
| ldbc_wcc | **0.69ms** | 697ms | 807ms | Grafeo 1,008x |
| ldbc_cdlp | **0.75ms** | 707ms | 852ms | Grafeo 946x |
| ldbc_lcc | **0.58ms** | 878ms | 818ms | Grafeo 1,417x |
| ldbc_sssp | **5.36ms** | 4646ms | 2467ms | Grafeo 460x |

**Grafeo wins all 6 LDBC benchmarks** with native algorithm implementations.

## Summary

### Grafeo Advantages

- **Fastest embedded graph database** tested
- **Native LDBC algorithm support** (460x - 19,052x faster than NetworkX fallback)
- Excellent write performance (node/edge insertion)
- Best traversal performance (BFS, DFS, hops)
- Best pattern matching (triangles, neighbors)

### Server Databases

Neo4j, Memgraph, and FalkorDB showed:

- Higher latency due to network overhead
- Some algorithm benchmarks failed (missing implementations)
- Better suited for distributed/production workloads

### Where Others Win

- **DuckDB/LadybugDB**: Batch reads (SQL optimization)
- **FalkorDB**: Competitive on some traversals

## Implementation Notes

### Grafeo Native Algorithms

Grafeo 0.2.6 provides native implementations for all LDBC algorithms:

- `bfs_layers` - BFS with depth tracking
- `pagerank` - PageRank iteration
- `connected_components` - Weakly Connected Components
- `label_propagation` - Community Detection (CDLP)
- `local_clustering_coefficient` - Local Clustering Coefficient
- `dijkstra` - Single-Source Shortest Paths

### NetworkX Fallback

Databases without native algorithm support use NetworkX:

1. Graph extracted via `get_neighbors()` calls
2. NetworkX algorithm runs on extracted graph
3. Extraction overhead included in benchmark time (fair comparison)

---

Full results: [results/bench_20260203_235328.json](results/bench_20260203_235328.json)
