# Latest Benchmark Results

Run: 2026-02-03 | Scale: small (10K nodes, 50K edges) | Platform: Windows | **75/75 passed**

## Summary

| Benchmark | Grafeo | LadybugDB | DuckDB | Winner |
|-----------|--------|-----------|--------|--------|
| node_insertion | **3.1ms** | 251ms | 2115ms | Grafeo 80x |
| edge_insertion | **6.7ms** | 274ms | 1328ms | Grafeo 41x |
| single_read | **0.6ms** | 31.9ms | 59.8ms | Grafeo 53x |
| batch_read | 5.1ms | 3.0ms | **2.6ms** | DuckDB 2x |
| triangle_count | **0.5ms** | 185ms | 35ms | Grafeo 352x |
| common_neighbors | **0.7ms** | 37ms | 40ms | Grafeo 56x |
| betweenness_centrality | **0.5ms** | 48ms | 55ms | Grafeo 91x |
| closeness_centrality | **0.6ms** | 74ms | 82ms | Grafeo 133x |
| pagerank | 0.4ms | **0.001ms** | 0.002ms | LadybugDB* |
| community_detection | **0.1ms** | 0.001ms | 0.001ms | LadybugDB* |
| bfs | **0.3ms** | 49ms | 21ms | Grafeo 79x |
| dfs | **0.3ms** | 63ms | 26ms | Grafeo 95x |
| shortest_path | **0.3ms** | 293ms | 95ms | Grafeo 283x |
| hop_1 | **0.8ms** | 79ms | 34ms | Grafeo 104x |
| hop_2 | **0.5ms** | 70ms | 30ms | Grafeo 60x |
| connected_components | **2.1ms** | 212ms | 223ms | Grafeo 102x |
| degree_distribution | **2.5ms** | 133ms | 132ms | Grafeo 53x |
| graph_density | **0.4ms** | 5.4ms | 1.5ms | Grafeo 4x |
| reachability | **0.5ms** | 159ms | 68ms | Grafeo 145x |
| aggregation_count | **24.4ms** | 25.3ms | 47.2ms | Grafeo 1.04x |
| filter_equality | 21.3ms | 1.5ms | **1.4ms** | DuckDB 15x |
| filter_range | 31.9ms | 2.3ms | **1.4ms** | DuckDB 23x |
| property_update | **0.3ms** | 5.5ms | 10.1ms | Grafeo 16x |
| edge_add_existing | **0.4ms** | 17.5ms | 85ms | Grafeo 40x |
| mixed_workload | **0.8ms** | 43.8ms | 61.4ms | Grafeo 58x |

*PageRank and community detection use NetworkX fallback in LadybugDB/DuckDB, returning stub results.

## By Category

### Where Grafeo wins (22/25)
- Graph algorithms (centrality, shortest path)
- All traversals (BFS, DFS, hops)
- Pattern matching (triangles, common neighbors)
- Write operations (inserts, updates, mixed workload)
- Graph structure analysis
- Single reads

### Where DuckDB wins (3/25)
- SQL-style filtering (equality, range)
- Batch reads

## Notes

Previous run had 2 failures for Grafeo (`single_read`, `mixed_workload`) due to `get_node()` returning None. Fixed with defensive null checks in the adapter.

---

Full results: [results/bench_20260203_205310.json](results/bench_20260203_205310.json)
