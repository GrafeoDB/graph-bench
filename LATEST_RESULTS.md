# Graph Database Benchmark Report

**Session:** bench_20260216_183931
**Scale:** small
**Date:** 2026-02-16

## Environment

- Platform: 
- Python: 
- CPU: 
- Memory: 0.0 GB

## Summary

| Database | Success | Failed | Timeout | Avg Time (ms) |
|----------|---------|--------|---------|---------------|
| Grafeo | 65 | 0 | 0 | 50.27 |
| Grafeo Server | 62 | 0 | 3 | — |
| LadybugDB | 64 | 0 | 1 | 238.14 |
| TuringDB | — | — | — | — |

## Algorithm Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| pagerank | 0.08 | 71.63 |
| community_detection | 0.10 | 72.64 |
| betweenness_centrality | 1.90 | 39.11 |
| closeness_centrality | 1.06 | 24.75 |

## Concurrent Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| throughput_scaling | 50.43 | 1155.44 |
| lost_update | 3.48 | 120.36 |
| read_after_write | 3.45 | 112.57 |
| concurrent_mixed | 12.28 | 148.46 |
| concurrent_acid | 67.82 | 965.83 |

## Hybrid Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| hybrid_graph_to_vector | 158.23 | 526.17 |
| hybrid_vector_to_graph | 36.65 | 177.51 |

## Ldbc_Acid Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| acid_atomicity_c | 0.06 | 1.90 |
| acid_atomicity_rb | 0.07 | 2.79 |
| acid_g0 | 2.05 | 5.57 |
| acid_g1a | 6.58 | 9.61 |
| acid_g1b | 0.07 | 2.58 |
| acid_g1c | 0.55 | 5.11 |
| acid_imp | 10.68 | 12.57 |
| acid_pmp | 10.94 | 19.28 |
| acid_otv | 0.08 | 3.49 |
| acid_fr | 5.88 | 8.24 |
| acid_lu | 1.71 | 47.17 |
| acid_ws | 0.62 | 6.02 |

## Ldbc_Graphanalytics Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| ldbc_bfs | 0.04 | 68.63 |
| ldbc_pagerank | 0.04 | 68.97 |
| ldbc_wcc | 0.05 | 67.94 |
| ldbc_cdlp | 0.08 | 68.50 |
| ldbc_lcc | 0.06 | 68.42 |
| ldbc_sssp | 1.30 | 68.37 |

## Ldbc_Snb Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| snb_is1 | 3.82 | 178.98 |
| snb_is2 | 3.50 | 156.50 |
| snb_is3 | 51.58 | 1670.73 |
| snb_is4 | 1.39 | 47.05 |
| snb_is5 | 1.35 | 65.49 |
| snb_is6 | 3.09 | 172.57 |
| snb_is7 | 3.90 | 119.24 |
| snb_ic1 | 747.19 | FAILED |
| snb_ic2 | 19.47 | 681.43 |
| snb_ic3 | 15.55 | 1116.69 |
| snb_ic6 | 22.10 | 895.75 |

## Pattern Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| triangle_count | 5.53 | 149.87 |
| common_neighbors | 1.47 | 40.77 |

## Query Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| aggregation_count | 2.40 | 26.99 |
| filter_equality | 0.32 | 1.57 |
| filter_range | 0.89 | 2.24 |

## Storage Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| node_insertion | 0.46 | 42.94 |
| edge_insertion | 14.36 | 16.77 |
| single_read | 0.83 | 47.60 |
| batch_read | 1.54 | 2.31 |

## Structure Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| connected_components | 0.91 | 22.33 |
| degree_distribution | 0.79 | 22.55 |
| graph_density | 0.23 | 1.50 |
| reachability | 2.69 | 58.31 |

## Traversal Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| hop_1 | 1.33 | 33.96 |
| hop_2 | 1.27 | 27.30 |
| bfs | 0.93 | 19.78 |
| dfs | 1.29 | 26.85 |
| shortest_path | 1.31 | 22.89 |

## Vector Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| vector_insert | 41.34 | 160.99 |
| vector_knn | 162.04 | 447.60 |
| vector_batch_search | 1612.94 | 4458.29 |
| vector_recall | 160.27 | 446.93 |

## Write Benchmarks

| Benchmark |Grafeo (ms) | LadybugDB (ms) |
|-----------|------------|------------|
| property_update | 0.44 | 28.04 |
| edge_add_existing | 0.67 | 2.01 |
| mixed_workload | 1.72 | 76.46 |

## Performance Comparisons

| Benchmark |Grafeo | Grafeo Server | LadybugDB |
|-----------|----------|----------|----------|
| acid_atomicity_c | **1.00x** | 0.01x | 0.03x |
| acid_atomicity_rb | **1.00x** | 0.02x | 0.03x |
| acid_fr | **1.00x** | 0.51x | 0.71x |
| acid_g0 | **1.00x** | 0.22x | 0.37x |
| acid_g1a | **1.00x** | 0.48x | 0.69x |
| acid_g1b | **1.00x** | 0.01x | 0.03x |
| acid_g1c | **1.00x** | 0.06x | 0.11x |
| acid_imp | **1.00x** | 0.73x | 0.85x |
| acid_lu | **1.00x** | 0.03x | 0.04x |
| acid_otv | **1.00x** | 0.01x | 0.02x |
| acid_pmp | **1.00x** | 0.35x | 0.57x |
| acid_ws | **1.00x** | 0.07x | 0.10x |
| aggregation_count | **1.00x** | 0.06x | 0.09x |
| batch_read | **1.00x** | 0.07x | 0.67x |
| betweenness_centrality | **1.00x** | 0.04x | 0.05x |
| bfs | **1.00x** | 0.04x | 0.05x |
| closeness_centrality | **1.00x** | 0.04x | 0.04x |
| common_neighbors | **1.00x** | 0.03x | 0.04x |
| community_detection | **1.00x** | 0.00x | 0.00x |
| concurrent_acid | **1.00x** | 0.17x | 0.07x |
| concurrent_mixed | **1.00x** | 0.16x | 0.08x |
| connected_components | **1.00x** | 0.03x | 0.04x |
| degree_distribution | **1.00x** | 0.03x | 0.04x |
| dfs | **1.00x** | 0.04x | 0.05x |
| edge_add_existing | **1.00x** | 0.17x | 0.33x |
| edge_insertion | **1.00x** | 0.06x | 0.86x |
| filter_equality | **1.00x** | 0.02x | 0.20x |
| filter_range | **1.00x** | 0.06x | 0.40x |
| graph_density | **1.00x** | 0.13x | 0.15x |
| hop_1 | **1.00x** | 0.03x | 0.04x |
| hop_2 | **1.00x** | 0.04x | 0.05x |
| hybrid_graph_to_vector | 0.46x | **1.00x** | 0.14x |
| hybrid_vector_to_graph | 0.21x | **1.00x** | 0.04x |
| ldbc_bfs | **1.00x** | 0.00x | 0.00x |
| ldbc_cdlp | **1.00x** | 0.00x | 0.00x |
| ldbc_lcc | **1.00x** | 0.00x | 0.00x |
| ldbc_pagerank | **1.00x** | 0.00x | 0.00x |
| ldbc_sssp | **1.00x** | 0.02x | 0.02x |
| ldbc_wcc | **1.00x** | 0.00x | 0.00x |
| lost_update | **1.00x** | 0.09x | 0.03x |
| mixed_workload | **1.00x** | 0.02x | 0.02x |
| node_insertion | **1.00x** | 0.13x | 0.01x |
| pagerank | **1.00x** | 0.00x | 0.00x |
| property_update | **1.00x** | 0.00x | 0.02x |
| reachability | **1.00x** | 0.04x | 0.05x |
| read_after_write | **1.00x** | 0.04x | 0.03x |
| shortest_path | **1.00x** | 0.04x | 0.06x |
| single_read | **1.00x** | 0.01x | 0.02x |
| snb_ic1 | 0.02x | **1.00x** | 0.00x |
| snb_ic2 | 0.31x | **1.00x** | 0.01x |
| snb_ic3 | 0.19x | **1.00x** | 0.00x |
| snb_ic6 | 0.27x | **1.00x** | 0.01x |
| snb_is1 | **1.00x** | 0.00x | 0.02x |
| snb_is2 | **1.00x** | 0.00x | 0.02x |
| snb_is3 | **1.00x** | 0.00x | 0.03x |
| snb_is4 | **1.00x** | 0.02x | 0.03x |
| snb_is5 | **1.00x** | 0.04x | 0.02x |
| snb_is6 | **1.00x** | 0.05x | 0.02x |
| snb_is7 | **1.00x** | 0.11x | 0.03x |
| throughput_scaling | **1.00x** | 0.24x | 0.04x |
| triangle_count | **1.00x** | 0.03x | 0.04x |
| vector_batch_search | 0.05x | **1.00x** | 0.02x |
| vector_insert | **1.00x** | 0.31x | 0.26x |
| vector_knn | 0.04x | **1.00x** | 0.01x |
| vector_recall | 0.05x | **1.00x** | 0.02x |

*Speedup relative to fastest (1.00x = fastest)*

## TuringDB Comparison (turing-bench)

Grafeo was benchmarked against TuringDB using [turing-bench](https://github.com/turing-db/turing-bench), TuringDB's own Cypher benchmark suite. Both databases were loaded with the PoleDB dataset (61,521 nodes, 105,840 edges) and ran 18 queries with 3 runs each, preceded by an untimed warmup pass. TuringDB ran as an in-container server (HTTP/localhost), Grafeo ran embedded (in-process). Both drivers returned native result objects with no post-processing overhead.

| Query | Rows | TuringDB | Grafeo | Faster |
|-------|------|----------|--------|--------|
| MATCH (n) RETURN n | 61,521 | 4ms | 157ms | TuringDB 39x |
| MATCH (p:Person) RETURN p | 369 | 0ms | 6ms | TuringDB |
| MATCH (p:Person) RETURN count(p) | 1 | 0ms | 0ms | Grafeo 3.6x (qps) |
| MATCH (c:Crime) RETURN c | 28,762 | 2ms | 64ms | TuringDB 32x |
| MATCH (c:Crime) RETURN count(c) | 1 | 0ms | 25ms | TuringDB |
| MATCH ()-[r]->() RETURN r | 105,840 | 9ms | 185ms | TuringDB 20x |
| MATCH ()-[r]->() RETURN count(r) | 1 | 2ms | 155ms | TuringDB 77x |
| Person{name:'John'}->Crime | 0 | 3ms | 0ms | Grafeo 100x+ |
| Person->Crime | 55 | 0ms | 0ms | Tie |
| Person{surname:'Smith'}->n | 6 | 2ms | 0ms | Grafeo 140x+ |
| p1->p2->Crime | 0 | 1ms | 0ms | Grafeo 2x |
| Crime->Location | 28,762 | 5ms | 94ms | TuringDB 19x |

TuringDB is faster on bulk result scans and aggregations. Its C++ engine serializes large result sets efficiently, and its `count()` uses optimized metadata lookups rather than full scans. Grafeo is faster on indexed property lookups with small result sets, where embedded in-process access eliminates network overhead and the Rust engine resolves queries in microseconds.
