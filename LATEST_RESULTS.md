# Graph Database Benchmark Report

**Session:** bench_20260219
**Scale:** small
**Date:** 2026-02-19

## Environment

- Platform: Windows 11 Pro
- Python: 3.13.0
- CPU: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
- Memory: 64 GB

## Summary

| Database | Success | Failed | Timeout | Avg Time (ms) |
|----------|---------|--------|---------|---------------|
| Grafeo | 65 | 0 | 0 | 50.27 |
| Grafeo Server | 31 | 0 | 0 | 30.16 |
| LadybugDB | 64 | 0 | 1 | 238.14 |
| TuringDB | 29 | 2 | 0 | 2,256.83 |

## Algorithm Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| pagerank | **0.08** | 2.13 | 71.63 | 4,447.45 |
| community_detection | **0.10** | 2.13 | 72.64 | 4,448.50 |
| betweenness_centrality | **1.90** | 41.05 | 39.11 | 3,403.43 |
| closeness_centrality | **1.06** | 28.06 | 24.75 | 1,583.60 |

## Concurrent Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| throughput_scaling | **50.43** | — | 1155.44 | — |
| lost_update | **3.48** | — | 120.36 | — |
| read_after_write | **3.45** | — | 112.57 | — |
| concurrent_mixed | **12.28** | — | 148.46 | — |
| concurrent_acid | **67.82** | — | 965.83 | — |

## Hybrid Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| hybrid_graph_to_vector | **158.23** | — | 526.17 | — |
| hybrid_vector_to_graph | **36.65** | — | 177.51 | — |

## Ldbc_Acid Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| acid_atomicity_c | **0.06** | — | 1.90 | — |
| acid_atomicity_rb | **0.07** | — | 2.79 | — |
| acid_g0 | **2.05** | — | 5.57 | — |
| acid_g1a | **6.58** | — | 9.61 | — |
| acid_g1b | **0.07** | — | 2.58 | — |
| acid_g1c | **0.55** | — | 5.11 | — |
| acid_imp | **10.68** | — | 12.57 | — |
| acid_pmp | **10.94** | — | 19.28 | — |
| acid_otv | **0.08** | — | 3.49 | — |
| acid_fr | **5.88** | — | 8.24 | — |
| acid_lu | **1.71** | — | 47.17 | — |
| acid_ws | **0.62** | — | 6.02 | — |

## Ldbc_Graphanalytics Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| ldbc_bfs | **0.04** | 2.66 | 68.63 | FAILED |
| ldbc_pagerank | **0.04** | 1.97 | 68.97 | 4,534.05 |
| ldbc_wcc | **0.05** | 2.11 | 67.94 | 4,535.31 |
| ldbc_cdlp | **0.08** | 2.09 | 68.50 | 4,533.57 |
| ldbc_lcc | **0.06** | 3.35 | 68.42 | 4,535.30 |
| ldbc_sssp | **1.30** | 2.62 | 68.37 | FAILED |

## Ldbc_Snb Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| snb_is1 | **3.82** | — | 178.98 | — |
| snb_is2 | **3.50** | — | 156.50 | — |
| snb_is3 | **51.58** | — | 1670.73 | — |
| snb_is4 | **1.39** | — | 47.05 | — |
| snb_is5 | **1.35** | — | 65.49 | — |
| snb_is6 | **3.09** | — | 172.57 | — |
| snb_is7 | **3.90** | — | 119.24 | — |
| snb_ic1 | **747.19** | — | FAILED | — |
| snb_ic2 | **19.47** | — | 681.43 | — |
| snb_ic3 | **15.55** | — | 1116.69 | — |
| snb_ic6 | **22.10** | — | 895.75 | — |

## Pattern Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| triangle_count | **5.53** | 173.45 | 149.87 | 9,686.07 |
| common_neighbors | **1.47** | 43.59 | 40.77 | 2,467.24 |

## Query Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| aggregation_count | **2.40** | 38.13 | 26.99 | 2,202.24 |
| filter_equality | **0.32** | 15.29 | 1.57 | 882.15 |
| filter_range | **0.89** | 19.26 | 2.24 | 881.89 |

## Storage Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| node_insertion | **0.46** | 3.39 | 42.94 | 131.46 |
| edge_insertion | **14.36** | 44.00 | 16.77 | 577.89 |
| single_read | **0.83** | 82.05 | 47.60 | 4,403.88 |
| batch_read | **1.54** | 9.98 | 2.31 | 53.21 |

## Structure Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| connected_components | **0.91** | 26.60 | 22.33 | 1,453.13 |
| degree_distribution | **0.79** | 26.59 | 22.55 | 1,452.95 |
| graph_density | **0.23** | 1.93 | 1.50 | 87.52 |
| reachability | **2.69** | 72.12 | 58.31 | 3,873.53 |

## Traversal Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| hop_1 | **1.33** | 38.91 | 33.96 | 2,203.50 |
| hop_2 | **1.27** | 32.29 | 27.30 | 1,761.77 |
| bfs | **0.93** | 23.94 | 19.78 | 1,321.70 |
| dfs | **1.29** | 31.64 | 26.85 | 1,760.06 |
| shortest_path | **1.31** | 29.40 | 22.89 | 2,072.17 |

## Vector Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| vector_insert | **41.34** | — | 160.99 | — |
| vector_knn | **162.04** | — | 447.60 | — |
| vector_batch_search | **1612.94** | — | 4458.29 | — |
| vector_recall | **160.27** | — | 446.93 | — |

## Write Benchmarks

| Benchmark | Grafeo (ms) | Grafeo Server (ms) | LadybugDB (ms) | TuringDB (ms) |
|-----------|-------------|---------------------|-----------------|----------------|
| property_update | **0.44** | 98.26 | 28.04 | 2,201.33 |
| edge_add_existing | **0.67** | 3.03 | 2.01 | 133.68 |
| mixed_workload | **1.72** | 107.06 | 76.46 | 6,712.59 |

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
