f31# GrafeoDB Benchmark Results

**Scale Factor:** SF0.1 (1,000 nodes, 18,000 edges)
**Date:**
**Platform:** Windows 11 Pro, AMD64, 64 GB RAM, Python 3.13

**Methodology:** 1 warmup + 3 measured iterations, median reported. 60s timeout per benchmark. Native implementations only — databases without native support for a benchmark show "—" (no NetworkX or brute-force fallback results).

---

## Summary

<table>
<tr>
  <th rowspan="2">Category</th>
  <th colspan="8" align="center">Server</th>
  <th colspan="2" align="center">Embedded</th>
</tr>
<tr>
  <th>ArangoDB</th>
  <th>FalkorDB</th>
  <th>Grafeo Server</th>
  <th>Memgraph</th>
  <th>Neo4j</th>
  <th>NebulaGraph</th>
  <th>TuGraph</th>
  <th>TuringDB</th>
  <th>Grafeo</th>
  <th>LadybugDB</th>
</tr>
<tr>
  <td>Storage</td>
  <td>241</td><td>129</td><td>176</td><td>155</td><td>221</td><td>157</td><td>270</td><td>7,974</td>
  <td>25</td><td>115</td>
</tr>
<tr>
  <td>Write</td>
  <td>10,000</td><td>85</td><td>199</td><td>102</td><td>426</td><td>187</td><td>128</td><td>9,246</td>
  <td>2</td><td>115</td>
</tr>
<tr>
  <td>Traversal</td>
  <td>4,657</td><td>93</td><td>164</td><td>130</td><td>234</td><td>274</td><td>206</td><td>9,090</td>
  <td>1</td><td>136</td>
</tr>
<tr>
  <td>Query</td>
  <td>3,975</td><td>54</td><td>75</td><td>70</td><td>115</td><td>205</td><td>351</td><td>3,968</td>
  <td>3</td><td>31</td>
</tr>
<tr>
  <td>Pattern</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Structure</td>
  <td>2,653</td><td>94</td><td>88</td><td>120</td><td>310</td><td>266</td><td>173</td><td>7,039</td>
  <td>1</td><td>108</td>
</tr>
<tr>
  <td>Algorithms</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>—</td>
  <td></td><td>—</td>
</tr>
<tr>
  <td>LDBC Graphanalytics</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>—</td>
  <td></td><td>—</td>
</tr>
<tr>
  <td>LDBC SNB Interactive</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>LDBC ACID</td>
  <td>2,117</td><td>90</td><td>190</td><td>109</td><td>268</td><td>174</td><td>115</td><td>—</td>
  <td>38</td><td>127</td>
</tr>
<tr>
  <td>Concurrent</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Vector †</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Hybrid †</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Turing-Bench ‡</td>
  <td>—</td><td>—</td><td>—</td><td>—</td><td></td><td>—</td><td>—</td><td>29</td>
  <td>669</td><td>—</td>
</tr>
</table>

All values in milliseconds (total per category). **Bold** = fastest in section.
† Brute-force fallback — no native vector index. Times reflect data retrieval + Python cosine similarity.
‡ Turing-Bench uses PoleDB dataset (61K nodes, 106K edges), not SF0.1.

---

## Embedded Databases

### Storage

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| node_insertion | 0.4 ms | 46.3 ms |
| edge_insertion | 21.4 ms | 17.9 ms |
| single_read | 0.5 ms | 48.7 ms |
| batch_read | 2.5 ms | 2.3 ms |
| **Total (ms)** | **24.8** | **115.2** |
| **Peak Memory** | | |

### Write

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| property_update | 0.3 ms | 29.0 ms |
| edge_add_existing | 0.5 ms | 1.9 ms |
| mixed_workload | 1.5 ms | 83.7 ms |
| **Total (ms)** | **2.3** | **114.6** |
| **Peak Memory** | | |

### Traversal

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| hop_1 | 0.4 ms | 35.1 ms |
| hop_2 | 0.3 ms | 27.9 ms |
| bfs | 0.2 ms | 20.6 ms |
| dfs | 0.2 ms | 28.8 ms |
| shortest_path | 0.2 ms | 23.9 ms |
| **Total (ms)** | **1.3** | **136.3** |
| **Peak Memory** | | |

### Query

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| aggregation_count | 2.2 ms | 27.5 ms |
| filter_equality | 0.1 ms | 1.5 ms |
| filter_range | 0.2 ms | 2.4 ms |
| **Total (ms)** | **2.5** | **31.4** |
| **Peak Memory** | | |

### Pattern

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| triangle_count | | |
| common_neighbors | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### Structure

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| connected_components | 0.5 ms | 23.6 ms |
| degree_distribution | 0.3 ms | 22.6 ms |
| graph_density | 0.0 ms | 1.7 ms |
| reachability | 0.3 ms | 60.0 ms |
| **Total (ms)** | **1.1** | **107.9** |
| **Peak Memory** | | |

### Algorithms

Native implementations only.

| Benchmark | Grafeo |
|-----------|--------|
| pagerank | |
| community_detection | |
| **Total (ms)** | |
| **Peak Memory** | |

### LDBC Graphanalytics

Native implementations only.

| Benchmark | Grafeo |
|-----------|--------|
| ldbc_bfs | |
| ldbc_pagerank | |
| ldbc_wcc | |
| ldbc_cdlp | |
| ldbc_lcc | |
| ldbc_sssp | |
| **Total (ms)** | |
| **Peak Memory** | |

### LDBC SNB Interactive

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| snb_is1 | | |
| snb_is2 | | |
| snb_is3 | | |
| snb_is4 | | |
| snb_is5 | | |
| snb_is6 | | |
| snb_is7 | | |
| snb_ic1 | | |
| snb_ic2 | | |
| snb_ic3 | | |
| snb_ic6 | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### LDBC ACID

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| acid_atomicity_c | 0.0 ms | 2.3 ms |
| acid_atomicity_rb | 0.0 ms | 2.9 ms |
| acid_g0 | 2.0 ms | 5.3 ms |
| acid_g1a | 6.6 ms | 9.7 ms |
| acid_g1b | 0.0 ms | 2.6 ms |
| acid_g1c | 0.5 ms | 5.1 ms |
| acid_imp | 10.5 ms | 12.8 ms |
| acid_pmp | 10.9 ms | 19.8 ms |
| acid_otv | 0.0 ms | 3.9 ms |
| acid_fr | 5.6 ms | 8.2 ms |
| acid_lu | 1.5 ms | 48.3 ms |
| acid_ws | 0.6 ms | 6.1 ms |
| **Total (ms)** | **38.2** | **127.0** |
| **Peak Memory** | | |

### Concurrent

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| throughput_scaling | | |
| lost_update | | |
| read_after_write | | |
| concurrent_mixed | | |
| concurrent_acid | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### Vector

Brute-force fallback — no native vector index. Times include data retrieval + Python cosine similarity.

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| vector_insert | | |
| vector_knn | | |
| vector_batch_search | | |
| vector_recall | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### Hybrid

Brute-force vector fallback. Graph traversal is native.

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| hybrid_graph_to_vector | | |
| hybrid_vector_to_graph | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### Turing-Bench (PoleDB)

External benchmark suite from [turing-bench](https://github.com/turing-db/turing-bench). PoleDB dataset: 61,521 nodes, 105,840 edges. 3 runs, median reported. LadybugDB was not tested.

| Benchmark | Rows | Grafeo | LadybugDB |
|-----------|------|--------|-----------|
| Full node scan | 61,521 | 144.0 ms | — |
| Label scan: Person | 369 | 0.0 ms | — |
| Count: Person | 1 | 0.0 ms | — |
| Label scan: Crime | 28,762 | 64.0 ms | — |
| Count: Crime | 1 | 23.0 ms | — |
| Full edge scan | 105,840 | 188.0 ms | — |
| Count: Edges | 1 | 158.0 ms | — |
| Indexed lookup → Crime | 0 | 0.0 ms | — |
| Person → Crime | 55 | 0.0 ms | — |
| Indexed lookup → Any | 6 | 0.0 ms | — |
| 2-hop → Crime | 0 | 0.0 ms | — |
| Crime → Location | 28,762 | 92.0 ms | — |
| **Total (ms)** | — | **669.0** | — |
| **Peak Memory** | — | | — |

---

## Server Databases

### Storage

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| node_insertion | 3.3 ms | 17.8 ms | 3.2 ms | 1.8 ms | 44.0 ms | 7.1 ms | 2.2 ms | 266.8 ms |
| edge_insertion | 81.3 ms | 53.1 ms | 58.0 ms | 57.7 ms | 47.4 ms | 160.6 ms | 7.7 ms | 3,239.4 ms |
| single_read | 80.6 ms | 123.6 ms | 72.6 ms | 58.3 ms | 104.8 ms | 81.5 ms | 143.6 ms | 4,412.8 ms |
| batch_read | 10.6 ms | 26.2 ms | 21.1 ms | 10.7 ms | 44.5 ms | 20.8 ms | 3.6 ms | 54.5 ms |
| **Total (ms)** | **175.8** | **220.7** | **154.9** | **128.5** | **240.7** | **270.0** | **157.1** | **7,973.5** |
| **Peak Memory** | | | | | | | | |

### Write

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| property_update | 99.7 ms | 218.5 ms | 34.3 ms | 27.3 ms | 6,607.5 ms | 40.5 ms | 34.7 ms | 2,204.5 ms |
| edge_add_existing | 3.0 ms | 10.2 ms | 1.7 ms | 1.6 ms | 43.5 ms | 3.5 ms | 1.8 ms | 130.7 ms |
| mixed_workload | 95.9 ms | 197.5 ms | 65.8 ms | 55.8 ms | 3,349.3 ms | 83.7 ms | 150.8 ms | 6,911.1 ms |
| **Total (ms)** | **198.6** | **426.2** | **101.8** | **84.7** | **10,000.3** | **127.7** | **187.3** | **9,246.3** |
| **Peak Memory** | | | | | | | | |

### Traversal

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| hop_1 | 39.4 ms | 69.2 ms | 42.4 ms | 29.1 ms | 2,213.2 ms | 51.3 ms | 97.1 ms | 2,210.8 ms |
| hop_2 | 47.4 ms | 54.4 ms | 32.1 ms | 23.7 ms | 885.1 ms | 40.5 ms | 60.1 ms | 1,762.5 ms |
| bfs | 24.0 ms | 40.8 ms | 24.9 ms | 17.9 ms | 438.5 ms | 29.5 ms | 40.3 ms | 1,262.0 ms |
| dfs | 24.4 ms | 52.5 ms | 30.1 ms | 22.7 ms | 680.9 ms | 39.9 ms | 76.7 ms | 1,763.7 ms |
| shortest_path | 29.0 ms | 17.3 ms | TIMEOUT | FAILED | 439.3 ms | 44.5 ms | FAILED | 2,091.0 ms |
| **Total (ms)** | **164.2** | **234.2** | **129.5** | **93.4** | **4,657.0** | **205.7** | **274.2** | **9,090.0** |
| **Peak Memory** | | | | | | | | |

### Query

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| aggregation_count | 38.9 ms | 52.6 ms | 31.8 ms | 26.1 ms | 2,208.2 ms | 304.6 ms | 175.6 ms | 2,208.0 ms |
| filter_equality | 15.7 ms | 24.2 ms | 13.2 ms | 12.2 ms | 881.1 ms | 16.4 ms | 14.5 ms | 880.5 ms |
| filter_range | 20.3 ms | 37.9 ms | 25.2 ms | 16.1 ms | 885.4 ms | 30.3 ms | 14.7 ms | 879.8 ms |
| **Total (ms)** | **74.9** | **114.7** | **70.2** | **54.4** | **3,974.7** | **351.3** | **204.8** | **3,968.3** |
| **Peak Memory** | | | | | | | | |

### Pattern

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| triangle_count | | | | | | | | |
| common_neighbors | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Structure

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| connected_components | 14.2 ms | 88.6 ms | 28.4 ms | 18.4 ms | 266.7 ms | 33.8 ms | 42.3 ms | 1,458.1 ms |
| degree_distribution | 26.9 ms | 69.0 ms | 24.9 ms | 19.2 ms | 1,462.6 ms | 34.2 ms | 64.8 ms | 1,456.6 ms |
| graph_density | 2.0 ms | 3.2 ms | 1.6 ms | 1.1 ms | 87.2 ms | 8.6 ms | 40.1 ms | 87.4 ms |
| reachability | 45.3 ms | 148.9 ms | 65.4 ms | 55.2 ms | 836.5 ms | 95.9 ms | 118.4 ms | 4,036.7 ms |
| **Total (ms)** | **88.4** | **309.7** | **120.3** | **93.9** | **2,653.0** | **172.5** | **265.6** | **7,038.8** |
| **Peak Memory** | | | | | | | | |

### Algorithms

Native implementations only.

| Benchmark | ArangoDB | FalkorDB | Grafeo Server | Memgraph | Neo4j | NebulaGraph | TuGraph |
|-----------|----------|----------|---------------|----------|-------|-------------|---------|
| pagerank | | | | | | | |
| community_detection | | | | | | | |
| **Total (ms)** | | | | | | | |
| **Peak Memory** | | | | | | | |

### LDBC Graphanalytics

Native implementations only.

| Benchmark | ArangoDB | FalkorDB | Grafeo Server | Memgraph | Neo4j | NebulaGraph | TuGraph |
|-----------|----------|----------|---------------|----------|-------|-------------|---------|
| ldbc_bfs | | | | | | | |
| ldbc_pagerank | | | | | | | |
| ldbc_wcc | | | | | | | |
| ldbc_cdlp | | | | | | | |
| ldbc_lcc | | | | | | | |
| ldbc_sssp | | | | | | | |
| **Total (ms)** | | | | | | | |
| **Peak Memory** | | | | | | | |

### LDBC SNB Interactive

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| snb_is1 | | | | | | | | |
| snb_is2 | | | | | | | | |
| snb_is3 | | | | | | | | |
| snb_is4 | | | | | | | | |
| snb_is5 | | | | | | | | |
| snb_is6 | | | | | | | | |
| snb_is7 | | | | | | | | |
| snb_ic1 | | | | | | | | |
| snb_ic2 | | | | | | | | |
| snb_ic3 | | | | | | | | |
| snb_ic6 | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### LDBC ACID

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| acid_atomicity_c | 5.4 ms | 8.6 ms | 2.3 ms | 1.8 ms | 90.0 ms | 2.7 ms | 4.2 ms | — |
| acid_atomicity_rb | 7.9 ms | 9.9 ms | 3.6 ms | 2.8 ms | 133.9 ms | 4.3 ms | 4.0 ms | — |
| acid_g0 | 11.7 ms | 15.5 ms | 4.9 ms | 4.4 ms | 132.1 ms | 5.7 ms | 11.1 ms | — |
| acid_g1a | 15.1 ms | 17.8 ms | 10.4 ms | 9.7 ms | 137.7 ms | 10.5 ms | 15.0 ms | — |
| acid_g1b | 6.9 ms | 10.7 ms | 3.1 ms | 2.3 ms | 130.6 ms | 3.4 ms | 5.4 ms | — |
| acid_g1c | 12.0 ms | 15.7 ms | 5.3 ms | 4.2 ms | 137.7 ms | 5.9 ms | 12.3 ms | — |
| acid_imp | 16.0 ms | 15.9 ms | 13.2 ms | 12.5 ms | 87.3 ms | 13.3 ms | 18.7 ms | — |
| acid_pmp | 29.8 ms | 36.2 ms | 21.2 ms | 19.6 ms | 267.0 ms | 23.8 ms | 35.1 ms | — |
| acid_otv | 9.8 ms | 14.1 ms | 4.6 ms | 3.4 ms | 181.6 ms | 5.1 ms | 7.9 ms | — |
| acid_fr | 12.2 ms | 13.8 ms | 8.8 ms | 7.9 ms | 174.2 ms | 9.0 ms | 14.2 ms | — |
| acid_lu | 50.0 ms | 93.9 ms | 26.4 ms | 16.4 ms | 506.1 ms | 24.2 ms | 34.2 ms | — |
| acid_ws | 12.8 ms | 15.7 ms | 5.6 ms | 5.0 ms | 138.4 ms | 6.6 ms | 12.1 ms | — |
| **Total (ms)** | **189.6** | **267.8** | **109.4** | **90.0** | **2,116.6** | **114.5** | **174.2** | **—** |
| **Peak Memory** | | | | | | | | |

### Concurrent

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| throughput_scaling | | | | | | | | |
| lost_update | | | | | | | | |
| read_after_write | | | | | | | | |
| concurrent_mixed | | | | | | | | |
| concurrent_acid | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Vector

Brute-force fallback — no native vector index. Times include data retrieval + Python cosine similarity.

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| vector_insert | | | | | | | | |
| vector_knn | | | | | | | | |
| vector_batch_search | | | | | | | | |
| vector_recall | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Hybrid

Brute-force vector fallback. Graph traversal is native.

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| hybrid_graph_to_vector | | | | | | | | |
| hybrid_vector_to_graph | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Turing-Bench (PoleDB)

External benchmark suite from [turing-bench](https://github.com/turing-db/turing-bench). PoleDB dataset: 61,521 nodes, 105,840 edges. 3 runs, median reported. Only TuringDB tested so far — see Embedded section for Grafeo's results.

| Benchmark | Rows | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| Full node scan | 61,521 | — | | — | — | — | — | — | 4.0 ms |
| Label scan: Person | 369 | — | | — | — | — | — | — | 0.0 ms |
| Count: Person | 1 | — | | — | — | — | — | — | 0.0 ms |
| Label scan: Crime | 28,762 | — | | — | — | — | — | — | 2.0 ms |
| Count: Crime | 1 | — | | — | — | — | — | — | 0.0 ms |
| Full edge scan | 105,840 | — | | — | — | — | — | — | 9.0 ms |
| Count: Edges | 1 | — | | — | — | — | — | — | 2.0 ms |
| Indexed lookup → Crime | 0 | — | | — | — | — | — | — | 3.0 ms |
| Person → Crime | 55 | — | | — | — | — | — | — | 1.0 ms |
| Indexed lookup → Any | 6 | — | | — | — | — | — | — | 2.0 ms |
| 2-hop → Crime | 0 | — | | — | — | — | — | — | 1.0 ms |
| Crime → Location | 28,762 | — | | — | — | — | — | — | 5.0 ms |
| **Total (ms)** | — | — | | — | — | — | — | — | **29.0** |
| **Peak Memory** | — | — | | — | — | — | — | — | |
