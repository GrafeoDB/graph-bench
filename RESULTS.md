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
  <td>12,177</td><td>44</td><td>226</td><td>225</td><td>322</td><td>140</td><td>282</td><td>12,344</td>
  <td>1</td><td>201</td>
</tr>
<tr>
  <td>Structure</td>
  <td>2,653</td><td>94</td><td>88</td><td>120</td><td>310</td><td>266</td><td>173</td><td>7,039</td>
  <td>1</td><td>108</td>
</tr>
<tr>
  <td>Algorithms</td>
  <td>—</td><td>—</td><td>70</td><td>46</td><td>179</td><td>—</td><td>331</td><td>—</td>
  <td>1</td><td>—</td>
</tr>
<tr>
  <td>LDBC Graphanalytics</td>
  <td>—</td><td>—</td><td>15</td><td>19</td><td>167</td><td>—</td><td>528</td><td>—</td>
  <td>1</td><td>—</td>
</tr>
<tr>
  <td>LDBC SNB Interactive</td>
  <td>37,719</td><td>5,183</td><td>22,293</td><td>7,283</td><td>6,110</td><td>4,998</td><td>5,758</td><td>FAILED</td>
  <td>3,119</td><td>4,737</td>
</tr>
<tr>
  <td>LDBC ACID</td>
  <td>2,117</td><td>90</td><td>190</td><td>109</td><td>268</td><td>174</td><td>115</td><td>—</td>
  <td>38</td><td>127</td>
</tr>
<tr>
  <td>Concurrent</td>
  <td>13,365</td><td>597</td><td>TIMEOUT</td><td>783</td><td>2,394</td><td>1,569</td><td>FAILED</td><td>TIMEOUT</td>
  <td>29</td><td>3,594</td>
</tr>
<tr>
  <td>Vector †</td>
  <td>13,348</td><td>5,480</td><td>241</td><td>12,708</td><td>12,542</td><td>15,326</td><td>634</td><td>TIMEOUT</td>
  <td>341</td><td>5,689</td>
</tr>
<tr>
  <td>Hybrid †</td>
  <td>10,049</td><td>3,196</td><td>79</td><td>1,352</td><td>1,460</td><td>1,855</td><td>86</td><td>TIMEOUT</td>
  <td>28</td><td>740</td>
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
| triangle_count | 0.2 ms | 161.1 ms |
| common_neighbors | 0.4 ms | 39.8 ms |
| **Total (ms)** | **0.6** | **200.9** |
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
| pagerank | 0.0 ms |
| community_detection | 0.0 ms |
| betweenness_centrality | 0.4 ms |
| closeness_centrality | 0.2 ms |
| **Total (ms)** | **0.6** |
| **Peak Memory** | |

### LDBC Graphanalytics

Native implementations only.

| Benchmark | Grafeo |
|-----------|--------|
| ldbc_bfs | 0.4 ms |
| ldbc_pagerank | 0.0 ms |
| ldbc_wcc | 0.0 ms |
| ldbc_cdlp | 0.0 ms |
| ldbc_lcc | 0.0 ms |
| ldbc_sssp | 0.4 ms |
| **Total (ms)** | **1.0** |
| **Peak Memory** | |

### LDBC SNB Interactive

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| snb_is1 | 25.7 ms | 182.0 ms |
| snb_is2 | 23.7 ms | 163.1 ms |
| snb_is3 | 344.6 ms | 1,777.4 ms |
| snb_is4 | 8.2 ms | 49.3 ms |
| snb_is5 | 8.4 ms | 64.3 ms |
| snb_is6 | 21.2 ms | 171.8 ms |
| snb_is7 | 16.4 ms | 119.8 ms |
| snb_ic1 | 2,458.2 ms | TIMEOUT |
| snb_ic2 | 81.9 ms | 655.6 ms |
| snb_ic3 | 68.4 ms | 607.7 ms |
| snb_ic6 | 62.4 ms | 945.6 ms |
| **Total (ms)** | **3,119.1** | **4,736.6** |
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
| throughput_scaling | 8.6 ms | 1,910.2 ms |
| lost_update | 1.1 ms | 120.0 ms |
| read_after_write | 2.0 ms | 117.7 ms |
| concurrent_mixed | 2.4 ms | 435.4 ms |
| concurrent_acid | 14.6 ms | 1,011.0 ms |
| **Total (ms)** | **28.7** | **3,594.3** |
| **Peak Memory** | | |

### Vector

Brute-force fallback — no native vector index. Times include data retrieval + Python cosine similarity.

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| vector_insert | 43.2 ms | 194.1 ms |
| vector_knn | 25.6 ms | 460.2 ms |
| vector_batch_search | 249.2 ms | 4,590.0 ms |
| vector_recall | 23.1 ms | 444.6 ms |
| **Total (ms)** | **341.1** | **5,688.9** |
| **Peak Memory** | | |

### Hybrid

Brute-force vector fallback. Graph traversal is native.

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| hybrid_graph_to_vector | 23.7 ms | 552.5 ms |
| hybrid_vector_to_graph | 4.3 ms | 187.7 ms |
| **Total (ms)** | **28.0** | **740.2** |
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
| triangle_count | 178.2 ms | 253.3 ms | 175.7 ms | 11.6 ms | 9,710.4 ms | 224.6 ms | 40.8 ms | 9,704.5 ms |
| common_neighbors | 48.2 ms | 68.3 ms | 49.6 ms | 32.6 ms | 2,466.1 ms | 57.6 ms | 99.4 ms | 2,639.2 ms |
| **Total (ms)** | **226.4** | **321.6** | **225.3** | **44.2** | **12,176.5** | **282.2** | **140.2** | **12,343.7** |
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

| Benchmark | Grafeo Server | Neo4j | Memgraph | TuGraph |
|-----------|---------------|-------|----------|---------|
| pagerank | 2.2 ms | 54.9 ms | 4.0 ms | 109.7 ms |
| community_detection | 2.3 ms | 60.5 ms | 3.6 ms | 110.0 ms |
| betweenness_centrality | 41.6 ms | 20.9 ms | TIMEOUT | 74.5 ms |
| closeness_centrality | 23.7 ms | 42.6 ms | 38.1 ms | 37.1 ms |
| **Total (ms)** | **69.8** | **178.9** | **45.7** | **331.3** |
| **Peak Memory** | | | | |

### LDBC Graphanalytics

Native implementations only.

| Benchmark | Grafeo Server | Neo4j | Memgraph | TuGraph |
|-----------|---------------|-------|----------|---------|
| ldbc_bfs | 2.8 ms | 23.7 ms | 1.5 ms | 88.7 ms |
| ldbc_pagerank | 2.4 ms | 53.5 ms | 2.9 ms | 87.7 ms |
| ldbc_wcc | 2.1 ms | 18.6 ms | 2.5 ms | 87.9 ms |
| ldbc_cdlp | 2.1 ms | 33.5 ms | 3.1 ms | 88.6 ms |
| ldbc_lcc | 2.7 ms | 20.4 ms | 6.6 ms | 87.4 ms |
| ldbc_sssp | 2.7 ms | 16.8 ms | 2.4 ms | 87.2 ms |
| **Total (ms)** | **14.8** | **166.5** | **19.0** | **527.5** |
| **Peak Memory** | | | | |

### LDBC SNB Interactive

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| snb_is1 | 326.7 ms | 273.1 ms | 329.3 ms | 231.3 ms | 4,487.7 ms | 248.3 ms | 1,062.1 ms | FAILED |
| snb_is2 | 254.6 ms | 226.5 ms | 246.4 ms | 183.8 ms | 1,099.0 ms | 192.2 ms | 108.9 ms | FAILED |
| snb_is3 | 3,798.4 ms | 3,234.3 ms | 3,836.7 ms | 2,706.5 ms | 6,030.6 ms | 2,913.6 ms | TIMEOUT | FAILED |
| snb_is4 | 107.4 ms | 92.3 ms | 108.5 ms | 77.0 ms | 97.1 ms | 81.6 ms | 154.0 ms | FAILED |
| snb_is5 | 105.5 ms | 89.4 ms | 139.2 ms | 81.0 ms | 2,207.7 ms | 84.3 ms | 666.3 ms | FAILED |
| snb_is6 | 268.9 ms | 221.9 ms | 263.8 ms | 199.3 ms | TIMEOUT | 202.2 ms | 199.0 ms | FAILED |
| snb_is7 | 177.3 ms | 153.9 ms | 194.9 ms | 131.3 ms | TIMEOUT | 139.3 ms | 124.3 ms | FAILED |
| snb_ic1 | 15,105.4 ms | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | FAILED |
| snb_ic2 | 850.3 ms | 739.6 ms | 869.8 ms | 574.3 ms | 10,086.2 ms | 646.6 ms | 1,047.8 ms | FAILED |
| snb_ic3 | 694.3 ms | 593.6 ms | 694.8 ms | 538.7 ms | TIMEOUT | 636.5 ms | 910.4 ms | FAILED |
| snb_ic6 | 604.2 ms | 485.3 ms | 599.0 ms | 459.3 ms | 13,710.9 ms | 613.7 ms | 724.7 ms | FAILED |
| **Total (ms)** | **22,293.0** | **6,109.9** | **7,282.4** | **5,182.5** | **37,719.2** | **5,758.3** | **4,997.5** | **FAILED** |
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
| throughput_scaling | TIMEOUT | 733.8 ms | 583.3 ms | 178.0 ms | 8,854.1 ms | FAILED | 386.4 ms | TIMEOUT |
| lost_update | TIMEOUT | 219.7 ms | FAILED | 41.0 ms | 1,169.5 ms | FAILED | 123.2 ms | TIMEOUT |
| read_after_write | TIMEOUT | 127.6 ms | 64.7 ms | 39.5 ms | 1,126.0 ms | FAILED | 199.5 ms | TIMEOUT |
| concurrent_mixed | TIMEOUT | 158.2 ms | 134.5 ms | 41.8 ms | 2,215.2 ms | FAILED | 105.7 ms | TIMEOUT |
| concurrent_acid | TIMEOUT | 1,154.4 ms | FAILED | 297.1 ms | TIMEOUT | FAILED | 754.0 ms | TIMEOUT |
| **Total (ms)** | **TIMEOUT** | **2,393.7** | **782.5** | **597.4** | **13,364.8** | **FAILED** | **1,568.8** | **TIMEOUT** |
| **Peak Memory** | | | | | | | | |

### Vector

Brute-force fallback — no native vector index. Times include data retrieval + Python cosine similarity.

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| vector_insert | 145.0 ms | 262.7 ms | 479.6 ms | 146.8 ms | 149.6 ms | 535.3 ms | 1,178.2 ms | TIMEOUT |
| vector_knn | 7.4 ms | 1,040.5 ms | 1,093.3 ms | 2,718.6 ms | 1,132.0 ms | 8.5 ms | 1,047.6 ms | TIMEOUT |
| vector_batch_search | 80.0 ms | 10,224.4 ms | 10,084.5 ms | TIMEOUT | 10,941.4 ms | 82.3 ms | 11,879.1 ms | TIMEOUT |
| vector_recall | 8.1 ms | 1,014.4 ms | 1,050.8 ms | 2,614.3 ms | 1,125.4 ms | 7.8 ms | 1,220.8 ms | TIMEOUT |
| **Total (ms)** | **240.5** | **12,542.0** | **12,708.2** | **5,479.7** | **13,348.4** | **633.9** | **15,325.7** | **TIMEOUT** |
| **Peak Memory** | | | | | | | | |

### Hybrid

Brute-force vector fallback. Graph traversal is native.

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| hybrid_graph_to_vector | 71.8 ms | 1,156.6 ms | 1,080.9 ms | 2,615.5 ms | 5,075.4 ms | 78.2 ms | 1,534.4 ms | TIMEOUT |
| hybrid_vector_to_graph | 7.4 ms | 303.3 ms | 271.0 ms | 580.0 ms | 4,973.8 ms | 8.1 ms | 320.9 ms | TIMEOUT |
| **Total (ms)** | **79.2** | **1,459.9** | **1,351.9** | **3,195.5** | **10,049.2** | **86.3** | **1,855.3** | **TIMEOUT** |
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
