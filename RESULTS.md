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
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Write</td>
  <td>10,000</td><td>85</td><td>200</td><td>102</td><td>426</td><td>846</td><td>126</td><td>9,676</td>
  <td>2</td><td>115</td>
</tr>
<tr>
  <td>Traversal</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Query</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Pattern</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
</tr>
<tr>
  <td>Structure</td>
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
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
  <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
  <td></td><td></td>
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
  <td>—</td><td>—</td><td>—</td><td>—</td><td></td><td>—</td><td>—</td><td></td>
  <td></td><td>—</td>
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
| node_insertion | | |
| edge_insertion | | |
| single_read | | |
| batch_read | | |
| **Total (ms)** | | |
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
| hop_1 | | |
| hop_2 | | |
| bfs | | |
| dfs | | |
| shortest_path | | |
| **Total (ms)** | | |
| **Peak Memory** | | |

### Query

| Benchmark | Grafeo | LadybugDB |
|-----------|--------|-----------|
| aggregation_count | | |
| filter_equality | | |
| filter_range | | |
| **Total (ms)** | | |
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
| connected_components | | |
| degree_distribution | | |
| graph_density | | |
| reachability | | |
| **Total (ms)** | | |
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
| acid_atomicity_c | | |
| acid_atomicity_rb | | |
| acid_g0 | | |
| acid_g1a | | |
| acid_g1b | | |
| acid_g1c | | |
| acid_imp | | |
| acid_pmp | | |
| acid_otv | | |
| acid_fr | | |
| acid_lu | | |
| acid_ws | | |
| **Total (ms)** | | |
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
| Full node scan | 61,521 | | — |
| Label scan: Person | 369 | | — |
| Count: Person | 1 | | — |
| Label scan: Crime | 28,762 | | — |
| Count: Crime | 1 | | — |
| Full edge scan | 105,840 | | — |
| Count: Edges | 1 | | — |
| Indexed lookup → Crime | 0 | | — |
| Person → Crime | 55 | | — |
| Indexed lookup → Any | 6 | | — |
| 2-hop → Crime | 0 | | — |
| Crime → Location | 28,762 | | — |
| **Total (ms)** | — | | — |
| **Peak Memory** | — | | — |

---

## Server Databases

### Storage

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| node_insertion | | | | | | | | |
| edge_insertion | | | | | | | | |
| single_read | | | | | | | | |
| batch_read | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Write

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| property_update | 99.3 ms | 218.5 ms | 34.3 ms | 27.3 ms | 6,607.5 ms | 40.4 ms | 36.0 ms | 2,206.2 ms |
| edge_add_existing | 3.1 ms | 10.2 ms | 1.7 ms | 1.6 ms | 43.5 ms | 3.2 ms | 671.4 ms | 131.9 ms |
| mixed_workload | 97.2 ms | 197.5 ms | 65.8 ms | 55.8 ms | 3,349.3 ms | 82.6 ms | 138.5 ms | 7,337.4 ms |
| **Total (ms)** | **199.6** | **426.2** | **101.8** | **84.7** | **10,000.3** | **126.2** | **845.9** | **9,675.5** |
| **Peak Memory** | | | | | | | | |

### Traversal

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| hop_1 | | | | | | | | |
| hop_2 | | | | | | | | |
| bfs | | | | | | | | |
| dfs | | | | | | | | |
| shortest_path | | | | | | | | |
| **Total (ms)** | | | | | | | | |
| **Peak Memory** | | | | | | | | |

### Query

| Benchmark | Grafeo Server | Neo4j | Memgraph | FalkorDB | ArangoDB | TuGraph | NebulaGraph | TuringDB |
|-----------|---------------|-------|----------|----------|----------|---------|-------------|----------|
| aggregation_count | | | | | | | | |
| filter_equality | | | | | | | | |
| filter_range | | | | | | | | |
| **Total (ms)** | | | | | | | | |
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
| connected_components | | | | | | | | |
| degree_distribution | | | | | | | | |
| graph_density | | | | | | | | |
| reachability | | | | | | | | |
| **Total (ms)** | | | | | | | | |
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
| acid_atomicity_c | | | | | | | | |
| acid_atomicity_rb | | | | | | | | |
| acid_g0 | | | | | | | | |
| acid_g1a | | | | | | | | |
| acid_g1b | | | | | | | | |
| acid_g1c | | | | | | | | |
| acid_imp | | | | | | | | |
| acid_pmp | | | | | | | | |
| acid_otv | | | | | | | | |
| acid_fr | | | | | | | | |
| acid_lu | | | | | | | | |
| acid_ws | | | | | | | | |
| **Total (ms)** | | | | | | | | |
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
| Full node scan | 61,521 | — | | — | — | — | — | — | |
| Label scan: Person | 369 | — | | — | — | — | — | — | |
| Count: Person | 1 | — | | — | — | — | — | — | |
| Label scan: Crime | 28,762 | — | | — | — | — | — | — | |
| Count: Crime | 1 | — | | — | — | — | — | — | |
| Full edge scan | 105,840 | — | | — | — | — | — | — | |
| Count: Edges | 1 | — | | — | — | — | — | — | |
| Indexed lookup → Crime | 0 | — | | — | — | — | — | — | |
| Person → Crime | 55 | — | | — | — | — | — | — | |
| Indexed lookup → Any | 6 | — | | — | — | — | — | — | |
| 2-hop → Crime | 0 | — | | — | — | — | — | — | |
| Crime → Location | 28,762 | — | | — | — | — | — | — | |
| **Total (ms)** | — | — | | — | — | — | — | — | |
| **Peak Memory** | — | — | | — | — | — | — | — | |
