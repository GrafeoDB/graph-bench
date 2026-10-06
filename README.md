# graph-bench

Benchmark suite for graph databases. Built to compare Grafeo against other options.

## Supported Databases

| Database | Type | Query Language | Status |
|----------|------|----------------|--------|
| **Grafeo** | Embedded | GQL (ISO) | Ready |
| **Grafeo Server** | Server | GQL (ISO) | Ready |
| **LadybugDB** | Embedded | Cypher | Ready |
| **FalkorDB Lite** | Embedded | Cypher | Ready |
| **Neo4j** | Server | Cypher | Ready |
| **Memgraph** | Server | Cypher | Ready |
| **ArangoDB** | Server | AQL | Ready |
| **FalkorDB** | Server (Redis) | Cypher | Ready |
| **TuGraph** | Server | Cypher | Ready |
| **NebulaGraph** | Server (Distributed) | nGQL | Ready |

## Quick Start

```bash
# Install with all adapters
uv sync --all-extras

# Run small benchmark on Grafeo
uv run graph-bench run -d grafeo -s small --verbose

# Compare embedded databases
uv run graph-bench run -d grafeo,ladybug,duckdb -s small --verbose

# Run core benchmarks
uv run graph-bench run -d grafeo -s medium
```

## Benchmarks

### Storage (4)
| Benchmark | What it measures |
|-----------|------------------|
| `node_insertion` | Batch node insert throughput |
| `edge_insertion` | Batch edge insert throughput |
| `single_read` | Point lookup latency |
| `batch_read` | Batch retrieval by label |

### Traversal (5)
| Benchmark | What it measures |
|-----------|------------------|
| `hop_1` | 1-hop neighbor expansion |
| `hop_2` | 2-hop neighbor expansion |
| `bfs` | Breadth-first search |
| `dfs` | Depth-first search |
| `shortest_path` | Path finding between nodes |

### Algorithms (4)
| Benchmark | What it measures |
|-----------|------------------|
| `pagerank` | PageRank centrality |
| `community_detection` | Louvain community detection |
| `betweenness_centrality` | Bridge node identification |
| `closeness_centrality` | Node reachability scoring |

### Query (3)
| Benchmark | What it measures |
|-----------|------------------|
| `aggregation_count` | COUNT aggregation |
| `filter_equality` | Equality filter performance |
| `filter_range` | Range filter performance |

### Pattern Matching (2)
| Benchmark | What it measures |
|-----------|------------------|
| `triangle_count` | Count triangles in graph |
| `common_neighbors` | Find mutual connections |

### Graph Structure (4)
| Benchmark | What it measures |
|-----------|------------------|
| `connected_components` | Find disconnected subgraphs |
| `degree_distribution` | Identify high-degree nodes |
| `graph_density` | Calculate edge density |
| `reachability` | Check node reachability |

### Write Operations (3)
| Benchmark | What it measures |
|-----------|------------------|
| `property_update` | Update node properties |
| `edge_add_existing` | Add edges between existing nodes |
| `mixed_workload` | 80% read / 20% write mix |

### LDBC-Inspired: SNB Interactive (11)
| Benchmark | What it measures |
|-----------|------------------|
| `snb_is1` - `snb_is7` | Short read queries (profile, posts, friends, content) |
| `snb_ic1` - `snb_ic3`, `snb_ic6` | Complex read queries (multi-hop friends, messages, tags) |

### LDBC-Inspired: GraphAnalytics (6 + 1)
| Benchmark | What it measures |
|-----------|------------------|
| `ldbc_bfs` | Breadth-first search (vertex depths) |
| `ldbc_pagerank` | PageRank centrality |
| `ldbc_wcc` | Weakly connected components |
| `ldbc_cdlp` | Community detection (label propagation) |
| `ldbc_lcc` | Local clustering coefficient, LDBC directed definition |
| `ldbc_sssp` | Single-source shortest paths |
| `lcc_undirected` | Local clustering coefficient with edges taken as undirected (not an LDBC definition) |

The two LCC definitions differ on a directed graph. `ldbc_lcc` follows LDBC Graphalytics: the neighbourhood
N(v) holds in- and out-neighbours, the directed edges between them are counted (an edge in each direction counts
twice) and divided by d * (d - 1). On the cycle 0->1->2->0 every vertex scores 0.5; `lcc_undirected` gives 1.0.
Each adapter lists the definitions it computes natively (`native_lcc_definitions`); the other one runs the
BaseAdapter fallback with that definition, and the result says so. `nx.clustering` on a directed graph (the
Fagiolo coefficient) is neither definition.

### LDBC-Inspired: ACID (12)
| Benchmark | What it measures |
|-----------|------------------|
| `acid_atomicity_c` | Commit visibility |
| `acid_atomicity_rb` | Rollback correctness |
| `acid_g0` | Dirty write detection |
| `acid_g1a` - `acid_g1c` | Read anomalies (aborted, intermediate, circular) |
| `acid_imp`, `acid_pmp` | Item/predicate many-preceders |
| `acid_otv`, `acid_fr` | Observed transaction vanishes, fractured read |
| `acid_lu`, `acid_ws` | Lost update, write skew |

### Concurrent (5)
| Benchmark | What it measures |
|-----------|------------------|
| `throughput_scaling` | Multi-threaded throughput |
| `lost_update` | Lost update under concurrency |
| `read_after_write` | Read-your-writes consistency |
| `concurrent_mixed` | Concurrent 80/20 read-write mix |
| `concurrent_acid` | ACID under concurrent load |

### Vector Scan (4): the same brute-force search for every database
| Benchmark | What it measures |
|-----------|------------------|
| `vector_scan_insert` | Insert 128-dimensional vector nodes (no index) |
| `vector_scan_knn` | k=10 search per cluster centroid: read every vector node, score in Python |
| `vector_scan_batch_search` | 100 such searches with perturbed queries |
| `vector_scan_recall` | Share of results in the query's cluster |

These were called `vector_insert`, `vector_knn`, `vector_batch_search` and `vector_recall`; the old names
(and `-c vector`) still select them. The hybrid benchmarks use the same scan.

### Vector ANN (3): the database's native index, no fallback
| Benchmark | What it measures |
|-----------|------------------|
| `vector_ann_build` | Insert the vectors, then build the native index (insert and index time reported apart) |
| `vector_ann_knn` | 100 k-NN queries at the configured `ef`, with recall@k against exact search |
| `vector_ann_delete` | The same queries after deleting a share of the vectors: recall before and after |

A database without a native vector index reports these as *unsupported* (SKIPPED), never as a fallback.
Recall@k compares the results with the exact top k over the vectors as the database stores them (the data
the scan reads). The HNSW parameters are shared by every database in a run: `--ann-m` (default 16),
`--ann-ef-construction` (128), `--ann-ef` (64), `--ann-k` (10), plus `--ann-quantization`
(`none`, `scalar`, `binary`, `product`: Grafeo embedded only) and `--ann-delete-fraction` (0.1).
`--ann-sweep` (or `--ann-sweep-ef 10,20,40`) adds a recall versus latency curve over `ef`.

### What each result measured

Every result carries an `implementation` label: *native* when every call in its timed iterations ran in the
database, *fallback* (with the operations, for example `fallback (traverse_dfs)`) when part of it ran in Python
(a NetworkX algorithm, a client-side traversal or the vector scan), *unsupported* when the database has no
native implementation of a benchmark that must not fall back. The labels are recorded while the benchmark
runs, not declared per adapter, and appear in the JSON (`metrics.custom`), CSV and Markdown reports.

## Scales

Scales follow the LDBC SNB scale factors (see `graph_bench/config.py`). `small`, `medium` and `large` are
aliases for `sf01`, `sf1` and `sf100`; `sf1` is the default. Each benchmark generates a fraction of these
counts (for example 1/10 of the nodes for the LDBC Graphalytics graph).

| Scale | Nodes | Edges | Warmup | Iterations | Timeout |
|-------|-------|-------|--------|------------|---------|
| `sf01` (`small`) | 1K | 18K | 1 | 3 | 60 s |
| `sf1` (`medium`) | 10K | 180K | 1 | 3 | 120 s |
| `sf3` | 27K | 540K | 3 | 10 | 300 s |
| `sf10` | 73K | 2M | 5 | 10 | 600 s |
| `sf30` | 180K | 6.5M | 5 | 10 | 1800 s |
| `sf100` (`large`) | 280K | 18M | 1 | 1 | 3600 s |

## Installation

```bash
git clone https://github.com/GrafeoDB/graph-bench
cd graph-bench

# Install everything
uv sync --all-extras

# Or just what you need
uv sync --extra grafeo --extra cli

# Grafeo Server over GWP (gRPC) needs the grafeo-server extra as well
uv sync --extra grafeo --extra grafeo-server --extra cli
```

### Grafeo versions

The Grafeo adapters are tested against **grafeo 0.5.44** and accept `grafeo>=0.5.44,<0.7`. Grafeo is
before 1.0, so a new minor version may break the adapters: the requirement pins the minor versions that are
tested, and becomes `grafeo~=0.6.0` once grafeo 0.6.0 is released (write all three parts: `grafeo~=0.6`
would also accept 0.7). A weekly CI run tests the newest grafeo on PyPI, pre-releases included.

- **Embedded** (`-d grafeo`): in memory by default. `GRAPH_BENCH_GRAFEO_PATH` selects a database file; a path
  without the `.grafeo` suffix gets it, because grafeo 0.6 replaces the WAL-directory layout with a single
  file. A checkpoint writes `<file>.checkpoint` first, so keep free disk space for a second copy.
- **Server** (`-d grafeo-server`): GQL over GWP to a `grafeo/grafeo-server:<version>-gwp` or `-full` image,
  at `GRAPH_BENCH_GRAFEO_SERVER_URI` (default `localhost:7688`, grafeo-server's GWP port).

A query that fails raises (`grafeo.GrafeoError`, or `GrafeoServerError` for the server), so its benchmark is
recorded as FAILED instead of being timed.

## CLI Usage

```bash
# Run on specific databases
uv run graph-bench run -d grafeo -s small --verbose
uv run graph-bench run -d grafeo,ladybug,arangodb -s medium

# Run specific categories
uv run graph-bench run -d grafeo -c storage -s small
uv run graph-bench run -d grafeo -c traversal,pattern -s small

# Run specific benchmarks
uv run graph-bench run -d grafeo -b triangle_count,common_neighbors -s small

# Output to specific directory
uv run graph-bench run -d grafeo -s small -o ./my_results/
```

### Grafeo-focused runs

```bash
# Native vector search with a recall versus latency curve
uv run graph-bench run -d grafeo -c vector_ann --ann-sweep -s small -f markdown

# Tune the shared HNSW parameters, with scalar quantization (embedded only)
uv run graph-bench run -d grafeo -c vector_ann --ann-m 32 --ann-ef 128 --ann-quantization scalar -s small

# Embedded and server side by side (one column each in the Markdown report)
uv run graph-bench run -d grafeo,grafeo-server -c ldbc_graphanalytics,vector_ann -s small -f markdown
```

The server's native vector search needs a grafeo-server with the vector index (the `-full` image); on other
builds the `vector_ann` results are *unsupported*.

## Python API

```python
from graph_bench.adapters import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.config import get_scale

# Connect
adapter = GrafeoAdapter()
adapter.connect()  # Uses :memory: by default

# Run a benchmark
scale = get_scale("small")
benchmark_cls = BenchmarkRegistry.get("triangle_count")
benchmark = benchmark_cls()
metrics = benchmark.run(adapter, scale)

print(f"Mean time: {metrics.timing.mean_ns / 1_000_000:.2f}ms")
print(f"Throughput: {metrics.throughput:.2f} ops/sec")

adapter.disconnect()
```

## Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# Embedded (no server needed)
GRAPH_BENCH_GRAFEO_PATH=:memory:
GRAPH_BENCH_LADYBUG_PATH=:memory:

# Grafeo Server (GWP)
GRAPH_BENCH_GRAFEO_SERVER_URI=localhost:7688

# Server databases (needs Docker or local install)
GRAPH_BENCH_NEO4J_URI=bolt://localhost:7687
GRAPH_BENCH_NEO4J_USER=neo4j
GRAPH_BENCH_NEO4J_PASSWORD=benchmark

GRAPH_BENCH_MEMGRAPH_URI=bolt://localhost:7688

GRAPH_BENCH_ARANGO_URI=http://localhost:8529
GRAPH_BENCH_ARANGO_USER=root
GRAPH_BENCH_ARANGO_PASSWORD=benchmark
GRAPH_BENCH_ARANGO_DATABASE=benchmark

GRAPH_BENCH_FALKORDB_HOST=localhost
GRAPH_BENCH_FALKORDB_PORT=6379

GRAPH_BENCH_NEBULA_HOST=localhost
GRAPH_BENCH_NEBULA_PORT=9669
GRAPH_BENCH_NEBULA_USER=root
GRAPH_BENCH_NEBULA_PASSWORD=nebula
```

## Docker Setup

```bash
# Start all server databases
docker compose up -d

# Or just some
docker compose up -d neo4j memgraph arangodb

# Check status
docker compose ps
```

## Latest Results

See [LATEST_RESULTS.md](LATEST_RESULTS.md) for the most recent benchmark run.

Results are saved as JSON in `results/`.

## Development

```bash
uv sync --all-extras

# Run tests (coverage is reported, the 88% target is not enforced yet)
uv run pytest --cov-fail-under=0

# Grafeo correctness tests: embedded always, grafeo-server when one answers at the URI
GRAPH_BENCH_GRAFEO_SERVER_URI=localhost:7688 uv run pytest tests/test_grafeo_correctness.py --no-cov -rs

# Lint
uv run ruff check graph_bench/

# Type check
uv run ty check graph_bench/

# Format
uv run ruff format graph_bench/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding benchmarks or adapters.

## Related

- [Grafeo](https://github.com/GrafeoDB/grafeo), the graph database being benchmarked
- [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph), interactive graph visualization for notebooks

## LDBC Disclaimer

This benchmark suite includes workloads **inspired by** [LDBC](https://ldbcouncil.org/) specifications, including the [Social Network Benchmark (SNB)](https://ldbcouncil.org/benchmarks/snb/), [Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/), and the [ACID test suite](https://ldbcouncil.org/benchmarks/snb/). These are **not official LDBC Benchmark results**. The results have not been audited or endorsed by the LDBC Council.

Key differences from official LDBC benchmarks:

- **No LDBC audit**: official LDBC results require formal auditing under the [LDBC Byelaws](https://ldbcouncil.org/benchmarks/fair-use-policies/).
- **Synthetic datasets**: graph-bench generates its own datasets that follow the LDBC SNB schema but are not produced by the official LDBC SNB data generator (Datagen).
- **Reduced scale**: default runs use SF0.1, far below the scale factors used in official LDBC publications.
- **Simplified query set**: only a subset of the official SNB Interactive queries are implemented.
- **No cost-of-ownership reporting**: official LDBC results include pricing and hardware disclosure.

LDBC specifications are used under the [Creative Commons Attribution 4.0 (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. LDBC and the LDBC logo are trademarks of the Linked Data Benchmark Council. See the [LDBC fair use policy](https://ldbcouncil.org/benchmarks/fair-use-policies/) for details.

## License

Apache 2.0
