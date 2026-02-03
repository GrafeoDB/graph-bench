# graph-bench

Benchmark suite for graph databases. Built to compare Grafeo against other options.

## Supported Databases

| Database | Type | Query Language | Status |
|----------|------|----------------|--------|
| **Grafeo** | Embedded | GQL (ISO) | Ready |
| **LadybugDB** | Embedded | Cypher | Ready |
| **DuckDB** | Embedded | SQL/PGQ | Ready |
| **Neo4j** | Server | Cypher | Ready |
| **Memgraph** | Server | Cypher | Ready |
| **ArangoDB** | Server | AQL | Ready |
| **FalkorDB** | Server (Redis) | Cypher | Ready |
| **NebulaGraph** | Server (Distributed) | nGQL | Ready |

## Quick Start

```bash
# Install with all adapters
uv sync --all-extras

# Run small benchmark on Grafeo
uv run graph-bench run -d grafeo -s small --verbose

# Compare embedded databases
uv run graph-bench run -d grafeo,ladybug,duckdb -s small --verbose

# Run all 25 benchmarks
uv run graph-bench run -d grafeo -s medium
```

## Benchmarks (25 total)

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

## Scales

| Scale | Nodes | Edges | Warmup | Iterations |
|-------|-------|-------|--------|------------|
| small | 10K | 50K | 2 | 5 |
| medium | 100K | 500K | 3 | 10 |
| large | 1M | 5M | 5 | 10 |

## Installation

```bash
git clone https://github.com/StevenBtw/graph-bench
cd graph-bench

# Install everything
uv sync --all-extras

# Or just what you need
uv sync --extra grafeo --extra cli
```

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
GRAPH_BENCH_DUCKDB_PATH=:memory:

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

# Run tests
uv run pytest

# Lint
uv run ruff check graph_bench/

# Type check
uv run ty check graph_bench/

# Format
uv run ruff format graph_bench/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding benchmarks or adapters.

## License

Apache 2.0
