# graph-bench

Comprehensive benchmark suite for graph databases.

## Supported Databases

| Database | Type | Query Language | Status |
|----------|------|----------------|--------|
| **Grafeo** | Embedded | GQL (ISO) | ✅ Full support |
| **LadybugDB** | Embedded | Cypher | ✅ Full support |
| **DuckDB** | Embedded | SQL/PGQ | ✅ Full support |
| **Neo4j** | Server | Cypher | ✅ Full support |
| **Memgraph** | Server | Cypher | ✅ Full support |
| **ArangoDB** | Server | AQL | ✅ Full support |
| **FalkorDB** | Server (Redis) | Cypher | ✅ Full support |
| **NebulaGraph** | Server (Distributed) | nGQL | ✅ Full support |

## Quick Start: Run Grafeo Benchmark

```bash
# Install with Grafeo support
uv sync --all-extras

# Run small benchmark on Grafeo only
uv run graph-bench run -d grafeo -s small --verbose

# Compare Grafeo against other embedded databases
uv run graph-bench run -d grafeo,ladybug,duckdb -s small --verbose

# Run all 25 benchmarks on Grafeo
uv run graph-bench run -d grafeo -s medium
```

## Benchmark Categories (25 total)

### Storage (4 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `node_insertion` | Batch node insert throughput |
| `edge_insertion` | Batch edge insert throughput |
| `single_read` | Point lookup latency |
| `batch_read` | Batch retrieval by label |

### Traversal (5 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `hop_1` | 1-hop neighbor expansion |
| `hop_2` | 2-hop neighbor expansion |
| `bfs` | Breadth-first search |
| `dfs` | Depth-first search |
| `shortest_path` | Path finding between nodes |

### Algorithms (4 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `pagerank` | PageRank centrality |
| `community_detection` | Louvain community detection |
| `betweenness_centrality` | Bridge node identification |
| `closeness_centrality` | Node reachability scoring |

### Query (3 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `aggregation_count` | COUNT aggregation |
| `filter_equality` | Equality filter performance |
| `filter_range` | Range filter performance |

### Pattern Matching (2 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `triangle_count` | Count triangles in graph |
| `common_neighbors` | Find mutual connections |

### Graph Structure (4 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `connected_components` | Find disconnected subgraphs |
| `degree_distribution` | Identify high-degree nodes |
| `graph_density` | Calculate edge density |
| `reachability` | Check node reachability |

### Write Operations (3 benchmarks)
| Benchmark | Description |
|-----------|-------------|
| `property_update` | Update node properties |
| `edge_add_existing` | Add edges between existing nodes |
| `mixed_workload` | 80% read / 20% write mix |

## Benchmark Scales

| Scale | Nodes | Edges | Warmup | Iterations |
|-------|-------|-------|--------|------------|
| small | 10K | 50K | 2 | 5 |
| medium | 100K | 500K | 3 | 10 |
| large | 1M | 5M | 5 | 10 |

## Installation

```bash
# Clone the repository
git clone https://github.com/StevenBtw/graph-bench
cd graph-bench

# Install with uv (recommended)
uv sync --all-extras

# Or install specific adapters only
uv sync --extra grafeo --extra cli
```

### Install from PyPI (when published)

```bash
# Basic install
uv add graph-bench

# With specific database support
uv add "graph-bench[grafeo,cli]"

# With all databases
uv add "graph-bench[all]"
```

## Running Benchmarks

### CLI Usage

```bash
# Run on specific databases
uv run graph-bench run -d grafeo -s small --verbose
uv run graph-bench run -d grafeo,ladybug,arangodb -s medium

# Run specific benchmark categories
uv run graph-bench run -d grafeo -c storage -s small
uv run graph-bench run -d grafeo -c traversal,pattern -s small

# Run specific benchmarks
uv run graph-bench run -d grafeo -b triangle_count,common_neighbors -s small

# Output results to specific directory
uv run graph-bench run -d grafeo -s small -o ./my_results/
```

### Python API

```python
from graph_bench.adapters import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.config import get_scale

# Connect to Grafeo (in-memory)
adapter = GrafeoAdapter()
adapter.connect()  # Uses :memory: by default

# Or with persistent storage
adapter.connect(path="./data/grafeo")

# Run a single benchmark
scale = get_scale("small")
benchmark_cls = BenchmarkRegistry.get("triangle_count")
benchmark = benchmark_cls()
metrics = benchmark.run(adapter, scale)

print(f"Mean time: {metrics.timing.mean_ns / 1_000_000:.2f}ms")
print(f"Throughput: {metrics.throughput:.2f} ops/sec")

# Clean up
adapter.disconnect()
```

### Using the Orchestrator

```python
from graph_bench.adapters import GrafeoAdapter, LadybugAdapter
from graph_bench.runner import BenchmarkOrchestrator

adapters = [GrafeoAdapter(), LadybugAdapter()]

for adapter in adapters:
    adapter.connect()

orchestrator = BenchmarkOrchestrator()
results = orchestrator.run(adapters, scale="small")

# Results are automatically saved to results/ directory
for result in results.results:
    if result.status == "SUCCESS":
        print(f"{result.database} - {result.benchmark}: {result.metrics.timing.mean_ns/1e6:.2f}ms")
```

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Embedded databases (no server needed)
GRAPH_BENCH_GRAFEO_PATH=:memory:      # Use :memory: for in-memory
GRAPH_BENCH_LADYBUG_PATH=:memory:
GRAPH_BENCH_DUCKDB_PATH=:memory:

# Server databases (requires Docker or local install)
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

### Docker Setup for Server Databases

```bash
# Start all server databases
docker compose up -d

# Start specific databases
docker compose up -d neo4j memgraph arangodb

# Check status
docker compose ps
```

## Results Format

Results are saved as JSON in the `results/` directory:

```json
{
  "session": {
    "id": "bench_20260202_145817",
    "scale": "small",
    "databases": ["Grafeo", "LadybugDB"]
  },
  "results": [
    {
      "benchmark": "triangle_count",
      "database": "Grafeo",
      "status": "SUCCESS",
      "metrics": {
        "timing": {
          "mean_ns": 58994720,
          "min_ns": 52145100,
          "max_ns": 65674100,
          "p99_ns": 65674100
        },
        "throughput": 16.95,
        "items_processed": 0
      }
    }
  ],
  "comparisons": {
    "triangle_count": {
      "Grafeo": 1.0,
      "LadybugDB": 0.27
    }
  }
}
```

## Recent Benchmark Results (Small Scale)

| Benchmark | Grafeo v0.2.4 | LadybugDB | Winner |
|-----------|---------------|-----------|--------|
| node_insertion | **3.5ms** | 289ms | Grafeo 83× |
| triangle_count | **59ms** | 219ms | Grafeo 3.7× |
| common_neighbors | **10ms** | 43ms | Grafeo 4.3× |
| betweenness_centrality | **11ms** | 57ms | Grafeo 5.2× |
| single_read | 1,131ms | **33ms** | LadybugDB 34× |
| property_update | 27ms | **7ms** | LadybugDB 3.9× |
| bfs | 24ms | **22ms** | LadybugDB 1.1× |

## Development

```bash
# Setup development environment
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=graph_bench --cov-report=term-missing

# Lint
uv run ruff check graph_bench/

# Type check
uv run ty check graph_bench/

# Format code
uv run ruff format graph_bench/
```

## Adding a New Benchmark

1. Create a benchmark class in the appropriate file under `graph_bench/benchmarks/`
2. Register it with `@BenchmarkRegistry.register("benchmark_name", category="category")`
3. Add the import to `graph_bench/benchmarks/__init__.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

## License

Apache 2.0
