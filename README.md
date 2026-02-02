# graph-bench

Comprehensive benchmark suite for graph databases.

## Overview

graph-bench provides standardized benchmarks for comparing graph database performance across:

- **Neo4j**
- **Memgraph**
- **Ladybug**
- **ArangoDB**
- **Grafeo**

## Features

- **Three benchmark scales**: small (10K nodes), medium (100K nodes), large (1M nodes)
- **Multiple benchmark categories**: storage, traversal, algorithms, queries
- **Standardized metrics**: timing statistics, throughput, comparisons
- **Multiple output formats**: JSON, CSV, Markdown reports

## Installation

```bash
uv add graph-bench
```

With CLI support:
```bash
uv add "graph-bench[cli]"
```

With all adapters:
```bash
uv add "graph-bench[all]"
```

## Quick Start

### Using the CLI

```bash
# Run benchmarks on available databases
graph-bench run -d neo4j,kuzu -s medium

# Generate a report from results
graph-bench report results/bench_20260201.json -f markdown

# List available adapters
graph-bench adapters list

# Generate synthetic dataset
graph-bench datasets generate synthetic -s medium
```

### Using the Python API

```python
from graph_bench import SCALES
from graph_bench.adapters import KuzuAdapter, Neo4jAdapter
from graph_bench.benchmarks import NodeInsertionBenchmark
from graph_bench.runner import BenchmarkOrchestrator

# Create adapters
adapters = [
    Neo4jAdapter(),
    KuzuAdapter(),
]

# Connect to databases
for adapter in adapters:
    adapter.connect()

# Run benchmarks
orchestrator = BenchmarkOrchestrator()
results = orchestrator.run(adapters, scale="medium")

# Print results
for result in results.results:
    if result.ok:
        print(f"{result.database} - {result.benchmark_name}: {result.metrics.timing.mean_ms:.2f}ms")
```

## Benchmark Categories

### Storage
- `node_insertion` - Node batch insert throughput
- `edge_insertion` - Edge batch insert throughput
- `single_read` - Point lookup latency
- `batch_read` - Batch retrieval throughput

### Traversal
- `hop_1` / `hop_2` - Neighbor expansion
- `bfs` / `dfs` - Graph traversal
- `shortest_path` - Path finding

### Algorithms
- `pagerank` - PageRank centrality
- `community_detection` - Louvain community detection

### Query
- `aggregation_count` - COUNT aggregation
- `filter_equality` / `filter_range` - Filter performance

## Configuration

### Environment Variables

```bash
# Neo4j
GRAPH_BENCH_NEO4J_URI=bolt://localhost:7687
GRAPH_BENCH_NEO4J_USER=neo4j
GRAPH_BENCH_NEO4J_PASSWORD=password

# Memgraph
GRAPH_BENCH_MEMGRAPH_URI=bolt://localhost:7687

# Kuzu
GRAPH_BENCH_KUZU_PATH=./data/kuzu

# ArangoDB
GRAPH_BENCH_ARANGO_URI=http://localhost:8529
GRAPH_BENCH_ARANGO_DATABASE=benchmark

# Grafeo
GRAPH_BENCH_GRAFEO_PATH=./data/grafeo
```

## Development

```bash
# Clone and setup
git clone https://github.com/StevenBtw/graph-bench
cd graph-bench
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check graph_bench/

# Type check
uv run ty check graph_bench/
```

## License

Apache 2.0
