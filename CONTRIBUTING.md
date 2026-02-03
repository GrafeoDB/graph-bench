# Contributing to graph-bench

Thank you for your interest in contributing to graph-bench! This document provides guidelines for contributing new benchmarks, database adapters, and general improvements.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/StevenBtw/graph-bench
cd graph-bench

# Install all dependencies including dev tools
uv sync --all-extras

# Verify setup
uv run pytest
uv run ruff check graph_bench/
uv run ty check graph_bench/
```

## Project Structure

```
graph-bench/
├── graph_bench/
│   ├── adapters/          # Database adapters
│   │   ├── base.py        # BaseAdapter class
│   │   ├── grafeo.py      # Grafeo adapter
│   │   ├── ladybug.py     # LadybugDB adapter
│   │   └── ...
│   ├── benchmarks/        # Benchmark implementations
│   │   ├── base.py        # BaseBenchmark class
│   │   ├── storage.py     # Storage benchmarks
│   │   ├── traversal.py   # Traversal benchmarks
│   │   ├── algorithms.py  # Algorithm benchmarks
│   │   ├── pattern.py     # Pattern matching benchmarks
│   │   ├── structure.py   # Graph structure benchmarks
│   │   ├── write.py       # Write operation benchmarks
│   │   └── query.py       # Query benchmarks
│   ├── cli/               # CLI interface
│   ├── runner/            # Benchmark orchestration
│   ├── config.py          # Configuration
│   ├── types.py           # Type definitions
│   └── protocols.py       # Protocol definitions
├── tests/                 # Test suite
└── results/               # Benchmark results (gitignored)
```

## Adding a New Benchmark

### Step 1: Choose the Right Category

Benchmarks are organized by category:

- `storage` - Node/edge CRUD operations
- `traversal` - Graph traversal operations
- `algorithm` - Graph algorithms (PageRank, centrality, etc.)
- `query` - Query operations (aggregation, filtering)
- `pattern` - Pattern matching (triangles, common neighbors)
- `structure` - Graph structure analysis
- `write` - Write-heavy operations

### Step 2: Create the Benchmark Class

Add your benchmark to the appropriate file in `graph_bench/benchmarks/`:

```python
from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig


@BenchmarkRegistry.register("my_benchmark", category="traversal")
class MyBenchmark(BaseBenchmark):
    """Benchmark description.

    Explain what this benchmark measures and its real-world use cases.
    """

    category = "traversal"

    @property
    def name(self) -> str:
        return "my_benchmark"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Prepare data for the benchmark."""
        adapter.clear()

        # Create test data based on scale
        node_count = min(5000, scale.nodes // 20)
        nodes = [{"id": f"node_{i}", "value": i} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="TestNode")

        # Store any state needed for run_iteration
        self._node_ids = [f"node_{i}" for i in range(node_count)]

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run one iteration of the benchmark.

        Returns:
            Number of items processed (for throughput calculation)
        """
        import random

        processed = 0
        for _ in range(10):
            node_id = random.choice(self._node_ids)
            result = adapter.get_node(node_id)
            if result:
                processed += 1

        return processed

    def teardown(self, adapter: GraphDatabaseAdapter) -> None:
        """Clean up after benchmark (optional)."""
        pass
```

### Step 3: Register the Benchmark

Add your benchmark to `graph_bench/benchmarks/__init__.py`:

```python
from graph_bench.benchmarks.traversal import (
    # ... existing imports ...
    MyBenchmark,
)

__all__ = [
    # ... existing exports ...
    "MyBenchmark",
]
```

### Step 4: Test Your Benchmark

```bash
# Run your benchmark on a single database
uv run graph-bench run -d grafeo -b my_benchmark -s small --verbose

# Run tests
uv run pytest tests/benchmarks/test_my_benchmark.py
```

## Adding a New Database Adapter

### Step 1: Create the Adapter Class

Create a new file in `graph_bench/adapters/`:

```python
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env


@AdapterRegistry.register("mydb")
class MyDBAdapter(BaseAdapter):
    """MyDB graph database adapter."""

    def __init__(self) -> None:
        self._db = None
        self._connected = False

    @property
    def name(self) -> str:
        return "MyDB"

    @property
    def version(self) -> str:
        try:
            import mydb
            return mydb.__version__
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            import mydb
        except ImportError as e:
            msg = "mydb package not installed. Install with: pip install mydb"
            raise ImportError(msg) from e

        uri = uri or get_env("MYDB_URI", "localhost:1234")
        self._db = mydb.connect(uri)
        self._connected = True

    def disconnect(self) -> None:
        if self._db:
            self._db.close()
        self._db = None
        self._connected = False

    def clear(self) -> None:
        # Delete all data
        self._db.execute("DELETE ALL")

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            for node in batch:
                self._db.create_node(label, node)
                count += 1
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        result = self._db.query(f"MATCH (n {{id: '{node_id}'}}) RETURN n")
        return result[0] if result else None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._db.query(f"MATCH (n:{label}) RETURN n LIMIT {limit}")

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        for src, tgt, edge_type, props in edges:
            self._db.create_edge(src, tgt, edge_type, props)
            count += 1
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if edge_type:
            query = f"MATCH (n {{id: '{node_id}'}})-[:{edge_type}]->(m) RETURN m.id"
        else:
            query = f"MATCH (n {{id: '{node_id}'}})-[]->(m) RETURN m.id"
        return [r["id"] for r in self._db.query(query)]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        # Implement using native DB function or fall back to base class BFS
        return super().shortest_path(source, target, edge_type=edge_type, weighted=weighted)

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._db.query(query, params or {})

    def count_nodes(self, *, label: str | None = None) -> int:
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"
        result = self._db.query(query)
        return result[0]["count"] if result else 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        if edge_type:
            query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"
        result = self._db.query(query)
        return result[0]["count"] if result else 0
```

### Step 2: Register the Adapter

Add to `graph_bench/adapters/__init__.py`:

```python
from graph_bench.adapters.mydb import MyDBAdapter

__all__ = [
    # ... existing exports ...
    "MyDBAdapter",
]
```

### Step 3: Add Configuration

Update `.env.example`:

```bash
# MyDB
GRAPH_BENCH_MYDB_URI=localhost:1234
```

### Step 4: Add Optional Dependency

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
mydb = ["mydb-python>=1.0"]
all-adapters = [
    # ... existing ...
    "mydb-python>=1.0",
]
```

## Code Style

We use:

- **ruff** for linting and formatting (line length: 120)
- **ty** for type checking
- **pytest** for testing (88% coverage target)

```bash
# Check style
uv run ruff check graph_bench/

# Format code
uv run ruff format graph_bench/

# Type check
uv run ty check graph_bench/

# Run tests with coverage
uv run pytest --cov=graph_bench --cov-report=term-missing
```

## Testing Guidelines

1. **Unit tests** for individual components
2. **Integration tests** for adapter operations (marked with `@pytest.mark.integration`)
3. **Benchmark tests** to verify benchmarks run correctly

Example test:

```python
import pytest
from graph_bench.adapters import GrafeoAdapter
from graph_bench.benchmarks import TriangleCountBenchmark
from graph_bench.config import get_scale


@pytest.fixture
def adapter():
    adapter = GrafeoAdapter()
    adapter.connect()
    yield adapter
    adapter.disconnect()


def test_triangle_count_benchmark(adapter):
    scale = get_scale("small")
    benchmark = TriangleCountBenchmark()

    metrics = benchmark.run(adapter, scale)

    assert metrics.timing.iterations == scale.measurement_iterations
    assert metrics.timing.mean_ns > 0
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting: `uv run pytest && uv run ruff check .`
5. Commit with descriptive message
6. Push to your fork
7. Create a Pull Request

## Benchmark Design Principles

1. **Reproducibility** - Same inputs should produce comparable results
2. **Isolation** - Each benchmark should clean up after itself
3. **Scalability** - Benchmarks should scale with the `ScaleConfig`
4. **Real-world relevance** - Measure operations that matter in practice
5. **Fair comparison** - Use adapter methods, not database-specific queries

## Questions?

Open an issue on GitHub or reach out to the maintainers.
