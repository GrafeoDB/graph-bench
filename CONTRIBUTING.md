# Contributing

Want to add a benchmark or database adapter? Here's how.

## Setup

```bash
git clone https://github.com/StevenBtw/graph-bench
cd graph-bench
uv sync --all-extras

# Check everything works
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

## Adding a Benchmark

### 1. Pick a category

- `storage` - Node/edge CRUD
- `traversal` - Graph traversal
- `algorithm` - PageRank, centrality, etc.
- `query` - Aggregation, filtering
- `pattern` - Triangles, common neighbors
- `structure` - Graph structure analysis
- `write` - Write-heavy operations

### 2. Create the benchmark

Add to the right file in `graph_bench/benchmarks/`:

```python
from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig


@BenchmarkRegistry.register("my_benchmark", category="traversal")
class MyBenchmark(BaseBenchmark):
    """What this benchmark measures."""

    category = "traversal"

    @property
    def name(self) -> str:
        return "my_benchmark"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Prepare data for the benchmark."""
        adapter.clear()
        node_count = min(5000, scale.nodes // 20)
        nodes = [{"id": f"node_{i}", "value": i} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="TestNode")
        self._node_ids = [f"node_{i}" for i in range(node_count)]

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run one iteration. Returns items processed for throughput calc."""
        import random

        processed = 0
        for _ in range(10):
            node_id = random.choice(self._node_ids)
            result = adapter.get_node(node_id)
            if result:
                processed += 1
        return processed

    def teardown(self, adapter: GraphDatabaseAdapter) -> None:
        """Clean up (optional)."""
        pass
```

### 3. Register it

Add to `graph_bench/benchmarks/__init__.py`:

```python
from graph_bench.benchmarks.traversal import MyBenchmark

__all__ = [
    # ...
    "MyBenchmark",
]
```

### 4. Test it

```bash
uv run graph-bench run -d grafeo -b my_benchmark -s small --verbose
```

## Adding a Database Adapter

### 1. Create the adapter

New file in `graph_bench/adapters/`:

```python
from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env


@AdapterRegistry.register("mydb")
class MyDBAdapter(BaseAdapter):
    """MyDB adapter."""

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
            msg = "mydb not installed. Install with: pip install mydb"
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

### 2. Register it

Add to `graph_bench/adapters/__init__.py`:

```python
from graph_bench.adapters.mydb import MyDBAdapter

__all__ = [
    # ...
    "MyDBAdapter",
]
```

### 3. Add config

Update `.env.example`:

```bash
GRAPH_BENCH_MYDB_URI=localhost:1234
```

### 4. Add dependency

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
mydb = ["mydb-python>=1.0"]
```

## Code Style

- **ruff** for linting and formatting (line length: 120)
- **ty** for type checking
- **pytest** for testing

```bash
uv run ruff check graph_bench/
uv run ruff format graph_bench/
uv run ty check graph_bench/
uv run pytest --cov=graph_bench --cov-report=term-missing
```

## Pull Requests

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests and linting
5. Push and create a PR

## Questions?

Open an issue.
