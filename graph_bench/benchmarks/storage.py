r"""
Storage benchmarks for graph databases.

Measures node/edge insertion, read, update, and delete operations.

    from graph_bench.benchmarks.storage import NodeInsertionBenchmark

    bench = NodeInsertionBenchmark()
    metrics = bench.run(adapter, scale)
"""

from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "NodeInsertionBenchmark",
    "EdgeInsertionBenchmark",
    "SingleReadBenchmark",
    "BatchReadBenchmark",
]


@BenchmarkRegistry.register("node_insertion", category="storage")
class NodeInsertionBenchmark(BaseBenchmark):
    """Benchmark node insertion throughput."""

    category = "storage"

    @property
    def name(self) -> str:
        return "node_insertion"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        batch_size = min(1000, scale.nodes // 10)
        nodes = self._generate_nodes(batch_size)
        return adapter.insert_nodes(nodes, label="Person", batch_size=batch_size)

    def _generate_nodes(self, count: int) -> list[dict[str, Any]]:
        return [{"id": f"person_{i}", "name": f"Person {i}", "age": 20 + (i % 60)} for i in range(count)]


@BenchmarkRegistry.register("edge_insertion", category="storage")
class EdgeInsertionBenchmark(BaseBenchmark):
    """Benchmark edge insertion throughput."""

    category = "storage"
    _node_count: int = 0

    @property
    def name(self) -> str:
        return "edge_insertion"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._node_count = min(1000, scale.nodes // 10)
        nodes = [{"id": f"person_{i}", "name": f"Person {i}"} for i in range(self._node_count)]
        adapter.insert_nodes(nodes, label="Person")

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        edge_count = min(500, scale.edges // 10)
        edges = self._generate_edges(edge_count)
        return adapter.insert_edges(edges)

    def _generate_edges(self, count: int) -> list[tuple[str, str, str, dict[str, Any]]]:
        edges = []
        for i in range(count):
            src = f"person_{i % self._node_count}"
            tgt = f"person_{(i + 1) % self._node_count}"
            edges.append((src, tgt, "FOLLOWS", {"since": 2020 + (i % 5)}))
        return edges


@BenchmarkRegistry.register("single_read", category="storage")
class SingleReadBenchmark(BaseBenchmark):
    """Benchmark single node lookup latency."""

    category = "storage"
    _node_ids: list[str] = []

    @property
    def name(self) -> str:
        return "single_read"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        node_count = min(10000, scale.nodes)
        nodes = [{"id": f"person_{i}", "name": f"Person {i}", "age": 20 + (i % 60)} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="Person")
        self._node_ids = [f"person_{i}" for i in range(node_count)]

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        import random

        count = 0
        for _ in range(100):
            node_id = random.choice(self._node_ids)
            result = adapter.get_node(node_id)
            if result:
                count += 1
        return count


@BenchmarkRegistry.register("batch_read", category="storage")
class BatchReadBenchmark(BaseBenchmark):
    """Benchmark batch node retrieval throughput."""

    category = "storage"

    @property
    def name(self) -> str:
        return "batch_read"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        node_count = min(10000, scale.nodes)
        nodes = [{"id": f"person_{i}", "name": f"Person {i}", "age": 20 + (i % 60)} for i in range(node_count)]
        adapter.insert_nodes(nodes, label="Person")

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        nodes = adapter.get_nodes_by_label("Person", limit=1000)
        return len(nodes)
