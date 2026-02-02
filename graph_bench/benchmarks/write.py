r"""
Write operation benchmarks for graph databases.

Measures update, delete, and mixed workload operations.

    from graph_bench.benchmarks.write import PropertyUpdateBenchmark

    bench = PropertyUpdateBenchmark()
    metrics = bench.run(adapter, scale)
"""

import random
import time
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "PropertyUpdateBenchmark",
    "EdgeAddExistingBenchmark",
    "MixedWorkloadBenchmark",
]


class WriteBenchmarkBase(BaseBenchmark):
    """Base class for write operation benchmarks."""

    category = "write"
    _node_ids: list[str] = []
    _node_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        self._node_count = min(2000, scale.nodes // 50)
        edge_count = min(5000, scale.edges // 100)

        nodes = [
            {"id": f"person_{i}", "name": f"Person {i}", "age": 20 + (i % 60), "score": 0}
            for i in range(self._node_count)
        ]
        adapter.insert_nodes(nodes, label="Person")

        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(edge_count):
            src = f"person_{i % self._node_count}"
            tgt = f"person_{(i * 3 + 1) % self._node_count}"
            if src != tgt:
                edges.append((src, tgt, "FOLLOWS", {"weight": 1.0}))

        adapter.insert_edges(edges)
        self._node_ids = [f"person_{i}" for i in range(self._node_count)]


@BenchmarkRegistry.register("property_update", category="write")
class PropertyUpdateBenchmark(WriteBenchmarkBase):
    """Benchmark property update operations.

    Property updates are common in:
    - User profile changes
    - Score/rating updates
    - Status changes
    """

    @property
    def name(self) -> str:
        return "property_update"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Update properties on random nodes."""
        update_count = 50
        updated = 0

        for _ in range(update_count):
            node_id = random.choice(self._node_ids)
            new_score = random.randint(1, 100)

            # Use execute_query for updates - most DBs support this pattern
            try:
                # Try Cypher-style update (Neo4j, Memgraph, LadybugDB, FalkorDB)
                adapter.execute_query(
                    "MATCH (n {id: $id}) SET n.score = $score RETURN n",
                    params={"id": node_id, "score": new_score},
                )
                updated += 1
            except Exception:
                try:
                    # Try GQL-style update (Grafeo)
                    adapter.execute_query(
                        "MATCH (n {id: $id}) SET n.score = $score",
                        params={"id": node_id, "score": new_score},
                    )
                    updated += 1
                except Exception:
                    try:
                        # Try AQL-style update (ArangoDB)
                        adapter.execute_query(
                            """
                            FOR doc IN Person
                                FILTER doc.id == @id
                                UPDATE doc WITH { score: @score } IN Person
                            """,
                            params={"id": node_id, "score": new_score},
                        )
                        updated += 1
                    except Exception:
                        pass

        return updated


@BenchmarkRegistry.register("edge_add_existing", category="write")
class EdgeAddExistingBenchmark(WriteBenchmarkBase):
    """Benchmark adding edges between existing nodes.

    Adding edges to existing nodes is common for:
    - New social connections
    - Relationship updates
    - Dynamic graph evolution
    """

    @property
    def name(self) -> str:
        return "edge_add_existing"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Add new edges between existing nodes."""
        edge_count = 30
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        for _ in range(edge_count):
            src = random.choice(self._node_ids)
            tgt = random.choice(self._node_ids)
            if src != tgt:
                edges.append((src, tgt, "NEW_CONNECTION", {"created_at": time.time()}))

        return adapter.insert_edges(edges)


@BenchmarkRegistry.register("mixed_workload", category="write")
class MixedWorkloadBenchmark(WriteBenchmarkBase):
    """Benchmark mixed read/write workload (80% read, 20% write).

    Mixed workloads simulate real application patterns:
    - OLTP-style access
    - Concurrent read-heavy with occasional writes
    - Real-world usage patterns
    """

    @property
    def name(self) -> str:
        return "mixed_workload"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Execute mixed read/write operations."""
        total_ops = 50
        read_ratio = 0.8
        completed = 0

        for i in range(total_ops):
            if random.random() < read_ratio:
                # Read operation
                node_id = random.choice(self._node_ids)
                if random.random() < 0.5:
                    # Point lookup
                    result = adapter.get_node(node_id)
                    if result:
                        completed += 1
                else:
                    # Neighbor lookup
                    neighbors = adapter.get_neighbors(node_id)
                    completed += 1
            else:
                # Write operation - add an edge
                src = random.choice(self._node_ids)
                tgt = random.choice(self._node_ids)
                if src != tgt:
                    try:
                        adapter.insert_edges([(src, tgt, "INTERACTS", {"ts": i})])
                        completed += 1
                    except Exception:
                        pass

        return completed
