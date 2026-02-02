r"""
Query benchmarks for graph databases.

Measures query performance including aggregations and filtering.

    from graph_bench.benchmarks.query import AggregationBenchmark

    bench = AggregationBenchmark()
    metrics = bench.run(adapter, scale)
"""

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "AggregationCountBenchmark",
    "FilterEqualityBenchmark",
    "FilterRangeBenchmark",
]


class QueryBenchmarkBase(BaseBenchmark):
    """Base class for query benchmarks."""

    category = "query"

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        adapter.clear()
        node_count = min(10000, scale.nodes // 10)

        nodes = [
            {"id": f"person_{i}", "name": f"Person {i}", "age": 20 + (i % 60), "city": f"City_{i % 100}"}
            for i in range(node_count)
        ]
        adapter.insert_nodes(nodes, label="Person")


@BenchmarkRegistry.register("aggregation_count", category="query")
class AggregationCountBenchmark(QueryBenchmarkBase):
    """Benchmark COUNT aggregation."""

    @property
    def name(self) -> str:
        return "aggregation_count"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        total = 0
        for _ in range(50):
            count = adapter.count_nodes(label="Person")
            total += count
        return total


@BenchmarkRegistry.register("filter_equality", category="query")
class FilterEqualityBenchmark(QueryBenchmarkBase):
    """Benchmark equality filter queries."""

    @property
    def name(self) -> str:
        return "filter_equality"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        total = 0
        for i in range(20):
            city = f"City_{i % 100}"
            try:
                query = "MATCH (n:Person {city: $city}) RETURN n LIMIT 100"
                results = adapter.execute_query(query, params={"city": city})
                total += len(results)
            except Exception:
                pass
        return total


@BenchmarkRegistry.register("filter_range", category="query")
class FilterRangeBenchmark(QueryBenchmarkBase):
    """Benchmark range filter queries."""

    @property
    def name(self) -> str:
        return "filter_range"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        total = 0
        for i in range(20):
            min_age = 20 + (i % 40)
            max_age = min_age + 10
            try:
                results = adapter.execute_query(
                    "MATCH (n:Person) WHERE n.age >= $min_age AND n.age <= $max_age RETURN n LIMIT 100",
                    params={"min_age": min_age, "max_age": max_age},
                )
                total += len(results)
            except Exception:
                pass
        return total
