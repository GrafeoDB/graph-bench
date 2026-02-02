r"""
Tests for graph_bench.benchmarks module.
"""

import pytest

from graph_bench.benchmarks import (
    BaseBenchmark,
    BenchmarkRegistry,
    NodeInsertionBenchmark,
    EdgeInsertionBenchmark,
    SingleReadBenchmark,
    BatchReadBenchmark,
    Hop1Benchmark,
    Hop2Benchmark,
    BFSBenchmark,
    DFSBenchmark,
    ShortestPathBenchmark,
    PageRankBenchmark,
    CommunityDetectionBenchmark,
    AggregationCountBenchmark,
    FilterEqualityBenchmark,
    FilterRangeBenchmark,
)


class TestBenchmarkRegistry:
    def test_registry_has_benchmarks(self):
        benchmarks = BenchmarkRegistry.list()
        assert "node_insertion" in benchmarks
        assert "edge_insertion" in benchmarks
        assert "hop_1" in benchmarks
        assert "pagerank" in benchmarks

    def test_get_benchmark_class(self):
        bench_cls = BenchmarkRegistry.get("node_insertion")
        assert bench_cls is not None
        assert issubclass(bench_cls, BaseBenchmark)

    def test_get_unknown_benchmark(self):
        bench_cls = BenchmarkRegistry.get("unknown")
        assert bench_cls is None

    def test_by_category_storage(self):
        storage_benchmarks = BenchmarkRegistry.by_category("storage")
        assert "node_insertion" in storage_benchmarks
        assert "edge_insertion" in storage_benchmarks

    def test_by_category_traversal(self):
        traversal_benchmarks = BenchmarkRegistry.by_category("traversal")
        assert "hop_1" in traversal_benchmarks
        assert "bfs" in traversal_benchmarks


class TestStorageBenchmarks:
    def test_node_insertion_benchmark(self):
        bench = NodeInsertionBenchmark()
        assert bench.name == "node_insertion"
        assert bench.category == "storage"

    def test_edge_insertion_benchmark(self):
        bench = EdgeInsertionBenchmark()
        assert bench.name == "edge_insertion"
        assert bench.category == "storage"

    def test_single_read_benchmark(self):
        bench = SingleReadBenchmark()
        assert bench.name == "single_read"

    def test_batch_read_benchmark(self):
        bench = BatchReadBenchmark()
        assert bench.name == "batch_read"


class TestTraversalBenchmarks:
    def test_hop1_benchmark(self):
        bench = Hop1Benchmark()
        assert bench.name == "hop_1"
        assert bench.category == "traversal"

    def test_hop2_benchmark(self):
        bench = Hop2Benchmark()
        assert bench.name == "hop_2"

    def test_bfs_benchmark(self):
        bench = BFSBenchmark()
        assert bench.name == "bfs"

    def test_dfs_benchmark(self):
        bench = DFSBenchmark()
        assert bench.name == "dfs"

    def test_shortest_path_benchmark(self):
        bench = ShortestPathBenchmark()
        assert bench.name == "shortest_path"


class TestAlgorithmBenchmarks:
    def test_pagerank_benchmark(self):
        bench = PageRankBenchmark()
        assert bench.name == "pagerank"
        assert bench.category == "algorithm"

    def test_community_detection_benchmark(self):
        bench = CommunityDetectionBenchmark()
        assert bench.name == "community_detection"


class TestQueryBenchmarks:
    def test_aggregation_count_benchmark(self):
        bench = AggregationCountBenchmark()
        assert bench.name == "aggregation_count"
        assert bench.category == "query"

    def test_filter_equality_benchmark(self):
        bench = FilterEqualityBenchmark()
        assert bench.name == "filter_equality"

    def test_filter_range_benchmark(self):
        bench = FilterRangeBenchmark()
        assert bench.name == "filter_range"
