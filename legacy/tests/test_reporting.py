r"""
Tests for graph_bench.reporting module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from graph_bench.reporting import CsvExporter, JsonExporter, MarkdownExporter, ResultCollector
from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig, Status, TimingStats


@pytest.fixture
def sample_result():
    scale = ScaleConfig(name="test", nodes=100, edges=200)
    timing = TimingStats(
        min_ns=1_000_000,
        max_ns=5_000_000,
        mean_ns=2_000_000.0,
        median_ns=2_000_000.0,
        std_ns=500_000.0,
        p99_ns=4_000_000.0,
        iterations=10,
    )
    metrics = Metrics(timing=timing, throughput=1000.0, items_processed=100)

    return BenchmarkResult(
        benchmark_name="node_insertion",
        database="test_db",
        scale=scale,
        metrics=metrics,
    )


@pytest.fixture
def collector_with_results(sample_result):
    collector = ResultCollector()
    collector.start_session(scale="test", databases=["test_db"])
    collector.add_result(sample_result)
    collector.end_session()
    return collector


class TestResultCollector:
    def test_create_collector(self):
        collector = ResultCollector()
        assert collector.results == []

    def test_start_session(self):
        collector = ResultCollector()
        collector.start_session(scale="medium", databases=["neo4j", "kuzu"])

        assert collector.session.scale == "medium"
        assert collector.session.databases == ["neo4j", "kuzu"]
        assert collector.session.session_id.startswith("bench_")

    def test_add_result(self, sample_result):
        collector = ResultCollector()
        collector.add_result(sample_result)

        assert len(collector.results) == 1
        assert collector.results[0] == sample_result

    def test_add_results(self, sample_result):
        collector = ResultCollector()
        collector.add_results([sample_result, sample_result])

        assert len(collector.results) == 2

    def test_get_results_by_database(self, sample_result):
        collector = ResultCollector()
        collector.add_result(sample_result)

        results = collector.get_results_by_database("test_db")
        assert len(results) == 1
        assert results[0].database == "test_db"

        results = collector.get_results_by_database("other_db")
        assert len(results) == 0

    def test_get_results_by_benchmark(self, sample_result):
        collector = ResultCollector()
        collector.add_result(sample_result)

        results = collector.get_results_by_benchmark("node_insertion")
        assert len(results) == 1

    def test_to_dict(self, collector_with_results):
        data = collector_with_results.to_dict()

        assert "session" in data
        assert "environment" in data
        assert "results" in data
        assert "comparisons" in data

    def test_compute_comparisons(self):
        collector = ResultCollector()

        scale = ScaleConfig(name="test", nodes=100, edges=200)

        for db, mean_ns in [("db1", 1_000_000.0), ("db2", 2_000_000.0)]:
            timing = TimingStats(
                min_ns=int(mean_ns),
                max_ns=int(mean_ns * 2),
                mean_ns=mean_ns,
                median_ns=mean_ns,
                std_ns=0.0,
                p99_ns=mean_ns * 1.5,
                iterations=10,
            )
            metrics = Metrics(timing=timing, throughput=1000.0, items_processed=100)
            result = BenchmarkResult(
                benchmark_name="test_bench",
                database=db,
                scale=scale,
                metrics=metrics,
            )
            collector.add_result(result)

        comparisons = collector.compute_comparisons()
        assert "test_bench" in comparisons
        assert comparisons["test_bench"]["db1"] == 1.0
        assert comparisons["test_bench"]["db2"] == 0.5


class TestJsonExporter:
    def test_to_string(self, collector_with_results):
        exporter = JsonExporter()
        output = exporter.to_string(collector_with_results)

        data = json.loads(output)
        assert "session" in data
        assert "results" in data

    def test_export_to_file(self, collector_with_results):
        exporter = JsonExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            exporter.export(collector_with_results, path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert "session" in data


class TestCsvExporter:
    def test_to_string(self, collector_with_results):
        exporter = CsvExporter()
        output = exporter.to_string(collector_with_results)

        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert "session_id" in lines[0]
        assert "node_insertion" in lines[1]

    def test_export_to_file(self, collector_with_results):
        exporter = CsvExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            exporter.export(collector_with_results, path)

            assert path.exists()


class TestMarkdownExporter:
    def test_to_string(self, collector_with_results):
        exporter = MarkdownExporter()
        output = exporter.to_string(collector_with_results)

        assert "# Graph Database Benchmark Report" in output
        assert "## Summary" in output
        assert "test_db" in output

    def test_export_to_file(self, collector_with_results):
        exporter = MarkdownExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            exporter.export(collector_with_results, path)

            assert path.exists()
            content = path.read_text()
            assert "# Graph Database Benchmark Report" in content
