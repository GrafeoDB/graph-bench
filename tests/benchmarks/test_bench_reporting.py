r"""
CodSpeed benchmarks for result aggregation, statistics and report exporters.

Run with:
    uv run pytest tests/benchmarks --codspeed --no-cov
"""

import random

import pytest

from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.benchmarks.traversal import Hop1Benchmark
from graph_bench.reporting import CsvExporter, JsonExporter, MarkdownExporter, ResultCollector
from graph_bench.reporting.template import TemplateExporter
from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig, TimingStats

# Display names, as reported by the adapters and expected by the template exporter.
DATABASES = ["Grafeo", "LadybugDB", "FalkorDB Lite", "Neo4j", "Memgraph", "ArangoDB", "FalkorDB", "TuGraph"]
SCALE = ScaleConfig(name="sf01", nodes=1_000, edges=18_000)


def _make_result(rng: random.Random, benchmark: str, database: str) -> BenchmarkResult:
    samples = sorted(rng.randint(100_000, 10_000_000) for _ in range(10))
    timing = TimingStats(
        min_ns=samples[0],
        max_ns=samples[-1],
        mean_ns=sum(samples) / len(samples),
        median_ns=float(samples[len(samples) // 2]),
        std_ns=float(samples[-1] - samples[0]) / 4,
        p99_ns=float(samples[-1]),
        iterations=len(samples),
    )
    metrics = Metrics(
        timing=timing,
        throughput=1_000_000_000 / timing.mean_ns,
        items_processed=rng.randint(1, 10_000),
        memory_bytes=rng.randint(10_000_000, 500_000_000),
    )
    return BenchmarkResult(benchmark_name=benchmark, database=database, scale=SCALE, metrics=metrics)


@pytest.fixture(scope="module")
def collector() -> ResultCollector:
    rng = random.Random(42)
    collector = ResultCollector()
    collector.start_session(scale=SCALE.name, databases=DATABASES)
    for benchmark in BenchmarkRegistry.list():
        for database in DATABASES:
            collector.add_result(_make_result(rng, benchmark, database))
    collector.end_session()
    return collector


@pytest.mark.benchmark
def test_compute_comparisons(collector):
    comparisons = collector.compute_comparisons()
    assert comparisons


@pytest.mark.benchmark
def test_json_export(collector):
    assert JsonExporter().to_string(collector)


@pytest.mark.benchmark
def test_csv_export(collector):
    assert CsvExporter().to_string(collector)


@pytest.mark.benchmark
def test_markdown_export(collector):
    assert MarkdownExporter().to_string(collector)


@pytest.mark.benchmark
def test_template_export(collector):
    assert TemplateExporter().to_string(collector)


def test_compute_timing_stats(benchmark):
    rng = random.Random(42)
    timings = [rng.randint(100_000, 10_000_000) for _ in range(10_000)]
    stats = benchmark(Hop1Benchmark()._compute_stats, timings)
    assert stats.iterations == len(timings)
