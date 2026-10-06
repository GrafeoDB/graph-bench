r"""
Every result says what it measured (native or fallback), and the reports show it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_bench.adapters.base import BaseAdapter
from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.benchmarks.vector import AnnSettings, configure_ann
from graph_bench.reporting import CsvExporter, JsonExporter, MarkdownExporter, ResultCollector
from graph_bench.reporting.collector import implementation_label
from graph_bench.reporting.template import TemplateExporter
from graph_bench.runner import BenchmarkOrchestrator, OrchestratorConfig
from graph_bench.types import BenchmarkResult, ScaleConfig, Status

from .dict_adapter import DictAdapter

SCALE = ScaleConfig(name="tiny", nodes=600, edges=1000, warmup_iterations=1, measurement_iterations=2)
BENCHMARKS = ["ldbc_pagerank", "ldbc_lcc", "lcc_undirected", "vector_ann_knn"]


def run(adapter: BaseAdapter, names: list[str]) -> list[BenchmarkResult[Any]]:
    orchestrator = BenchmarkOrchestrator(config=OrchestratorConfig(scale=SCALE))
    classes = [BenchmarkRegistry.get(name) for name in names]
    benchmarks = [bench_cls() for bench_cls in classes if bench_cls is not None]
    return orchestrator.run([adapter], benchmarks=benchmarks, scale=SCALE).results


@pytest.fixture(scope="module")
def collector() -> Iterator[ResultCollector]:
    configure_ann(**vars(AnnSettings()))
    configure_ann(sweep=(10, 40), queries=20)
    grafeo = GrafeoAdapter()
    grafeo.connect(path=":memory:")
    dict_adapter = DictAdapter()
    collected = ResultCollector()
    collected.start_session(scale=SCALE.name, databases=["Grafeo", "dict"])
    collected.add_results(run(grafeo, BENCHMARKS))
    collected.add_results(run(dict_adapter, BENCHMARKS))
    collected.end_session()
    grafeo.disconnect()
    yield collected
    configure_ann(**vars(AnnSettings()))


def result(collector: ResultCollector, benchmark: str, database: str) -> BenchmarkResult[Any]:
    (found,) = [r for r in collector.results if r.benchmark_name == benchmark and r.database == database]
    return found


@pytest.mark.parametrize(
    ("benchmark", "database", "label"),
    [
        ("ldbc_pagerank", "Grafeo", "native"),
        ("ldbc_lcc", "Grafeo", "fallback (local_clustering_coefficient)"),
        ("lcc_undirected", "Grafeo", "native"),
        ("vector_ann_knn", "Grafeo", "native"),
        ("ldbc_pagerank", "dict", "fallback (pagerank)"),
        ("ldbc_lcc", "dict", "fallback (local_clustering_coefficient)"),
        ("vector_ann_knn", "dict", "unsupported"),
    ],
)
def test_results_are_labelled(collector: ResultCollector, benchmark: str, database: str, label: str) -> None:
    assert implementation_label(result(collector, benchmark, database)) == label


def test_unsupported_is_skipped(collector: ResultCollector) -> None:
    skipped = result(collector, "vector_ann_knn", "dict")
    assert skipped.status == Status.SKIPPED
    assert skipped.metrics is None


def test_json_keeps_labels_and_recall(collector: ResultCollector) -> None:
    data = json.loads(JsonExporter().to_string(collector))
    by_name = {(r["benchmark"], r["database"]): r for r in data["results"]}
    ann = by_name[("vector_ann_knn", "Grafeo")]["metrics"]["custom"]
    assert ann["implementation"] == "native"
    assert 0.0 <= ann["recall_at_k"] <= 1.0
    assert [p["ef"] for p in ann["sweep"]] == [10, 40]
    assert by_name[("vector_ann_knn", "dict")]["status"] == "SKIPPED"


def test_csv_has_implementation_and_recall(collector: ResultCollector) -> None:
    lines = CsvExporter().to_string(collector).splitlines()
    assert lines[0].endswith(",implementation,recall_at_k")
    assert any("ldbc_lcc" in line and '"fallback (local_clustering_coefficient)"' in line for line in lines)
    ann = next(line for line in lines if "vector_ann_knn" in line and ",Grafeo," in line)
    assert float(ann.rsplit(",", 1)[1]) >= 0.0


def test_markdown_shows_labels_recall_and_sweep(collector: ResultCollector) -> None:
    text = MarkdownExporter().to_string(collector)
    assert "(fallback)" in text
    assert "unsupported" in text
    assert "## Vector ANN (native index, recall@k against exact search)" in text
    assert "### Grafeo vector_ann_knn: recall versus latency" in text


def test_template_shows_measured_support(collector: ResultCollector) -> None:
    text = TemplateExporter().to_string(collector)
    assert "| LCC (directed, LDBC) |" in text
    assert "| LCC (undirected) |" in text
    assert "fallback (local_clustering_coefficient)" in text
    assert "## Vector ANN" in text


def test_report_command_reads_labels_back(collector: ResultCollector, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from graph_bench.cli.main import app

    results = tmp_path / "results.json"
    JsonExporter().export(collector, results)
    report = tmp_path / "report.md"
    outcome = CliRunner().invoke(app, ["report", str(results), "-f", "markdown", "-o", str(report)])
    assert outcome.exit_code == 0, outcome.output
    assert "(fallback)" in report.read_text()
