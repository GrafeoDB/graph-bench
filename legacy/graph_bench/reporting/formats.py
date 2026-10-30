r"""
Export formats for benchmark results.

    from graph_bench.reporting.formats import JsonExporter, MarkdownExporter

    exporter = JsonExporter()
    exporter.export(collector, "results.json")
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from graph_bench.reporting.collector import ResultCollector, implementation_label
from graph_bench.types import BenchmarkResult, Status

__all__ = ["BaseExporter", "CsvExporter", "JsonExporter", "MarkdownExporter", "ann_section"]


def ann_section(collector: ResultCollector) -> list[str]:
    """Markdown for the vector_ann results: latency next to recall@k, sweeps and deletes."""
    ann = [r for r in collector.results if r.benchmark_name.startswith("vector_ann_")]
    if not ann:
        return []
    lines = [
        "## Vector ANN (native index, recall@k against exact search)",
        "",
        "| Database | Benchmark | ms / query | recall@k | k | ef | m | ef_construction | quantization |",
        "|----------|-----------|-----------:|---------:|--:|---:|--:|----------------:|--------------|",
    ]
    for r in ann:
        if r.status == Status.SKIPPED:
            lines.append(f"| {r.database} | {r.benchmark_name} | unsupported | | | | | | |")
            continue
        if not r.ok or not r.metrics:
            lines.append(f"| {r.database} | {r.benchmark_name} | FAILED | | | | | | |")
            continue
        c = r.metrics.custom
        queries = c.get("queries")
        per_query = f"{r.metrics.timing.mean_ms / queries:.3f}" if queries else ""
        recall = f"{c['recall_at_k']:.4f}" if "recall_at_k" in c else ""
        lines.append(
            f"| {r.database} | {r.benchmark_name} | {per_query} | {recall} | {c.get('k', '')} | "
            f"{c.get('ef', '')} | {c.get('m', '')} | {c.get('ef_construction', '')} | "
            f"{c.get('quantization') or 'none'} |"
        )
    lines.append("")
    for r in ann:
        c = r.metrics.custom if r.ok and r.metrics else {}
        if "recall_before_delete" in c:
            lines.append(
                f"- **{r.database} {r.benchmark_name}:** deleted {c['deleted']} vectors "
                f"({c['delete_fraction']:.0%}); recall@{c['k']} {c['recall_before_delete']:.4f} before, "
                f"{c['recall_at_k']:.4f} after (drop {c['recall_drop']:+.4f}); "
                f"results naming a deleted vector: {c['deleted_returned']}; short results: {c['short_results']}"
            )
    lines.append("")
    for r in ann:
        sweep = r.metrics.custom.get("sweep") if r.ok and r.metrics else None
        if sweep:
            lines.append(f"### {r.database} {r.benchmark_name}: recall versus latency")
            lines.append("")
            lines.append("| ef | recall@k | ms / query |")
            lines.append("|---:|---------:|-----------:|")
            lines.extend(f"| {p['ef']} | {p['recall_at_k']:.4f} | {p['mean_query_ms']:.3f} |" for p in sweep)
            lines.append("")
    return lines


def _cell(result: BenchmarkResult[Any] | None) -> str:
    if result is None:
        return " N/A |"
    if result.status == Status.SKIPPED:
        return " unsupported |"
    if not result.ok or not result.metrics:
        return " FAILED |"
    label = " (fallback)" if result.metrics.custom.get("implementation") == "fallback" else ""
    return f" {result.metrics.timing.mean_ms:.2f}{label} |"


class BaseExporter(ABC):
    """Base class for result exporters."""

    @abstractmethod
    def export(self, collector: ResultCollector, path: str | Path) -> None:
        """Export results to file."""
        ...

    @abstractmethod
    def to_string(self, collector: ResultCollector) -> str:
        """Export results to string."""
        ...


class JsonExporter(BaseExporter):
    """Export results to JSON format."""

    def __init__(self, *, indent: int = 2) -> None:
        self._indent = indent

    def export(self, collector: ResultCollector, path: str | Path) -> None:
        """Export results to JSON file."""
        Path(path).write_text(self.to_string(collector))

    def to_string(self, collector: ResultCollector) -> str:
        """Export results to JSON string."""
        return json.dumps(collector.to_dict(), indent=self._indent)


class CsvExporter(BaseExporter):
    """Export results to CSV format."""

    def export(self, collector: ResultCollector, path: str | Path) -> None:
        """Export results to CSV file."""
        Path(path).write_text(self.to_string(collector))

    def to_string(self, collector: ResultCollector) -> str:
        """Export results to CSV string."""
        lines = [
            "session_id,benchmark,category,database,scale,status,mean_ms,p99_ms,throughput,items_processed,"
            "implementation,recall_at_k"
        ]

        session_id = collector.session.session_id

        for result in collector.results:
            mean_ms = ""
            p99_ms = ""
            throughput = ""
            items = ""
            recall = ""

            if result.metrics:
                mean_ms = f"{result.metrics.timing.mean_ms:.3f}"
                p99_ms = f"{result.metrics.timing.p99_ms:.3f}"
                throughput = f"{result.metrics.throughput:.2f}"
                items = str(result.metrics.items_processed)
                if "recall_at_k" in result.metrics.custom:
                    recall = f"{result.metrics.custom['recall_at_k']:.4f}"

            line = ",".join([
                session_id,
                result.benchmark_name,
                self._get_category(result.benchmark_name),
                result.database,
                result.scale.name,
                result.status.name,
                mean_ms,
                p99_ms,
                throughput,
                items,
                f'"{implementation_label(result)}"',
                recall,
            ])
            lines.append(line)

        return "\n".join(lines)

    def _get_category(self, benchmark_name: str) -> str:
        """Get category from benchmark name."""
        from graph_bench.benchmarks.base import BenchmarkRegistry

        bench_cls = BenchmarkRegistry.get(benchmark_name)
        if bench_cls:
            return bench_cls.category
        return "unknown"


class MarkdownExporter(BaseExporter):
    """Export results to Markdown format."""

    def export(self, collector: ResultCollector, path: str | Path) -> None:
        """Export results to Markdown file."""
        Path(path).write_text(self.to_string(collector))

    def to_string(self, collector: ResultCollector) -> str:
        """Export results to Markdown string."""
        lines: list[str] = []
        session = collector.session
        env = collector.environment

        lines.append("# Graph Database Benchmark Report")
        lines.append("")
        lines.append(f"**Session:** {session.session_id}")
        lines.append(f"**Scale:** {session.scale}")
        lines.append(f"**Date:** {session.started_at[:10] if session.started_at else 'N/A'}")
        lines.append("")

        lines.append("## Environment")
        lines.append("")
        lines.append(f"- Platform: {env.platform}")
        lines.append(f"- Python: {env.python_version}")
        lines.append(f"- CPU: {env.cpu}")
        lines.append(f"- Memory: {env.memory_gb} GB")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        self._add_summary_table(collector, lines)

        categories = self._get_categories(collector)
        for category in categories:
            lines.append(f"## {category.title()} Benchmarks")
            lines.append("")
            self._add_category_table(collector, category, lines)

        lines.extend(ann_section(collector))

        comparisons = collector.compute_comparisons()
        if comparisons:
            lines.append("## Performance Comparisons")
            lines.append("")
            self._add_comparison_table(comparisons, lines)

        return "\n".join(lines)

    def _get_categories(self, collector: ResultCollector) -> list[str]:
        """Get unique categories from results."""
        from graph_bench.benchmarks.base import BenchmarkRegistry

        categories = set()
        for result in collector.results:
            bench_cls = BenchmarkRegistry.get(result.benchmark_name)
            if bench_cls:
                categories.add(bench_cls.category)
        return sorted(categories)

    def _add_summary_table(self, collector: ResultCollector, lines: list[str]) -> None:
        """Add summary table to lines."""
        databases = collector.session.databases

        lines.append("| Database | Success | Native | Fallback | Unsupported | Failed | Avg Time (ms) |")
        lines.append("|----------|---------|--------|----------|-------------|--------|---------------|")

        for db in databases:
            db_results = collector.get_results_by_database(db)
            success = sum(1 for r in db_results if r.ok)
            labels = [r.metrics.custom.get("implementation") for r in db_results if r.ok and r.metrics]
            skipped = sum(1 for r in db_results if r.status == Status.SKIPPED)
            failed = sum(1 for r in db_results if r.status not in (Status.SUCCESS, Status.SKIPPED))

            times = [r.metrics.timing.mean_ms for r in db_results if r.ok and r.metrics]
            avg_time = f"{sum(times) / len(times):.2f}" if times else "N/A"

            lines.append(
                f"| {db} | {success} | {labels.count('native')} | {labels.count('fallback')} | "
                f"{skipped} | {failed} | {avg_time} |"
            )

        lines.append("")

    def _add_category_table(self, collector: ResultCollector, category: str, lines: list[str]) -> None:
        """Add category-specific benchmark table."""
        from graph_bench.benchmarks.base import BenchmarkRegistry

        databases = collector.session.databases
        benchmarks = []

        for result in collector.results:
            bench_cls = BenchmarkRegistry.get(result.benchmark_name)
            if bench_cls and bench_cls.category == category:
                if result.benchmark_name not in benchmarks:
                    benchmarks.append(result.benchmark_name)

        if not benchmarks:
            lines.append("No benchmarks in this category.")
            lines.append("")
            return

        header = "| Benchmark |" + " | ".join(f"{db} (ms)" for db in databases) + " |"
        separator = "|-----------|" + "|".join("-" * 12 for _ in databases) + "|"
        lines.append(header)
        lines.append(separator)

        for bench in benchmarks:
            row = f"| {bench} |"
            for db in databases:
                result = next((r for r in collector.results if r.benchmark_name == bench and r.database == db), None)
                row += _cell(result)
            lines.append(row)

        lines.append("")
        lines.append("*(fallback): part of the measured work ran in Python, not in the database.*")
        lines.append("")

    def _add_comparison_table(self, comparisons: dict[str, dict[str, float]], lines: list[str]) -> None:
        """Add comparison table showing speedups."""
        if not comparisons:
            return

        all_dbs = set()
        for speedups in comparisons.values():
            all_dbs.update(speedups.keys())
        databases = sorted(all_dbs)

        header = "| Benchmark |" + " | ".join(databases) + " |"
        separator = "|-----------|" + "|".join("-" * 10 for _ in databases) + "|"
        lines.append(header)
        lines.append(separator)

        for bench, speedups in sorted(comparisons.items()):
            row = f"| {bench} |"
            for db in databases:
                speed = speedups.get(db, 0.0)
                if speed == 1.0:
                    row += " **1.00x** |"
                else:
                    row += f" {speed:.2f}x |"
            lines.append(row)

        lines.append("")
        lines.append("*Speedup relative to fastest (1.00x = fastest)*")
        lines.append("")
