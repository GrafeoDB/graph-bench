r"""
Export formats for benchmark results.

    from graph_bench.reporting.formats import JsonExporter, MarkdownExporter

    exporter = JsonExporter()
    exporter.export(collector, "results.json")
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from graph_bench.reporting.collector import ResultCollector

__all__ = ["BaseExporter", "JsonExporter", "CsvExporter", "MarkdownExporter"]


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
        lines = ["session_id,benchmark,category,database,scale,status,mean_ms,p99_ms,throughput,items_processed"]

        session_id = collector.session.session_id

        for result in collector.results:
            mean_ms = ""
            p99_ms = ""
            throughput = ""
            items = ""

            if result.metrics:
                mean_ms = f"{result.metrics.timing.mean_ms:.3f}"
                p99_ms = f"{result.metrics.timing.p99_ms:.3f}"
                throughput = f"{result.metrics.throughput:.2f}"
                items = str(result.metrics.items_processed)

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

        lines.append("| Database | Success | Failed | Avg Time (ms) |")
        lines.append("|----------|---------|--------|---------------|")

        for db in databases:
            db_results = collector.get_results_by_database(db)
            success = sum(1 for r in db_results if r.ok)
            failed = sum(1 for r in db_results if not r.ok)

            times = [r.metrics.timing.mean_ms for r in db_results if r.ok and r.metrics]
            avg_time = f"{sum(times) / len(times):.2f}" if times else "N/A"

            lines.append(f"| {db} | {success} | {failed} | {avg_time} |")

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
                if result and result.ok and result.metrics:
                    row += f" {result.metrics.timing.mean_ms:.2f} |"
                elif result and not result.ok:
                    row += " FAILED |"
                else:
                    row += " N/A |"
            lines.append(row)

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
