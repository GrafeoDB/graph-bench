r"""
Result collection and reporting.

Aggregates benchmark results and exports to
JSON, CSV, and Markdown formats.

    from graph_bench.reporting import ResultCollector, MarkdownExporter

    collector = ResultCollector()
    collector.add_result(result)
    MarkdownExporter().export(collector, "report.md")
"""

from graph_bench.reporting.collector import ResultCollector, SessionInfo
from graph_bench.reporting.formats import CsvExporter, JsonExporter, MarkdownExporter

__all__ = [
    "CsvExporter",
    "JsonExporter",
    "MarkdownExporter",
    "ResultCollector",
    "SessionInfo",
]
