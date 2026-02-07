r"""
Template-style exporter matching LATEST_RESULTS.md format.

Generates markdown output with:
- Summary tables by category (SNB Interactive, Graph Analytics, Combinatorial)
- Per-database detailed results in collapsible sections
- Feature availability matrix
- Database characteristics table
- Footnotes for caveats

    from graph_bench.reporting.template import TemplateExporter

    exporter = TemplateExporter()
    exporter.export(collector, "RESULTS.md")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graph_bench.reporting.formats import BaseExporter

if TYPE_CHECKING:
    from graph_bench.reporting.collector import ResultCollector
    from graph_bench.types import BenchmarkResult

__all__ = ["TemplateExporter"]


# Database ordering for tables
DB_ORDER = [
    "Grafeo",
    "LadybugDB",
    "Neo4j",
    "Memgraph",
    "FalkorDB",
    "ArangoDB",
    "NebulaGraph",
    "TuGraph",
]

# Database types
DB_TYPES = {
    "Grafeo": "Embedded",
    "LadybugDB": "Embedded",
    "Neo4j": "Server",
    "Memgraph": "Server",
    "FalkorDB": "Server",
    "ArangoDB": "Server",
    "NebulaGraph": "Distributed",
    "TuGraph": "Server",
}

# Benchmark categories and their benchmarks (must match actual registered names)
CATEGORY_BENCHMARKS = {
    "SNB Interactive - Short Reads": [
        "snb_is1",
        "snb_is2",
        "snb_is3",
        "snb_is4",
        "snb_is5",
        "snb_is6",
        "snb_is7",
    ],
    "SNB Interactive - Complex Reads": [
        "snb_ic1",
        "snb_ic2",
        "snb_ic3",
        "snb_ic6",
    ],
    # Combined SNB Interactive for summary tables
    "SNB Interactive": [
        "snb_is1", "snb_is2", "snb_is3", "snb_is4", "snb_is5", "snb_is6", "snb_is7",
        "snb_ic1", "snb_ic2", "snb_ic3", "snb_ic6",
    ],
    "LDBC Graphanalytics": [
        "ldbc_bfs",
        "ldbc_pagerank",
        "ldbc_wcc",
        "ldbc_cdlp",
        "ldbc_lcc",
        "ldbc_sssp",
    ],
    # Alias for summary tables
    "Graph Analytics": [
        "ldbc_bfs",
        "ldbc_pagerank",
        "ldbc_wcc",
        "ldbc_cdlp",
        "ldbc_lcc",
        "ldbc_sssp",
    ],
    "LDBC ACID - Atomicity": [
        "acid_atomicity_c",
        "acid_atomicity_rb",
    ],
    "LDBC ACID - Isolation": [
        "acid_g0",
        "acid_g1a",
        "acid_g1b",
        "acid_g1c",
        "acid_imp",
        "acid_pmp",
        "acid_otv",
        "acid_fr",
        "acid_lu",
        "acid_ws",
    ],
    "Combinatorial - Writes": [
        "node_insertion",
        "edge_insertion",
    ],
    "Combinatorial - Reads": [
        "single_read",
        "batch_read",
    ],
    "Combinatorial - Traversals": [
        "bfs",
        "dfs",
        "hop_1",
        "hop_2",
        "triangle_count",
        "common_neighbors",
    ],
    "Concurrent ACID": [
        "concurrent_mixed",
        "throughput_scaling",
        "lost_update",
        "read_after_write",
        "concurrent_acid",
    ],
    "Vector Search": [
        "vector_insert",
        "vector_knn",
        "vector_batch_search",
        "vector_recall",
    ],
    "Hybrid Graph+Vector": [
        "hybrid_graph_to_vector",
        "hybrid_vector_to_graph",
    ],
}

# Databases with native graph analytics support (not using NetworkX fallback)
NATIVE_ANALYTICS = {
    "Grafeo": [
        "ldbc_bfs", "ldbc_pagerank", "ldbc_wcc", "ldbc_cdlp", "ldbc_lcc", "ldbc_sssp"
    ],
    "Neo4j": ["ldbc_bfs", "ldbc_pagerank", "ldbc_wcc", "ldbc_cdlp", "ldbc_sssp"],
    "Memgraph": [
        "ldbc_bfs", "ldbc_pagerank", "ldbc_wcc", "ldbc_cdlp", "ldbc_lcc", "ldbc_sssp"
    ],
    "TuGraph": [
        "ldbc_bfs", "ldbc_pagerank", "ldbc_wcc", "ldbc_cdlp", "ldbc_lcc", "ldbc_sssp"
    ],
}

# Databases with native vector search support
NATIVE_VECTOR = {
    "Grafeo": [
        "vector_insert", "vector_knn", "vector_batch_search",
        "vector_recall", "hybrid_graph_to_vector",
        "hybrid_vector_to_graph",
    ],
}

# Database info for detailed sections
DB_INFO = {
    "Grafeo": {
        "type": "Embedded (in-process, Rust)",
        "data_model": "LPG + RDF",
        "query_languages": "GQL (ISO), Cypher, Gremlin, GraphQL, SPARQL",
        "acid": "Full (snapshot isolation, WAL)",
        "consistency": "Strong",
        "license": "Apache 2.0",
    },
    "LadybugDB": {
        "type": "Embedded",
        "data_model": "LPG",
        "query_languages": "Cypher",
        "acid": "Full (snapshot isolation)",
        "consistency": "Strong",
        "license": "MIT",
    },
    "Neo4j": {
        "type": "Server (Bolt RPC)",
        "data_model": "LPG",
        "query_languages": "Cypher",
        "acid": "Full (read committed isolation)",
        "consistency": "Strong",
        "license": "GPL / Commercial",
    },
    "Memgraph": {
        "type": "Server (Bolt RPC)",
        "data_model": "LPG",
        "query_languages": "Cypher",
        "acid": "Yes (single-node, snapshot isolation)",
        "consistency": "Strong",
        "license": "BSL / Enterprise",
    },
    "FalkorDB": {
        "type": "Server (Redis protocol)",
        "data_model": "LPG",
        "query_languages": "Cypher",
        "acid": "Partial (Redis-level durability, AOF fsync policy)",
        "consistency": "Strong",
        "license": "SSPL",
    },
    "ArangoDB": {
        "type": "Server (HTTP / TCP)",
        "data_model": "Multi-model (document, key-value, graph)",
        "query_languages": "AQL, Gremlin, GraphQL",
        "acid": "Full (read committed isolation)",
        "consistency": "Strong",
        "license": "Apache 2.0",
    },
    "NebulaGraph": {
        "type": "Distributed (Thrift RPC)",
        "data_model": "LPG",
        "query_languages": "nGQL",
        "acid": "Partial (eventual consistency)",
        "consistency": "Eventual (tunable replica factor)",
        "license": "Apache 2.0",
    },
    "TuGraph": {
        "type": "Server (Bolt RPC)",
        "data_model": "LPG",
        "query_languages": "Cypher, GQL (ISO)",
        "acid": "Full (snapshot isolation)",
        "consistency": "Strong",
        "license": "Apache 2.0",
    },
}


class TemplateExporter(BaseExporter):
    """Export results in LATEST_RESULTS.md template format."""

    def export(self, collector: ResultCollector, path: str | Path) -> None:
        """Export results to markdown file."""
        Path(path).write_text(self.to_string(collector), encoding="utf-8")

    def to_string(self, collector: ResultCollector) -> str:
        """Export results to markdown string."""
        lines: list[str] = []
        session = collector.session
        env = collector.environment

        # Header
        date = datetime.now().strftime("%Y-%m-%d")
        platform = env.platform.title() if env.platform else "Unknown"

        lines.append("# Benchmark Results")
        lines.append("")
        lines.append(f"**Run:** {date} | **Platform:** {platform}  ")
        lines.append(
            "**Benchmark suite:** [graph-bench](https://github.com/GrafeoDB/graph-bench)"
        )
        lines.append("")

        # Scale factors
        self._add_scale_factors(lines, session.scale)

        # Summary section
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")

        # Get databases in order
        dbs = self._get_ordered_databases(collector)

        # SNB Interactive summary
        self._add_summary_section(
            collector, lines, "SNB Interactive", CATEGORY_BENCHMARKS["SNB Interactive"], dbs
        )

        # Graph Analytics summary (native only)
        self._add_analytics_summary(collector, lines, dbs)

        # Combinatorial summary
        self._add_combinatorial_summary(collector, lines, dbs)

        # Reading the results section
        self._add_reading_results(lines)

        # Per-database detailed results
        lines.append("---")
        lines.append("")
        lines.append("## Per-Database Results")
        lines.append("")

        for db in dbs:
            self._add_database_details(collector, lines, db)

        # Query Languages & Data Models
        self._add_query_languages(lines, dbs)

        # Native Algorithm Support
        self._add_native_support(lines, dbs)

        # Methodology
        self._add_methodology(lines)

        # Raw Data
        self._add_raw_data(lines, session.session_id)

        return "\n".join(lines)

    def _get_ordered_databases(self, collector: ResultCollector) -> list[str]:
        """Get databases in preferred order."""
        available = set(collector.session.databases)
        return [db for db in DB_ORDER if db in available]

    def _add_scale_factors(self, lines: list[str], scale: str) -> None:
        """Add scale factors section (LDBC standard)."""
        lines.append("## Scale Factors (LDBC Standard)")
        lines.append("")
        lines.append("| Scale | Persons | KNOWS Edges | Reference |")
        lines.append("|-------|--------:|------------:|-----------|")
        lines.append("| SF0.1 (sf01) | 1K | 18K | Quick validation |")
        lines.append("| SF1 (sf1) | 10K | 180K | Standard benchmark |")
        lines.append("| SF3 (sf3) | 27K | 540K | Medium scale |")
        lines.append("| SF10 (sf10) | 73K | 2M | Large scale |")
        lines.append("| SF30 (sf30) | 180K | 6.5M | Very large |")
        lines.append("| SF100 (sf100) | 280K | 18M | Full scale |")
        lines.append("")
        lines.append(f"**Current run:** {scale}")
        lines.append("")
        lines.append(
            "All times in milliseconds. Best result per benchmark in **bold**. "
        )
        lines.append("")

    def _add_summary_section(
        self,
        collector: ResultCollector,
        lines: list[str],
        title: str,
        benchmarks: list[str],
        dbs: list[str],
    ) -> None:
        """Add a summary section for a category."""
        lines.append(f"### {title} (total ms)")
        lines.append("")

        scale = collector.session.scale
        lines.append(f"| Database | Type | {scale.title()} |")
        lines.append("|----------|------|------:|")

        # Calculate totals and find fastest
        totals: dict[str, float] = {}
        for db in dbs:
            total = 0.0
            has_all = True
            for bench in benchmarks:
                result = self._get_result(collector, bench, db)
                if result and result.ok and result.metrics:
                    total += result.metrics.timing.mean_ms
                else:
                    has_all = False
            if has_all or total > 0:
                totals[db] = total

        fastest_db = min(totals, key=lambda x: totals[x]) if totals else None

        for db in dbs:
            db_type = DB_TYPES.get(db, "Unknown")
            total: float | None = totals.get(db)
            if total is not None:
                if db == fastest_db:
                    lines.append(f"| **{db}** | {db_type} | **{total:,.1f}** |")
                else:
                    lines.append(f"| {db} | {db_type} | {total:,.0f} |")
            else:
                lines.append(f"| {db} | {db_type} | |")

        lines.append("")

    def _add_analytics_summary(
        self, collector: ResultCollector, lines: list[str], dbs: list[str]
    ) -> None:
        """Add Graph Analytics summary (native only)."""
        lines.append("### Graph Analytics (native implementations only, total ms)")
        lines.append("")

        scale = collector.session.scale
        lines.append(f"| Database | Type | {scale.title()} |")
        lines.append("|----------|------|------:|")

        # Only show databases with native support
        native_dbs = [db for db in dbs if db in NATIVE_ANALYTICS]

        for db in native_dbs:
            db_type = DB_TYPES.get(db, "Unknown")
            total = 0.0
            for bench in CATEGORY_BENCHMARKS["Graph Analytics"]:
                result = self._get_result(collector, bench, db)
                if result and result.ok and result.metrics:
                    total += result.metrics.timing.mean_ms
            if total > 0:
                lines.append(f"| **{db}** | {db_type} | {total:,.1f} |")
            else:
                lines.append(f"| {db} | {db_type} | |")

        lines.append("")
        lines.append(
            "Only databases with native in-database algorithm implementations are included. "
            "Databases that fall back to extracting the graph into Python/NetworkX measure "
            "extraction overhead, not database performance. That extraction typically adds "
            "100-1,000x overhead."
        )
        lines.append("")

    def _add_combinatorial_summary(
        self, collector: ResultCollector, lines: list[str], dbs: list[str]
    ) -> None:
        """Add Combinatorial Workload summary."""
        lines.append("### Combinatorial Workload (total ms)")
        lines.append("")
        lines.append("| Database | Type | Writes | Reads | Traversals | ACID |")
        lines.append("|----------|------|-------:|------:|-----------:|-----:|")

        # Calculate category totals
        categories = {
            "Writes": CATEGORY_BENCHMARKS["Combinatorial - Writes"],
            "Reads": CATEGORY_BENCHMARKS["Combinatorial - Reads"],
            "Traversals": CATEGORY_BENCHMARKS["Combinatorial - Traversals"],
            "ACID": CATEGORY_BENCHMARKS["Concurrent ACID"],
        }

        # Find fastest per category
        fastest: dict[str, str] = {}
        for cat_name, benchmarks in categories.items():
            cat_totals: dict[str, float] = {}
            for db in dbs:
                total = 0.0
                has_failed = False
                for bench in benchmarks:
                    result = self._get_result(collector, bench, db)
                    if result and result.ok and result.metrics:
                        total += result.metrics.timing.mean_ms
                    elif result and not result.ok:
                        has_failed = True
                if not has_failed and total > 0:
                    cat_totals[db] = total
            if cat_totals:
                fastest[cat_name] = min(cat_totals, key=lambda x: cat_totals[x])

        for db in dbs:
            db_type = DB_TYPES.get(db, "Unknown")
            row = f"| {'**' + db + '**' if db == fastest.get('Writes') else db} | {db_type} |"

            for cat_name, benchmarks in categories.items():
                total = 0.0
                has_failed = False
                for bench in benchmarks:
                    result = self._get_result(collector, bench, db)
                    if result and result.ok and result.metrics:
                        total += result.metrics.timing.mean_ms
                    elif result and not result.ok:
                        has_failed = True

                if has_failed:
                    row += " FAILED |"
                elif total > 0:
                    is_fastest = fastest.get(cat_name) == db
                    if is_fastest:
                        row += f" **{total:,.1f}** |"
                    else:
                        row += f" {total:,.0f} |"
                else:
                    row += " |"

            lines.append(row)

        lines.append("")
        # Add footnote for NebulaGraph if present
        if "NebulaGraph" in dbs:
            lines.append(
                "\\* NebulaGraph write speed reflects async writes with eventual consistency "
                "(`replica_factor=1`), not durable commits."
            )
            lines.append("")

    def _add_reading_results(self, lines: list[str]) -> None:
        """Add 'Reading the results' section."""
        lines.append("### Reading the results")
        lines.append("")
        lines.append(
            "These benchmarks compare databases with fundamentally different architectures. "
            "Before drawing conclusions, consider:"
        )
        lines.append("")
        lines.append(
            "- **Embedded vs. server.** Grafeo and LadybugDB run in-process - "
            "no network serialization, no protocol overhead. Server databases pay ~0.1-1ms "
            "per round-trip."
        )
        lines.append(
            "- **Consistency model.** NebulaGraph uses eventual consistency by default. "
            "Its write speeds are not comparable to ACID-compliant databases without qualification."
        )
        lines.append(
            "- **Memory model.** Memgraph is in-memory first with optional WAL persistence. "
            "FalkorDB inherits Redis persistence semantics."
        )
        lines.append(
            "- **Scale factor.** Small (10K nodes) fits in L2 cache. "
            "Medium and large benchmarks reveal architectural differences."
        )
        lines.append("")

    def _add_database_details(
        self, collector: ResultCollector, lines: list[str], db: str
    ) -> None:
        """Add detailed results for a single database."""
        lines.append("<details>")
        lines.append(f"<summary><h3>{db}</h3></summary>")
        lines.append("")

        # Database info table
        info = DB_INFO.get(db, {})
        if info:
            lines.append("| | |")
            lines.append("|---|---|")
            lines.append(f"| **Type** | {info.get('type', 'Unknown')} |")
            lines.append(f"| **Data model** | {info.get('data_model', 'LPG')} |")
            lines.append(f"| **Query languages** | {info.get('query_languages', 'Unknown')} |")
            lines.append(f"| **ACID** | {info.get('acid', 'Unknown')} |")
            lines.append(f"| **Consistency** | {info.get('consistency', 'Unknown')} |")
            lines.append(f"| **License** | {info.get('license', 'Unknown')} |")
            lines.append("")

        # Add category tables
        scale = collector.session.scale
        for cat_name, benchmarks in CATEGORY_BENCHMARKS.items():
            results = [
                (bench, self._get_result(collector, bench, db)) for bench in benchmarks
            ]
            # Skip if no results in this category
            if not any(r for _, r in results):
                continue

            lines.append(f"#### {cat_name} - {scale.title()}")
            lines.append("")
            lines.append("| Benchmark | Time |")
            lines.append("|-----------|-----:|")

            total = 0.0
            for bench, result in results:
                if result and result.ok and result.metrics:
                    ms = result.metrics.timing.mean_ms
                    total += ms
                    lines.append(f"| {bench} | {ms:.2f}ms |")
                elif result and not result.ok:
                    lines.append(f"| {bench} | FAILED |")

            if total > 0:
                lines.append(f"| **Total** | **{total:.2f}ms** |")
            lines.append("")

        lines.append("</details>")
        lines.append("")

    def _add_query_languages(self, lines: list[str], dbs: list[str]) -> None:
        """Add Query Languages & Data Models section."""
        lines.append("---")
        lines.append("")
        lines.append("## Query Languages & Data Models")
        lines.append("")

        header = "| |" + " | ".join(dbs) + " |"
        sep = "|---|" + "|".join(":---:" for _ in dbs) + "|"
        lines.append(header)
        lines.append(sep)

        features = [
            ("**LPG**", lambda db: True),
            ("**RDF**", lambda db: db == "Grafeo"),
            ("**GQL (ISO)**", lambda db: db == "Grafeo"),
            ("**Cypher**", lambda db: db in ["Grafeo", "LadybugDB", "Neo4j", "Memgraph", "FalkorDB", "TuGraph"]),
            ("**Gremlin**", lambda db: db in ["Grafeo", "ArangoDB"]),
            ("**GraphQL**", lambda db: db in ["Grafeo", "ArangoDB"]),
            ("**SPARQL**", lambda db: db == "Grafeo"),
            ("**AQL**", lambda db: db == "ArangoDB"),
            ("**nGQL**", lambda db: db == "NebulaGraph"),
        ]

        for feature_name, check_fn in features:
            row = f"| {feature_name} |"
            for db in dbs:
                if check_fn(db):
                    row += " ✅ |"
                else:
                    row += " |"
            lines.append(row)

        lines.append("")

    def _add_native_support(self, lines: list[str], dbs: list[str]) -> None:
        """Add Native Algorithm Support section."""
        lines.append("---")
        lines.append("")
        lines.append("## Native Algorithm Support")
        lines.append("")

        native_dbs = [db for db in dbs if db in NATIVE_ANALYTICS]
        if not native_dbs:
            lines.append("No databases with native algorithm support in this run.")
            lines.append("")
            return

        header = "| |" + " | ".join(native_dbs) + " |"
        sep = "|---|" + "|".join(":---:" for _ in native_dbs) + "|"
        lines.append(header)
        lines.append(sep)

        algorithms = [
            ("BFS", "ldbc_bfs"),
            ("PageRank", "ldbc_pagerank"),
            ("WCC", "ldbc_wcc"),
            ("CDLP", "ldbc_cdlp"),
            ("LCC", "ldbc_lcc"),
            ("SSSP", "ldbc_sssp"),
        ]

        for algo_name, bench_name in algorithms:
            row = f"| {algo_name} |"
            for db in native_dbs:
                if bench_name in NATIVE_ANALYTICS.get(db, []):
                    row += " ✅ |"
                else:
                    row += " |"
            lines.append(row)

        lines.append("")

    def _add_methodology(self, lines: list[str]) -> None:
        """Add Methodology section."""
        lines.append("---")
        lines.append("")
        lines.append("## Methodology")
        lines.append("")
        lines.append("- **Warmup:** 3 runs discarded before measurement")
        lines.append("- **Iterations:** 10 measured runs, median reported")
        lines.append("- **Isolation:** Each database gets a clean dataset load before benchmarking")
        lines.append("- **Timeout:** 600 seconds per benchmark; exceeded shown as `T/O`")
        lines.append("")

    def _add_raw_data(self, lines: list[str], session_id: str) -> None:
        """Add Raw Data section."""
        lines.append("---")
        lines.append("")
        lines.append("## Raw Data")
        lines.append("")
        lines.append(f"- Results: [`results/{session_id}.json`](results/{session_id}.json)")
        lines.append("")

    def _get_result(
        self, collector: ResultCollector, bench: str, db: str
    ) -> BenchmarkResult[Any] | None:
        """Get result for a specific benchmark and database."""
        for r in collector.results:
            if r.benchmark_name == bench and r.database == db:
                return r
        return None
