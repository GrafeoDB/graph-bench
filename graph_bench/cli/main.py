r"""
Command-line interface for graph-bench.

    graph-bench run -d neo4j,kuzu -s medium
    graph-bench report results/bench.json -f markdown
"""

from pathlib import Path
from typing import Annotated

try:
    import typer
except ImportError:
    typer = None  # type: ignore

__all__ = ["app", "main"]


def _check_typer() -> None:
    if typer is None:
        msg = "typer package not installed. Install with: pip install 'graph-bench[cli]'"
        raise ImportError(msg)


if typer is not None:
    app = typer.Typer(
        name="graph-bench",
        help="Comprehensive benchmark suite for graph databases.",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        databases: Annotated[
            str | None, typer.Option("-d", "--databases", help="Databases to benchmark (comma-separated)")
        ] = None,
        benchmarks: Annotated[
            str | None, typer.Option("-b", "--benchmarks", help="Benchmarks to run (comma-separated)")
        ] = None,
        category: Annotated[
            str | None, typer.Option("-c", "--category", help="Benchmark category filter")
        ] = None,
        scale: Annotated[str, typer.Option("-s", "--scale", help="Scale: small, medium, large")] = "medium",
        output: Annotated[Path, typer.Option("-o", "--output", help="Output directory")] = Path("./results"),
        format_: Annotated[
            str, typer.Option("-f", "--format", help="Output format: json, csv, markdown, template, all")
        ] = "json",
        verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
        dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would run")] = False,
        sequential: Annotated[
            bool, typer.Option("--sequential", help="Run each database separately and aggregate results")
        ] = False,
        timeout: Annotated[
            int | None, typer.Option("--timeout", help="Timeout in seconds per benchmark")
        ] = None,
        all_benchmarks: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Run comprehensive suite: SNB (small/medium/large), Graph Analytics (medium), Concurrent (small)",
            ),
        ] = False,
    ) -> None:
        """Run benchmarks on graph databases."""
        from graph_bench.adapters import AdapterRegistry
        from graph_bench.benchmarks import BenchmarkRegistry
        from graph_bench.config import get_scale
        from graph_bench.reporting import CsvExporter, JsonExporter, MarkdownExporter, ResultCollector
        from graph_bench.reporting.template import TemplateExporter
        from graph_bench.runner import BenchmarkOrchestrator, OrchestratorConfig

        db_list = databases.split(",") if databases else AdapterRegistry.list()
        bench_list = benchmarks.split(",") if benchmarks else None
        categories = [category] if category else None

        if verbose:
            typer.echo(f"Databases: {', '.join(db_list)}")
            typer.echo(f"Scale: {scale}")
            typer.echo(f"Output: {output}")

        if dry_run:
            typer.echo("\n[DRY RUN] Would run:")
            if all_benchmarks:
                typer.echo("  Comprehensive benchmark suite (--all):")
                benchmark_plan = [
                    ("ldbc_snb", ["small", "medium", "large"]),
                    ("ldbc_graphanalytics", ["medium"]),
                    ("concurrent", ["small"]),
                ]
                for cat, scales in benchmark_plan:
                    cat_benches = BenchmarkRegistry.by_category(cat)
                    typer.echo(f"    {cat} @ {', '.join(scales)}: {', '.join(cat_benches)}")
                typer.echo(f"  Databases: {', '.join(db_list)}")
            else:
                for db in db_list:
                    bench_names = bench_list or BenchmarkRegistry.list()
                    if categories:
                        bench_names = [
                            b for b in bench_names
                            if BenchmarkRegistry.get(b)
                            and BenchmarkRegistry.get(b).category in categories  # type: ignore
                        ]
                    typer.echo(f"  {db}: {', '.join(bench_names)}")
            return

        adapters = []
        for db_name in db_list:
            try:
                adapter = AdapterRegistry.create(db_name)
                adapter.connect()
                adapters.append(adapter)
                if verbose:
                    typer.echo(f"Connected to {adapter.name}")
            except Exception as e:
                typer.echo(f"Warning: Could not connect to {db_name}: {e}", err=True)

        if not adapters:
            typer.echo("Error: No databases available", err=True)
            raise typer.Exit(1)

        scale_config = get_scale(scale)
        config = OrchestratorConfig(
            scale=scale_config,
            benchmarks=bench_list,
            categories=categories,
            verbose=verbose,
            timeout_override=timeout,
        )

        orchestrator = BenchmarkOrchestrator(config=config)

        # Define progress callback
        def progress(db: str, bench: str, status: str) -> None:
            if verbose:
                typer.echo(f"  [{db}] {bench}: {status}")

        if verbose:
            orchestrator.set_progress_callback(progress)

        collector = ResultCollector()
        collector.start_session(scale=scale, databases=[a.name for a in adapters])

        if all_benchmarks:
            # Comprehensive benchmark suite
            typer.echo("\n=== Running comprehensive benchmark suite ===")

            # Define the benchmark plan: (category, scales)
            benchmark_plan = [
                ("ldbc_snb", ["small", "medium", "large"]),
                ("ldbc_graphanalytics", ["medium"]),
                ("concurrent", ["small"]),
            ]

            for cat, scales in benchmark_plan:
                for s in scales:
                    typer.echo(f"\n--- {cat} @ {s} ---")
                    s_config = get_scale(s)
                    cat_config = OrchestratorConfig(
                        scale=s_config,
                        categories=[cat],
                        verbose=verbose,
                        timeout_override=timeout,
                    )
                    cat_orchestrator = BenchmarkOrchestrator(config=cat_config)
                    if verbose:
                        cat_orchestrator.set_progress_callback(progress)

                    if sequential and len(adapters) > 1:
                        for adapter in adapters:
                            single_result = cat_orchestrator.run([adapter], scale=s_config)
                            collector.add_results(single_result.results)
                    else:
                        cat_result = cat_orchestrator.run(adapters, scale=s_config)
                        collector.add_results(cat_result.results)
        else:
            typer.echo(f"\nRunning benchmarks at scale '{scale}'...")

            if sequential and len(adapters) > 1:
                # Run each database separately for better isolation
                for adapter in adapters:
                    typer.echo(f"\n--- Running {adapter.name} ---")
                    single_result = orchestrator.run([adapter], scale=scale_config)
                    collector.add_results(single_result.results)
                    typer.echo(f"Completed {adapter.name}: {single_result.success_count} passed")
            else:
                result = orchestrator.run(adapters, scale=scale_config)
                collector.add_results(result.results)

        collector.end_session()

        output.mkdir(parents=True, exist_ok=True)
        session_id = collector.session.session_id

        formats_to_export = format_.split(",") if "," in format_ else [format_]
        if "all" in formats_to_export:
            formats_to_export = ["json", "csv", "markdown", "template"]

        for fmt in formats_to_export:
            fmt = fmt.strip()
            if fmt == "json":
                path = output / f"{session_id}.json"
                JsonExporter().export(collector, path)
                typer.echo(f"Exported JSON: {path}")
            elif fmt == "csv":
                path = output / f"{session_id}.csv"
                CsvExporter().export(collector, path)
                typer.echo(f"Exported CSV: {path}")
            elif fmt == "markdown":
                path = output / f"{session_id}.md"
                MarkdownExporter().export(collector, path)
                typer.echo(f"Exported Markdown: {path}")
            elif fmt == "template":
                path = output / f"{session_id}_RESULTS.md"
                TemplateExporter().export(collector, path)
                typer.echo(f"Exported Template: {path}")

        for adapter in adapters:
            adapter.disconnect()

        success = sum(1 for r in collector.results if r.ok)
        failed = sum(1 for r in collector.results if not r.ok)
        typer.echo(f"\nCompleted: {success} successful, {failed} failed")

    @app.command()
    def report(
        results_path: Annotated[Path, typer.Argument(help="Path to results JSON file or directory")],
        format_: Annotated[str, typer.Option("-f", "--format", help="Output format: json, csv, markdown")] = "markdown",
        output: Annotated[Path | None, typer.Option("-o", "--output", help="Output file path")] = None,
        compare: Annotated[bool, typer.Option("--compare", help="Generate comparison table")] = False,
    ) -> None:
        """Generate reports from benchmark results."""
        import json

        from graph_bench.reporting import CsvExporter, JsonExporter, MarkdownExporter, ResultCollector
        from graph_bench.types import BenchmarkResult, Metrics, ScaleConfig, Status, TimingStats

        if results_path.is_dir():
            json_files = list(results_path.glob("*.json"))
            if not json_files:
                typer.echo(f"No JSON files found in {results_path}", err=True)
                raise typer.Exit(1)
            results_path = max(json_files, key=lambda p: p.stat().st_mtime)

        if not results_path.exists():
            typer.echo(f"File not found: {results_path}", err=True)
            raise typer.Exit(1)

        with open(results_path) as f:
            data = json.load(f)

        collector = ResultCollector()

        session = data.get("session", {})
        collector._session.session_id = session.get("id", "")
        collector._session.started_at = session.get("started_at", "")
        collector._session.completed_at = session.get("completed_at", "")
        collector._session.scale = session.get("scale", "")
        collector._session.databases = session.get("databases", [])

        for r in data.get("results", []):
            metrics = None
            if "metrics" in r:
                m = r["metrics"]
                t = m.get("timing", {})
                metrics = Metrics(
                    timing=TimingStats(
                        min_ns=t.get("min_ns", 0),
                        max_ns=t.get("max_ns", 0),
                        mean_ns=t.get("mean_ns", 0.0),
                        median_ns=t.get("median_ns", 0.0),
                        std_ns=t.get("std_ns", 0.0),
                        p99_ns=t.get("p99_ns", 0.0),
                        iterations=t.get("iterations", 0),
                    ),
                    throughput=m.get("throughput", 0.0),
                    items_processed=m.get("items_processed", 0),
                )

            result = BenchmarkResult(
                benchmark_name=r.get("benchmark", ""),
                database=r.get("database", ""),
                scale=ScaleConfig(name=r.get("scale", ""), nodes=0, edges=0),
                metrics=metrics,
                status=Status[r.get("status", "FAILED")],
                error=r.get("error"),
            )
            collector.add_result(result)

        if output is None:
            ext = {"json": ".json", "csv": ".csv", "markdown": ".md"}.get(format_, ".md")
            output = results_path.with_suffix(ext)

        if format_ == "json":
            JsonExporter().export(collector, output)
        elif format_ == "csv":
            CsvExporter().export(collector, output)
        else:
            MarkdownExporter().export(collector, output)

        typer.echo(f"Generated report: {output}")

        if compare:
            comparisons = collector.compute_comparisons()
            if comparisons:
                typer.echo("\nPerformance Comparisons:")
                for bench, speedups in sorted(comparisons.items()):
                    fastest = max(speedups.items(), key=lambda x: x[1])[0]
                    typer.echo(f"  {bench}: fastest={fastest}")

    @app.command()
    def adapters(
        action: Annotated[str, typer.Argument(help="Action: list, test")] = "list",
        name: Annotated[str | None, typer.Option("-n", "--name", help="Adapter name")] = None,
        uri: Annotated[str | None, typer.Option("--uri", help="Connection URI")] = None,
    ) -> None:
        """List and test database adapters."""
        from graph_bench.adapters import AdapterRegistry

        if action == "list":
            typer.echo("Available adapters:")
            for adapter_name in AdapterRegistry.list():
                typer.echo(f"  - {adapter_name}")
        elif action == "test":
            if not name:
                typer.echo("Error: --name required for test", err=True)
                raise typer.Exit(1)

            try:
                adapter = AdapterRegistry.create(name)
                kwargs = {"uri": uri} if uri else {}
                adapter.connect(**kwargs)
                typer.echo(f"Successfully connected to {adapter.name} (version: {adapter.version})")
                adapter.disconnect()
            except Exception as e:
                typer.echo(f"Failed to connect to {name}: {e}", err=True)
                raise typer.Exit(1)
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(1)

    @app.command()
    def datasets(
        action: Annotated[str, typer.Argument(help="Action: generate, list")] = "list",
        name: Annotated[str, typer.Option("-n", "--name", help="Dataset name")] = "synthetic",
        scale: Annotated[str, typer.Option("-s", "--scale", help="Scale: small, medium, large")] = "medium",
        output: Annotated[Path, typer.Option("-o", "--output", help="Output directory")] = Path("./data"),
        seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    ) -> None:
        """Manage benchmark datasets."""
        import json

        from graph_bench.config import get_scale
        from graph_bench.datasets import SyntheticSocialNetwork

        if action == "list":
            typer.echo("Available datasets:")
            typer.echo("  - synthetic: Synthetic social network (Person, FOLLOWS)")
        elif action == "generate":
            scale_config = get_scale(scale)
            typer.echo(f"Generating {name} dataset at scale '{scale}'...")
            typer.echo(f"  Nodes: {scale_config.nodes:,}")
            typer.echo(f"  Edges: {scale_config.edges:,}")

            if name == "synthetic":
                dataset = SyntheticSocialNetwork(seed=seed)
            else:
                typer.echo(f"Unknown dataset: {name}", err=True)
                raise typer.Exit(1)

            nodes, edges = dataset.generate(scale_config)

            output.mkdir(parents=True, exist_ok=True)
            nodes_path = output / f"{name}_{scale}_nodes.json"
            edges_path = output / f"{name}_{scale}_edges.json"

            with open(nodes_path, "w") as f:
                json.dump(nodes, f)
            with open(edges_path, "w") as f:
                json.dump([(e[0], e[1], e[2], e[3]) for e in edges], f)

            typer.echo("\nGenerated:")
            typer.echo(f"  Nodes: {nodes_path}")
            typer.echo(f"  Edges: {edges_path}")
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(1)

    def main() -> None:
        """Main entry point."""
        _check_typer()
        app()

else:

    def app() -> None:
        _check_typer()

    def main() -> None:
        _check_typer()


if __name__ == "__main__":
    main()
