r"""
Command-line interface for graph-bench.

    graph-bench run -d neo4j,kuzu -s medium
    graph-bench report results/bench.json -f markdown
"""

from graph_bench.cli.main import app, main

__all__ = [
    "app",
    "main",
]
