"""Command line entry point."""

import typer

from graph_bench import __version__

app = typer.Typer(no_args_is_help=True, help="Per-process, answer-checked comparison of embedded graph databases.")


@app.callback()
def _root() -> None:
    """Keep subcommands as subcommands."""


@app.command()
def version() -> None:
    """Print the graph-bench version."""
    typer.echo(__version__)


def main() -> None:
    app()
