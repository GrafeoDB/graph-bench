"""Command line entry point."""

import typer

from graph_bench import __version__
from graph_bench.databases import KNOWN
from graph_bench.datasets import cache
from graph_bench.suites import catalog

app = typer.Typer(no_args_is_help=True, help="Per-process, answer-checked comparison of embedded graph databases.")
data_app = typer.Typer(no_args_is_help=True, help="Prepare the datasets and reference answers.")
app.add_typer(data_app, name="data")

_SELECTION = typer.Option(..., "--selection", help="Selection name: push or full.")


@app.callback()
def _root() -> None:
    """Keep subcommands as subcommands."""


def _check_selection(name: str) -> str:
    try:
        catalog.selection(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--selection") from None
    return name


def _dbs(value: str | None) -> list[str]:
    if value is None:
        return list(KNOWN)
    return [name.strip() for name in value.split(",") if name.strip()]


@app.command()
def version() -> None:
    """Print the graph-bench version."""
    typer.echo(__version__)


@app.command("list")
def list_benchmarks(
    selection: str = _SELECTION,
    db: str | None = typer.Option(None, "--db", help="Comma-separated database names; default: all built-in."),
) -> None:
    """Print the benchmark ids of a selection, sorted, one per line."""
    name = _check_selection(selection)
    dbs = _dbs(db)
    try:
        ids = [c.bench_id for c in catalog.cases(name, dbs)] + catalog.load_ids(name, dbs)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--db") from None
    for bench_id in sorted(ids):
        typer.echo(bench_id)


@data_app.command("prepare")
def data_prepare(selection: str = _SELECTION) -> None:
    """Generate (or reuse) the datasets and reference answers of a selection; print their directories."""
    datasets = cache.prepare(_check_selection(selection))
    for key in sorted(datasets):
        typer.echo(datasets[key].root)


@data_app.command("key")
def data_key(selection: str = _SELECTION) -> None:
    """Print the dataset cache key of a selection."""
    typer.echo(cache.dataset_key(_check_selection(selection)))


def main() -> None:
    app()
