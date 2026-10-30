"""Command line entry point."""

import shutil
import tempfile
from pathlib import Path

import typer

from graph_bench import __version__
from graph_bench.databases import KNOWN
from graph_bench.datasets import cache
from graph_bench.runner import supervisor
from graph_bench.runner.worker import DEFAULT_CASE_TIMEOUT_S
from graph_bench.suites import catalog

app = typer.Typer(no_args_is_help=True, help="Per-process, answer-checked comparison of embedded graph databases.")
data_app = typer.Typer(no_args_is_help=True, help="Prepare the datasets and reference answers.")
app.add_typer(data_app, name="data")

_SELECTION = typer.Option(..., "--selection", help="Selection name: push or full.")
_WORKDIR = typer.Option(None, "--workdir", help="Directory for database files; default: a temporary one.")
_OUT = typer.Option(..., "--out", help="Output directory for results.json and the raw worker output.")


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


@app.command()
def bench(
    selection: str = _SELECTION,
    db: str | None = typer.Option(
        None, "--db", help="Comma-separated database names, run in this order; default: all built-in."
    ),
    grafeo: str | None = typer.Option(None, "--grafeo", help="Grafeo source to install first (not available yet)."),
    case_timeout: float = typer.Option(
        DEFAULT_CASE_TIMEOUT_S,
        "--case-timeout",
        min=0.1,
        help=(
            "Seconds of worker silence before it is killed and the case marked timeout;"
            " the worker reports progress after every run."
        ),
    ),
    workdir: Path | None = _WORKDIR,
    out: Path = _OUT,
) -> None:
    """Run the databases one after another, one worker process each; write and print DIR/results.json."""
    if grafeo is not None:
        typer.echo("--grafeo is not available yet", err=True)
        raise typer.Exit(2)
    name = _check_selection(selection)
    dbs = _dbs(db)
    try:
        catalog.cases(name, dbs)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--db") from None
    own_workdir = workdir is None
    work = Path(tempfile.mkdtemp(prefix="graph-bench-")) if workdir is None else workdir
    try:
        supervisor.run_bench(name, dbs, out, workdir=work, case_timeout_s=case_timeout)
    finally:
        if own_workdir:
            shutil.rmtree(work, ignore_errors=True)
    typer.echo(out / supervisor.RESULTS_FILE)


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
