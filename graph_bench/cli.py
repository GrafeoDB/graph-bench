"""Command line entry point."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import typer

from graph_bench import __version__
from graph_bench.databases import KNOWN
from graph_bench.datasets import cache
from graph_bench.grafeo_setup import install_grafeo
from graph_bench.report.readme import render_readme, update_readme
from graph_bench.report.summary import render_summary
from graph_bench.results import Results
from graph_bench.runner import supervisor
from graph_bench.runner.worker import DEFAULT_CASE_TIMEOUT_S
from graph_bench.suites import catalog

app = typer.Typer(no_args_is_help=True, help="Per-process, answer-checked comparison of embedded graph databases.")
data_app = typer.Typer(no_args_is_help=True, help="Prepare the datasets and reference answers.")
app.add_typer(data_app, name="data")

_SELECTION = typer.Option(..., "--selection", help="Selection name: push or full.")
_WORKDIR = typer.Option(None, "--workdir", help="Directory for database files; default: a temporary one.")
_GRAFEO_SOURCE = typer.Option(..., "--grafeo", help="A grafeo .whl file or a grafeo checkout (built with maturin).")
_OUT = typer.Option(..., "--out", help="Output directory for results.json and the raw worker output.")
_REPORT_IN = typer.Option(..., "--in", help="The results.json to report on.")
_REPORT_WRITE = typer.Option(
    None, "--write", help="README to update between its results markers (readme only); default: print."
)


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


def _install_requested(grafeo: str | None) -> dict[str, str] | None:
    """Install the Grafeo from `--grafeo`, else from GRAPH_BENCH_GRAFEO; None when neither is given."""
    source = grafeo or os.environ.get("GRAPH_BENCH_GRAFEO")
    if not source:
        return None
    try:
        return install_grafeo(Path(source))
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None


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
def setup(
    grafeo: Path = _GRAFEO_SOURCE,
) -> None:
    """Install the Grafeo under test and print what was installed as JSON.

    Building a checkout needs a lot of disk: set CARGO_TARGET_DIR to a drive with ~90 GB free.
    """
    info = _install_requested(str(grafeo))
    typer.echo(json.dumps(info))


@app.command()
def bench(
    selection: str = _SELECTION,
    db: str | None = typer.Option(
        None, "--db", help="Comma-separated database names, run in this order; default: all built-in."
    ),
    grafeo: str | None = typer.Option(
        None, "--grafeo", help="Grafeo wheel or checkout to install first; default: GRAPH_BENCH_GRAFEO."
    ),
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
    name = _check_selection(selection)
    dbs = _dbs(db)
    try:
        catalog.cases(name, dbs)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--db") from None
    _install_requested(grafeo)
    own_workdir = workdir is None
    work = Path(tempfile.mkdtemp(prefix="graph-bench-")) if workdir is None else workdir
    try:
        supervisor.run_bench(name, dbs, out, workdir=work, case_timeout_s=case_timeout)
    finally:
        if own_workdir:
            shutil.rmtree(work, ignore_errors=True)
    typer.echo(out / supervisor.RESULTS_FILE)


def _echo_utf8(text: str) -> None:
    """Write `text` to stdout as UTF-8, whatever the console code page (the report uses symbols such as ⊘)."""
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
        return
    sys.stdout.flush()
    buffer.write(text.encode("utf-8"))
    buffer.flush()


@app.command()
def report(
    fmt: str = typer.Option(..., "--format", help="summary (grafeo's README block) or readme (graph-bench's matrix)."),
    in_path: Path = _REPORT_IN,
    write: Path | None = _REPORT_WRITE,
) -> None:
    """Print the summary block or the README matrix of a results.json, or write the matrix into a README."""
    if fmt not in ("summary", "readme"):
        raise typer.BadParameter(f"{fmt!r} is not one of: summary, readme", param_hint="--format")
    if write is not None and fmt != "readme":
        raise typer.BadParameter("--write is only valid with --format readme", param_hint="--write")
    try:
        results = Results.read(in_path)
        text = render_summary(results) if fmt == "summary" else render_readme(results)
        if write is not None:
            update_readme(write, text)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"report: {exc}", err=True)
        raise typer.Exit(2) from None
    if write is None:
        _echo_utf8(text)
    else:
        typer.echo(write)


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
