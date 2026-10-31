"""Command line entry point."""

import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Callable
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
_GRAFEO_SHA = typer.Option(
    None, "--grafeo-sha", help="The grafeo commit a wheel was built from, recorded in results.json as git_sha."
)
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


def _install_requested(grafeo: str | None, *, git_sha: str | None = None) -> dict[str, str | None] | None:
    """Install the Grafeo from `--grafeo`, else from GRAPH_BENCH_GRAFEO; None when neither is given."""
    source = grafeo or os.environ.get("GRAPH_BENCH_GRAFEO")
    if not source:
        return None
    try:
        return install_grafeo(Path(source), git_sha=git_sha)
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None


@app.command()
def version() -> None:
    """Print the graph-bench version."""
    typer.echo(__version__)


def database_requirements(pyproject: Path) -> list[str]:
    """The packages of the database extras, version bounds dropped, environment markers kept.

    The extras are the ones the `embedded` extra lists; `uv pip install --upgrade -r` on the result moves each
    database to its newest stable release (uv never picks a pre-release unless asked) without touching uv.lock.
    """
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    listed = re.search(r"\[([^\]]*)\]", extras["embedded"][0])
    names = [name.strip() for name in listed.group(1).split(",")] if listed else []
    requirements: list[str] = []
    for extra in names:
        for requirement in extras[extra]:
            spec, _, marker = requirement.partition(";")
            package = re.match(r"[A-Za-z0-9._-]+", spec.strip())
            if package is None:
                raise ValueError(f"cannot read the requirement {requirement!r}")
            requirements.append(package.group(0) + (f" ;{marker}" if marker else ""))
    return requirements


@app.command()
def packages() -> None:
    """Print the database packages of the `embedded` extra, one requirement per line, without version bounds."""
    root = Path(__file__).resolve().parent.parent
    typer.echo("\n".join(database_requirements(root / "pyproject.toml")))


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
    grafeo_sha: str | None = _GRAFEO_SHA,
) -> None:
    """Install the Grafeo under test and print what was installed as JSON.

    Building a checkout needs a lot of disk: set CARGO_TARGET_DIR to a drive with ~90 GB free.
    """
    info = _install_requested(str(grafeo), git_sha=grafeo_sha)
    typer.echo(json.dumps(info))


_RUN_DB = typer.Option(None, "--db", help="Comma-separated database names, run in this order; default: all built-in.")
_RUN_GRAFEO = typer.Option(
    None, "--grafeo", help="Grafeo wheel or checkout to install first; default: GRAPH_BENCH_GRAFEO."
)
_CASE_TIMEOUT = typer.Option(
    DEFAULT_CASE_TIMEOUT_S,
    "--case-timeout",
    min=0.1,
    help=(
        "Seconds of worker silence before it is killed and the case marked timeout;"
        " the worker reports progress after every run."
    ),
)


def _run(
    run: Callable[..., Results],
    selection: str,
    db: str | None,
    grafeo: str | None,
    case_timeout: float,
    workdir: Path | None,
    out: Path,
) -> None:
    """Validate the selection and databases, then install Grafeo if asked, then run and print DIR/results.json."""
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
        run(name, dbs, out, workdir=work, case_timeout_s=case_timeout)
    finally:
        if own_workdir:
            shutil.rmtree(work, ignore_errors=True)
    typer.echo(out / supervisor.RESULTS_FILE)


@app.command()
def bench(
    selection: str = _SELECTION,
    db: str | None = _RUN_DB,
    grafeo: str | None = _RUN_GRAFEO,
    case_timeout: float = _CASE_TIMEOUT,
    workdir: Path | None = _WORKDIR,
    out: Path = _OUT,
) -> None:
    """Run the databases one after another, one worker process each; write and print DIR/results.json."""
    _run(supervisor.run_bench, selection, db, grafeo, case_timeout, workdir, out)


@app.command()
def codspeed(
    selection: str = _SELECTION,
    db: str | None = _RUN_DB,
    grafeo: str | None = _RUN_GRAFEO,
    case_timeout: float = _CASE_TIMEOUT,
    workdir: Path | None = _WORKDIR,
    out: Path = _OUT,
) -> None:
    """CI entry point: one `pytest benches --codspeed` process per database; write and print DIR/results.json.

    Keeps CODSPEED_PROFILE_FOLDER as it is. results.json gets answers and preflight times, no timings.

    Exits 0 once results.json is written, whatever pytest's exit codes.
    """
    _run(supervisor.run_codspeed, selection, db, grafeo, case_timeout, workdir, out)


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
