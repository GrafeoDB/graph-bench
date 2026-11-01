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
from graph_bench import local_check as local_check_module
from graph_bench import merge as merge_module
from graph_bench import notify as notify_module
from graph_bench.databases import KNOWN
from graph_bench.datasets import cache
from graph_bench.grafeo_setup import install_grafeo
from graph_bench.report.discord import render_discord
from graph_bench.report.overview import render_overview
from graph_bench.report.readme import render_readme, update_readme
from graph_bench.report.summary import RESULTS_URL, render_summary
from graph_bench.results import Results
from graph_bench.runner import RUN_DEADLINE_S, supervisor
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
    None,
    "--write",
    help=(
        "readme: the README to update between its results markers; overview: the page to write (docs/results.md)."
        " Default: print."
    ),
)


_RUN_URL = typer.Option(None, "--run-url", help="Link to the CI run (discord only).")
_README_URL = typer.Option(RESULTS_URL, "--readme-url", help="Link to the README results section (discord only).")


_NOTIFY_PROGRESS = typer.Option(
    None,
    "--progress",
    help="File holding how many messages went out: read to resume a failed post, updated after each message.",
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
        ids = catalog.all_ids(name, dbs)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--db") from None
    for bench_id in ids:
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


_DEADLINE = typer.Option(
    None,
    "--deadline-s",
    min=0,
    help=(
        "Seconds after the start of the run past which nothing new starts; every case not yet run is marked"
        " timeout 'run deadline' and results.json is written. Default: none."
    ),
)


_DEADLINE_TOTAL = typer.Option(
    None,
    "--deadline-total-s",
    min=0,
    help=(
        "The whole run deadline the 'run deadline' message names, when --deadline-s is only what is left of it"
        " (CI: the 48 minutes, counted from the job's first step). Default: --deadline-s."
    ),
)


_YARDSTICK = typer.Option(
    None,
    "--yardstick",
    help=(
        "A results.json of earlier databases on the same hardware (CI's earlier per-database jobs), of the same"
        " selection and mode; its ok/agree entries set the in-run budgets as earlier databases of this run would."
        " Repeat for several files."
    ),
)


def _yardsticks(paths: list[Path] | None, selection: str, mode: str) -> list[Results]:
    """Read and check the `--yardstick` files; a missing, unreadable or mismatched file exits 2 naming it."""
    docs = []
    for path in paths or []:
        try:
            doc = Results.read(path)
            supervisor.check_yardstick(doc, selection, mode)
        except FileNotFoundError:
            typer.echo(f"--yardstick: {path} does not exist", err=True)
            raise typer.Exit(2) from None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            typer.echo(f"--yardstick: {path}: {exc}", err=True)
            raise typer.Exit(2) from None
        docs.append(doc)
    return docs


def _run(
    run: Callable[..., Results],
    mode: str,
    selection: str,
    db: str | None,
    grafeo: str | None,
    case_timeout: float,
    workdir: Path | None,
    out: Path,
    deadline_s: float | None = None,
    yardstick: list[Path] | None = None,
    deadline_total_s: float | None = None,
) -> None:
    """Validate the selection, databases and yardsticks, then install Grafeo if asked, then run and print
    DIR/results.json."""
    if deadline_total_s is not None and deadline_s is None:
        raise typer.BadParameter("needs --deadline-s", param_hint="--deadline-total-s")
    if deadline_total_s is not None and deadline_s is not None and deadline_total_s < deadline_s:
        # The message would name a deadline earlier than the real cutoff.
        raise typer.BadParameter("must be at least --deadline-s", param_hint="--deadline-total-s")
    name = _check_selection(selection)
    dbs = _dbs(db)
    try:
        catalog.cases(name, dbs)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--db") from None
    docs = _yardsticks(yardstick, name, mode)
    _install_requested(grafeo)
    own_workdir = workdir is None
    work = Path(tempfile.mkdtemp(prefix="graph-bench-")) if workdir is None else workdir
    try:
        run(
            name,
            dbs,
            out,
            workdir=work,
            case_timeout_s=case_timeout,
            deadline_s=deadline_s,
            yardstick=docs,
            deadline_total_s=deadline_total_s,
        )
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
    deadline_s: float | None = _DEADLINE,
    yardstick: list[Path] | None = _YARDSTICK,
    deadline_total_s: float | None = _DEADLINE_TOTAL,
) -> None:
    """Run the databases one after another, one worker process each; write and print DIR/results.json."""
    _run(
        supervisor.run_bench,
        "bench",
        selection,
        db,
        grafeo,
        case_timeout,
        workdir,
        out,
        deadline_s,
        yardstick,
        deadline_total_s,
    )


@app.command()
def codspeed(
    selection: str = _SELECTION,
    db: str | None = _RUN_DB,
    grafeo: str | None = _RUN_GRAFEO,
    case_timeout: float = _CASE_TIMEOUT,
    workdir: Path | None = _WORKDIR,
    out: Path = _OUT,
    deadline_s: float | None = _DEADLINE,
    yardstick: list[Path] | None = _YARDSTICK,
    deadline_total_s: float | None = _DEADLINE_TOTAL,
) -> None:
    """CI entry point: one `pytest benches --codspeed` process per database; write and print DIR/results.json.

    Keeps CODSPEED_PROFILE_FOLDER as it is. results.json gets answers and preflight times, no timings.

    Exits 0 once results.json is written, whatever pytest's exit codes.
    """
    _run(
        supervisor.run_codspeed,
        "codspeed",
        selection,
        db,
        grafeo,
        case_timeout,
        workdir,
        out,
        deadline_s,
        yardstick,
        deadline_total_s,
    )


_MERGE_IN = typer.Argument(..., help="The results.json files to merge, one per CI job.")
_MERGE_DB = typer.Option(
    None,
    "--db",
    help=(
        "Comma-separated databases the merged results cover; one without an input is listed as 'no results' and"
        " recorded failed. Default: the inputs' databases."
    ),
)


_MERGE_LOCAL = typer.Option(
    None,
    "--local",
    help=(
        "The local snapshot (results/local/results.json): its databases outside the CI roster and not in the inputs"
        " or --db are added, tagged local, and never ranked or budgeted against the CI ones."
    ),
)


@app.command()
def merge(
    out: Path = _OUT, inputs: list[Path] = _MERGE_IN, db: str | None = _MERGE_DB, local: Path | None = _MERGE_LOCAL
) -> None:
    """Merge the results.json files of runs over different databases into DIR/results.json and print its path.

    The inputs must agree on selection, mode, graph-bench commit and Grafeo source, and hold different databases.
    The final 10x budget rule is applied once, over the merged CI entries; the local ones keep their own verdicts.
    """
    try:
        docs = [(path, Results.read(path)) for path in inputs]
        snapshot = (local, Results.read(local)) if local is not None else None
        merged, missing = merge_module.merge(docs, _dbs(db) if db is not None else (), local=snapshot)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.echo(f"merge: {exc}", err=True)
        raise typer.Exit(2) from None
    for name in missing:
        typer.echo(f"no results: {name}")
    for name in merge_module.local_databases(merged):
        typer.echo(f"local: {name}")
    path = out / supervisor.RESULTS_FILE
    try:
        merged.write(path)
    except OSError as exc:
        typer.echo(f"merge: cannot write {path}: {exc}", err=True)
        raise typer.Exit(2) from None
    typer.echo(path)


_LOCAL_SNAPSHOT = typer.Option(..., "--local", help="The local snapshot, results/local/results.json.")


@app.command("local-check")
def local_check(local: Path = _LOCAL_SNAPSHOT) -> None:
    """Print, one per line, the snapshot's databases outside the CI roster whose newest stable release differs
    from the version the snapshot measured (Neo4j: its pinned release); exit 0.

    stdout holds only those lines (CI turns them into a notice). Databases whose release could not be looked up
    (unknown) and a summary go to stderr. An unreadable snapshot exits 2.
    """
    try:
        snapshot = Results.read(local)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.echo(f"local-check: {exc}", err=True)
        raise typer.Exit(2) from None
    checks = local_check_module.checks(snapshot)
    changes = [check for check in checks if check.changed]
    for change in changes:
        typer.echo(local_check_module.describe(change))
    for check in checks:
        if check.latest is None:
            typer.echo(f"local-check: {local_check_module.describe(check)}", err=True)
    date = snapshot.started_at[:10]
    checked = sum(check.latest is not None for check in checks)
    unknown = len(checks) - checked
    if changes:
        summary = f"{len(changes)} of {checked} checked databases outside the CI roster differ from the local snapshot"
        summary += f" of {date}" + (f", {unknown} unknown" if unknown else "")
        typer.echo(f"local-check: {summary}; refresh it with .github/scripts/local-bench.sh", err=True)
    elif checked == 0 and unknown:
        summary = f"none of the {unknown} databases outside the CI roster could be checked ({unknown} unknown)"
        typer.echo(f"local-check: {summary}", err=True)
    elif unknown:
        summary = f"{checked} of {len(checks)} databases outside the CI roster checked, all match the local snapshot"
        typer.echo(f"local-check: {summary} of {date}; {unknown} unknown", err=True)
    else:
        typer.echo(f"local-check: every database outside the CI roster matches the local snapshot of {date}", err=True)


@app.command("ci-settings")
def ci_settings() -> None:
    """Print the CI run settings as NAME=VALUE lines (for $GITHUB_OUTPUT): `deadline_s`, the `--deadline-s` of
    every macro-runner job, and `databases`, the roster `catalog.CI_DATABASES` in run order."""
    typer.echo(f"deadline_s={RUN_DEADLINE_S}")
    typer.echo("databases=" + ",".join(catalog.CI_DATABASES))


_YARDSTICK_DB = typer.Option(..., "--db", help="The leg's database, one of the CI roster.")
_YARDSTICK_DIR = typer.Argument(..., help="Where the leg downloaded the run's results: DIR/<artifact>/results.json.")


@app.command("ci-yardsticks")
def ci_yardsticks(db: str = _YARDSTICK_DB, directory: Path = _YARDSTICK_DIR) -> None:
    """Print a CI leg's `--yardstick FILE` arguments on one line: each DIR/*/results.json whose databases all come
    before DB in the roster (`catalog.CI_DATABASES`), so a re-run leg, or one started out of matrix order, never
    measures against a later leg. Nothing when there is none; an unreadable file is skipped with a warning on
    stderr; a DB outside the roster exits 2."""
    try:
        taken, unreadable = merge_module.earlier_legs(db, sorted(directory.glob("*/results.json")))
    except ValueError as exc:
        typer.echo(f"ci-yardsticks: {exc}", err=True)
        raise typer.Exit(2) from None
    for path, error in unreadable:
        typer.echo(f"ci-yardsticks: skipping {path}: {error}", err=True)
    typer.echo(" ".join(f"--yardstick {path}" for path in taken))


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
    fmt: str = typer.Option(
        ...,
        "--format",
        help=(
            "summary (grafeo's README block), readme (graph-bench's matrix), overview (the full page, every"
            " version) or discord (JSON)."
        ),
    ),
    in_path: Path = _REPORT_IN,
    write: Path | None = _REPORT_WRITE,
    run_url: str | None = _RUN_URL,
    readme_url: str = _README_URL,
) -> None:
    """Print the summary block, the README matrix, the overview page or the Discord payloads of a results.json;
    or write the matrix into a README, or the overview page to a file."""
    if fmt not in ("summary", "readme", "overview", "discord"):
        raise typer.BadParameter(f"{fmt!r} is not one of: summary, readme, overview, discord", param_hint="--format")
    if write is not None and fmt not in ("readme", "overview"):
        raise typer.BadParameter("--write is only valid with --format readme or overview", param_hint="--write")
    try:
        results = Results.read(in_path)
        if fmt == "discord":
            payloads = render_discord(results, run_url=run_url, readme_url=readme_url)
            text = json.dumps(payloads, ensure_ascii=False) + "\n"
        elif fmt == "overview":
            text = render_overview(results)
        else:
            text = render_summary(results) if fmt == "summary" else render_readme(results)
        if write is not None and fmt == "overview":
            write.parent.mkdir(parents=True, exist_ok=True)
            write.write_text(text, encoding="utf-8", newline="\n")
        elif write is not None:
            update_readme(write, text)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"report: {exc}", err=True)
        raise typer.Exit(2) from None
    if write is None:
        _echo_utf8(text)
    else:
        typer.echo(write)


@app.command()
def notify(
    in_path: Path = _REPORT_IN,
    webhook_env: str = typer.Option(..., "--webhook-env", help="Environment variable that holds the webhook URL."),
    run_url: str | None = _RUN_URL,
    readme_url: str = _README_URL,
    progress: Path | None = _NOTIFY_PROGRESS,
) -> None:
    """Post a results.json to a Discord webhook; skip, with a log line, when the variable is empty or unset."""
    url = os.environ.get(webhook_env, "").strip()
    if not url:
        typer.echo(f"notify: {webhook_env} is not set; skipping the Discord post")
        return
    try:
        payloads = render_discord(Results.read(in_path), run_url=run_url, readme_url=readme_url)
        start = int(progress.read_text(encoding="utf-8").strip() or 0) if progress and progress.exists() else 0

        def record(count: int) -> None:
            if progress is not None:
                progress.write_text(f"{count}\n", encoding="utf-8")

        sent = notify_module.post_all(url, payloads, start=start, on_sent=record)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"notify: {str(exc).replace(url, '<webhook>')}", err=True)
        raise typer.Exit(2) from None
    except notify_module.NotifyError as exc:
        done = progress.read_text(encoding="utf-8").strip() if progress and progress.exists() else "0"
        typer.echo(f"notify: {exc} ({done} of {len(payloads)} messages had gone out)", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"notify: posted {sent} messages to {webhook_env} ({start} were already out)")


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
