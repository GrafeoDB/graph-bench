"""The summary block for grafeo's README.

It names no database other than Grafeo (user decision, 2026-10-30): it holds counts only, and never a combined
score. It covers the single-threaded results at the largest scale present, plus the Startup row (whose scale is
the empty database). Grafeo is ranked only against the databases of its own source (`report.ranking`): with a
local snapshot merged in, the header counts its databases, marked, and a footer line names its date and machine.
"""

import statistics
from dataclasses import dataclass

from graph_bench.report.ranking import LOCAL_MARK, Ranked, is_load, local_databases, rank, scale_key
from graph_bench.results import Entry, GrafeoSource, Machine, Results
from graph_bench.runner import CASE_CEILING_S, LOAD_CEILING_S, RUN_DEADLINE_S, deadline_over, load_over
from graph_bench.suites.catalog import STARTUP_SCALE
from graph_bench.types import CATEGORY_NAMES, CATEGORY_ORDER, marker

RESULTS_URL = "https://github.com/GrafeoDB/graph-bench#results"
# The generated overview page (`graph-bench report --format overview`): every version, source and run.
OVERVIEW_URL = "https://github.com/GrafeoDB/graph-bench/blob/main/docs/results.md"
# ⏱ covers the final budget rule and the run-time limits (FAIRNESS.md rules 8 and 9), and the cases CI's run
# deadline left unstarted.
LEGEND = (
    "⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average, "
    f"over {CASE_CEILING_S:g} s, {load_over(LOAD_CEILING_S)} or not started by the {deadline_over(RUN_DEADLINE_S)}"
)
GRAFEO = "grafeo"


def grafeo_text(grafeo: GrafeoSource) -> str:
    """`Grafeo <version> (<sha[:7]>)`, with `-dirty` kept and the install source in place of a missing sha.

    The grafeo commit (`git_sha`) is preferred over a wheel's file SHA-256, which names no commit.
    """
    version = grafeo.get("version") or "unknown"
    sha = grafeo.get("git_sha") or grafeo.get("sha")
    if not sha:
        return f"Grafeo {version} ({grafeo.get('source', 'unknown')})"
    dirty = "-dirty" if sha.endswith("-dirty") else ""
    return f"Grafeo {version} ({sha[:7]}{dirty})"


def grafeo_label(results: Results) -> str:
    """`grafeo_text` of the CI run's Grafeo."""
    return grafeo_text(results.grafeo)


def machine_text(m: Machine) -> str:
    return f"{m['cpu']}, {m['cores']} cores, {m['os']}"


def machine_label(results: Results) -> str:
    return machine_text(results.machine)


def local_run_label(results: Results) -> str:
    """`<date> on <machine>` of the local snapshot's run."""
    run = results.run("local")
    return f"{run['started_at'][:10]} on {machine_text(run['machine'])}"


def _agreed(entry: Entry) -> bool:
    """Whether an entry counts towards n in "Grafeo fastest k / n": its answer agreed and it is ok or ⏱.

    An entry that failed (✗) after its answer agreed does not count. A load (Load or Load (query language)) counts
    when it completed and its node and edge count check did not disagree (`unchecked` when the database has no
    `counts()`).
    """
    if is_load(entry):
        return entry.status == "ok" and entry.answer != "disagree"
    return entry.answer == "agree" and entry.status in ("ok", "timeout")


def _median_rank(positions: list[int]) -> str:
    if not positions:
        return "n/a"
    value = statistics.median(positions)
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


@dataclass(frozen=True)
class SummaryRow:
    """One category of the summary: its name, the databases that ran it, and Grafeo's counts."""

    category: str
    databases: int
    fastest: int
    agreed: int
    median_rank: str


@dataclass(frozen=True)
class SummaryData:
    """What the summary block holds, before it is laid out: rows, the scale, and Grafeo's markers."""

    rows: list[SummaryRow]
    scale: str
    marked: list[tuple[str, str]]


def summary_data(results: Results) -> SummaryData:
    """Grafeo's per-category counts at the largest scale it ran at, single-threaded."""
    single = [e for e in results.entries if e.threads == 1]
    grafeo_entries = [e for e in single if e.database == GRAFEO]
    if not grafeo_entries:
        raise ValueError("the results hold no single-threaded Grafeo entries to summarise")
    # The largest scale Grafeo ran at, so a larger-scale run of another database cannot empty the table. The
    # Startup row has a scale of its own (the empty database) and is always in.
    scaled = [e.scale for e in grafeo_entries if e.category != "startup"]
    scale = max(scaled, key=scale_key) if scaled else STARTUP_SCALE
    grafeo_entries = [e for e in grafeo_entries if e.scale == scale or e.category == "startup"]
    ranked = [r for r in rank(results) if (r.scale == scale or r.category == "startup") and r.threads == 1]

    rows = []
    for category in CATEGORY_ORDER:
        if not any(e.category == category for e in grafeo_entries):
            continue
        in_row: list[Ranked] = [r for r in ranked if r.category == category]
        dbs = {db for r in in_row for db in r.peers(GRAFEO)}  # Grafeo's own source only
        positions = [p for r in in_row if (p := r.position(GRAFEO)) is not None]
        fastest = sum(1 for p in positions if p == 1)
        agreed = sum(1 for e in grafeo_entries if e.category == category and _agreed(e))
        rows.append(SummaryRow(CATEGORY_NAMES[category], len(dbs), fastest, agreed, _median_rank(positions)))
    return SummaryData(rows, scale, _marked(grafeo_entries, ranked))


def render_summary(results: Results, *, results_url: str = RESULTS_URL) -> str:
    """The summary block, between `<!-- graph-bench:start -->` and `<!-- graph-bench:end -->`."""
    data = summary_data(results)
    rows = [f"| {r.category} | {r.databases} | {r.fastest} / {r.agreed} | {r.median_rank} |" for r in data.rows]
    local = local_databases(results)
    count = f"{len(results.databases)} embedded databases"
    if local:
        count += f" ({len(local)}{LOCAL_MARK} from a local snapshot)"
    lines = [
        "<!-- graph-bench:start -->",
        f"**graph-bench** · {grafeo_label(results)} · {machine_label(results)} · {results.started_at[:10]} · {count}",
        "",
        "| Category | DBs | Grafeo fastest | Grafeo median rank |",
        "|---|---|---|---|",
        *rows,
    ]
    # Each footer line is its own paragraph, so GitHub shows them on separate lines.
    footer = [f"Single-threaded, {data.scale} scale; answers checked; wrong answers not ranked."]
    footer += [f"Grafeo: {m} on {benchmark}" for benchmark, m in data.marked]
    if data.marked:
        footer.append(LEGEND)
    if local:
        run = results.run("local")
        footer.append(
            f"{LOCAL_MARK} {len(local)} of the {len(results.databases)} databases come from a local snapshot"
            f" ({run['started_at'][:10]}, {machine_text(run['machine'])}): Grafeo is ranked only against the"
            " databases measured with it."
        )
    footer.append(f"Full results, with every database named: {results_url}")
    footer.append(f"Every database's version, source and run: {OVERVIEW_URL}")
    for line in footer:
        lines += ["", line]
    lines.append("<!-- graph-bench:end -->")
    return "\n".join(lines) + "\n"


def _marked(entries: list[Entry], ranked: list[Ranked]) -> list[tuple[str, str]]:
    """Grafeo's own markers as (benchmark, marker), in matrix order."""
    order = {r.benchmark: i for i, r in enumerate(ranked)}
    found = [(e.benchmark, m) for e in entries if (m := marker(e.status, e.answer))]
    return sorted(found, key=lambda pair: order.get(pair[0], len(order)))
