"""The summary block for grafeo's README.

It names no database other than Grafeo (user decision, 2026-10-30): it holds counts only, and never a combined
score. It covers the single-threaded results at the largest scale present.
"""

import statistics

from graph_bench.report.ranking import Ranked, is_load, rank, scale_key
from graph_bench.results import Entry, Results
from graph_bench.types import CATEGORY_NAMES, CATEGORY_ORDER, marker

RESULTS_URL = "https://github.com/GrafeoDB/graph-bench#results"
LEGEND = "⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average"
GRAFEO = "grafeo"


def grafeo_label(results: Results) -> str:
    """`Grafeo <version> (<sha[:7]>)`, with `-dirty` kept and the install source in place of a missing sha.

    The grafeo commit (`git_sha`) is preferred over a wheel's file SHA-256, which names no commit.
    """
    version = results.grafeo.get("version") or "unknown"
    sha = results.grafeo.get("git_sha") or results.grafeo.get("sha")
    if not sha:
        return f"Grafeo {version} ({results.grafeo.get('source', 'unknown')})"
    dirty = "-dirty" if sha.endswith("-dirty") else ""
    return f"Grafeo {version} ({sha[:7]}{dirty})"


def machine_label(results: Results) -> str:
    m = results.machine
    return f"{m['cpu']}, {m['cores']} cores, {m['os']}"


def _agreed(entry: Entry) -> bool:
    """Whether an entry counts towards n in "Grafeo fastest k / n": its answer agreed and it is ok or ⏱.

    An entry that failed (✗) after its answer agreed does not count. A load counts when it completed and its node
    and edge count check did not disagree (`unchecked` when the database has no `counts()`).
    """
    if is_load(entry):
        return entry.status == "ok" and entry.answer != "disagree"
    return entry.answer == "agree" and entry.status in ("ok", "timeout")


def _median_rank(positions: list[int]) -> str:
    if not positions:
        return "n/a"
    value = statistics.median(positions)
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def render_summary(results: Results, *, results_url: str = RESULTS_URL) -> str:
    """The summary block, between `<!-- graph-bench:start -->` and `<!-- graph-bench:end -->`."""
    single = [e for e in results.entries if e.threads == 1]
    grafeo_entries = [e for e in single if e.database == GRAFEO]
    if not grafeo_entries:
        raise ValueError("the results hold no single-threaded Grafeo entries to summarise")
    # The largest scale Grafeo ran at, so a larger-scale run of another database cannot empty the table.
    scale = max((e.scale for e in grafeo_entries), key=scale_key)
    grafeo_entries = [e for e in grafeo_entries if e.scale == scale]
    ranked = [r for r in rank(results) if r.scale == scale and r.threads == 1]

    rows = []
    for category in CATEGORY_ORDER:
        if not any(e.category == category for e in grafeo_entries):
            continue
        in_row: list[Ranked] = [r for r in ranked if r.category == category]
        dbs = {db for r in in_row for db, _ in r.order}
        positions = [p for r in in_row if (p := r.position(GRAFEO)) is not None]
        fastest = sum(1 for p in positions if p == 1)
        agreed = sum(1 for e in grafeo_entries if e.category == category and _agreed(e))
        rows.append(f"| {CATEGORY_NAMES[category]} | {len(dbs)} | {fastest} / {agreed} | {_median_rank(positions)} |")

    lines = [
        "<!-- graph-bench:start -->",
        f"**graph-bench** · {grafeo_label(results)} · {machine_label(results)} · {results.started_at[:10]}"
        f" · {len(results.databases)} embedded databases",
        "",
        "| Category | DBs | Grafeo fastest | Grafeo median rank |",
        "|---|---|---|---|",
        *rows,
    ]
    # Each footer line is its own paragraph, so GitHub shows them on separate lines.
    footer = [f"Single-threaded, {scale} scale; answers checked; wrong answers not ranked."]
    marked = _marked(grafeo_entries, ranked)
    footer += [f"Grafeo: {m} on {benchmark}" for benchmark, m in marked]
    if marked:
        footer.append(LEGEND)
    footer.append(f"Full results, with every database named: {results_url}")
    for line in footer:
        lines += ["", line]
    lines.append("<!-- graph-bench:end -->")
    return "\n".join(lines) + "\n"


def _marked(entries: list[Entry], ranked: list[Ranked]) -> list[tuple[str, str]]:
    """Grafeo's own markers as (benchmark, marker), in matrix order."""
    order = {r.benchmark: i for i, r in enumerate(ranked)}
    found = [(e.benchmark, m) for e in entries if (m := marker(e.status, e.answer))]
    return sorted(found, key=lambda pair: order.get(pair[0], len(order)))
