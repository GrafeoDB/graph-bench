"""The results section of graph-bench's README: an overview, one detail table per category, the Memory and Disk
tables and a footer.

Every database is named here. Columns are Grafeo first, then the others alphabetically by display name. There
is never a combined score across categories. Memory and Disk (results.json `resources`) are not categories: they
appear in neither the overview nor the summary block.
"""

from collections import Counter
from pathlib import Path

from graph_bench.report.ranking import (
    Ranked,
    all_databases,
    benchmark_key,
    format_sig3,
    is_load,
    is_ranked,
    rank,
    safe_display_name,
    scale_key,
    speed,
    title,
)
from graph_bench.report.summary import LEGEND, grafeo_label, machine_label
from graph_bench.results import Entry, Resource, Results
from graph_bench.suites.catalog import answer_only, load_benchmark
from graph_bench.types import CATEGORY_NAMES, LOAD_CATEGORIES, Category, marker

START = "<!-- results:start -->"
END = "<!-- results:end -->"

_MARKER_ORDER = ("⊘", "✗", "≠", "⏱")
MIB = 1 << 20
IN_MEMORY = "in memory"
MEMORY_NOTE = (
    '"Loaded" is the memory the load added (after minus before); "peak" is the process tree\'s absolute peak during'
    " the load, interpreter and started engine included."
)

Section = tuple[Category, str, int | None]  # category, scale, threads


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _sections(ranked: list[Ranked]) -> dict[Section, list[Ranked]]:
    sections: dict[Section, list[Ranked]] = {}
    for r in ranked:
        sections.setdefault((r.category, r.scale, r.threads), []).append(r)
    return sections


def _overview_cell(entries: list[Entry], n: int) -> str:
    if not entries:
        return ""
    markers = Counter(m for e in entries if (m := marker(e.status, e.answer)))
    if markers["⊘"] == len(entries):
        return "⊘"
    k = sum(1 for e in entries if is_ranked(e))
    counts = [f"{m}{markers[m]}" for m in _MARKER_ORDER if markers[m]]
    return " ".join([f"✓ {k}/{n}", *counts])


# The detail cell of an answer-only item whose answer agrees: it has no time to show.
ANSWER_ONLY_CELL = "✓"
ANSWER_ONLY_NOTE = f"{ANSWER_ONLY_CELL}: an answer-check-only case agrees; it has no time budget and is not ranked."


def _detail_cell(entry: Entry | None, r: Ranked, mode: str) -> str:
    if entry is None:
        return ""
    if not is_ranked(entry):
        return marker(entry.status, entry.answer)
    if answer_only(entry.benchmark):
        return ANSWER_ONLY_CELL
    value = speed(entry, mode)
    if value is None:
        return ""
    text = format_sig3(value / (1e9 if is_load(entry) else 1e6))
    return f"**{text}**" if entry.database in r.fastest() else text


def _section_label(section: Section, *, with_scale: bool) -> str:
    category, scale, threads = section
    label = CATEGORY_NAMES[category]
    if with_scale and category != "startup":  # Startup has one scale of its own, the empty database
        label += f", {scale}"
    if threads is None:
        label += ", all cores"
    return label


def _overview_rows(
    sections: dict[Section, list[Ranked]],
    by_key: dict[tuple[str, str, int | None, str], Entry],
    dbs: list[str],
    *,
    with_scale: bool,
) -> list[tuple[str, list[str]]]:
    """One (label, cell per database) pair per section; the cells are `✓ k/n` and the marker counts."""
    rows = []
    for section, members in sections.items():
        cells = []
        for db in dbs:
            entries = [e for r in members if (e := by_key.get((r.benchmark, r.scale, r.threads, db))) is not None]
            cells.append(_overview_cell(entries, len(members)))
        rows.append((_section_label(section, with_scale=with_scale), cells))
    return rows


def overview(results: Results) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The README overview as data: the databases (Grafeo first) and, per section, its label and one cell each."""
    dbs = all_databases(results)
    sections = _sections(rank(results))
    by_key = {(e.benchmark, e.scale, e.threads, e.database): e for e in results.entries}
    scales = {scale for category, scale, _ in sections if category != "startup"}
    return dbs, _overview_rows(sections, by_key, dbs, with_scale=len(scales) > 1)


def render_readme(results: Results) -> str:
    """The text that goes between the README's results markers."""
    dbs = all_databases(results)
    names = [safe_display_name(db) for db in dbs]
    ranked = rank(results)
    sections = _sections(ranked)
    by_key: dict[tuple[str, str, int | None, str], Entry] = {
        (e.benchmark, e.scale, e.threads, e.database): e for e in results.entries
    }
    scales = list(dict.fromkeys(scale for category, scale, _ in sections if category != "startup"))
    with_scale = len(scales) > 1

    lines = ["### Overview", ""]
    setting = ["Single-threaded"] if all(threads == 1 for _, _, threads in sections) else []
    if not with_scale and scales:
        setting.append(f"{scales[0]} scale")
    if setting:
        lines += [", ".join(setting) + ".", ""]
    lines += [
        "✓ k/n: correct answers within the time budget, out of the category's n benchmarks; then the count of"
        " each marker.",
        "",
    ]
    overview_rows = [[label, *cells] for label, cells in _overview_rows(sections, by_key, dbs, with_scale=with_scale)]
    lines += _table(["Category", *names], overview_rows)

    for section, members in sections.items():
        category, scale, threads = section
        unit = "s" if category in LOAD_CATEGORIES else "ms"
        cores = ", all cores" if threads is None else ""
        where = "empty database" if category == "startup" else f"{scale} scale"
        heading = f"### {CATEGORY_NAMES[category]} ({where}{cores}, {unit})"
        rows = [
            [r.title, *(_detail_cell(by_key.get((r.benchmark, r.scale, r.threads, db)), r, results.mode) for db in dbs)]
            for r in members
        ]
        lines += ["", heading, "", *_table(["Benchmark", *names], rows)]

    lines += _resource_tables(results, dbs, names)

    lines += ["", LEGEND]
    if any(answer_only(e.benchmark) for e in results.entries):
        lines += ["", ANSWER_ONLY_NOTE]
    lines += ["", _run_line(results)]
    if results.mode == "codspeed":
        note = "Times are single preflight runs; CodSpeed holds the measured timings."
        if results.resources:
            note += " Memory and disk are graph-bench's own readings: CodSpeed holds no memory or disk data."
        lines += ["", note]
    lines += ["", *_table(["Database", "Version", "Process", "Storage", "Threads"], _database_rows(results, dbs))]
    return "\n".join(lines) + "\n"


def _resource_ranked(r: Resource) -> bool:
    """A record whose numbers count: measured (`ok`) with a count check that did not disagree."""
    return r.status == "ok" and r.answer != "disagree"


def _resource_row(label: str, records: dict[str, Resource], dbs: list[str], value: str) -> list[str]:
    """One row: `value` (a `Resource` field, bytes) in MiB per database, the lowest bold; markers for records that
    did not measure, and "in memory" for a missing disk size of an in-memory store."""
    numbers: dict[str, int] = {}
    for db, r in records.items():
        number = getattr(r, value)
        if _resource_ranked(r) and number is not None:
            numbers[db] = number
    lowest = min(numbers.values(), default=None)
    cells = []
    for db in dbs:
        r = records.get(db)
        if r is None:
            cells.append("")
        elif (mark := marker(r.status, r.answer)) != "":
            cells.append(mark)
        elif db in numbers:
            text = format_sig3(numbers[db] / MIB)
            cells.append(f"**{text}**" if numbers[db] == lowest else text)
        else:
            cells.append(IN_MEMORY if value == "disk_bytes" and r.storage == "in-memory" else "")
    return [label, *cells]


def _resource_tables(results: Results, dbs: list[str], names: list[str]) -> list[str]:
    """Per scale (and thread setting): "### Memory (<scale> scale, MiB)", a loaded and a peak row per dataset and
    store, then "### Disk (<scale> scale, MiB)", one row per dataset and store."""
    sections: dict[tuple[str, int | None], dict[str, dict[str, Resource]]] = {}
    for r in results.resources:
        load = load_benchmark(r.dataset, r.store)
        sections.setdefault((r.scale, r.threads), {}).setdefault(load, {})[r.database] = r
    lines: list[str] = []
    for (scale, threads), loads in sorted(sections.items(), key=lambda kv: (scale_key(kv[0][0]), kv[0][1] is None)):
        where = f"{scale} scale" + (", all cores" if threads is None else "")
        order = sorted(loads, key=benchmark_key)
        memory = []
        for load in order:
            memory.append(_resource_row(f"{title(load)}, loaded", loads[load], dbs, "footprint_bytes"))
            memory.append(_resource_row(f"{title(load)}, peak during load", loads[load], dbs, "peak_bytes"))
        disk = [_resource_row(title(load), loads[load], dbs, "disk_bytes") for load in order]
        lines += ["", f"### Memory ({where}, MiB)", "", *_table(["Dataset", *names], memory), "", MEMORY_NOTE]
        lines += ["", f"### Disk ({where}, MiB)", "", *_table(["Dataset", *names], disk)]
    return lines


def _run_line(results: Results) -> str:
    return " · ".join(
        [
            f"graph-bench {results.graph_bench_commit[:7]}",
            grafeo_label(results),
            machine_label(results),
            results.started_at[:10],
            f"selection {results.selection}",
            f"{results.mode} mode",
        ]
    )


def _database_rows(results: Results, dbs: list[str]) -> list[list[str]]:
    rows = []
    for db in dbs:
        info = results.databases.get(db)
        if info is None:
            rows.append([safe_display_name(db), "", "", "", ""])
            continue
        version = info.version + (f" (Java {info.java})" if info.java else "")
        rows.append([safe_display_name(db), version, info.process, info.storage, info.threads_note])
    return rows


def update_readme(path: Path, body: str) -> None:
    """Replace the text between the README's results markers with `body`; raise ValueError without them."""
    text = path.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END, start + len(START)) if start != -1 else -1
    if start == -1 or end == -1:
        raise ValueError(f"{path} needs a {START} line followed by a {END} line")
    head, tail = text[: start + len(START)], text[end:]
    inner = body.strip("\n")
    path.write_text(f"{head}\n{inner}\n{tail}", encoding="utf-8", newline="\n")
