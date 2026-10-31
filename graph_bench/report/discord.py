"""The Discord post for a Grafeo release: the trigger line, the overview per category, Grafeo's summary and links.

The output is a list of webhook bodies. Discord's "Execute Webhook" accepts `content` of up to 2000 characters
(https://discord.com/developers/docs/resources/webhook#execute-webhook), so the text is split over several
messages at category boundaries, each table inside a code block. Every body suppresses link previews (flag 4,
SUPPRESS_EMBEDS) and allows no mentions (`allowed_mentions.parse` empty), so the post cannot ping anyone.
"""

from typing import Any

from graph_bench.report.ranking import safe_display_name
from graph_bench.report.readme import overview
from graph_bench.report.summary import LEGEND, RESULTS_URL, grafeo_label, machine_label, summary_data
from graph_bench.results import Results

CONTENT_LIMIT = 2000  # characters of `content` in one webhook message
_BUDGET = 1900  # what a message body may hold, leaving headroom under the limit
_FENCE = "```"
_SUPPRESS_EMBEDS = 4


def trigger_line(results: Results) -> str:
    return f"**Grafeo {results.grafeo.get('version') or 'unknown'} released**"


def _message(content: str) -> dict[str, Any]:
    return {"content": content, "flags": _SUPPRESS_EMBEDS, "allowed_mentions": {"parse": []}}


def _pack(groups: list[list[str]], budget: int) -> list[str]:
    """Code-block texts, each within `budget` characters, filled greedily one group of lines at a time.

    A group stays whole unless it alone exceeds the budget, then it is split by line.
    """
    room = budget - 2 * len(_FENCE) - 2
    blocks: list[list[str]] = []
    size = 0
    for group in groups:
        pieces = [group]
        if len("\n".join(group)) > room:
            pieces = [[line] for line in group]
        for piece in pieces:
            need = len("\n".join(piece)) + 1
            if not blocks or size + need > room:
                blocks.append([])
                size = 0
            blocks[-1] += piece
            size += need
    return [f"{_FENCE}\n" + "\n".join(lines) + f"\n{_FENCE}" for lines in blocks]


def _overview_groups(results: Results) -> list[list[str]]:
    dbs, rows = overview(results)
    names = [safe_display_name(db) for db in dbs]
    width = max((len(n) for n in names), default=0)
    groups = []
    for label, cells in rows:
        group = [label]
        group += [f"  {name.ljust(width)}  {cell}" for name, cell in zip(names, cells, strict=True) if cell]
        groups.append(group)
    return groups


def _summary_text(results: Results) -> str:
    data = summary_data(results)
    width = max((len(r.category) for r in data.rows), default=0)
    lines = [f"Grafeo fastest, {data.scale} scale, single-threaded (k / n, median rank)"]
    lines += [f"{r.category.ljust(width)}  {r.fastest} / {r.agreed}  rank {r.median_rank}" for r in data.rows]
    lines += [f"Grafeo: {m} on {benchmark}" for benchmark, m in data.marked]
    if data.marked:
        lines.append(LEGEND)
    return f"{_FENCE}\n" + "\n".join(lines) + f"\n{_FENCE}"


def render_discord(
    results: Results, *, run_url: str | None = None, readme_url: str = RESULTS_URL
) -> list[dict[str, Any]]:
    """The webhook bodies of the post, in the order they are sent."""
    head = [
        trigger_line(results),
        f"{grafeo_label(results)} · {machine_label(results)} · {results.started_at[:10]}"
        f" · selection {results.selection}",
    ]
    if run_url:
        head.append(f"CI run: {run_url}")
    head.append(f"Results: {readme_url}")
    explain = "Overview, per category and database: ✓ k/n correct answers within the time budget, then marker counts."

    messages = [_message("\n".join(head))]
    blocks = _pack(_overview_groups(results), _BUDGET - len(explain) - 1)
    for i, block in enumerate(blocks):
        messages.append(_message(f"{explain}\n{block}" if i == 0 else block))
    messages.append(_message(_summary_text(results)))
    for message in messages:
        if len(message["content"]) > CONTENT_LIMIT:  # a single over-long line; only URLs could cause it
            message["content"] = message["content"][: CONTENT_LIMIT - 3] + "..."
    return messages
